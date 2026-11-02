"""Train or evaluate the two VQA branches of EarthVLNet on EarthVLSet (multiple-choice questions).

* LLM branch, ``configs/EarthVLNet.py`` (judging and analysis questions), see ``scripts/train_generation.sh``;
* counting branch, ``configs/EarthVLCount.py`` (counting questions), see ``scripts/train_count.sh``.

Training (DeepSpeed)::

    deepspeed --include localhost:0,1,2,3 train_generation.py --config_path=EarthVLNet --model_dir=./log/EarthVLNet \
        train.num_iters 40000 data.train.params.batch_size 6

Evaluation of a checkpoint (``scripts/eval_cls.sh``, ``scripts/eval_count.sh``)::

    python train_generation.py --config_path=EarthVLCount --model_dir=./log/EarthVLCount \
        --eval_only --ckpt ./log/EarthVLCount/model-40000.pth

Both write ``<model_dir>/prediction-<step>.json``; ``evaluate.py`` merges the predictions of the two branches into
the final answers and scores them. Trailing ``key value`` pairs override config entries (see ``engine/trainer.py``).
"""
import json
import os

import torch
from tqdm import tqdm

import ever as er
from ever.core.dist import all_gather, synchronize

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
er.registry.register_all()  # imports ./data, ./module and ./callback (run from the repository root)

from data.earthvl_mc import EarthVLDataset  # noqa: E402
from data.vocab import COUNT_QUESTIONS, COUNT_TYPES  # noqa: E402
from module.metric import (Count_RMSE_MetricV2, VQA_OA_Metric, equal_str, has_ground_truth,  # noqa: E402
                           skipped_qas_note)


def res_all_gather(inputs):
    """Gather lists (questions, answers, predictions, ...) from all ranks and concatenate them in rank order."""
    assert isinstance(inputs, list)
    outputs = []
    for data in inputs:
        res_data = []
        for x in all_gather(data):
            res_data += x
        outputs.append(res_data)
    synchronize()
    return outputs


def evaluate_cls_fn(ctx, test_dataloader):
    """Evaluate on ``data.test``: accuracy per question type / question, counting RMSE, and the prediction file.

    Runs on every rank; predictions are gathered from all ranks, metrics are logged and
    ``<model_dir>/prediction-<step>.json`` ({image: [{Type, Question, Answer, Correct, GT_Answer}]}) is written by
    rank 0. Works for both branches (the model returns one answer string per question).

    QAs without a ground-truth answer (the EarthVQA Test labels are withheld for the EarthVQA challenge on Codabench)
    are predicted and written with ``GT_Answer`` '' and ``Correct`` null, but not scored; their number is logged.
    """
    model = ctx.model
    model.eval()
    pred_save_path = os.path.join(ctx.model_dir, f'prediction-{ctx.global_step}.json')
    pred_dict = dict()
    metric = VQA_OA_Metric(EarthVLDataset.QUESTION_TYPES, logger=ctx.logger)
    detail_metric = VQA_OA_Metric(EarthVLDataset.QUESTIONS, logger=ctx.logger)
    count_metric = Count_RMSE_MetricV2(COUNT_TYPES, logger=ctx.logger)
    detail_count_metric = Count_RMSE_MetricV2(COUNT_QUESTIONS, logger=ctx.logger)
    with torch.no_grad():
        for img, ret in tqdm(test_dataloader, disable=not ctx.is_main):
            ques, questypes, ans, imagen = ret['question'], ret['questype'], ret['answer'], ret['imagen']
            pred = model(img, ret)
            ques, questypes, ans, imagen, pred = res_all_gather([ques, questypes, ans, imagen, pred])
            if not ctx.is_main:
                continue
            metric(pred, ans, questypes)
            detail_metric(pred, ans, ques)
            count_metric(pred, ans, questypes, ques)
            detail_count_metric(pred, ans, ques, ques)

            for q_i_str, qt_i, ans_i, imagen_i, gt_i in zip(ques, questypes, pred, imagen, ans):
                qa_list = pred_dict.get(imagen_i, [])
                instance_qa = dict()
                instance_qa['Type'] = qt_i
                instance_qa['Question'] = q_i_str
                instance_qa['Answer'] = ans_i
                instance_qa['Correct'] = equal_str(ans_i, gt_i) if has_ground_truth(gt_i) else None
                instance_qa['GT_Answer'] = gt_i
                qa_list.append(instance_qa)
                pred_dict[imagen_i] = qa_list

    if ctx.is_main:
        os.makedirs(ctx.model_dir, exist_ok=True)
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(pred_dict, ensure_ascii=False, indent=1))
        ctx.logger.info(f'predictions written to {pred_save_path}')
        metric.summary()
        detail_metric.summary()
        count_metric.summary()
        detail_count_metric.summary()
        if metric.num_skipped:
            ctx.logger.warning(skipped_qas_note(metric.num_skipped, metric.num_skipped + sum(metric.cls_total_num)))
    synchronize()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from engine.trainer import main

    main(evaluate_fn=evaluate_cls_fn)
