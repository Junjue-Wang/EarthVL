"""Merge the predictions of the two EarthVLNet branches and compute the multiple-choice VQA metrics.

Inputs are the prediction files written by the evaluation of each branch (``prediction-<step>.json``)::

    {"<image>": [{"Type": ..., "Question": ..., "Answer": <prediction>, "Correct": ..., "GT_Answer": ...}, ...]}

* LLM branch (``--cls_pred``): EarthVLNet, answers every question type with a sentence.
* Counting branch (``--count_pred``): EarthVLCount, predicts the number of the counting questions.

Merging rule (per image and question):

* Basic Counting / Reasoning-based Counting: the counting-branch answer;
* Comprehensive Analysis questions answered by the counting branch (the two traffic questions): the LLM sentence
  with its numbers replaced by the counting-branch number;
* everything else: the LLM answer.

The merged answers are scored with the accuracy of each question type and the overall accuracy (OA), and with the
RMSE of the counting answers (``module/metric.py``). Only numpy and prettytable are required.

QAs without a ground-truth answer (``GT_Answer`` ''), e.g. of the EarthVQA Test split, whose labels are withheld for
the EarthVQA challenge on Codabench, are merged but not scored; their number is reported after the tables.

Usage::

    python evaluate.py [--cls_pred PATH] [--count_pred PATH] [--save_merged PATH]
"""
import argparse
import json
import sys

from data.vocab import COUNT_QUESTIONS, COUNT_TYPES, QUESTIONS, QUESTION_TYPES
from module.metric import Count_RMSE_MetricV2, VQA_OA_Metric, find_pos_index, skipped_qas_note


def merge_predictions(pr_cls_dict, pr_count_dict):
    """Combine the LLM-branch and counting-branch predictions into {image: [{Type, Question, Answer, GT_Answer}]}."""
    missing = [fname for fname in pr_cls_dict if fname not in pr_count_dict]
    if missing:
        raise KeyError(f'{len(missing)} images of the LLM-branch predictions have no counting-branch predictions, '
                       f'e.g. {missing[0]!r}')
    qa_dict = dict()
    for fname, qas_list in pr_cls_dict.items():
        qas = []
        count_qas_list = pr_count_dict[fname]
        fname_ques_stack_name = dict()  # questions of this image that are already merged
        for qa in qas_list:
            type_, ques, answer, gt_ans = qa['Type'], qa['Question'], qa['Answer'], qa['GT_Answer']
            for count_qa in count_qas_list:
                count_ques, count_ans, count_type, gt_count_ans = \
                    count_qa['Question'], count_qa['Answer'], count_qa['Type'], count_qa['GT_Answer']
                if count_ques == ques and not fname_ques_stack_name.get(count_ques, False):
                    # the counting branch answers this question: put its number into the LLM sentence
                    fname_ques_stack_name[count_ques] = True
                    final_answer = find_pos_index(answer, '#')[1].replace('#', count_ans)
                    qas.append(dict(Type=type_, Question=ques, Answer=final_answer, GT_Answer=gt_ans))

                if not fname_ques_stack_name.get(count_ques, False) and count_type != 'Comprehensive Analysis':
                    # counting question: the counting-branch answer
                    fname_ques_stack_name[count_ques] = True
                    qas.append(dict(Type=count_type, Question=count_ques, Answer=count_ans, GT_Answer=gt_count_ans))

                if not fname_ques_stack_name.get(ques, False) and type_ != 'Comprehensive Analysis':
                    # any other question: the LLM answer
                    fname_ques_stack_name[ques] = True
                    qas.append(dict(Type=type_, Question=ques, Answer=answer, GT_Answer=gt_ans))

        qa_dict[fname] = qas
    return qa_dict


def score(qa_dict, logger=None):
    """Print (or log) the accuracy and RMSE tables of the merged predictions and return the four metric objects.

    QAs without a ground-truth answer are skipped by every metric (``metric.num_skipped``) and reported after the
    tables.
    """
    metric = VQA_OA_Metric(QUESTION_TYPES, logger=logger)
    detail_metric = VQA_OA_Metric(QUESTIONS, logger=logger)
    count_metric = Count_RMSE_MetricV2(COUNT_TYPES, logger=logger)
    detail_count_metric = Count_RMSE_MetricV2(COUNT_QUESTIONS, logger=logger)

    for qas_list in qa_dict.values():
        for qa in qas_list:
            type_, ques, answer, gt_ans = qa['Type'], qa['Question'], qa['Answer'], qa['GT_Answer']
            metric([answer], [gt_ans], [type_])
            detail_metric([answer], [gt_ans], [ques])
            count_metric([answer], [gt_ans], [type_], [ques])
            detail_count_metric([answer], [gt_ans], [ques], [ques])

    metric.summary()
    detail_metric.summary()
    count_metric.summary()
    detail_count_metric.summary()
    if metric.num_skipped:
        note = skipped_qas_note(metric.num_skipped, metric.num_skipped + sum(metric.cls_total_num))
        if logger is None:
            print('\n' + note)
        else:
            logger.warning(note)
    return metric, detail_metric, count_metric, detail_count_metric


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Merge the LLM-branch and counting-branch predictions of '
                                                 'EarthVLNet and report accuracy / RMSE.')
    # defaults: where scripts/eval_cls.sh and scripts/eval_count.sh (or training) write the predictions
    parser.add_argument('--cls_pred', default='./log/EarthVLNet/prediction-40000.json',
                        help='prediction JSON of the LLM branch (EarthVLNet)')
    parser.add_argument('--count_pred', default='./log/EarthVLCount/prediction-40000.json',
                        help='prediction JSON of the counting branch (EarthVLCount)')
    parser.add_argument('--save_merged', default=None,
                        help='optional path to write the merged predictions as JSON')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # the RMSE tables print U+2193 ('↓RMSE'); do not fail on consoles with a legacy code page
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')

    with open(args.cls_pred, encoding='utf-8') as f:
        pr_cls_dict = json.load(f)
    with open(args.count_pred, encoding='utf-8') as f:
        pr_count_dict = json.load(f)

    qa_dict = merge_predictions(pr_cls_dict, pr_count_dict)
    print(f'Merged {sum(len(v) for v in qa_dict.values())} QAs of {len(qa_dict)} images.')
    if args.save_merged:
        with open(args.save_merged, 'w', encoding='utf-8') as f:
            f.write(json.dumps(qa_dict, ensure_ascii=False, indent=1))
        print(f'Merged predictions written to {args.save_merged}')

    score(qa_dict)


if __name__ == '__main__':
    main()
