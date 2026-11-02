"""evaluate.py: merging the two branches' predictions and scoring them (only numpy + prettytable needed), and the
evaluation of train_generation.py on QAs without a ground-truth answer."""
import json
import math
import os
import subprocess
import sys

import pytest

TRAFFIC_Q = 'What are the comprehensive traffic situations in this scene?'
LAND_USE_Q = 'What are the land use types in this scene?'


def qa(type_, question, answer, gt):
    return dict(Type=type_, Question=question, Answer=answer, Correct=answer == gt, GT_Answer=gt)


def tiny_predictions():
    """LLM-branch and counting-branch prediction files for two images, as written by train_generation.py."""
    cls = {
        '4191.png': [
            qa('Basic Judging', 'Are there any roads in this scene?', 'Yes.', 'Yes.'),
            qa('Direction Analysis', 'What are the directions of the main roads?',
               'Some roads run east-west.', 'Some roads run north-south.'),
            qa('Reasoning-based Counting', 'How many intersections are in this scene?', '9.', '3.'),  # replaced
            qa('Comprehensive Analysis', TRAFFIC_Q, 'This is an important traffic area with 9 intersections.',
               'This is an important traffic area with 3 intersections.'),
            qa('Comprehensive Analysis', LAND_USE_Q, 'There are residential areas.', 'There are residential areas.'),
        ],
        '6001.png': [
            qa('Basic Counting', 'What is the area ratio of water?', '0%-10%.', '10%-20%.'),  # replaced
            qa('Reasoning-based Judging', 'Are there any bridges in this scene?', 'No.', 'Yes.'),
        ],
    }
    count = {
        '4191.png': [
            qa('Reasoning-based Counting', 'How many intersections are in this scene?', '3', '3.'),
            qa('Comprehensive Analysis', TRAFFIC_Q, '3', 'This is an important traffic area with 3 intersections.'),
            qa('Comprehensive Analysis', LAND_USE_Q, '1', 'There are residential areas.'),
        ],
        '6001.png': [
            qa('Basic Counting', 'What is the area ratio of water?', '10%-20%', '10%-20%.'),
        ],
    }
    return cls, count


def test_merge_rules():
    from evaluate import merge_predictions
    cls, count = tiny_predictions()
    merged = merge_predictions(cls, count)
    answers = {(img, q['Question']): q for img, qas in merged.items() for q in qas}
    assert len(answers) == sum(len(v) for v in merged.values()) == 7  # every question exactly once
    # counting questions: the counting-branch answer
    assert answers['4191.png', 'How many intersections are in this scene?']['Answer'] == '3'
    assert answers['6001.png', 'What is the area ratio of water?']['Answer'] == '10%-20%'
    # traffic question: the LLM sentence with the counting-branch number
    assert answers['4191.png', TRAFFIC_Q]['Answer'] == 'This is an important traffic area with 3 intersections.'
    # other questions: the LLM answer (a sentence without numbers is left as is)
    assert answers['4191.png', LAND_USE_Q]['Answer'] == 'There are residential areas.'
    assert answers['4191.png', 'What are the directions of the main roads?']['Answer'] == 'Some roads run east-west.'
    assert answers['6001.png', 'Are there any bridges in this scene?']['Answer'] == 'No.'

    with pytest.raises(KeyError):
        merge_predictions(cls, {'4191.png': count['4191.png']})


def test_score_tables(capsys):
    from evaluate import merge_predictions, score
    cls, count = tiny_predictions()
    metric, detail_metric, count_metric, detail_count_metric = score(merge_predictions(cls, count))
    # correct: BJ, RC (3), CA traffic, CA land use, BC (10%-20%); wrong: DirA, RJ -> OA 5/7
    assert sum(metric.cls_true_num) == 5 and sum(metric.cls_total_num) == 7
    assert metric.num_skipped == count_metric.num_skipped == 0
    out = capsys.readouterr().out
    assert 'OA' in out and 'Overall_RMSE' in out and 'Overall_RSME' not in out
    assert 'no ground-truth answer' not in out
    # all counted numbers are right -> RMSE 0
    assert all(g == p for gs, ps in zip(count_metric.gts, count_metric.preds) for g, p in zip(gs, ps))


def unlabelled(predictions, image):
    """`predictions` with the ground truth of `image` removed, as written for the withheld EarthVQA Test labels."""
    out = json.loads(json.dumps(predictions))
    for q in out[image]:
        q.update(GT_Answer='', Correct=None)
    return out


def test_metrics_skip_qas_without_ground_truth(capsys):
    from data.vocab import COUNT_TYPES, QUESTION_TYPES
    from module.metric import Count_RMSE_MetricV2, VQA_OA_Metric, has_ground_truth
    assert not any(has_ground_truth(gt) for gt in ('', ' ', None))
    assert all(has_ground_truth(gt) for gt in ('0.', 'No.', 0))

    metric = VQA_OA_Metric(QUESTION_TYPES)
    metric(['Yes.', 'Yes.', '3.'], ['Yes.', '', None], ['Basic Judging', 'Basic Judging', 'Reasoning-based Counting'])
    assert metric.cls_true_num[0] == metric.cls_total_num[0] == 1 and sum(metric.cls_total_num) == 1
    assert metric.num_skipped == 2
    count_metric = Count_RMSE_MetricV2(COUNT_TYPES)
    count_metric(['3.', '5.'], ['4.', ''], ['Reasoning-based Counting'] * 2,
                 ['How many intersections are in this scene?'] * 2)
    assert count_metric.gts[1] == [3] and count_metric.preds[1] == [4] and count_metric.num_skipped == 1

    # nothing left to score: the tables show None
    capsys.readouterr()
    empty = VQA_OA_Metric(QUESTION_TYPES)
    empty(['Yes.'], [''], ['Basic Judging'])
    empty.summary()
    empty_count = Count_RMSE_MetricV2(COUNT_TYPES)
    empty_count(['3.'], [''], ['Basic Counting'], ['What is the area ratio of water?'])
    empty_count.summary()
    rows = {line.split('|')[1].strip(): line.split('|')[2].strip()
            for line in capsys.readouterr().out.splitlines() if line.startswith('|')}
    assert rows['OA'] == rows['Overall_RMSE'] == 'None'


def test_score_reports_qas_without_ground_truth(capsys):
    from evaluate import merge_predictions, score
    from module.metric import EARTHVQA_CHALLENGE_URL
    cls, count = tiny_predictions()
    merged = merge_predictions(unlabelled(cls, '4191.png'), unlabelled(count, '4191.png'))
    assert sum(len(v) for v in merged.values()) == 7  # still merged
    assert all(q['GT_Answer'] == '' for q in merged['4191.png'])
    metric, detail_metric, count_metric, detail_count_metric = score(merged)
    # only 6001.png is scored: BC right, RJ wrong
    assert sum(metric.cls_true_num) == 1 and sum(metric.cls_total_num) == 2
    assert metric.num_skipped == detail_metric.num_skipped == count_metric.num_skipped == 5
    assert detail_count_metric.num_skipped == 5
    assert count_metric.gts[0] == [12] and not count_metric.gts[1] and not count_metric.gts[2]
    out = capsys.readouterr().out
    assert '5 of 7 QAs have no ground-truth answer' in out and EARTHVQA_CHALLENGE_URL in out


def test_cli(tmp_path, repo_root):
    cls, count = tiny_predictions()
    (tmp_path / 'cls.json').write_text(json.dumps(cls), encoding='utf-8')
    (tmp_path / 'count.json').write_text(json.dumps(count), encoding='utf-8')
    merged_path = tmp_path / 'merged.json'
    proc = subprocess.run([sys.executable, 'evaluate.py', '--cls_pred', str(tmp_path / 'cls.json'),
                           '--count_pred', str(tmp_path / 'count.json'), '--save_merged', str(merged_path)],
                          cwd=repo_root, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    assert proc.returncode == 0, proc.stderr
    assert 'Merged 7 QAs of 2 images.' in proc.stdout
    assert json.loads(merged_path.read_text(encoding='utf-8')).keys() == {'4191.png', '6001.png'}


def test_cli_without_ground_truth(tmp_path, capsys):
    import evaluate
    cls, count = tiny_predictions()
    (tmp_path / 'cls.json').write_text(json.dumps(unlabelled(cls, '4191.png')), encoding='utf-8')
    (tmp_path / 'count.json').write_text(json.dumps(unlabelled(count, '4191.png')), encoding='utf-8')
    merged_path = tmp_path / 'merged.json'
    evaluate.main(['--cls_pred', str(tmp_path / 'cls.json'), '--count_pred', str(tmp_path / 'count.json'),
                   '--save_merged', str(merged_path)])
    out = capsys.readouterr().out
    assert 'Merged 7 QAs of 2 images.' in out and '5 of 7 QAs have no ground-truth answer' in out
    merged = json.loads(merged_path.read_text(encoding='utf-8'))
    assert [q['GT_Answer'] for q in merged['4191.png']] == [''] * 5


def test_evaluate_cls_fn_writes_every_prediction(vqa_data, unlabelled_test_qa, tmp_path, caplog):
    """train_generation.py: QAs without a ground-truth answer are predicted and written, but not scored."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('albumentations')
    import logging
    import train_generation
    from data.earthvl_mc import EarthVLDataset
    from engine.trainer import EngineContext
    from module.metric import EARTHVQA_CHALLENGE_URL

    qa_path, unlabelled_images = unlabelled_test_qa
    select_type = ['Basic Judging', 'Reasoning-based Judging']
    dataset = EarthVLDataset(qa_path=qa_path, image_load=False, mask_load=False, feat_load=False, encoding=False,
                             training=False, select_type=select_type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    class YesModel(torch.nn.Module):
        def forward(self, x, ret):
            return ['Yes.'] * len(ret['question'])

    logger = logging.Logger('test_evaluate_cls_fn')  # like the engine's logger, not attached to the root logger
    logger.addHandler(caplog.handler)
    ctx = EngineContext(model=YesModel(), model_dir=str(tmp_path / 'run'), global_step=7, logger=logger, is_main=True)
    train_generation.evaluate_cls_fn(ctx, loader)

    with open(tmp_path / 'run' / 'prediction-7.json', encoding='utf-8') as f:
        pred = json.load(f)
    qas = [(image, q) for image, image_qas in pred.items() for q in image_qas]
    assert len(qas) == len(dataset) == len(select_type) * len(vqa_data.images['Test'])
    for image, q in qas:
        assert q['Answer'] == 'Yes.'
        if image in unlabelled_images:
            assert q['GT_Answer'] == '' and q['Correct'] is None
        else:
            assert q['Correct'] == (q['GT_Answer'] == 'Yes.')
    num_unlabelled = len(select_type) * len(unlabelled_images)
    note = [r.getMessage() for r in caplog.records if 'no ground-truth answer' in r.getMessage()]
    assert len(note) == 1 and note[0].startswith(f'{num_unlabelled} of {len(dataset)} QAs')
    assert EARTHVQA_CHALLENGE_URL in note[0]


def test_open_ended_metric_table(capsys):
    from module.metric import Metric
    Metric().summary()
    header = capsys.readouterr().out.strip().splitlines()[1]
    assert header.split('|')[1].strip() == 'Metric'


def test_cli_defaults_match_eval_scripts(repo_root):
    """scripts/eval_cls.sh and eval_count.sh write where evaluate.py (scripts/test.sh) reads by default."""
    from evaluate import parse_args
    args = parse_args([])
    for script, default in (('eval_cls.sh', args.cls_pred), ('eval_count.sh', args.count_pred)):
        text = open(os.path.join(repo_root, 'scripts', script), encoding='utf-8').read()
        model_dir = text.split('model_dir=${2:-')[1].split('}')[0]
        assert os.path.normpath(default) == os.path.normpath(os.path.join(model_dir, 'prediction-40000.json'))


@pytest.mark.skipif(not (os.environ.get('EARTHVL_CLS_PRED') and os.environ.get('EARTHVL_COUNT_PRED')),
                    reason="set EARTHVL_CLS_PRED and EARTHVL_COUNT_PRED to the authors' unreleased prediction files")
def test_paper_numbers():
    """The authors' stored predictions reproduce the EarthVLNet row of the paper (OA 81.06, overall RMSE 2.340).

    They are not released: they contain the withheld EarthVQA Test labels.
    """
    from evaluate import merge_predictions, score
    from module.metric import get_rmse
    with open(os.environ['EARTHVL_CLS_PRED'], encoding='utf-8') as f:
        cls = json.load(f)
    with open(os.environ['EARTHVL_COUNT_PRED'], encoding='utf-8') as f:
        count = json.load(f)
    metric, _, count_metric, _ = score(merge_predictions(cls, count))
    assert metric.num_skipped == 0  # every stored QA has its ground truth
    oa = sum(metric.cls_true_num) / sum(metric.cls_total_num)
    rmse = get_rmse(sum(count_metric.gts, []), sum(count_metric.preds, []))
    assert math.isclose(oa, 0.8105819896982495, abs_tol=1e-9)
    assert math.isclose(rmse, 2.3406364894414913, abs_tol=1e-9)
