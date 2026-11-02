"""Evaluation metrics of EarthVLSet.

* ``VQA_OA_Metric``: accuracy per question type / question and overall accuracy (OA) of multiple-choice answers.
* ``Count_RMSE_MetricV2``: RMSE of the numbers in counting answers (the EarthVL paper protocol).
* ``Metric``: BLEU-1..4, ROUGE-L and CIDEr for open-ended answers (needs nltk, imported on first use). With the
  open-ended loader (``data/earthvl_open.py``) and the coco-caption code in ``score/``, it is an evaluation tool for
  the open-ended split of EarthVL-GLOBAL; this release contains no open-ended model, config or training script.

The multiple-choice metrics skip QAs without a ground-truth answer ('' or None) and count them in ``num_skipped``:
the EarthVQA Test labels are withheld for the EarthVQA challenge on Codabench (``EARTHVQA_CHALLENGE_URL``).

Importing this module only requires numpy and prettytable.
"""
import math
import re

import numpy as np
import prettytable as pt


class Metric(object):
    """Corpus-level BLEU-1..4 (nltk), ROUGE-L and CIDEr (coco-caption code in score/) of open-ended answers.

    ``references``: per sample, a list of tokenised reference answers; ``hypotheses``: per sample, the tokenised
    prediction (the input format of ``nltk.translate.bleu_score.corpus_bleu``).
    """

    def __init__(self, n=[1, 2, 3, 4], logger=None):
        self.n = n
        self.bleu_scores = [0 for _ in range(len(self.n))]
        self.logger = logger
        self.smoothing_function = None
        self.rouge_score = 0
        self.cider_score = 0

    def __call__(self, references, hypotheses):
        from nltk.translate.bleu_score import corpus_bleu  # nltk is only needed for open-ended evaluation
        from score.cider import Cider
        from score.rouge import Rouge

        self.bleu_scores[0] = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                                          smoothing_function=self.smoothing_function)
        self.bleu_scores[1] = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                                          smoothing_function=self.smoothing_function)
        self.bleu_scores[2] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                                          smoothing_function=self.smoothing_function)
        self.bleu_scores[3] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                                          smoothing_function=self.smoothing_function)
        hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
        ref = [[' '.join(reft) for reft in reftmp] for reftmp in
               [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
        self.rouge_score, _ = Rouge().compute_score(ref, hypo)
        self.cider_score, _ = Cider().compute_score(ref, hypo)

    def summary(self):
        tb = pt.PrettyTable()
        tb.field_names = ['Metric', 'Score']
        for idx, bleu_score in enumerate(self.bleu_scores):
            tb.add_row([f'Bleu{idx+1}', bleu_score])
        tb.add_row(['ROUGE_L', self.rouge_score])
        tb.add_row(['CIDER', self.cider_score])

        if self.logger is None:
            print('\n' + tb.get_string())
        else:
            self.logger.info('\n' + tb.get_string())


def equal_str(pr_str: str, gt_str: str):
    """An answer is correct when its set of sentences equals that of the ground truth (case and order ignored)."""
    pr_str_set = set(pr_str.lower().strip('.').split('. '))
    gt_str_set = set(gt_str.lower().strip('.').split('. '))

    return pr_str_set == gt_str_set


# The EarthVQA Test labels are withheld for this challenge, which scores predictions of the EarthVQA Test QAs.
EARTHVQA_CHALLENGE_URL = 'https://www.codabench.org/competitions/2922/'


def has_ground_truth(answer):
    """False for a QA without a ground-truth answer ('' or None), e.g. of the EarthVQA Test split."""
    return answer is not None and str(answer).strip() != ''


def skipped_qas_note(num_skipped, num_total):
    """Report on the QAs that the metrics skipped because they have no ground-truth answer."""
    return (f'{num_skipped} of {num_total} QAs have no ground-truth answer and are not included in the tables above. '
            f'The EarthVQA Test labels are withheld for the EarthVQA Visual Question Answering challenge, which '
            f'scores those QAs: {EARTHVQA_CHALLENGE_URL}')


class VQA_OA_Metric(object):
    def __init__(self, ques_classes: list, logger=None):
        self.ques_classes = ques_classes
        self.logger = logger
        self.cls_true_num = [0 for _ in range(len(self.ques_classes))]
        self.cls_total_num = [0 for _ in range(len(self.ques_classes))]
        self.num_skipped = 0  # QAs without a ground-truth answer

    def __call__(self, pred: np.array, gt: np.array, questype):
        for ques_t, pr_i, gt_i in zip(questype, pred, gt):
            if not has_ground_truth(gt_i):
                self.num_skipped += 1
                continue
            cls_idx = self.ques_classes.index(ques_t)
            self.cls_total_num[cls_idx] += 1
            self.cls_true_num[cls_idx] += equal_str(str(pr_i), str(gt_i))

    def summary(self):
        tb = pt.PrettyTable()
        tb.field_names = ['Classes', 'Acc']
        for idx in range(len(self.cls_true_num)):
            if self.cls_total_num[idx] != 0:
                tb.add_row([self.ques_classes[idx], self.cls_true_num[idx] / self.cls_total_num[idx]])
            else:
                tb.add_row([self.ques_classes[idx], None])

        total = sum(self.cls_total_num)
        tb.add_row(['OA', sum(self.cls_true_num) / total if total else None])

        if self.logger is None:
            print('\n' + tb.get_string())
        else:
            self.logger.info('\n' + tb.get_string())


def get_rmse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        if len(records_real) == 0:
            return None
        else:
            return math.sqrt(sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real))
    else:
        return None


# Numerical answer strings -> index used by the counting branch and by the RMSE:
# 0-10 counts, 11-20 area-ratio bins, 21 '< = 5', 22 '> 5', 23 'more than 10'.
# The keys are searched in this order by find_pos_index.
Count_TABLE = {
    'More than 10.': 23, 'more than 10': 23,
    '0%-10%.': 11, '0%-10%': 11,
    '10%-20%.': 12, '10%-20%': 12,
    '20%-30%.': 13, '20%-30%': 13,
    '30%-40%.': 14, '30%-40%': 14,
    '40%-50%.': 15, '40%-50%': 15,
    '50%-60%.': 16, '50%-60%': 16,
    '60%-70%.': 17, '60%-70%': 17,
    '70%-80%.': 18, '70%-80%': 18,
    '80%-90%.': 19, '80%-90%': 19,
    '90%-100%.': 20, '90%-100%': 20,
    '< = 5.': 21, '> 5.': 22,
    '< = 5': 21, '> 5': 22,
    '10.': 10, '10': 10,
    '0.': 0, '0': 0,
    '1.': 1, '1': 1,
    '2.': 2, '2': 2,
    '3.': 3, '3': 3,
    '4.': 4, '4': 4,
    '5.': 5, '5': 5,
    '6.': 6, '6': 6,
    '7.': 7, '7': 7,
    '8.': 8, '8': 8,
    '9.': 9, '9': 9,
}

# Index -> answer string (inverse of Count_TABLE, without the trailing period).
Reverse_Lookup = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '0%-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%', '70%-80%', '80%-90%', '90%-100%',
    '< = 5', '> 5', 'more than 10'
]


def find_pos_index(text_str: str, tag=''):
    """Find the numerical answers in `text_str`.

    Returns the list of their Count_TABLE indices and `text_str` with each of them replaced by `tag`.

    Note: the keys are used as regular expressions, so their '.' matches any character (e.g. '1.' also matches
    '1 '), while the replacement is literal. The reported results were computed with exactly this behaviour; do not
    escape the patterns.
    """
    pos_list = []
    for candidate_i, index in Count_TABLE.items():
        find_idxes = [substr.start() for substr in re.finditer(candidate_i, text_str)]
        pos_list += len(find_idxes) * [index]
        if len(find_idxes) > 0:  # if we can find the current numerical strs, we replace this.
            text_str = text_str.replace(candidate_i, tag).replace(candidate_i.strip('.'), tag)
        if text_str == '':
            break
    return pos_list, text_str


class Count_RMSE_MetricV2(object):
    """RMSE between the numbers of predicted and ground-truth answers (indices of Count_TABLE).

    Counting questions are scored with all their numbers; of the Comprehensive Analysis questions only the two
    traffic questions (which mention numbers of intersections) are scored. When the prediction and the ground truth
    contain a different number of numbers, the missing ones count as index -1. QAs without a ground-truth answer are
    skipped (``num_skipped``).
    """

    def __init__(self, ques_classes: list, logger=None):
        self.ques_classes = ques_classes
        self.logger = logger
        self.gts = [[] for _ in range(len(self.ques_classes))]
        self.preds = [[] for _ in range(len(self.ques_classes))]
        self.num_skipped = 0  # QAs without a ground-truth answer

    def __call__(self, pred: list, gt: list, questype: list, ques: list):
        for gt_i, pred_i, ques_t, ques_i in zip(gt, pred, questype, ques):
            if not has_ground_truth(gt_i):
                self.num_skipped += 1
                continue
            if ques_t in self.ques_classes:
                if ques_t == 'Comprehensive Analysis':
                    if ques_i in ['What are the comprehensive traffic situations in this scene?',
                                  'What are the traffic situations near the school?']:
                        self.add_pos_check(gt_i, pred_i, ques_t)
                else:
                    self.add_pos_check(gt_i, pred_i, ques_t)

    def add_pos_check(self, gt_i, pred_i, ques_t):
        cls_idx = self.ques_classes.index(ques_t)
        gt_i_idxes = find_pos_index(gt_i)[0]
        pr_i_idxes = find_pos_index(pred_i)[0]

        if len(gt_i_idxes) > len(pr_i_idxes):
            pr_i_idxes += (len(gt_i_idxes) - len(pr_i_idxes)) * [-1]  # default penalty
        else:
            gt_i_idxes += (len(pr_i_idxes) - len(gt_i_idxes)) * [-1]  # default penalty
        # (gts/preds are swapped here, which does not change the RMSE)
        self.gts[cls_idx] += pr_i_idxes
        self.preds[cls_idx] += gt_i_idxes

    def summary(self):
        tb = pt.PrettyTable()
        tb.field_names = ['Classes', '↓RMSE']
        for idx in range(len(self.ques_classes)):
            ques = self.ques_classes[idx]
            rmse = get_rmse(self.gts[idx], self.preds[idx])
            tb.add_row([ques, rmse])

        overall_gt, overall_pred = sum(self.gts, []), sum(self.preds, [])
        tb.add_row(['Overall_RMSE', get_rmse(overall_gt, overall_pred)])

        if self.logger is None:
            print('\n' + tb.get_string())
        else:
            self.logger.info('\n' + tb.get_string())
