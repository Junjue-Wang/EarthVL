# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
#
# From the MS COCO caption evaluation code (https://github.com/tylin/coco-caption), BSD 2-Clause License,
# see THIRD_PARTY_NOTICES.md. Modified for EarthVL: compute_score takes lists (one entry per sample) instead of
# dicts keyed by image id.

from score.cider_scorer import CiderScorer

class Cider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param gts: list : per sample, a list of tokenized reference sentences (str)
        :param res: list : per sample, a list with one tokenized hypothesis / candidate sentence (str)
        :return: cider (float) : computed CIDEr score for the corpus
                 scores (numpy array) : CIDEr score of every sample
        """

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for i in range(len(res)):
            hypo = res[i]
            ref = gts[i]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"