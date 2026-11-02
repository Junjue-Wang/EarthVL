"""Albumentations transforms of the VQA training pipeline.

The geometric transforms also accept an ``answer`` target and rewrite the direction words of the answer so that it
stays correct for the transformed image, e.g. after a 90-degree rotation 'Some roads run east-west.' becomes
'Some roads run north-south.' (only direction words at the end of a sentence, i.e. followed by '.', are rewritten).
They are subclasses of the albumentations transforms (https://github.com/albumentations-team/albumentations, MIT
License); the configs use them with ``albumentations.Compose(..., is_check_shapes=False)``.
"""
import numpy as np
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandomDropChannel(ImageOnlyTransform):
    """Set int(C * drop_rate) randomly chosen channels (with repetition) of an H x W x C feature map to zero.

    Args:
        drop_rate (float)

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            drop_rate=0.1,
            always_apply=False,
            p=0.5,
    ):
        super(RandomDropChannel, self).__init__(always_apply=always_apply, p=p)
        self.drop_rate = drop_rate

    def apply(self, img, **params):
        h, w, c = img.shape
        zero_channel = int(c * self.drop_rate)
        masked_indexes = np.random.randint(0, c, size=zero_channel)
        img[:, :, masked_indexes] = 0
        return img

    def get_transform_init_args_names(self):
        return ("drop_rate",)


due_directions = ['north-south.', 'east-west.']
deviation_directions = ['southwest-northeast.', 'southeast-northwest.']


def hflip_ans(text):
    """Answer after a horizontal flip: the two diagonal directions swap."""
    new_words = []
    words = text.split(' ')
    for word in words:
        if word == 'southwest-northeast.':
            new_words.append('southeast-northwest.')
        elif word == 'southeast-northwest.':
            new_words.append('southwest-northeast.')
        else:
            new_words.append(word)
    return ' '.join(new_words)


def vflip_ans(text):
    """Answer after a vertical flip: the two diagonal directions swap."""
    new_words = []
    words = text.split(' ')
    for word in words:
        if word == 'southwest-northeast.':
            new_words.append('southeast-northwest.')
        elif word == 'southeast-northwest.':
            new_words.append('southwest-northeast.')
        else:
            new_words.append(word)
    return ' '.join(new_words)


def rot90_ans(text, factor=1):
    """Answer after `factor` 90-degree rotations: an odd factor swaps north-south/east-west and the diagonals."""
    new_words = []
    words = text.split(' ')
    for word in words:
        if word in due_directions:
            new_index = (due_directions.index(word) + factor) % len(due_directions)
            new_words.append(due_directions[new_index])
        elif word in deviation_directions:
            new_index = (deviation_directions.index(word) + factor) % len(deviation_directions)
            new_words.append(deviation_directions[new_index])
        else:
            new_words.append(word)
    return ' '.join(new_words)


class VQAHorizontalFlip(HorizontalFlip):
    """HorizontalFlip that also updates the direction words of the ``answer`` target.

    Targets:
        image, mask, bboxes, keypoints, answer
    """

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            return F.hflip_cv2(img)
        else:
            return F.hflip(img)

    def apply_to_answer(self, answer, **params):
        return hflip_ans(answer)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "answer": self.apply_to_answer
        }


class VQAVerticalFlip(VerticalFlip):
    """VerticalFlip that also updates the direction words of the ``answer`` target.

    Targets:
        image, mask, bboxes, keypoints, answer
    """

    def apply(self, img, **params):
        return F.vflip(img)

    def apply_to_answer(self, answer, **params):
        return vflip_ans(answer)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "answer": self.apply_to_answer
        }


class VQARandomRotate90(RandomRotate90):
    """RandomRotate90 (0-3 times) that also updates the direction words of the ``answer`` target.

    Targets:
        image, mask, bboxes, keypoints, answer
    """

    def apply(self, img, factor, **params):
        return np.ascontiguousarray(np.rot90(img, factor))

    def apply_to_answer(self, answer, factor=0, **params):
        return rot90_ans(answer, factor)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "answer": self.apply_to_answer
        }
