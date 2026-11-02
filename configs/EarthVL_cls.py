"""Data and schedule of the LLM branch of EarthVLNet (judging and analysis questions); used by configs/EarthVLNet.py.

Paper protocol: train on the Train splits of EarthVQA and EarthVL-GLOBAL (without Rotterdam, see ROTTERDAM_IMAGES),
evaluate on their Test splits. DATA_ROOT is the Hugging Face EarthVLSet download; FEAT_ROOT holds the stage-1 features
(``<FEAT_ROOT>/<EarthVQA|EarthVL-GLOBAL>/<split>/<image_id>.hdf5``, see scripts/extract_features.sh).
The EarthVQA Test labels are withheld for the EarthVQA challenge on Codabench: test QAs without an answer are
predicted but not scored (train_generation.py, evaluate.py).
"""
from albumentations import Compose

from module.data_augs import RandomDropChannel, VQAHorizontalFlip, VQARandomRotate90, VQAVerticalFlip

DATA_ROOT = './EarthVLSet'
FEAT_ROOT = './features/sfpn_convnext_large'

train_qa_path = [
    f'{DATA_ROOT}/EarthVQA/Train_QA.json',
    f'{DATA_ROOT}/EarthVL-GLOBAL/earthvl_global_multiplechoice/Train.json',
]
train_feat_dir = [
    f'{FEAT_ROOT}/EarthVQA/Train',
    f'{FEAT_ROOT}/EarthVL-GLOBAL/Train',
]
test_qa_path = [
    f'{DATA_ROOT}/EarthVQA/Test_QA.json',
    f'{DATA_ROOT}/EarthVL-GLOBAL/earthvl_global_multiplechoice/Test.json',
]
test_feat_dir = [
    f'{FEAT_ROOT}/EarthVQA/Test',
    f'{FEAT_ROOT}/EarthVL-GLOBAL/Test',
]
# Rotterdam (Netherlands) is in the Val split of the paper, but the Hugging Face EarthVL-GLOBAL Train split contains it
# (ids found by an exact byte-size match with the original images). The paper's VQA runs did not train on it.
ROTTERDAM_IMAGES = [f'{i}.png' for i in range(9118, 9486)]

# the counting questions are answered by the counting branch (configs/EarthVLCount.py)
# data config of the released LLM-branch weights: these 6 types, RandomDropChannel drop_rate 0.2, Rotterdam excluded
select_type = ['Basic Judging', 'Reasoning-based Judging', 'Object Situation Analysis', 'Distribution Analysis',
               'Direction Analysis', 'Comprehensive Analysis']

data = dict(
    train=dict(
        type='EarthVLLoader',
        params=dict(
            image_feat_dir=train_feat_dir,
            qa_path=train_qa_path,
            image_load=False,
            pred_mask_load=True,
            feat_load=True,
            encoding=False,
            mask_load=False,
            vqa_transforms=Compose([
                VQAHorizontalFlip(p=0.5),
                VQAVerticalFlip(p=0.5),
                VQARandomRotate90(p=0.5),
                VQARandomRotate90(p=0.5),
                VQARandomRotate90(p=0.5),
                RandomDropChannel(p=0.5, drop_rate=0.2)
            ], is_check_shapes=False),
            training=True,
            batch_size=10,
            num_workers=2,
            select_type=select_type,
            exclude_images=ROTTERDAM_IMAGES,
        ),
    ),
    test=dict(
        type='EarthVLLoader',
        params=dict(
            image_feat_dir=test_feat_dir,
            qa_path=test_qa_path,
            image_load=False,
            pred_mask_load=True,
            feat_load=True,
            encoding=False,
            mask_load=False,
            vqa_transforms=None,
            training=False,
            batch_size=10,
            num_workers=2,
            select_type=select_type,
        ),
    ),
)

# engine/trainer.py; intervals in optimizer steps, 0 = off
train = dict(
    num_iters=40000,
    log_interval_step=50,
    save_ckpt_interval_step=40000,
    eval_interval_step=0,
    eval_after_train=True,
)

test = dict()
