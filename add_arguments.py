import argparse

MODEL = 'AE'
BATCH_SIZE = 1
NUM_WORKERS = 4
GPU = 1

# source data
INPUT_SIZE = '512,512'
DATA_DIRECTORY_IMG = '../data/orivncdata/img/'
DATA_DIRECTORY_LABEL = '../data/orivncdata/lab/'
DATA_LIST_PATH = './dataset/vncdata_list/train.txt'

# target data
INPUT_SIZE_TARGET = '512,512'
DATA_DIRECTORY_TARGET = '../data/cvlabdata/train/img/'
DATA_DIRECTORY_TARGET_LABEL = '../data/cvlabdata/train/lab/'
DATA_LIST_PATH_TARGET = './dataset/cvlabdata_list/train.txt'

# target validation
DATA_DIRECTORY_VAL = '../data/cvlabdata/val/img/'
DATA_DIRECTORY_VAL_LABEL = '../data/cvlabdata/val/lab/'
DATA_LIST_PATH_VAL = './dataset/cvlabdata_list/val.txt'

# target validation
DATA_DIRECTORY_TEST = '../data/cvlabdata/test/img/'
DATA_DIRECTORY_TEST_LABEL = '../data/cvlabdata/test/lab/'
DATA_LIST_PATH_TEST = './dataset/cvlabdata_list/test.txt'

# model setting
ITER_START = 0
PRETRAIN = 0
RESTORE_FROM = ''
D1RESTORE_FROM = ''
D2RESTORE_FROM = ''

NUM_CLASSES = 2
NUM_STEPS = 100000

# auto-encoder
LEARNING_RATE = 0.00005
STEP_SIZE = 6000

# label discrimator
LEARNING_RATE_Dl = 0.0001
STEP_SIZE_Dl = 3000

# feature discrimator
LEARNING_RATE_Df = 0.0001
STEP_SIZE_Df = 3000

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 300
SNAPSHOT_DIR = './snapshots/'
SAVE_DIR = './testResult/'
TEST_MODEL_PATH = './snapshots/CV_300.pth'

LAMBDA_ADV_LABEL = 0.001
LAMBDA_ADV_FEATURE = 0.001
LAMBDA_REC = 0.0005

TEST_AUG = 4 # 1: original 4: original + different flips
BEST_TJAC = 0.50
FROZEN_LAYER = []  # 'Conv_down1', 'Conv_down2', 'Conv_down3','Conv_down4','Conv_down5', 'Conv_up1', 'Conv_up2','Conv_up3', 'Conv_up4'
RANDOM_LAYER = []  # ,'Conv_out', 'Conv_out1','Conv_out2'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="The segmentation network.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir-img", type=str, default=DATA_DIRECTORY_IMG,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-label", type=str, default=DATA_DIRECTORY_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-test", type=str, default=DATA_DIRECTORY_TEST,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-test-label", type=str, default=DATA_DIRECTORY_TEST_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-val", type=str, default=DATA_DIRECTORY_VAL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-val-label", type=str, default=DATA_DIRECTORY_VAL_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-dir-target-label", type=str, default=DATA_DIRECTORY_TARGET_LABEL,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-val", type=str, default=DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-test", type=str, default=DATA_LIST_PATH_TEST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--step-size", type=int, default=STEP_SIZE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--step-size-Dl", type=int, default=STEP_SIZE_Dl,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-Dl", type=float, default=LEARNING_RATE_Dl,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--step-size-Df", type=int, default=STEP_SIZE_Df,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-Df", type=float, default=LEARNING_RATE_Df,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-label", type=float, default=LAMBDA_ADV_LABEL,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-feature", type=float, default=LAMBDA_ADV_FEATURE,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-rec", type=float, default=LAMBDA_REC,
                        help="lambda_rec for reconstruction training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--iter-start", type=int, default=ITER_START,
                        help="Number of training steps.")
    parser.add_argument("--pretrain", type=int, default=PRETRAIN,
                        help="whether to pretrain the model.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--D1restore-from", type=str, default=D1RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--D2restore-from", type=str, default=D2RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--test-model-path", type=str, default=TEST_MODEL_PATH,
                        help="Where restore test model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Save dir.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--test_aug", type=int, default=TEST_AUG,
                        help="Test augmentation")
    parser.add_argument("--best_tjac", type=float, default=BEST_TJAC,
                        help="The best tjac")
    parser.add_argument("--frozen_layer", type=str, default=FROZEN_LAYER,
                        help="The layer which is need to frozen")
    parser.add_argument("--random_layer", type=str, default=RANDOM_LAYER,
                        help="The layer which is dont need to copy (i.e, random initilation)")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()