import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=False, default= './PASCAL_VOC/')
    parser.add_argument("--image_size", type=int, required=False, default=416)
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--num_classes", type=int, required=False, default=20)
    parser.add_argument("--loss", type=str, required=False, default="Dice_BCE")
    parser.add_argument("--lr", type=float, required=False, default=8e-5)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--epochs", type=int, default=21, required=False)
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../input/hubmap-folds/train_unsplit_data.csv",
        required=False,
    )
    return parser.parse_args()

args = parse_args()

DATASET = args.folder
GPU = 0
seed = 42 #seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 8
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
NUM_CLASSES = args.num_classes
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 700
CLIP = None # 0.99999
CONF_THRESHOLD = 0.2
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
BACKBONE_FREEZE = False
PIN_MEMORY = True
RESUME = ''
SAVE_MODEL = True
CHECKPOINT_DIR = "exp1"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
STEP_SIZE = 300  #epochs after lr linear decay by 0.1
ACCUM_FACTOR = 1  # to turn on use 2,4, or more
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
#
