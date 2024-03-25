# Paths and folder names
CLASSES                 = "./class.names"
DATA_DIR                = "./data"
F_FULL_START_DATA       = "full-data"
F_FULL_WORKDATA         = "full-workdata"
F_TRAIN                 = "train"
F_VAL                   = "val"
F_IMAGES                = "imgs"
F_ANNS                  = "anns"
OUTPUT_DIR              = "./output"
F_ROUND_WORKDATA        = "ALround"

# Training params
LEARNING_RATE           = 0.0001
BATCH_SIZE              = 2
ITERATIONS              = 1000
CHECKPOINT_PERIOD       = 500
MODEL                   = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
AL_ROUNDS               = 4
DEVICE                  = "gpu"
AL_OBJ_SCORE_THRESHOLD  = 0.3 # Original used 0.05, but we cant afford it
UD_SPLIT                = [1, 0.75, 0.5, 0.25] # 1 = full uncertainty, 0 = full diversity

# Program params
DEBUG                   = 0
DT2_CONFIG_FILENAME     = "modelConfig.pkl"
METADATA_FILENAME       = "metadata.pkl"
IMG_EXTENSION           = ".jpg"
ANN_EXTENSION           = ".txt" # I mean we only support .txt with yolo format so dont change that
SEED                    = 10