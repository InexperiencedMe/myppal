import argparse
import shutil
import os
from utils import *
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes',                default=config.CLASSES)
    parser.add_argument('--data-dir',               default=config.DATA_DIR)
    parser.add_argument('--output-dir',             default=config.OUTPUT_DIR)
    parser.add_argument('--device',                 default=config.DEVICE)
    parser.add_argument('--learning-rate',          default=config.LEARNING_RATE,       type=float)
    parser.add_argument('--batch-size',             default=config.BATCH_SIZE,          type=int)
    parser.add_argument('--iterations',             default=config.ITERATIONS,          type=int)
    parser.add_argument('--checkpoint-period',      default=config.CHECKPOINT_PERIOD,   type=int)
    parser.add_argument('--model',                  default=config.MODEL)
    parser.add_argument('--al-rounds',              default=config.AL_ROUNDS,           type=int)
    parser.add_argument('--debug',                  default=config.DEBUG,               type=int)
    parser.add_argument('--UD-split', nargs='*',    default=config.UD_SPLIT,            type=float)
    parser.add_argument('--seed',                   default=config.SEED,                type=int)
    args = parser.parse_args()

    seedEverything(args.seed)

    print(f"Starting training. First training, and then {int(args.al_rounds)} rounds of AL training")
    if int(args.debug) < 2:
        trainSamplesCount = int(countFiles(f"{config.DATA_DIR}/{config.F_FULL_START_DATA}/{config.F_TRAIN}/{config.F_ANNS}")/(int(args.al_rounds) + 1))
        valSamplesCount = int(countFiles(f"{config.DATA_DIR}/{config.F_FULL_START_DATA}/{config.F_VAL}/{config.F_ANNS}")/(int(args.al_rounds) + 1))
    else:
        trainSamplesCount = 10
        valSamplesCount = 10
    print(f"### For each round we'll be using {trainSamplesCount} train samples and {valSamplesCount} validation samples")
    if not os.path.isdir(f"{config.DATA_DIR}/{config.F_FULL_WORKDATA}"):
        shutil.copytree(f"{config.DATA_DIR}/{config.F_FULL_START_DATA}", f"{config.DATA_DIR}/{config.F_FULL_WORKDATA}")
    else:
        shutil.rmtree(f"{config.DATA_DIR}/{config.F_FULL_WORKDATA}")
        shutil.copytree(f"{config.DATA_DIR}/{config.F_FULL_START_DATA}", f"{config.DATA_DIR}/{config.F_FULL_WORKDATA}")
    makeStandardFoldersAndMovePartialData(f"{config.F_ROUND_WORKDATA}-0", config.F_FULL_WORKDATA, args.output_dir, trainSamplesCount, valSamplesCount, 0)

    # INITIAL TRAINING - AL ROUND 0
    print(f"\n### STARTING INITIAL TRAINING - ROUND 0 ###")
    train(     args.output_dir + f"/output-{config.F_ROUND_WORKDATA}-0",
               args.data_dir + f"/{config.F_ROUND_WORKDATA}-0",
               args.classes,
               device=args.device,
               learning_rate=float(args.learning_rate),
               batch_size=int(args.batch_size),
               iterations=int(args.iterations)//(int(args.al_rounds)+1),
               checkpoint_period=int(args.checkpoint_period),
               model=args.model)
    if int(args.debug) < 1:
        shutil.rmtree(args.data_dir + f"/{config.F_ROUND_WORKDATA}-0")

    # AL ROUNDS STARTING FROM 1
    for i in range(int(args.al_rounds)):
        print(f"\n### STARTING TRAINING - ROUND {i+1}###")
        makeStandardFoldersAndMovePartialData(f"{config.F_ROUND_WORKDATA}-{i+1}", config.F_FULL_WORKDATA, args.output_dir, trainSamplesCount, valSamplesCount, i, args.UD_split[i])
        train(  args.output_dir + f"/output-{config.F_ROUND_WORKDATA}-{i+1}",
                args.data_dir + f"/{config.F_ROUND_WORKDATA}-{i+1}",
                args.classes,
                device=args.device,
                learning_rate=float(args.learning_rate),
                batch_size=int(args.batch_size),
                iterations=int(args.iterations)//(int(args.al_rounds)+1),
                checkpoint_period=int(args.checkpoint_period),
                model=f"{args.output_dir}/output-{config.F_ROUND_WORKDATA}-{i}/model_final.pth",
                customConfig=config.DT2_CONFIG_FILENAME)
        if int(args.debug) < 1:
            shutil.rmtree(args.data_dir + f"/{config.F_ROUND_WORKDATA}-{i+1}")
    if int(args.debug) < 1:
        shutil.rmtree(args.data_dir + f"/{config.F_FULL_WORKDATA}", ignore_errors=True)