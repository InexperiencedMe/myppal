from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import pickle
import torch
from utils import *
import config
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import csv
from collections import OrderedDict
import argparse

random.seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--filename', default="scores")
        args = parser.parse_args()

        image = cv2.imread("./data/full-data/val/imgs/csgo1628634028312018200_png_jpg.rf.6d1f4ee49f11a0794644caffcea25301.jpg")

        seedEverything(config.SEED)

        allScores = []
        for model in ["TESTING1"]:#["fullUN", "175525", "0907050", "755250", "fullDIV", "constant05", "959085", "958575", "constant90", "constant95", "002505", "250575"]:
                for i in range(5): # Trained all models in 5 rounds
                        with open('modelConfig.pkl', 'rb') as file:
                                cfg = pickle.load(file)
                        with open('metadata.pkl', 'rb') as file:
                                metadata = pickle.load(file)
                        cfg.MODEL.WEIGHTS = f'./output-{model}/output-ALround-{i}/model_final.pth'
                        cfg.MODEL.DEVICE = 'cuda:0'
                        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
                        cfg.SEED = config.SEED
                        predictor = DefaultPredictor(cfg)
                        register_datasets("./data/full-data", config.CLASSES)
                        evaluator = COCOEvaluator("val", output_dir="./output")
                        val_loader = build_detection_test_loader(cfg, "val")
                        resultingScores = inference_on_dataset(predictor.model, val_loader, evaluator)['bbox']
                        resultingScores["model"] = model
                        resultingScores["round"] = i

                        allScores.append(resultingScores)

        with open(f'{args.filename}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                orderedColumns = ["model", "round"] + [key for key in allScores[0].keys() if key not in ['model', 'round']]
                writer.writerow(orderedColumns)
                for score in allScores:
                        orderedScores = [score[key] for key in orderedColumns]
                        writer.writerow(orderedScores)