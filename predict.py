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

image = cv2.imread("./data/full-data/val/imgs/csgo1628634028312018200_png_jpg.rf.6d1f4ee49f11a0794644caffcea25301.jpg")

seedEverything(config.SEED)

# Perform prediction
# for model in ["model_0000999.pth", "model_0001999.pth", "model_0002999.pth", "model_0003499.pth",  "model_final.pth", ]:
for i in range(5):
        with open('modelConfig.pkl', 'rb') as file:
                cfg = pickle.load(file)
        with open('metadata.pkl', 'rb') as file:
                metadata = pickle.load(file)
        cfg.MODEL.WEIGHTS = f'./output-fullUN/output-ALround-{i}/model_final.pth'
        cfg.MODEL.DEVICE = 'cuda:0'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.SEED = config.SEED
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image)
        print(f"Predicted classes: {outputs['instances'].pred_classes} with scores {outputs['instances'].scores}")
        # Display predictions
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(f"Prediction for final model of round {i}", out.get_image()[:, :, ::-1])
cv2.waitKey(0)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
register_datasets("./data/full-data", config.CLASSES)
evaluator = COCOEvaluator("val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
