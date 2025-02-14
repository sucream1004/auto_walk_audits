# Toward Complete Automated Walkability Audits with Street View Images: Leveraging Virtual Reality for Enhanced Semantic Segmentation


## Dataset
dataset 은 요청하시면 줄게요 (Google forms link)  

Training sample 은 다음처럼 생겼습니다.
[example1.png] # 벤치

[_bollard_] # 볼라드

Testing sample 은 다음과 같이 생겼습니다.

## Transfer learning training with custom data

### Requirements
* Detectron2 by Meta

이거 참고하세요. https://detectron2.readthedocs.io/en/latest/tutorials/install.html. 설치는 Docker 를 추천합니다.

## Benchmark



## inference

```
import os

from PIL import Image
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer

model_iter = "model_iter.pth"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # COCO mask_rcnn_R_50_FPN pretrained weight
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_{model_iter}.pth")  # path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set threshold
predictor = DefaultPredictor(cfg)

im = cv2.imread("./gsv_search/screen.png")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
display(Image.fromarray(out.get_image()[:, :, ::-1]))
```
