# Toward Complete Automated Walkability Audits with Street View Images: Leveraging Virtual Reality for Enhanced Semantic Segmentation


## Dataset
dataset 은 요청하시면 줄게요 (Google forms link)  

Training sample 은 다음처럼 생겼습니다.
[img/example1.png]

Testing sample 은 다음과 같이 생겼습니다.
[41_jpg.rf.d4decd52af681d8f86b1b8139e4e8950.jpg] # testing 데이터

### Requirements
* Detectron2 by Meta
* 이거 참고하세요. https://detectron2.readthedocs.io/en/latest/tutorials/install.html. 설치는 Docker 를 추천합니다.
* Windows 에서 설치하고 싶으시면: https://helloshreyas.com/how-to-install-detectron2-on-windows-machine 참고하세요.

## Benchmark
* existing data (benchmark data) 로는 ade20k 를 사용했습니다 (https://ade20k.csail.mit.edu/).

## inference
* model 을 받아서 한번 해보세요.
* model 은 [구글 드라이브] 에서 받으세요.
* 코드는 다음과 같이하면 됩니다. inference.ipynb 를 확인하세요.

[inference 예시]
  
```
from PIL import Image
import cv2
import numpy as np

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch

cfg.INPUT.MASK_FORMAT='bitmask'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "best_model_bench.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
# change the img path if you want to test your own image.

im = cv2.imread("61_jpg.rf.4e147b7acbea1a7ded08a22b42d1f0ec.jpg")
ori_size = im.shape[:2]
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
                scale=1,
                instance_mode=ColorMode.IMAGE_BW
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
display(Image.fromarray(out.get_image()[:, :, ::-1]))
```

## Testing
* Testing 은 마찬가지로 testing.ipynb 를 사용하시면 됩니다.
* 직접 데이터를 받아서 training 을 해서 사용하셔도 되고, best model 을 받으셔서 해보셔도 됩니다.

[img/infer.png]
