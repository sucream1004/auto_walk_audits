# Toward Complete Automated Walkability Audits with Street View Images: Leveraging Virtual Reality for Enhanced Semantic Segmentation


## Dataset
dataset 은 요청하시면 줄게요 (Google forms link)  

Training sample 은 다음처럼 생겼습니다.
[example1.png]

Testing sample 은 다음과 같이 생겼습니다.

## Transfer learning training with custom data

### Requirements
* Detectron2 by Meta

이거 참고하세요. https://detectron2.readthedocs.io/en/latest/tutorials/install.html. 설치는 Docker 를 추천합니다.

## Benchmark



## inference

```
# Visualize
from PIL import Image
from glob import glob
import os 

from detectron2.utils.visualizer import ColorMode, Visualizer


im = cv2.imread("./gsv_search/screen.png")
outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im[:, :, ::-1],
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
display(Image.fromarray(out.get_image()[:, :, ::-1]))
```
