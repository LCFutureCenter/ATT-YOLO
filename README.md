# ATT-YOLO: An accurate YOLO-style object detector for surface defect detection in electronics manufacturing
PyTorch implementation of ATT-YOLO (An accurate YOLO-style object detector for surface defect detection in electronics manufacturing)

## Introduction
In electronics manufacturing, surface defect detection is very important for product quality control, and defective products can cause severe customer complaints. At the same time, in the manufacturing process, the cycle time of each product is usually very short. Furthermore, high-resolution input images from high-resolution industrial cameras are necessary to meet the requirements for high quality control standards. Hence, how to design an accurate object detector with real-time inference speed that can accept high-resolution input is an important task. In this work, an accurate YOLO-style object detector was designed, ATT-YOLO, which uses only one self-attention module, many-scale feature extraction and integration in the backbone and feature pyramid, and an improved auto-anchor design to address this problem. 
  For testing, we curated a dataset consisting of 14,478 laptop surface defects, on which ATT-YOLO achieved 92.8% mAP0.5 for the binary-class object detection task. We also further verified our design on the COCO benchmark dataset. Among object detectors with similar computational costs, ATT-YOLO outperforms existing YOLO-style object detectors, achieving 44.9% in terms of the mAP measure, which is better than the performance of YOLO-style small models, including YOLOv7-tiny-SiLU (38.7%), YOLOv6-small (43.1%), pp-YOLOE-small (42.7%), YOLOX-small (39.6%), and YOLOv5-small (36.7%). 
  We hope that this work can serve as a useful reference for the utilization of attention-based networks in real-world situations.

## Citation
if you find our work useful in your research, please consider citing:
```
@inproceedings{,
  title={ATT-YOLO: An accurate YOLO-style object detector for surface defect detection in electronics manufacturing},
  author={Jyun-Rong Wang, Hua-Feng Dai, Tao-Gen Chen, Hao Liu, Xue-Gang Zhang, and Quan Zhong},
  booktitle={},
  pages={},
  year={}
}
```

## Usage
### Installation
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.

### Data Prepare
The coco dataset will be downloaded from [here](http://cocodataset.org).

The remaining code will be open source soon!!!

## License
Our code is released under MIT License (see LICENSE file for details).