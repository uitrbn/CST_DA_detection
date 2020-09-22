
# CST_DA_detection



The pytorch implementation of paper "Collaborative Training between Region Proposal Localization and Classification for Domain Adaptive Object Detection" in ECCV2020.



## Introduction

The project is based on this [repository](https://github.com/jwyang/faster-rcnn.pytorch). Please follow the instructions to setup the enviroment. We used Pytorch 0.4.0 and torchvision 0.2.2 for this project.



## Data Preparation

- **Cityscapes** Please refer to this [website](https://www.cityscapes-dataset.com/) to download the data. The training set includes 2964 images.
- **FoggyCityscapes** Please refer to this [website](https://www.cityscapes-dataset.com/) to download the data.

Unzip both dataset in `./data`, then run `boxes_for_cityscapes.py` to convert the annotation format.



## Training and Evaluation

1. Pretrain with source data.

   `CUDA_VISIBLE_DEVICES=${gpu_id} python cityscapes_pretrain.py`

2. Domain adaptation with target data.

   `CUDA_VISIBLE_DEVICES=${gpu_id} python cityscapes_to_foggycityscapes_da.py`

3. Test with target data.

   `CUDA_VISIBLE_DEVICES=${gpu_id} python cityscapes_to_foggycityscapes_da_test.py --checksession ${session_id} --checkepoch ${epoch_num} --checkpoint ${point_num}`

   

## Pretrained Model

The pretrained model will be released soon.



## Citation

Please cite the following reference if you utilize this repository for your project.

```
@article{zhao2020collaborative,
  title={Collaborative Training between Region Proposal Localization and Classi? cation for Domain Adaptive Object Detection},
  author={Zhao, Ganlong and Li, Guanbin and Xu, Ruijia and Lin, Liang},
  journal={arXiv preprint arXiv:2009.08119},
  year={2020}
}
```

