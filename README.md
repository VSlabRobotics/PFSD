# PFSD: A Multi-Modal Pedestrian-Focus Scene Dataset for Rich Tasks in Semi-Structured Environments

This is the official website to post information about ***PFSD***. For more details, please refer to: 

**PFSD [[Paper](https://arxiv.org/abs/2502.15342)]** <br />

The link to get dataset: https://pan.baidu.com/s/19DN4csVlMnrIYOKByBrZvQ 
Code: pfsd

## Getting Started
### Installation

#### a. Clone this repository
```shell
git clone https://github.com/VSlabRobotics/PFSD.git && cd PFSD
```
#### b. Install the environment

Follow the install documents for [OpenPCDet](docs/INSTALL.md and prepare needed packages.

#### c. Prepare the dataset 

Get dataset from the previous link.
For nuScenes format datasets, please follow the [document](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) in OpenPCDet.

### Training

```shell
python train.py --cfg_file PATH_TO_CONFIG_FILE
#For example,
python train.py --cfg_file cfgs/nuscenes_models/hmfn.yaml
```

## License
All datasets are published under the Creative Commons Attribution-NonCommercial-ShareAlike. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.
