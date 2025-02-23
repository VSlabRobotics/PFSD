# PFSD

This is the official website to post information about ***PFSD***. For more details, please refer to: xxx

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
1
