# AutoNeRF

AutoNeRF is a repository for implementing Neural Radiance Fields (NeRF) using automated techniques. 

## Overview

Neural Radiance Fields (NeRF) is a powerful method for synthesizing novel views of a scene by learning a continuous 3D representation from a set of 2D images. AutoNeRF aims to automate the process of annotating image data for NeRF model input in one shot.

AutoNeRF introduces a novel method for preprocessing images without ground truth poses as input for the NeRF models. This technique uses recent advances in metric monocular depth estimation and fairly simple linear algebra to estimate camera poses from the input images, allowing the NeRF models to be trained without the need for explicit pose annotations. By eliminating the requirement for ground truth poses, AutoNeRF significantly simplifies the data preparation process and expands the applicability of NeRF to a wider range of datasets. 

## Notebooks

### slam.ipynb

The `slam.ipynb` notebook provides visualizations to help validate the SLAM (Simultaneous Localization and Mapping) process for measuring 3D poses and viewing angles.

### nerf.ipynb

The `nerf.ipynb` notebook contains client-side code usage for the Neural Radiance Fields (NeRF) model. It demonstrates how to utilize the NeRF model to synthesize novel views of a scene based on a learned continuous 3D representation. The notebook provide a step-by-step example.

## Features

- Automated training of NeRF models
- Efficient rendering of novel views
- Support for various datasets and scene types
- Easy-to-use API for customization and extension

## Installation

To install AutoNeRF, simply clone this repository and install the required dependencies:

```
git clone https://github.com/piyushmundhra/AutoNeRF.git
cd AutoNeRF
pip install -r requirements.txt
```

## Citations

```bibtex
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}

@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
