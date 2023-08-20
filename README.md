# Segmentation and morphmetry of vasucular bundle

> Tsuyama et al: Quantitative morphological transformation of vascular bundles in the culm of moso bamboo (Phyllostachys pubescens) DOI: 10.1371/journal.pone.0290732
>
> 

### Directory structure

The original dataset, pairs of image and the correponding mask, for building u-net model should be placed in '_original_pdg'  directory. Under the job number (ex. 003), following directory will be generated.The 'test' directory contains the target images to be segmented.

```
.
├── _original_pdg
├── _run
│   └── 003
│       ├── extracted_VB_images
│       ├── model
│       ├── morphology
│       ├── segmentation
│       └── train
│           ├── image
│           │   ├── 0.png
│           │   ├── 1.png
│           │   └── 2.png
│           └── mask
│               ├── 0.png
│               ├── 1.png
│               └── 2.png
└── _test
```



### Jupyter notebook

001_model_builing.ipynb: Building u-net model

002_segment_analysis.ipynb: Segmentation using u-net model

003_morphometry.ipynb: Measurement of morphological parameters



### Flow chart 

Step 1.
The original microscopic image and the corresponding mask image are divided into 512x512 pixel images. A set of around 200 images was used as a training dataset, followed by augmentation such as shift, scale, rotate,  The U-net was successfully trained and a model with 98% accuracy was constantly obtained.
Phase 2
The actual test microscopic images were divided into image patches, and for each image patch segmentation was performed by U-net. Finally, patches were stitched into a single image again to complete the process.

<img src="./img/1.png" alt="1" style="zoom:80%;" />

Extracted vascular bundles are exemplified in the following images. Those images are sequentially numbered with respect to the from the relative radial distance from the outer surface of the column.

![](/Users/sugiyama/Documents/python/GitHub/vb_unet_2023/img/2.png)

Finally, morphological parameters were obtained using scikit-image package. Typical output was exemplified as follows.

![](/Users/sugiyama/Documents/python/GitHub/vb_unet_2023/img/3.png)





### Data used in the paper

26 microscope images and corresponding mask images (1575 × 6150) are available upon request.

Please mail to the corresponding author.



### References

#### The gerenative network for segmentation was referenced from  U-Net

> [ U-Net: Convolutional Networks for Biomedical Image Segmentation ](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
>
> Olaf Ronneberger, Philipp Fischer, Thomas Brox:  Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at [arXiv:1505.04597 [cs.CV\]](http://arxiv.org/abs/1505.04597)

#### U-Net coding

> The network structure was constructed by referring to the following website.https://github.com/zhixuhao/unet/blob/master/README.md

