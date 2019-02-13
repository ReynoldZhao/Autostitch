# Autostitch

## Introduction

In this project, we will implement a system to combine a series of horizontally overlapping photographs into a single panoramic image. We are given the ORB feature detector and descriptor. we will use ORB to first detect discriminating features in the images and find the best matching features in the other images. Then, using RANSAC, we will automatically align the photographs (determine their overlap and relative positions) and then blend the resulting images into a single seamless panorama.

Click [here](http://www.cs.cornell.edu/courses/cs5670/2018sp/projects/pa3/index.html) to view projects introduction. 

## Features

* Using **RANSAC** to automatically align the photographs (determine their overlap and relative positions) and then blend the resulting images into a single seamless panorama

* With previous features detection and matching knowledge, we find the matched images and create program that can calculate **Homography approach** transformation parameters, and transfrom them so that we align images from different angels in a consistent platform

* Create panoramas with **Spherical Warp Mappings method**

* In compute homography , we use **SVD (Singular Value Decomposition) method**

* Normalize the image with the accumulated weight channel, we deal with diffrent images' edges when creating panorama with **edge feather method** 

* We can create **360 degree* panorama

  

## Structure

| Name             | Function                                                     |
| ---------------- | ------------------------------------------------------------ |
| resources/       | available images to create  panorama                         |
| src/warp.py      | Warp each image into spherical coordinates.                  |
| src/alignment.py | Compute the alignment of image pairs.                        |
| src/blend.py     | Stitch and crop the resulting aligned images.                |
| src/test.py      | test whether function in features.py is same with task's command |
| src/gui.py       | An gui for users to create panorama                          |

## Usages

### Requirements

* Linux / Windows / MacOS
* python 2.7 / python 3.5
* cv2
* numpy
* pandas
* scipy

### Compilation

``` python
cd python test.py
cd python gui.py
```

## Examples

### A series of images

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/resources/yosemite/panorama/yosemite1.jpg)

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/resources/yosemite/panorama/yosemite2.jpg)

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/resources/yosemite/panorama/yosemite3.jpg)

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/resources/yosemite/panorama/yosemite4.jpg)
### Panorama

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/Project3_Results/yosemite_pano_homography_blendwidth50.png)

![](https://github.com/ReynoldZhao/Project3_Autostitch/raw/master/Project3_Results/yosemite_pano_translation_blendwidth50.png)
