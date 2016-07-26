This software package contains code used for conducting experiments in
following work:

A. Bendale, T. Boult “[Towards Open Set Deep Networks](http://vast.uccs.edu/~abendale/papers/0348.pdf)” IEEE Conference on 
Computer Vision and Pattern Recognition (CVPR), 2016 [pdf](http://vast.uccs.edu/~abendale/papers/0348.pdf)

Authors: Abhijit Bendale (abendale@vast.uccs.edu)
Terrance Boult (tboult@vast.uccs.edu)
Vision and Security Technology Lab
University of Colorado at Colorado Springs

The code is provided "as is", without any guarantees. Please refer
COPYRIGHT.txt and libMR/COPYRIGHT_Libmr.txt for more details
about license and usage restrictions.


The package is divided into two parts: LibMR and OpenMax. 

1) LibMR contains code for Extreme Value Theory based Weibull fitting (and other
code used in the work for Meta-Recognition). Weibull fitting performed by 
LibMR is slightly different than that done by other popular packages such as
MATLAB's wblfit() function. We have added a short tutorial to illustrate
these differences.

2) OpenMax contains code used in the CVPR 2016 paper on "Towards Open Set Deep 
Networks". The software packages for OpenMax code calls LibMR functions for 
performing Weibull fitting. In this code, we are providing code to compute
OpenMax probabilities from pre-computed Mean Activation Vectors and Softmax
probabilities. Mean Activation Vectors and Softmax probabilities are obtained
by passing images through Caffe's architecture. 



## Usage:


#### 1) Compiling LibMR

Compile LibMR and python interface to LibMR using following commands.
For pythong interfaces to work, you would require Cython to be pre-installed
on your machine
```bash
cd libMR/
chmod +x compile.sh
./compile.sh
```

#### 2) Precomputed Features
##### 2.a) Extract features using a pre-trained AlexNet network. The extracted features
are saved in a mat file. We save fc7, fc8, prob and scores from Caffe framework.
Example of saved files can be found in data/train_features/

E.g. loading the file data/train_features/n01440764/n01440764_9981.JPEG.mat in ipython
will lead to 

```python
>>> from scipy.io import loadmat
>>> import scipy as sp
>>> features = loadmat('data/train_features/n01440764/n01440764_9981.JPEG.mat')
>>> print features.keys()
['fc7', 'fc8', '__header__', '__globals__', 'scores', 'IMG_NAME', '__version__', 'prob']
>>> print features['fc8'].shape
(10, 1000)
>>> print sp.mean(features['prob'], axis=0)[0]
0.999071
>>> print features['scores'][0][0]
0.9990708
```

where fc7, fc8, prob, scores are outputs of respective layers provided by Caffe Library.
Further, fc8 layer contains 10 channels for 1000 classes of ImageNet. These channels are
referred to as crops in Caffe library [classifier.py](https://github.com/BVLC/caffe/blob/master/python/caffe/classifier.py), check the function predict() ). The average of prob layer (i.e. SoftMax layer) is the
final probability value reported by AlexNet as architected in Caffe Library. In the paper (and
code), each of these 10 crops is referred to as channel. Hence, average of features['prob'],
is essentially features['scores']. 

You will have to install Caffe library on your machine with python interface activated. If you do
have it configured, then features for entire ImageNet data can be extracted using the script,
python preprocessing/imageNet_Features.py

##### 2.b) Compute Mean Activation Vector
Mean Activation Vector is defined as the Mean Vector in the pen-ultimate layer. In our paper,
we considered fc8 as the pen-ultimate layer. Hence, MAV is the mean of features in fc8. 
The Mean is computed for each channel. The script to compute mean can be found in 

python preprocessing/MAV_Compute.py

Since this process is very time consuming, we are proving pre-computed mean activation vectors
on ILSVRC 2012 dataset with this code repository. You can download the mean activation vectors
using following command

```bash
wget http://vast.uccs.edu/OSDN/data.tar
tar -xvf data.tar
```
The structure of Mean Files is very simple. For e.g.

```python
>>> from scipy.io import loadmat
>>> mav = loadmat('data/mean_files/n01440764.mat')
>>> print mav['n01440764'].shape
(10, 1000)
```

The above is the MAV for object category n01440764. MAV is computed in similar manner for each
of the 1000 ImageNet Categories. Note, for computing MAV, only those images are considered that 
classified correctly by the network.

##### 2.c) Compute Distances from Mean Activation Vector

Once the mean activation vector for each category is computed, the next step is to 
compute category specific distance distribution. 

```python
>>> from scipy.io import loadmat
>>> distance_distribution = loadmat('data/mean_distance_files/n01440764_distances.mat')
>>> print distance_distribution.keys()
['__header__', 'eucos', '__globals__', 'euclidean', 'cosine', '__version__']
>>> print distance_distribution['euclidean'].shape
(10, 1224)
```

Distances (e.g. euclidean or cosine or eucos) are computed from MAV to correctly classified 
training examples. In the above example, n01440764 category has 1300 training images.
During training process, only 1224 images were classified correctly by the network. Thus, we build
distance distribution using these correctly classified images. These distances are computed 
for each channel (for defn of channel see point 2.a). This process forms a distance 
distribution from the MAV. This distance distribution is used for Weibull fitting. In our 
experiments, we consider tail size for Weibull fitting as 20. This means, that 20 distances
farthest away from MAV were used to estimate Weibull parameters. Again, since the distance
 computation process is very time comsuming, we are providing pre-computed distances for
each class, so that you have access to the distance distribution from MAV for each
category. It can be downloaded from

```bash
wget http://vast.uccs.edu/OSDN/data.tar
```

There are 1000 distance distributions, one for each category. The script used for computing
distance distributions can be found in

```python
python preprocessing/compute_distances.py
```

##### 2.d) Details of Weibull tail fitting on distance distributions can be found in the
function weibull_tailfitting() in file evt_fitting.py . However, compute_openmax.py performs
Weibull tailfitting for each category while computing probability for OpenMax
	
#### 3) Computing Probability values using OpenMax algorithm

OpenMax probability for given image can be computed using following command.

```python
python compute_openmax.py --image_arrname data/train_features/n01440764/n01440764_14280.JPEG.mat
```

The script accepts image feature files (features extracted from caffe as mentioned above). It computes openmax probability for the said image using default weibull tail sizes and other parameters. For more details, check the paper or get in touch with authors.

#### 4) Fooling Images
Fooling images are generously provided by Anh Nguyen and Prof. Jeff Clune from University of Wyoming. We will upload the fooling images and features extracted for fooling images in few days. When you use fooling images, please be sure to cite the following paper

```
@InProceedings{nguyen2015deep,
title={Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images},
author={Nguyen, Anh and Yosinski, Jason and Clune, Jeff},
booktitle={Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on},
year={2015},
organization={IEEE}
}
```


