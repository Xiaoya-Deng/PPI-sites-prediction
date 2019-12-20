# Prediction of protein-protein interaction sites using convolutional neural network and improved data sets.


Authors
-------
Zengyan Xie, Xiaoya Deng, Kunxian shu.

Chongqing Key Laboratory of Big Data for Bio Intelligence, Chongqing University of Posts and Telecommunications, Chongqing 400065, China;

Description
-----------
Protein-protein interaction (PPI) sites plays a key role in the formation of protein complex which is the basis of a variety of biological process. Experimental methods to solve PPI sites are expensive and time-consuming, which leads to the development of different kinds of prediction algorithms. We proposed a convolutional neural network for PPI sites prediction and used residue binding propensity to improve the positive samples. Our method obtains a remarkable result of AUC=0.912 on the improved data set. Comparison of the result of samples with high propensity to that of randomly selected samples suggests that there are considerable false positive PPI sites in the positive samples defined by distance between residue atoms.

Usage
-----

__DEPENDENCIES__

Our tools depends upon the following:

* Python 3.5

* Tensorflow 1.10.0

* Python modules: Numpy, Matplotlib, re, sys, os, random, sklearn

* Tools: PSAIA, PSI-BLAST

Please install these dependencies before using our tools. 

__USAGE__

1. Feature Extraction:

You can extract the features described in our paper. 
	
2. The training and testing is pretty simple. Just follow the following steps:

* Put feature files of each complex in a fold. 

* Run leave_one_complex.py, then you can get AUC of each complex by using leave-one-complex-out validation.

* Run kfold.py, then you can get the result of 5-fold cross-validation.

_We tested our model on 8  Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz and NVIDIA Corporation GP102 [TITAN Xp] (rev a1)._
