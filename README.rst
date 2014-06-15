
Reweighted Wake-Sleep
=====================

This repository contains the implementation of the machine learning 
method described in http://arxiv.org/abs/1406.2751 . 

Installation & Requirements 
---------------------------

This implementation in written in Python and uses Theano. To automatically
install all dependencies run

 pip install -r requirements.txt

In order to reproduce the experiments in the paper you need to download about 
500 MB of training data:

 cd data
 sh download.sh

