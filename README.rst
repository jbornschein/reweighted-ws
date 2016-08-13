.. image:: https://img.shields.io/shippable/557c82e6edd7f2c05214d9b0/master.svg
    :target: https://app.shippable.com/projects/557c82e6edd7f2c05214d9b0/builds/latest

.. image:: https://requires.io/github/jbornschein/reweighted-ws/requirements.svg?branch=master
    :target: https://requires.io/github/jbornschein/reweighted-ws/requirements/?branch=master
    :alt: Requirements Status

.. image:: https://img.shields.io/github/license/jbornschein/reweighted-ws.svg
    :target: http://choosealicense.com/licenses/agpl-3.0/
    :alt: AGPLv3


Reweighted Wake-Sleep
=====================

This repository contains the implementation of the machine learning 
method described in http://arxiv.org/abs/1406.2751 . 

*Note: There is an alternative implementation based on Blocks/Theano in https://github.com/jbornschein/bihm*

Installation & Requirements 
---------------------------

This implementation in written in Python and uses Theano. To automatically
install all dependencies run

 pip install -r requirements.txt

In order to reproduce the experiments in the paper you need to download about 
500 MB of training data:

 cd data
 sh download.sh

