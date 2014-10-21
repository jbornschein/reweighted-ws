#!/bin/sh
#

DOWN=wget
#DOWN=curl

$DOWN http://www.capsec.org/datasets/adult.h5
$DOWN http://www.capsec.org/datasets/connect4.h5
$DOWN http://www.capsec.org/datasets/dna.h5
$DOWN http://www.capsec.org/datasets/mnist.pkl.gz
$DOWN http://www.capsec.org/datasets/mnist_salakhutdinov.pkl.gz
$DOWN http://www.capsec.org/datasets/mushrooms.h5
$DOWN http://www.capsec.org/datasets/ocr_letters.h5
$DOWN http://www.capsec.org/datasets/rcv1.h5
$DOWN http://www.capsec.org/datasets/web.h5

