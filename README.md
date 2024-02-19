# QNN
Quaternion neural network  

This repository contains code for implementation of quaternion neural network (QNN) in Pytorch framework

The files contain slightly modified  original code from hTorch library (https://github.com/ispamm/hTorch)

The repository contains also example of simple 2 layer QNN model, trained with Adam optimization algorithm: example1.py

and the same QNN model trained with L-BFGS (Limited memory Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm: example2.py

The L-BFGS method is sensitive to initial values of hyperparameters, however in many cases it offers much faster convergence or trainig speed


# Installation

After cloning the repository, the pytorch installation should be installed (https://pytorch.org/get-started/locally/)

Pytorch installation with GPU support is strongly connected with correct CUDA version... 

Anaconda package software is recommended or creating virtual environment
