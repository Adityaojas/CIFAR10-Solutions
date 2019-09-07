# CIFAR10-Solutions
Various approaches to the Cifar 10 Image Classification Problem

Download the data from: https://www.kaggle.com/quanbk/cifar10
Place the files in the Cifar10 Directory

keras_cifar10.py consists of a simple Fully-Connected NN approach. It fetches an accuracy of about 58%. This can be improved much.

shallownet.py consists of the ShallowNet architecture. It's the smallest effective Deep Learning architecture. It only obtains an accuracy of 60% but it's better than the Fully Connected NN approach. The less accuracy can also be justified by the fact that that no effor was made to combat overfitting, like regularization.

minivggnet_aug.py applies the minivggnet architecture (A mini version of the VGG architectur) to the cifar 10 dataset along with explicit augmentation for regularization. I run 40 epochs initially, and after realizing and analysing the scope of imporovement, I run 15 more epochs. You can edit the code to run on 60 epochs straight away. the accuracy comes out to be about 82% which is way better than the previous attempts.
