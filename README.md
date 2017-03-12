# SVM MNIST digit classification in python using scikit-learn

Project presents well known problem of [MNIST handwritten digit classification](https://en.wikipedia.org/wiki/MNIST_database). For the puropose of this tutorial I will use [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) algorithm with raw pixel features. Solution is written in python with use of [scikit-learn](http://scikit-learn.org/stable/) easy to use machine learning library.


The goal of this project is to not to achieve the state of the art performance, rather to teach you **how to train SVM classifier on image data**. 
If you want to hit the top performance, this two resources will show you current state of the art

* [Who is the best in MNIST ?](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
* [Kaggle digit recognizer comptetition](https://www.kaggle.com/c/digit-recognizer)

Table below shows comparision with other models:


| Method                                     | Accuracy | Comments     |
|--------------------------------------------|----------|--------------|
| Random forest                              | 0.937    |              |
| Simple one-layer neural network            | 0.926    |              |
| Simple 2 layer convolutional network       | 0.981    |              |
| SVM RBF                                    |          | C=?, gamma=? |
| Linear SVM + Nystroem kernel approximation |          |              |
| Linear SVM +Fourier kernel approximation   |          |              |
| Linear SVM +Random Kitchen Sinks           |          |              |


## Solution

In this tutorial I use two approches for SVM learning. First, uses classical SVM with RBF kernel. The drawback of this solution is rather long training on big datasets, although the accuracy with good parameters is high. The second, uses Linear SVM, which allows for training in O(n) time. In order to achieve high accuracy we use some trick. We aproximate RBF kernel in a high dimensional space by embedings. The teory behind is quite complicated, however [scikit-learn has ready to use clases for kernel approximation](http://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation
). We will use:

* Nystroem kernel approximation
* Fourier kernel approximation
* Random Kitchen Sinks



## How the project is organised

Project consist of three files:

* mnist_helpers.py - contains some visualization functions
* svm_mnist_classification.py - main file for SVM with RBF kernel classification
* svm_mnist_embedings.py - script for linear SVM with embedings

**SVM with RBF kernel**

The svm_mnist_classification.py script downloads the MNIST database and visualize some random digits. Next, it standarize the data (mean=0, std=1) and lauchn grid search with cross validation for finding the best parameters.

Grid search is very time consuming process, so you can use my best parameters (from the range C=[], gamma=[]):
* C = ??
* gamma = ??

With this params:

* training time =
* accuracy: 



**Linear SVM with different embedings**





## Further improvements
 
* Augmenting the training set with artificial samples


## Source code

* Download code from GitHub: [https://github.com/ksirg/svm_mnist_digit_classification](https://github.com/ksirg/svm_mnist_digit_classification)
* Run project at PLON.io: [https://plon.io/explore/svm-mnist-handwritten-digit/USpQjoNcO8QHlmG6T](https://plon.io/explore/svm-mnist-handwritten-digit/USpQjoNcO8QHlmG6T)


## Useful SVM MNIST learning materials

* [MNIST handwritten digit recognition](http://brianfarris.me/static/digit_recognizer.html) - author compares a accuracy of a few machine learning classification algorithms (Random Forest, Stochastic Gradient Descent, Support Vector Machine, Nearest Neighbors)
* [Digit Recognition using OpenCV, sklearn and Python](http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html) - this blog post presents using HOG features and a multiclass Linear SVM.
* [Fast and Accurate Digit Classification- technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-159.html) - there is also download page with custom [LibLinear intersection kernel](http://ttic.uchicago.edu/~smaji/projects/digits/)
* [Random features for large-scale kernel machines](http://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf) Rahimi, A. and Recht, B. - Advances in neural information processing 2007,
* [Efficient additive kernels via explicit feature maps](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf) Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010
* [Generalized RBF feature maps for Efficient Detection](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/sreekanth10generalized.pdf) Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010

 
