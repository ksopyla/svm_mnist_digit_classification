# SVM MNIST digit classification in python using scikit-learn

The project presents the well-known problem of [MNIST handwritten digit classification](https://en.wikipedia.org/wiki/MNIST_database).
For the purpose of this tutorial, I will use [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) 
the algorithm with raw pixel features. 
The solution is written in python with use of [scikit-learn](http://scikit-learn.org/stable/) easy to use machine learning library.

![Sample MNIST digits visualization](/images/mnist_digits.png)



The goal of this project is not to achieve the state of the art performance, rather teach you 
**how to train SVM classifier on image data** with use of SVM from sklearn. 
Although the solution isn't optimized for high accuracy, the results are quite good (see table below). 

If you want to hit the top performance, this two resources will show you current state of the art solutions:

* [Who is the best in MNIST ?](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
* [Kaggle digit recognizer comptetition](https://www.kaggle.com/c/digit-recognizer)

The table below shows some results in comparison with other models:


| Method                                     | Accuracy | Comments     |
|--------------------------------------------|----------|--------------|
| Random forest                              | 0.937    |              |
| Simple one-layer neural network            | 0.926    |              |
| Simple 2 layer convolutional network       | 0.981    |              |
| SVM RBF                                    | 0.9852   | C=5, gamma=0.05 |
| Linear SVM + Nystroem kernel approximation |          |              |
| Linear SVM + Fourier kernel approximation  |          |              |


## Project Setup

This tutorial was written and tested on Ubuntu 18.10.
Project contains the Pipfile with all necessary libraries

* Python - version >= 3.6 
* pipenv - package and virtual environment management 
* numpy
* matplotlib
* scikit-learn


1. Install Python.
1. [Install pipenv](https://pipenv.readthedocs.io/en/latest/install/#pragmatic-installation-of-pipenv)
1. Git clone the repository
1. Install all necessary python packages executing this command in terminal

```
git clone https://github.com/ksopyla/svm_mnist_digit_classification.git
cd svm_mnist_digit_classification
pipenv install
```




## Solution

In this tutorial, I use two approaches to SVM learning. 
First, uses classical SVM with RBF kernel. The drawback of this solution is rather long training on big datasets, although the accuracy with good parameters is high. 
The second, use Linear SVM, which allows for training in O(n) time. In order to achieve high accuracy, we use some trick. We approximate RBF kernel in a high dimensional space by embeddings. The theory behind is quite complicated, 
however [sklearn has ready to use classes for kernel approximation](http://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation). 
We will use:

* Nystroem kernel approximation
* Fourier kernel approximation

The code was tested with python 3.6.


## How the project is organized

Project consist of three files:

* _mnist_helpers.py_ - contains some visualization functions: MNIST digits visualization and confusion matrix
* _svm_mnist_classification.py_ - script for SVM with RBF kernel classification
* _svm_mnist_embedings.py_ - script for linear SVM with embedings

### SVM with RBF kernel

The **svm_mnist_classification.py** script downloads the MNIST database and visualizes some random digits.
Next, it standardizes the data (mean=0, std=1) and launch grid search with cross-validation for finding the best parameters.

1. MNIST SVM kernel RBF Param search C=[0.1,0.5,1,5], gamma=[0.01,0.0.05,0.1,0.5].

Grid search was done for params C and gamma, where C=[0.1,0.5,1,5], gamma=[0.01,0.0.05,0.1,0.5].
I have examined only 4x4 different param pairs with 3 fold cross validation so far (4x4x3=48 models), 
this procedure takes 3687.2min :) (2 days, 13:56:42.531223 exactly) on one core CPU.

Param space was generated with numpy logspace and outer matrix multiplication. 
```
C_range = np.outer(np.logspace(-1, 0, 2),np.array([1,5]))
# flatten matrix, change to 1D numpy array
C_range = C_range.flatten()

gamma_range = np.outer(np.logspace(-2, -1, 2),np.array([1,5]))
gamma_range = gamma_range.flatten()

```
Of course, you can broaden the range of parameters, but this will increase the computation time.


![SVM RBF param space](https://plon.io/files/58d3af091b12ce00012bd6e1)

Grid search is very time consuming process, so you can use my best parameters 
(from the range c=[0.1,5], gamma=[0.01,0.05]):
* C = 5
* gamma = 0.05
* accuracy = 0.9852


```
Confusion matrix:
[[1014    0    2    0    0    2    2    0    1    3]
 [   0 1177    2    1    1    0    1    0    2    1]
 [   2    2 1037    2    0    0    0    2    5    1]
 [   0    0    3 1035    0    5    0    6    6    2]
 [   0    0    1    0  957    0    1    2    0    3]
 [   1    1    0    4    1  947    4    0    5    1]
 [   2    0    1    0    2    0 1076    0    4    0]
 [   1    1    8    1    1    0    0 1110    2    4]
 [   0    4    2    4    1    6    0    1 1018    1]
 [   3    1    0    7    5    2    0    4    9  974]]
Accuracy=0.985238095238
```


2. MNIST SVM kernel RBF Param search C=[0.1,0.5,1,5, 10, 50], gamma=[0.001, 0.005, 0.01,0.0.05,0.1,0.5].

This much broaden search 6x8 params with 3 fold cross validation gives 6x8x3=144 models, 
this procedure takes **13024.3min**  (9 days, 1:33:58.999782 exactly) on one core CPU.

![SVM RBF param space](https://plon.io/files/58e171451b12ce00012bd71d)

Best parameters:
* C = 5
* gamma = 0.05
* accuracy = 0.9852



### Linear SVM with different embeddings

Linear SVM's (SVM with linear kernels) have this advantages that there are many O(n)
training algorithms. They are really fast in comparison with other nonlinear SVM (where most of them are O(n^2)).
This technique is really useful if you want to train on big data.

Linear SVM algortihtms examples(papers and software):

* [Pegasos](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)
* [Stochastic gradient descent](http://leon.bottou.org/projects/sgd)
* [Averaged Stochastic gradient descent](https://arxiv.org/abs/1107.2490)
* [Liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
* [Stochastic Gradient Descent with Barzilai–Borwein update step for SVM](http://www.sciencedirect.com/science/article/pii/S0020025515002467)
* [Primal SVM by Olivier Chappelle](http://olivier.chapelle.cc/primal/) - there also exists [Primal SVM in Python](https://github.com/ksopyla/primal_svm)

Unfortunately, linear SVM isn't powerful enough to classify data with accuracy 
comparable to RBF SVM.

Learning SVM with RBF kernel could be time-consuming. In order to be more expressive, we try to approximate
nonlinear kernel, map vectors into higher dimensional space explicitly and use fast linear SVM in this new space. This works extremely well!


The script _svm_mnist_embedings.py_ presents accuracy summary and training times for 
full RBF kernel, linear SVC, and linear SVC with two kernel approximation 
Nystroem and Fourier.




## Further improvements
 
* Augmenting the training set with artificial samples
* Using Randomized param search


## Useful SVM MNIST learning materials

* [MNIST handwritten digit recognition](http://brianfarris.me/static/digit_recognizer.html) - author compares an accuracy of a few machine learning classification algorithms (Random Forest, Stochastic Gradient Descent, Support Vector Machine, Nearest Neighbors)
* [Digit Recognition using OpenCV, sklearn and Python](http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html) - this blog post presents using HOG features and a multiclass Linear SVM.
* [Grid search for RBF SVM parameters](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
* [Fast and Accurate Digit Classification- technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-159.html) - there is also download page with custom [LibLinear intersection kernel](http://ttic.uchicago.edu/~smaji/projects/digits/)
* [Random features for large-scale kernel machines](http://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf) Rahimi, A. and Recht, B. - Advances in neural information processing 2007,
* [Efficient additive kernels via explicit feature maps](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf) Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010
* [Generalized RBF feature maps for Efficient Detection](http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/sreekanth10generalized.pdf) Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010

 
