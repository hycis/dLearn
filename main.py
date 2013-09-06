
from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.mlp import MLP
from dLearn.error_function import cross_entropy_theano
from dLearn.datasets import Dataset

from numpy import random, asarray, split
from theano import shared

def main():
    
    #import pdb
    #pdb.set_trace()
    
    ###################
    #BUILD THE DATASET#
    ###################
    print 'build the dataset'
    train_set = MNIST(which_set='train', one_hot=True)
    test_set = MNIST(which_set='test', one_hot=True)
    
    new_train_setX, new_valid_setX = split(train_set.X, [50000], axis=0)
    new_train_sety, new_valid_sety = split(train_set.y, [50000], axis=0)
    
    train_set = Dataset(X=shared(new_train_setX), y=shared(new_train_sety))
    valid_set = Dataset(X=shared(new_valid_setX), y=shared(new_valid_sety))
    test_set = Dataset(X=shared(test_set.X), y=shared(test_set.y))
    
    ##################
    #BUILD THE LAYERS#
    ##################
    print 'build the layers'
    input_shape = list(new_train_setX[0].shape) + [1]

    h1 = Sigmoid(prev_layer_shape=input_shape, this_layer_shape=[10,1])
    #h2 = Sigmoid(prev_layer_shape=h1.this_layer_shape, this_layer_shape=[100,1])
    #y_layer = Sigmoid(prev_layer_shape=h2.this_layer_shape, this_layer_shape=[10,1])
    
    print 'build the model'
    mlp = MLP(input_shape=input_shape,
              layers=[h1],
              train_set=train_set,
              valid_set=valid_set,
              test_set=test_set)

    mlp.train_batch(100000)
    
if __name__ == '__main__':
    import os
    os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()