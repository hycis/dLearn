
from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.mlp import MLP
from dLearn.error_function import cross_entropy_theano, loglikehood
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
    train_set = MNIST(which_set='train', one_hot=False)
    test_set = MNIST(which_set='test', one_hot=False)
    
    train_setX, valid_setX = split(train_set.X, [50000], axis=0)
    train_sety, valid_sety = split(train_set.y, [50000], axis=0)
    

    #import pdb
    #pdb.set_trace()
    ##################
    #BUILD THE LAYERS#
    ##################
    print 'build the layers'
    input_size = len(train_setX[0])

    h1 = Sigmoid(prev_layer_size=input_size, this_layer_size=10)
    #h2 = Sigmoid(prev_layer_size=h1.this_layer_size, this_layer_size=[100,1])
    #y_layer = Sigmoid(prev_layer_size=h2.this_layer_size, this_layer_size=[10,1])
    
    print 'build the model'
    mlp = MLP(input_size=input_size,
              layers=[h1],
              train_set=[train_setX, train_sety],
              valid_set=[valid_setX, valid_sety],
              test_set=[test_set.X, test_set.y],
              error_function=loglikehood )

    mlp.train_batch(100000)
    
if __name__ == '__main__':
    import os
    os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()