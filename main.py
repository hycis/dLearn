
from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.mlp import MLP
from dLearn.error_function import cross_entropy_theano

from numpy import random, asarray, split

def main():
    
    #import pdb
    #pdb.set_trace()
    
    ###################
    #BUILD THE DATASET#
    ###################
    train_set = MNIST(which_set='train', one_hot=True)
    test_set = MNIST(which_set='test', one_hot=True)

    # reshape to (60000, 784, 1)
    new_train_setX = train_set.X.reshape(60000, 784, 1)
    new_train_sety = train_set.y.reshape(60000, 10, 1)
    
    new_train_setX, new_valid_setX = split(new_train_setX, [50000], axis=0)
    new_train_sety, new_valid_sety = split(new_train_sety, [50000], axis=0)
    
    train_set = asarray([new_train_setX, new_train_sety])
    valid_set = asarray([new_valid_setX, new_valid_sety])
    test_set = asarray([test_set.X.reshape(10000,784,1), 
                        test_set.y.reshape(10000,10,1)])
    
    input_shape = list(new_train_setX[0].shape)
    
    ##################
    #BUILD THE LAYERS#
    ##################
    h1 = Sigmoid(prev_layer_shape=input_shape, this_layer_shape=[1000,1])
    h2 = Sigmoid(prev_layer_shape=h1.this_layer_shape, this_layer_shape=[1000,1])
    y_layer = Sigmoid(prev_layer_shape=h2.this_layer_shape, this_layer_shape=[10,1])
    
    mlp = MLP(input_shape=input_shape,
              layers=[h1,h2,y_layer],
              train_set=train_set,
              valid_set=valid_set,
              test_set=test_set)

    mlp.train_batch(10)
    
if __name__ == '__main__':
    import os
    os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()