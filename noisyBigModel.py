

from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.svhn import SVHN_On_Memory

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.layer.softmax import Softmax
from dLearn.learning_model.layer.noisyRELU import NoisyRELU
from dLearn.learning_model.layer.tanh import Tanh


from dLearn.learning_model.layer.tanh import Tanh

from dLearn.learning_model.mlp import MLP
from dLearn.error_function import cross_entropy_theano, loglikehood
from dLearn.datasets import Dataset

from numpy import random, asarray, split, nonzero
from theano import shared

import matplotlib.pyplot as plt 

import cPickle

def main():
    
    #import pdb
    #pdb.set_trace()
    
    ###################
    #BUILD THE DATASET#
    ###################
    print 'build the dataset'
    
    import pdb
    pdb.set_trace()
    train_set = SVHN_On_Memory(which_set='train')
    test_set = SVHN_On_Memory(which_set='test')
    
    
    
    train_setX, valid_setX = split(train_set.X, [50000], axis=0)
    train_sety, valid_sety = split(train_set.y, [50000], axis=0)
    
    test_set.y = nonzero(test_set.y)[1]
    train_sety = nonzero(train_sety)[1] 
    valid_sety = nonzero(valid_sety)[1]
    

    #import pdb
    #pdb.set_trace()
    ##################
    #BUILD THE LAYERS#
    ##################
    print 'build the layers'
    input_size = len(train_setX[0])
    
    

    h1 = NoisyRELU(prev_layer_size=input_size, this_layer_size=1000, threshold=5, noise_factor=1)
    h2 = NoisyRELU(prev_layer_size=h1.get_size(), this_layer_size=1000, threshold=5, noise_factor=1)
    h3 = NoisyRELU(prev_layer_size=h2.get_size(), this_layer_size=1000, threshold=5, noise_factor=1)

    
    
    output_layer = Softmax(prev_layer_size=h1.this_layer_size, this_layer_size=10, 
                           W_range=[0,0], b_range=[0,0])
    #y_layer = Sigmoid(prev_layer_size=h2.this_layer_size, this_layer_size=[10,1])
    

    mlp = MLP(input_size=input_size,
              layers=[h1, h2, h3, output_layer],
              train_set=[train_setX, train_sety],
              valid_set=[valid_setX, valid_sety],
              test_set=[test_set.X, test_set.y],
              error_function=loglikehood,
              batch_size=20,
              learning_rate=0.1)
     
    print 'start training'
    mlp.train(save_path='1000-1000-1000-noisy.pkl', save_freq=1)
        #p = plt.plot(mlp.epoch, mlp.valid_error)
        #plots.append(p)
    
    #data = [mlp.epoch, mlp.valid_error]
    
    #with open('1000-1000-1000-noisy.pkl', 'wb') as batch_err:
     #   cPickle.dump(data, batch_err)
    
    #plt.legend(plots, legends)
    #plt.savefig('plot.png')


def mnist():
    
    #import pdb
    #pdb.set_trace()
    
    ###################
    #BUILD THE DATASET#
    ###################
    print 'build the dataset'
    

    train_set = MNIST(which_set='train')
    test_set = MNIST(which_set='test')
    
    
    
    train_setX, valid_setX = split(train_set.X, [50000], axis=0)
    train_sety, valid_sety = split(train_set.y, [50000], axis=0)
    

    #import pdb
    #pdb.set_trace()
    ##################
    #BUILD THE LAYERS#
    ##################
    print 'build the layers'
    input_size = len(train_setX[0])
    
    

    h1 = NoisyRELU(prev_layer_size=input_size, this_layer_size=1000, threshold=5, noise_factor=1)
    h2 = NoisyRELU(prev_layer_size=h1.get_size(), this_layer_size=1000, threshold=5, noise_factor=1)
    #h3 = NoisyRELU(prev_layer_size=h2.get_size(), this_layer_size=1000, threshold=5, noise_factor=1)

    
    
    output_layer = Softmax(prev_layer_size=h1.this_layer_size, this_layer_size=10, 
                           W_range=[0,0], b_range=[0,0])
    #y_layer = Sigmoid(prev_layer_size=h2.this_layer_size, this_layer_size=[10,1])
    

    mlp = MLP(input_size=input_size,
              layers=[h1, h2, output_layer],
              train_set=[train_setX, train_sety],
              valid_set=[valid_setX, valid_sety],
              test_set=[test_set.X, test_set.y],
              error_function=loglikehood,
              batch_size=20,
              learning_rate=0.1)
     
    print 'start training'
    mlp.train(save_freq=1)
        #p = plt.plot(mlp.epoch, mlp.valid_error)
        #plots.append(p)
    
    #data = [mlp.epoch, mlp.valid_error]
    

    
    #plt.legend(plots, legends)
    #plt.savefig('plot.png')
        
           
        
    
if __name__ == '__main__':
    import os
    #os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()
    
    