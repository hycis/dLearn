

from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.layer.softmax import Softmax
from dLearn.learning_model.layer.noisyRELU import NoisyRELU

from dLearn.learning_model.layer.tanh import Tanh

from dLearn.learning_model.mlp import MLP
from dLearn.error_function import cross_entropy_theano, loglikehood
from dLearn.datasets import Dataset

from numpy import random, asarray, split
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
    
    

    h1 = NoisyRELU(prev_layer_size=input_size, this_layer_size=2000, noise_factor=1, threshold=5)
    output_layer = Softmax(prev_layer_size=h1.this_layer_size, this_layer_size=10, 
                           W_range=[0,0], b_range=[0,0])
    #y_layer = Sigmoid(prev_layer_size=h2.this_layer_size, this_layer_size=[10,1])
    
#     import pdb
#     pdb.set_trace()

#, 0.1, 0.5, 0.001, 0.005

    plots = []
    legends = []
    batches = []
    errors = []
    for learning_rate in [0.1, 0.5]:
        print 'build the model'
        print 'learning rate', learning_rate
        mlp = MLP(input_size=input_size,
                  layers=[h1, output_layer],
                  train_set=[train_setX, train_sety],
                  valid_set=[valid_setX, valid_sety],
                  test_set=[test_set.X, test_set.y],
                  error_function=loglikehood,
                  batch_size=20,
                  learning_rate=learning_rate)
     
        print 'start training'
        mlp.train()
        #p = plt.plot(mlp.epoch, mlp.valid_error)
        #plots.append(p)
        batches.append(mlp.epoch)
        errors.append(mlp.valid_error)
        legends.append(learning_rate)
    
    with open('batches1.pkl', 'wb') as bat:
        cPickle.dump(batches, bat)
    with open('errors1.pkl', 'wb') as err:
        cPickle.dump(errors, err)
    
    with open('legends1.pkl', 'wb') as leg:
        cPickle.dump(legends, leg)
    
    #plt.legend(plots, legends)
    #plt.savefig('plot.png')
        
        
    
if __name__ == '__main__':
    import os
    #os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()