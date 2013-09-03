
from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.layer.sigmoid import Sigmoid
from dLearn.learning_model.mlp import MLP
from dLearn.error_function import mse

from numpy import random, asarray

def main():
    
    h1 = Sigmoid(prev_layer_shape=[3,5], this_layer_shape=[3,5])
    h2 = Sigmoid(prev_layer_shape=[3,5], this_layer_shape=[3,5])
    y_layer = Sigmoid(prev_layer_shape=[3,5], this_layer_shape=[3,5])
                
    mlp = MLP(layers=[h1, h2, y_layer],
              error_function = mse,
              batch_size = 100
              )
    
    input = asarray(random.uniform(low=-1, high=1, size=(3,5)))
    
    #o1 = h1.fprop(input)
    #o2 = h2.fprop(o1)
    #o3 = y_layer.fprop(o2)
    
    y = mlp.fprop(input)
    
    print 'input',  input
    print 'y', y
    
    import pdb
    pdb.set_trace()
    #print o2
    #print o3
    #print y
    
    
    
    
    


if __name__ == '__main__':
    import os
    os.environ['PYLEARN2_DATA_PATH'] = '/Users/zhenzhou/Desktop/pylearn2/data'
    main()