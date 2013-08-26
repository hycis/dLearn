'''
Created on Aug 25, 2013

@author: zhenzhou
'''

from dLearn.learning_model.mlp_layer.layer import Layer

class SoftMax(Layer):
    
    def __init__(self, size=[], init_range=None, output_space=None):
        pass
    
       
    def fprop(self, X):
        
    
    def get_weight(self):
        raise NotImplementedError(str(type(self)) + 'did not implement get_weight')
    
    def get_bias(self):
        raise NotImplementedError(str(type(self)) + 'did not implement get_bias()')
    
    def shape(self):
        raise NotImplementedError(str(type(self)) + 'did not implement shape()')
    
    
