'''
Created on Aug 25, 2013

@author: zhenzhou
'''
from dLearn.learning_model.model import LearningModel


class MLP(LearningModel):
    
    def __init__(self, 
                 input_space, 
                 layers, 
                 learning_algorithm, 
                 loss_function, 
                 dataset, 
                 batch_size):
        pass
    
    def setup(self):
        pass
    
    def train(self):
        pass
    
    def fprop(self, X):
        pass
    
    def append_layer(self, layer):
        pass
    
    def append_layers(self, layers):
        pass
    
    def remove_layer(self, layer_name):
        pass
    
    def remove_layers(self, layers_names):
        pass
    
    def get_layer_param(self, layer):
        pass
    
    def loss(self, y, y_hat):
        pass
    
    
    