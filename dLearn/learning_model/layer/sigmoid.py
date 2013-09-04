'''
Created on Aug 25, 2013

@author: zhenzhou
'''


'''
Created on Aug 25, 2013

@author: zhenzhou
'''
from theano import shared, config, function, tensor
from numpy import random, asarray, float32, float64


from dLearn.learning_model.layer import Layer

class Sigmoid(Layer):
    
    def __init__(self, prev_layer_shape, this_layer_shape, W_range=[-0.5,0.5], b_range=[-0.5,0.5], type=['NORMAL']):
        '''
        params:
            prev_layer_shape: shape of previous layer in tuples of [n1, [n2]], 
                where n1, n2 are the row and column dimensions of the layer.
            this_layer_shape: shape of previous layer in tuples of [n1, [n2]], 
                where n1, n2 are the row and column dimensions of the layer.
            W_range: if W_range = [min, max], then the weights are initialized
                from Uniform distribution w ~ U(min, max).
                if W_range = [val], then the weights are initialized to val
            b_range: if b_range = [min, max], then the biases are initialized
                from Uniform distribution b ~ U(min, max).
                if b_range = [val], then the biases are initialized to val
            type: 
                ['NORMAL'] - normal mlp layer
                ['DROPOUT', dropout_rate] - dropout layer
                ['MAXOUT'] - maxout layer
        '''

        self.prev_layer_shape = prev_layer_shape
        self.this_layer_shape = this_layer_shape
        self.type = type
        
        W_dim = self.this_layer_shape + self.prev_layer_shape
        b_dim = self.this_layer_shape
        self.W_values = asarray(random.uniform(low=W_range[0],
                                               high=W_range[1],
                                               size=(W_dim)))

        self.b_values = asarray(random.uniform(low=b_range[0],
                                               high=b_range[1],
                                               size=(b_dim)))
        
        self.W_theano = shared(self.W_values)
        self.b_theano = shared(self.b_values)
        
        self.params_theano = [self.W_theano, self.b_theano]
        
        #assert len(prev_layer_shape) is 2, 'give prev_layer_shape=[n1,n2]'
        assert len(this_layer_shape) is 2, 'give this_layer_shape=[n1,n2]'
        assert self.type[0] in ['NORMAL', 'DROPOUT', 'MAXOUT']
        if self.type[0] is 'DROPOUT': 
            assert self.type[1] > 0 and self.type[1] < 1  
    
        
    def set_params(self, W, b):
        self.W_values = W
        self.b_values = b
        self.W_theano = shared(self.W_values)
        self.b_theano = shared(self.b_values)
        
        self.params_theano = [self.W_theano, self.b_theano]
        
    def fprop_theano(self, input_theano):
        
        if self.type[0] is 'NORMAL':
            output_theano = (input_theano * self.W_theano).sum(axis=3).sum(axis=2) \
                            + self.b_theano
        elif self.type[0] is 'DROPOUT':
            pass
        elif self.type[0] is 'MAXOUT':
            pass
        
        return tensor.nnet.sigmoid(output_theano)
        
    def fprop(self, X):
        '''
        params: X - numpy ndarray
        return: a numpy ndarray after applying softmax(X)
        '''
        input_theano = tensor.dmatrix()      
        f = function(inputs=[input_theano], outputs=self.fprop_theano(input_theano))
        return f(X)
        
    def get_shape(self):
        return self.shape
    
    
