

'''
Created on Aug 25, 2013

@author: zhenzhou
'''
from theano import shared, config, function, tensor
from numpy import random, asarray, zeros, log, count_nonzero


from dLearn.learning_model.layer import Layer

class NoisyRELU(Layer):
    
    def __init__(self, prev_layer_size, this_layer_size, 
                 W_range=[-0.5,0.5], b_range=[-0.5,0.5], 
                 type=['NORMAL'], noise_factor = 1):
        '''
        params:
            prev_layer_size: size of previous layer in tuples of [n1, [n2]], 
                where n1, n2 are the row and column dimensions of the layer.
            this_layer_size: size of previous layer in tuples of [n1, [n2]], 
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

        self.prev_layer_size = prev_layer_size
        self.this_layer_size = this_layer_size
        self.type = type
                
        W_dim = [self.prev_layer_size, self.this_layer_size]
        b_dim = self.this_layer_size
        
        self.W_values = asarray(random.uniform(low=W_range[0],
                                               high=W_range[1],
                                               size=W_dim),
                                dtype=config.floatX)

        self.b_values = asarray(random.uniform(low=W_range[0],
                                               high=W_range[1], 
                                               size=b_dim), 
                                dtype=config.floatX)
        
        self.W_theano = shared(self.W_values)
        self.b_theano = shared(self.b_values)
        
        self.noise_factor = noise_factor
        
        self.params_theano = [self.W_theano, self.b_theano]
        
        #assert len(prev_layer_size) is 2, 'give prev_layer_size=[n1,n2]'
        #assert len(this_layer_size) is 2, 'give this_layer_size=[n1,n2]'
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
        
        u = random.uniform(low=0.001, high=0.999, size = self.this_layer_size)
        noise = log(u/(1-u))
        
        if self.type[0] is 'NORMAL':
            output_theano = tensor.dot(input_theano, self.W_theano) + self.b_theano + self.noise_factor * noise
        elif self.type[0] is 'DROPOUT':
            pass
        elif self.type[0] is 'MAXOUT':
            pass
                
        return tensor.maximum(0, output_theano)
    
    def get_active_rate(self, X, batch_size):
        return float(count_nonzero(self.fprop(X))) / self.this_layer_size / batch_size
    
    def fprop(self, X):
        '''
        params: X - numpy ndarray
        return: a numpy ndarray after applying softmax(X)
        '''
        input_theano = tensor.matrix()      
        f = function(inputs=[input_theano], outputs=self.fprop_theano(input_theano))
        return f(X)
        
    def get_size(self):
        return self.size
    
    
