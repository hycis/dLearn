'''
Created on Aug 25, 2013

@author: zhenzhou
'''

from dLearn.learning_model import LearningModel
from dLearn.error_function import cross_entropy_theano
from numpy import random, asarray, sqrt, inf, mean
from theano import shared, tensor, function

import cPickle

import time

from pylearn2.datasets.mnist import MNIST


class MLP(LearningModel):
    
    def __init__(self,
                 input_shape,  
                 layers, 
                 train_set,
                 valid_set, 
                 test_set,
                 error_function=cross_entropy_theano,  
                 batch_size=100,
                 learning_rate=0.01):
        '''
        params:
            input_space - 
            layers - a list of layer object
            learning_algorithm - a string in ['SGD', 'BGD'] where 'SGD' means Stochastic
                Gradient Descend, and 'BGD' means Batch Gradient Descend
            error_function - the error function for calculating errors         
        '''
        
        self.layers = layers
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.error_function = error_function
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        
        # setup the train_model, test_model, valid_model and params
        self.__setup()
     
    def __setup(self):
        
        '''
        setup the train_model, test_model, valid_model and params
        '''
             
        self.params_theano = []
        for layer in self.layers:
            self.params_theano += layer.params_theano

        # x_{brc : batch, row, col}
        batch_x_theano = tensor.dtensor3()
        batch_y_theano = tensor.dtensor3()
        
        batch_y_hat_theano = self.batch_fprop_theano(batch_x_theano)

        cost = self.cost_L1_theano(batch_y_theano=batch_y_theano, 
                                   batch_y_hat_theano=batch_y_hat_theano)


        gparams = []
        for param in self.params_theano:
            gparam = tensor.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params_theano, gparams):
            updates.append((param, param - self.learning_rate*gparam))
        
        index = tensor.lscalar()
        
        import pdb
        pdb.set_trace()
        self.train_model = function(inputs=[index], outputs=cost, updates=updates,
                               givens={batch_x_theano: 
                                       self.train_set[0][index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.train_set[1][index*self.batch_size:
                                                      (index+1)*self.batch_size]})
        self.valid_model = function(inputs=[index], outputs=cost, updates=updates,
                               givens={batch_x_theano: 
                                       self.valid_set[0][index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.valid_set[1][index*self.batch_size:
                                                      (index+1)*self.batch_size]})
        
        self.test_model = function(inputs=[index], outputs=cost, updates=updates,
                               givens={batch_x_theano: 
                                       self.test_set[0][index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.test_set[1][index*self.batch_size:
                                                      (index+1)*self.batch_size]})
        
        import pdb
        pdb.set_trace()
            
    def train(self):
        
        validation_freq = 1000
        n_train_batch = self.train_set[0].shape[0]

        
        
        assert validation_freq < n_train_batch / 10, 'at least 10 validations per epoch'
        
        start_time = time.clock()
        
        best_valid_loss = inf
        improve_threshold = 0.995
        
        epoch = 0
        continue_training = True
        while continue_training:
            epoch += 1
            # loop for one epoch
            continue_training = False
            for batch_index in xrange(n_train_batch):
                batch_avg_cost = self.train_model(batch_index)
                
                if batch_index + 1 % validation_freq == 0:
                    validation_losses = [self.valid_model(i) for i
                                         in xrange(self.valid_set.shape[0])]
                    this_valid_loss = mean(validation_losses)
                    print ('epoch %i, batch number %i/%i, validation error %f %%' %
                           (epoch, batch_index, n_train_batch, this_valid_loss * 100.))          
                    
                    if this_valid_loss < best_valid_loss:
                        best_valid_loss = this_valid_loss
                        best_iter = [epoch, batch_index]
                        
                        test_losses = [self.test_model(i) for i 
                                       in xrange(self.test_set.shape[0])]
                        this_test_loss = mean(test_losses)
                        
                        print ('epoch %i, batch number %i/%i, test error %f %%' %
                               (epoch, batch_index, n_train_batch, this_test_loss * 100.))    

                        if best_valid_loss < improve_threshold * this_valid_loss:
                            continue_training = True
                              
    def train_batch(self, num_batches):
        
        assert num_batches >= 10000, 'at least 10k batches'
        validation_freq = 1000 # the number of batches to train
                                    # before the next validation
        best_valid_loss = inf
        best_iter = 0
        batch = 0
        batch_index = batch
        n_train_batch = self.train_set[0].shape[0]
        n_valid_batch = self.valid_set[0].shape[0]
        n_test_batch = self.test_set[0].shape[0]
        while batch < num_batches:
            
            if batch >= n_train_batch:
                batch_index = batch % n_train_batch
            
            batch_avg_cost = self.train_model(batch_index)
            
            if batch + 1 % validation_freq == 0:
                validation_losses = [self.valid_model(i) for i in
                                     xrange(n_valid_batch)]
                this_valid_loss = mean(validation_losses)
                
                print ('batch number %i/%i, validation error %f %%' %
                (batch, num_batches, this_valid_loss * 100.))
                
                if this_valid_loss < best_valid_loss:
                    best_valid_loss = this_valid_loss
                    best_iter = batch
                    
                    test_losses = [self.test(i) for i in 
                                   xrange(n_test_batch)]
                    this_test_loss = mean(test_losses)
                    print ('batch number %i/%i, test error %f %%' %
                    (batch, num_batches, this_test_loss * 100.))
            
            batch = batch + 1
            batch_index = batch                
    

                   
            
    def cost_L1_theano(self, batch_y_theano, batch_y_hat_theano, L1_reg=10e-2):
        
        cost = self.error_function(batch_y_theano, batch_y_hat_theano) / self.batch_size
        
        L1 = tensor.dscalar()
        for layer in self.layers:
            L1 += abs(layer.W_theano).sum()
        
        #b1 = [abs(layer.b_theano).sum() for layer in self.layers]
        cost += L1_reg * L1
        return cost    
    
    def batch_fprop_theano(self, batch_input_theano):
        batch_output_theano = tensor.dtensor3()
        shape = batch_input_theano[0].shape
        for i in xrange(self.batch_size):
            batch_output_theano = tensor.concatenate([batch_output_theano,
                self.fprop_theano(batch_input_theano[i]).reshape((1,) + shape, ndim=3)], axis=0)
        return batch_output_theano
    
    def fprop_theano(self, input_theano):
        for layer in self.layers:
            output_theano = layer.fprop_theano(input_theano)
            input_theano = output_theano
        
        return output_theano
            
            
    def batch_fprop(self, batch):
        batch_input_theano = tensor.dmatrix()
        f = function([batch_input_theano], self.batch_fprop_theano(batch_input_theano))
        return f(batch)
    
    def fprop(self, x):
        input_theano = tensor.dmatrix()
        f = function([input_theano], self.fprop_theano(input_theano))
        return f(x)
    
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
    
    def save(self, save_path='model.pkl'):
        file = open(save_path, 'wb')
        cPickle.dump(self, file)
        file.close()
        
        
    
    
    