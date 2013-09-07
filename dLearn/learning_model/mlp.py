'''
Created on Aug 25, 2013

@author: zhenzhou
'''

from dLearn.learning_model import LearningModel
from dLearn.error_function import loglikehood
from numpy import random, asarray, sqrt, inf, mean, float64, float32
from theano import shared, tensor, function, config

import cPickle
import math
import time

from pylearn2.datasets.mnist import MNIST



class MLP(LearningModel):
    
    def __init__(self,
                 input_size,  
                 layers, 
                 train_set,
                 valid_set, 
                 test_set,
                 error_function=loglikehood,  
                 batch_size=200,
                 learning_rate=0.01,
                 L1_reg = 0.00,
                 L2_reg = 0.0001):
        '''
        params:
            input_space - 
            layers - a list of layer object
            learning_algorithm - a string in ['SGD', 'BGD'] where 'SGD' means Stochastic
                Gradient Descend, and 'BGD' means Batch Gradient Descend
            error_function - the error function for calculating errors         
        '''
        

        self.floatX = train_set[0].dtype
        self.intX = train_set[1].dtype
        
        
        self.layers = layers
        self.train_set_X = shared(train_set[0].astype(self.floatX))
        self.train_set_y = shared(train_set[1].astype(self.intX))
        self.valid_set_X = shared(valid_set[0].astype(self.floatX))
        self.valid_set_y = shared(valid_set[1].astype(self.intX))
        self.test_set_X = shared(test_set[0].astype(self.floatX))
        self.test_set_y = shared(test_set[1].astype(self.intX))

        self.batch_size = batch_size
        self.error_function = error_function
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        
        # setup the train_model, test_model, valid_model and params
        self.__setup()
     
    def __setup(self):
        
        '''
        setup the train_model, test_model, valid_model and params
        '''
          
        self.params_theano = []
        for layer in self.layers:
            self.params_theano += layer.params_theano

        batch_x_theano = tensor.matrix(dtype=self.floatX)
        batch_y_theano = tensor.vector(dtype=self.intX)
        
        batch_y_hat_theano = self.fprop_theano(batch_x_theano)


        #error = -tensor.mean(tensor.log(batch_y_hat_theano)[tensor.arange(batch_y_theano.shape[0]), batch_y_theano])
        
        #f = function([batch_x_theano, batch_y_theano], error)

        #import pdb
        #pdb.set_trace()
        cost = self.cost_L1_L2_theano(batch_y_theano=batch_y_theano, 
                                   batch_y_hat_theano=batch_y_hat_theano)
        
        error = self.error(batch_y_theano=batch_y_theano, 
                           batch_y_hat_theano=batch_y_hat_theano)


        gparams = []
        for param in self.params_theano:
            gparam = tensor.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params_theano, gparams):
            updates.append((param, param - self.learning_rate*gparam))
        
        index = tensor.lscalar()
        #import pdb
        #pdb.set_trace()
        self.train_model = function(inputs=[index], outputs=error, updates=updates,
                               givens={batch_x_theano: 
                                       self.train_set_X[index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.train_set_y[index*self.batch_size:
                                                      (index+1)*self.batch_size]},
                                    allow_input_downcast=True)
        
        self.valid_model = function(inputs=[index], outputs=error,
                               givens={batch_x_theano: 
                                       self.valid_set_X[index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.valid_set_y[index*self.batch_size:
                                                      (index+1)*self.batch_size]},
                                    allow_input_downcast=True)

        self.test_model = function(inputs=[index], outputs=error,
                               givens={batch_x_theano: 
                                       self.test_set_X[index*self.batch_size:
                                                      (index+1)*self.batch_size],
                                       batch_y_theano:
                                       self.test_set_y[index*self.batch_size:
                                                      (index+1)*self.batch_size]},
                                    allow_input_downcast=True)
        
        #import pdb
        #pdb.set_trace()
            
#     def train(self):
#         
#         validation_freq = 1000
#         n_train_batch = self.train_set[0].size[0]
# 
#         
#         
#         assert validation_freq < n_train_batch / 10, 'at least 10 validations per epoch'
#         
#         start_time = time.clock()
#         
#         best_valid_loss = inf
#         improve_threshold = 0.995
#         
#         epoch = 0
#         continue_training = True
#         while continue_training:
#             epoch += 1
#             # loop for one epoch
#             continue_training = False
#             for batch_index in xrange(n_train_batch):
#                 batch_avg_cost = self.train_model(batch_index)
#                 
#                 if batch_index + 1 % validation_freq == 0:
#                     print 'self.valid_set.size[0]', self.valid_set.size[0]                            
#                     validation_losses = [self.valid_model(i) for i
#                                          in xrange(self.valid_set.size[0])]
#                     this_valid_loss = mean(validation_losses)
#                     print ('epoch %i, batch number %i/%i, validation error %f %%' %
#                            (epoch, batch_index, n_train_batch, this_valid_loss * 100.))          
#                     
#                     if this_valid_loss < best_valid_loss:
#                         best_valid_loss = this_valid_loss
#                         best_iter = [epoch, batch_index]
#                         
#                         test_losses = [self.test_model(i) for i 
#                                        in xrange(self.test_set.size[0])]
#                         this_test_loss = mean(test_losses)
#                         
#                         print ('epoch %i, batch number %i/%i, test error %f %%' %
#                                (epoch, batch_index, n_train_batch, this_test_loss * 100.))    
# 
#                         if best_valid_loss < improve_threshold * this_valid_loss:
#                             continue_training = True
                              
    def train_batch(self, num_batches):
        
        assert num_batches >= 10000, 'at least 10k batches'
        validation_freq = 1000 # the number of batches to train
                                    # before the next validation

        
        best_valid_loss = inf
        best_iter = 0
        batch = 1
        batch_index = batch - 1
        n_train_batch = self.train_set_X.eval().shape[0] / self.batch_size
        n_valid_batch = self.valid_set_X.eval().shape[0] / self.batch_size
        n_test_batch = self.test_set_X.eval().shape[0] / self.batch_size
        while batch < num_batches:
             
            batch_avg_cost = self.train_model(batch_index)
            
            if (batch) % validation_freq == 0:
                print 'validation in progress'
#                 validation_losses = []
#                 for i in xrange(n_valid_batch):
#                     print i
#                     print self.valid_set_X.eval()[i*self.batch_size : (i+1)*self.batch_size]
# 
#                     if math.isnan(self.valid_model(i)):
#                         print 'nan', i
#                         print self.valid_set_X.eval()[i*self.batch_size : (i+1)*self.batch_size]
#                         print self.valid_set_y.eval()[i*self.batch_size : (i+1)*self.batch_size]
#                         import pdb
#                         pdb.set_trace()
#                     validation_losses.append(self.valid_model(i))
                validation_losses = [self.valid_model(i) for i in
                                     xrange(n_valid_batch)]
                this_valid_loss = mean(validation_losses)
                
                #print validation_losses
                #print 'this_valid_loss', this_valid_loss
                
                print ('batch number %i/%i, validation error %f %%' %
                (batch, num_batches, this_valid_loss * 100.))
                
                if this_valid_loss < best_valid_loss:
                    best_valid_loss = this_valid_loss
                    best_iter = batch
                    
                    test_losses = [self.test_model(i) for i in 
                                   xrange(n_test_batch)]
                    this_test_loss = mean(test_losses)
                    print ('batch number %i/%i, test error %f %%' %
                    (batch, num_batches, this_test_loss * 100.))            
            batch = batch + 1
            batch_index = batch % n_train_batch - 1              
    

                   
            
    def cost_L1_L2_theano(self, batch_y_theano, batch_y_hat_theano):
        
        error = self.error_function(batch_y_theano, batch_y_hat_theano)
        
        L1 = shared(0)
        L2 = shared(0)
        for layer in self.layers:
            L1 += abs(layer.W_theano).sum()
            L2 += (layer.W_theano ** 2).sum()
            
        
        cost = error + self.L1_reg * L1 + self.L2_reg * L2
        return cost    
    
    def error(self, batch_y_theano, batch_y_hat_theano):
        return tensor.mean(tensor.neq(batch_y_theano,
                   tensor.argmax(batch_y_hat_theano, axis=1)))
    
    def fprop_theano(self, input_theano):
        for layer in self.layers:
            output_theano = layer.fprop_theano(input_theano)
            input_theano = output_theano
        
        return output_theano
    
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
        
        
    
    
    