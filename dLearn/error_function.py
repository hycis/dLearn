'''
Created on Aug 26, 2013

@author: zhenzhou
'''
from theano import tensor

def cross_entropy_theano(batch_y_theano, batch_y_hat_theano, batch_size):
    '''
    params:
        y_batch: theano matrix of y in a batch
        y_hat_batch: theano matrix of y_hat in a batch 
    '''
    return (-batch_y_theano * tensor.log(batch_y_hat_theano) - 
            (1-batch_y_theano) * tensor.log(1-batch_y_hat_theano)).sum() / batch_size

def abs_error(y_batch_theano, y_hat_batch_theano, batch_size):
    return tensor.abs_(y_batch_theano - y_hat_batch_theano).sum() / batch_size

def mse(y_batch, y_hat_batch):

    
    pass

def sum_of_errors(y, y_hat):
    pass