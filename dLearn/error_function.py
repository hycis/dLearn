'''
Created on Aug 26, 2013

@author: zhenzhou
'''
from theano import tensor

def cross_entropy_theano(batch_y_theano, batch_y_hat_theano):
    '''
    params:
        y_batch: theano matrix of y in a batch
        y_hat_batch: theano matrix of y_hat in a batch 
    '''
    
    
    
    
    return (-batch_y_theano * tensor.log(batch_y_hat_theano) - 
            (1-batch_y_theano) * tensor.log(1-batch_y_hat_theano)).sum() / batch_y_theano.shape[0]


def mse(y_batch, y_hat_batch):

    
    pass

def loglikehood(batch_y_theano, batch_y_hat_theano):
    return -tensor.mean(tensor.log(batch_y_hat_theano)[tensor.arange(batch_y_theano.shape[0]), batch_y_theano])

def sum_of_errors(y, y_hat):
    pass