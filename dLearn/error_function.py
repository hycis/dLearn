'''
Created on Aug 26, 2013

@author: zhenzhou
'''
import theano.tensor as T

def cross_entropy_theano(y_batch, y_hat_batch):
    '''
    params:
        y_batch: theano matrix of y in a batch
        y_hat_batch: theano matrix of y_hat in a batch 
    '''
    return (-y_batch * T.log(y_hat_batch) - (1-y_batch) * T.log(1-y_hat_batch)).sum()

def mse(y, y_hat):

    
    pass

def sum_of_errors(y, y_hat):
    pass