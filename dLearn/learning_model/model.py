'''
Created on Aug 25, 2013

@author: zhenzhou
'''
class LearningModel(object):
    
    def train(self):
        raise NotImplementedError(str(type(self)) + 'did not implement train()')
    
    def continue_learning(self):
        raise NotImplementedError(str(type(self)) + 'did not implement continue_learning()')

    def train_batch(self):
        raise NotImplementedError(str(type(self)) + 'did not implement train_batch()')
