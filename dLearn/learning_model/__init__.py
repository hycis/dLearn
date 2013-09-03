class LearningModel(object):
    
    def train(self):
        raise NotImplementedError(str(type(self)) + 'did not implement train()')
    
    def continue_learning(self):
        raise NotImplementedError(str(type(self)) + 'did not implement continue_learning()')

    def train_batch(self, num_of_batches):
        raise NotImplementedError(str(type(self)) + 'did not implement train_batch(num_of_batches)')

    def save(self):
        raise NotImplementedError(str(type(self)) + 'did not implement save()')
