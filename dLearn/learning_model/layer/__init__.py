


class Layer(object):
    
    def fprop(self, X):
        raise NotImplementedError(str(type(self)) + 'did not implement fprop(X)')
    
    def get_weight(self):
        raise NotImplementedError(str(type(self)) + 'did not implement get_weight')
    
    def get_bias(self):
        raise NotImplementedError(str(type(self)) + 'did not implement get_bias()')
    
    def shape(self):
        raise NotImplementedError(str(type(self)) + 'did not implement shape()')

        
