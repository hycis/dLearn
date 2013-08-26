
from pylearn2.datasets.mnist import MNIST

from dLearn.learning_model.mlp_layer.softmax import SoftMax
from dLearn.learning_model.mlp import MLP
from dlearn.error_function.mse import MSE

def main():
    
    h1 = SoftMax()
    h2 = SoftMax()
    h3 = SoftMax()
    y_layer = SoftMax()
                
    mlp = MLP(input_space,
              layers=[h1, h2, h3, y_layer],
              error_function = MSE(),
              dataset = MNIST(),
              batch_size)
    
    mlp.setup()
    mlp.train()
    

    
    
    
    


if __name__ == '__main__':
    main()