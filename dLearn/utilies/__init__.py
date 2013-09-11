

import cPickle
import matplotlib.pyplot as plt

def plot_error(batches, errors, legends, save_path):
    '''
    batches: list of batch pkl files
    errors: list of error pkl files
    legends: list of legends pkl files
    '''
    
    assert len(batches) == len(errors) == len(legends)
    
    plots = []
    for i in xrange(len(batches)):
        bat = cPickle.load(open(batches[i], 'rb'))
        err = cPickle.load(open(errors[i], 'rb'))  
        plots.append(plt.plot(bat, err))
        
    leg = cPickle.load(open(legends[i], 'rb'))
    
    plt.legend(plots, leg)
    plt.savefig(save_path)
 
        