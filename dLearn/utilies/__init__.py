
import matplotlib
matplotlib.use('Agg')
import cPickle
import matplotlib.pyplot as plt

def plot_error(batches, errors, legends, save_path='plot.png'):
    '''
    batches: list of batch pkl files
    errors: list of error pkl files
    legends: list of legends pkl files
    '''
    
    assert len(batches) == len(errors) == len(legends)
    
    plots = []
    legs =[]
    for i in xrange(len(batches)):
        bat = cPickle.load(open(batches[i], 'rb'))
        err = cPickle.load(open(errors[i], 'rb'))
        leg = cPickle.load(open(legends[i], 'rb'))
        legs += leg
        plots.append(plt.plot(bat, err))
        
    
    plt.legend(plots, legs)
    plt.savefig(save_path)
 
        