
import matplotlib
matplotlib.use('Agg')
import cPickle
import matplotlib.pyplot as plt

def plot_error(batch_error_list, legends, save_path='plot.png', axis_limits=[0, 25000, 0, 0.15]):
    '''
    error_list: list of pkl files, where each pkl file contains [batch_num, test_error]
    legends: list of legends for the datasets
    '''
    
    assert len(batch_error_list) == len(legends)
    
    plots = []
    for i in xrange(len(batch_error_list)):
        bat_err_ls = cPickle.load(open(batch_error_list[i], 'rb'))
        plots.append(plt.plot(bat_err_ls[0], bat_err_ls[1]))
        
    plt.axis(axis_limits)
    plt.legend(plots, legends)
    plt.savefig(save_path)


    
def test_learning_rates(model, learning_rates=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 
                        num_batches=10000, plot=False, axis_limits=[0, 25000, 0, 0.15],
                        save_path='learning_rate.png'):
    
    plots = []
    legends = []
    for learning_rate in learning_rates:
        model.set_learning_rate(learning_rate)
        model.train_batch(num_batches)
        
        with open('{}.pkl'.format(learning_rate)) as f:
            list = [model.batch_num, model.test_error]
            cPickle.dump(list, f)
        
        if plot:
            plots.append(plt.plot(model.batch_num, model.test_error))
            legends.append(learning_rate)
    
    if plot:
        plt.axis(axis_limits)
        plt.legend(plots, legends)
        plt.savefig(save_path)
            
 
        