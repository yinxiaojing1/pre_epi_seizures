import matplotlib.pyplot as plt
import itertools


def single_param_plot(hp_opt_results, param_name, single_param_var):
    
    # Get table from the structure
    cv_results = hp_opt_results['cv_results']

    
    # Loop through the metrics
    metrics = [key 
               for key in cv_results.columns
               if 'param' not in key
               and 'rank' not in key]
    
    for metric in metrics:
        
        print metric
        
        metric_result = cv_results[metric]

        param_var = single_param_var

        plt.figure()
        plt.plot(param_var, metric_result, 'o')
        plt.title(metric)
        plt.xlabel(param_name)
        plt.ylabel(metric)
        #plt.savefig(path_to_save + '/' + key)
        plt.show()


def single_mean_std_param_plot(directory, 
                               hp_opt_results,
                               param_name,
                               single_param_var,
                               feature_slot):
    
    # Get table from the structure
    cv_results = hp_opt_results['cv_results']

    
    # Loop through the metrics to get mean
    mean_metrics = [key 
               for key in cv_results.columns
               if 'param' not in key
               and 'rank' not in key
               and 'mean' in key
               and 'time' not in key]
    
   # Loop through the metrics to get standard deviation
    std_metrics = [key 
               for key in cv_results.columns
               if 'param' not in key
               and 'rank' not in key
               and 'std' in key
               and 'time' not in key]
    
    
    for mean_metric, std_metric in zip(mean_metrics, std_metrics):
        
 
        
        # Get the results from the return structure
        mean_metric_result = cv_results[mean_metric]
        std_metric_result = cv_results[std_metric]
        param_var = single_param_var
        print 'This is the mean metric'
        print mean_metric
        print mean_metric_result
        
        print 'This is the std metric'
        print std_metric
        print std_metric_result
        # Plot the figures
        plt.figure()
        plt.errorbar(x=param_var, y=mean_metric_result, yerr=std_metric_result, fmt='o',
                     capsize=10, elinewidth=1)
        plt.title(mean_metric)
        plt.xlabel(param_name)
        plt.ylabel(mean_metric)
        #plt.savefig(path_to_save + '/' + key)
        plt.show()