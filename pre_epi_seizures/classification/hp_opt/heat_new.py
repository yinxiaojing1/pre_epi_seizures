import matplotlib.pyplot as plt
import seaborn as sns

import itertools


def get_hp_opt_results_args(hp_opt_results):
    
    # Get table from the structure
    cv_results = hp_opt_results['cv_results']

    # get variation of parameters
    estimator_params = [key 
                        for key in cv_results.columns
                        if 'param' in key]
    
    # Get metrics from optimizations
    metrics = [key 
               for key in cv_results.columns
               if 'param' not in key
               and 'rank' not in key]
    
    # create combinations of 2 parameter for heatmap
    param_bi_comb = itertools.combinations(estimator_params, r=2)
    for bi_comb in param_bi_comb:
        
       # Loop through the metrics
        for metric in metrics:
            
            metric_result = cv_results[metric]
            params = [cv_results[estimator_param]
                      for estimator_param
                      in estimator_params]
            
            xticklabels=[2**i for i in xrange(-5, 11)]
            yticklabels=[2**i for i in xrange(-15, 1)]
            
            df = metric_result.reshape(len(xticklabels), len(yticklabels))

            
            plt.figure()
            sns.heatmap(df,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        cbar_kws={'label': metric},
                        )
            plt.title(key)
            plt.xlabel(bi_comb[0])
            plt.ylabel(bi_comb[1])
            #plt.savefig(path_to_save + '/' + key)
            plt.show()
            

