import scipy
import scipy.stats
import numpy as np


def dist_estimation(univariate_data,
                    dist_name):
    size = len(univariate_data)
    x = np.linspace(0, 1, size)
    y = univariate_data


    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x,
                          *param[:-2],
                          loc=param[-2],
                          scale=param[-1])

    return x, pdf_fitted

