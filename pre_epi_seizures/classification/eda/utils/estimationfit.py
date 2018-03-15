import scipy
import scipy.stats


def dist_estimation(univariate_data,
                    dist_name):
    size = len(univariate_data)
    x = scipy.arange(size)
    y = univariate_data

    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x,
                          *param[:-2],
                          loc=param[-2],
                          scale=param[-1]) * size

    return pdf_fitted

