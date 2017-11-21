
def C_gamma_grid():
    C_list = [2**i for i in xrange(-5, 11)]
    gamma_list = [2**i for i in xrange(-15, 1)]
    parameter_list = [{'C': C, 'gamma': gamma}
                       for C in C_list
                       for gamma in gamma_list]

    return parameter_list



