from memory_profiler import profile
import numpy as np
@profile
def main():
    [1]*1000*2*3600
    d = [1]*1000*2*3600
    # c = np.asarray(d)
    # c = d
    print 'Done!'
if __name__ == '__main__':
    main()