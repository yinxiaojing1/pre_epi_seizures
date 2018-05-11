import numpy as np
import pylab as pl
from numpy import linalg
import matplotlib.pyplot as plt
# import sys
# sys.path.append("./")

class Fisher_classif(object):
	# Init
    def __init__(self):
        self.boudary = None
        self.w = None
        self.c = None
        self.error = None    
            
    #gets two classes, and calculates the Fisher Linear Discriminant boundary
    def calc_boundary(self, data, label): #label = None,
        #gets two classes, and calculates the parameters of Fisher Linear Discriminant boundary
            class_label = [i for i in set(label)]
            if len(class_label) != 2:
                raise TypeError, "Please provide two classes."
            self.class_label = class_label
            class1 = data[label==class_label [0]]
            class2 = data[label==class_label [1]]

                
            #calc w
            cov1 = np.cov(class1.T)
            cov2 = np.cov(class2.T)
            mean1 = np.mean(class1.T,1)
            mean2 = np.mean(class2.T,1)
            w = dot (linalg.inv(cov1+cov2) , (mean2 - mean1)) 
            
            #calculo do c
            projections = []
            for x in data:
                projections.append( dot(w.T,x) )
            
            pm1 = dot(w,mean1)
            pm2 = dot(w,mean2)
            c_min = np.min ([pm1,pm2])
            c_max = np.max ([pm1,pm2])
            c_try = np.arange (-c_max, -c_min , 0.1)
            error_c = []
            
            #finds the c that lead to a lower number of errors
            for t in c_try:
                predict = []
                for p in projections:
                    if p < -t:
                        predict.append (class_label[0])
                    else: 
                        predict.append (class_label[1])
                verify = [predict [i] == label[i] for i in range(len(label))]
                #print verify.count(False)/(len (verify)+0.0)
                error_c.append (verify.count(False)/(len (verify)+0.0))
            index = np.where (error_c == np.min(error_c))[0]
            #if there is more than one minimun error value, it choses the middle one
            if len (index) > 1:
                c = c_try [index [len(index)/2.0]]
            else : c = c_try[index[0]]
            
            #c = c_try [error_c.index(np.min(error_c))]
            
            
            # Updates variables
            self.w = w
            self.c = c
            self.error = np.min(error_c)

    def predict(self, obs):
        #determines if the observation(s) belong to class 1 or class 2
        #obs should be in the shape of a list of lists, even if it only has one observation - - obs = [[x1], [x2], [x3]] 
        if not (hasattr(obs[0], '__iter__')):
            raise TypeError, "obs should be in the shape of a list of lists, even if it only has one observation - ex: obs = [[x1], [x2], [x3]]."
        w = self.w
        c = self.c
        test_label = []
        for x in obs:
            if dot(w.T,x) < -c: 
                test_label.append(self.class_label[0]) #self.
            else:
                test_label.append(self.class_label[1])
        return test_label
        
        

if __name__=='__main__':
    
    #===========================================================================
    # Generate Data

    np.random.seed(0)
    mean1, cov1, n1 = [1, 5], [[1,1],[1,2]], 200  # 200 samples of class 1
    x1 = np.random.multivariate_normal(mean1, cov1, n1)
    y1 = np.ones(n1, dtype=np.int)
    mean2, cov2, n2 = [2.5, 2.5], [[1,0],[0,1]], 300 # 300 samples of class -1
    x2 = np.random.multivariate_normal(mean2, cov2, n2)
    y2 = -np.ones(n2, dtype=np.int)
    data = np.concatenate((x1, x2), axis=0) # concatenate the samples
    label = np.concatenate((y1, y2))
    #===========================================================================
    # training
    #w2=np.array([ 2.5948979,  -2.58553746])
    #b=5.63727441841
    FC= Fisher_classif()
    FC.calc_boundary (data, label)
    c = FC.c
    w = FC.w
    xx = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.01)
    plot1 = plt.plot(x1[:, 0], x1[:, 1], 'ob', x2[:, 0], x2[:, 1], 'or')
    #c_try_plot = [-dot(w,mean1), -dot(w,mean2)]
    #for c in c_try_plot:
    yy = - (w[0] * xx + c) / w[1]
    plot2 = plt.plot(xx, yy, '--k')
    print 'Classifier error:', FC.error
    
    #classification
    
    test_label = FC.predict(data)
    verify = [test_label [i] == label[i] for i in range(len(label))]
    prediction_error = verify.count(False)/(len (verify)+0.0)
    print 'Prediction error:', prediction_error
    

