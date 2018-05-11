"""
.. module:: dimreduction
   :platform: Unix, Windows
   :synopsis: This module provides various functions to...

.. moduleauthor:: Filipe Canento


"""

# Notes:
# Falar com Marta para saber algoritmo de PCA que usou.

# Imports
import scipy
from scipy import linalg
import pylab
import traceback
from itertools import izip
import sklearn.decomposition as skldec
import numpy as np



def selector(method):
    """
    Selector for the dimensionality reduction functions and methods.
    
    Input:
        method (str): The desired function or method.
    
    Output:
        fcn (function): The function pointer.
    
    Configurable fields:
    
    See Also:
    
    
    Example:
    
    
    References:
        .. [1]
    """
    
    if method == 'pca_Opop':
        fcn = pca_OPop
    elif method == 'pca_Ind':
        fcn = pca_Ind
    elif method == 'lda':
        fcn = lda
    else:
        raise TypeError, "Method %s not implemented." % method
    
    return fcn


# Principal Component Analysis - Overall Population Eigen-Heartbeat
class pca_OPop(object):
    # Init
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        #self.transform_matrix = None ---- sao os eigenvectores
        self.is_trained = False
        self.train_mean = None
        self.ev_energy = None
        
    # Perform PCA on input data; determine eigenvalues, eigenvectors, transform matrix
    def train(self, data = None,  label=None, energy = 1): #label = None,
        """
        Input:
            data : 3 dimentional list or array [individual, heartbeat, values]
            energy : value between 0 and 1 for dimentionality reduction #### nao sei se e para fazer isto aqui
        
        """

        # Check inputs
        if data is None or (not isinstance (data , list) and not isinstance (data, np.ndarray)): ##check
            raise TypeError, "Please provide input data."
        if not 0 < energy <= 1:
            raise TypeError, "Please provide input energy between 0 and 1." 
        # if label is None:
            # raise TypeError, "Please provide input data label." 
            
        success = False
            
        try:	
            if isinstance (data, list):
                # transforms the dataMix list in to an array
                Xt=np.zeros([len(data),len(data[0])])
                for i in range(len(data)):
                    Xt[i,:]=data[i]	#Xt is the 2D array used as training data [heartbeat, values]
            else: Xt= data
            #PCA com toda a populacao
            ncomponents= min(len(data),len(data[0]))-1 #setting the number of eigenvalues considered as 100 per cent energy
            pca = skldec.PCA(n_components=ncomponents)
            F=pca.fit(Xt) #Fit the model with X
            comt=F.components_ #eigenvalues [c1, c2, ... , cn]
            Tt=F.fit_transform(Xt) #eigenvectores [V1, V2, ... , Vn]
            E=pca.explained_variance_ratio_ #energy associated to the eigenvalues [e1, e2, ... , en]
                
            
            neig=len(comt)+0.0
            if energy!=1: #quando a energia diferente de 1 corta os eigenvectores (Tt) e conta-os (neig)
                sum=0
                neig=0
                while sum<energy: #energy wanted
                    sum=sum+E[neig] 
                    neig=neig+1 #number os eigencomponents
                comt=comt[:(neig+1)]
                Tt=Tt[:,:(neig+1)]
    

            # Updates variables
            self.eigen_values = Tt					
            self.eigen_vectors = comt 
            self.train_mean = np.mean(Xt,axis=0)
            self.ev_energy = E 
            
            success = True
            self.is_trained = True
        except Exception as e: 
            print e
            print traceback.format_exc()
        return success
        
    # Project test data
    def project(self, data = None ): #label = None
        """
        Input:
            data : test data, a 2 dimentional list or array [heartbeat, values] with the segments of one individue
            eigen_vectors : eigenvectors' matrix coming from  data training
            train_mean : mean of the training data
        Output:
            Projections : projection coeficients of the individual's observation in the test data  
        """
            
        Segmentsout = np.copy(data)
        Segmentsout = (Segmentsout - self.train_mean)
        Projections=np.dot(Segmentsout,self.eigen_vectors.T)

        return Projections
  
# Principal Component Analysis - Individualized Eigen-Heartbeat

class pca_Ind(object):
    # Init
    def __init__(self):
        self.eigen_values_list = None
        self.eigen_vectors_list = None
        
        self.is_trained = False
        
    # Perform PCA on input data; determine eigenvalues, eigenvectors, transform matrix
    def train(self, data = None, label = None, energy = 1): #
        """
        Input:
            data : 3 dimentional list or array [individual, heartbeat, values]
            energy : value between 0 and 1 for dimentionality reduction (fraction)
        
        """
        
        # Check inputs
        if data is None or (not isinstance (data , list) and not isinstance (data, np.ndarray)): ##check
            raise TypeError, "Please provide input data."
        if not 0 <= energy <= 1:
            raise TypeError, "Please provide input energy between 0 and 1." 
        if label is None:
            raise TypeError, "Please provide input data label." 
            
        success = False
            
        try:	
            #variables initiation
            Xmean = []
            Tlist = []
            comlist = []
            Elist = []
            for ID in set(label): # perform PCA for each individual
                
                Segments=[data[i] for i in range(len(label)) if label[i]==ID]
                
                #transformar o Segments num array 
                X=np.zeros([len(Segments),len(Segments[0])])
                for i in range(len(Segments)):
                        X[i,:]=Segments[i]
                
                Xmean.append(np.mean(X,axis=0))
                ncomponents= min(len(Segments),len(Segments[0]))-1
                pca = skldec.PCA(n_components=ncomponents)
                #print ncomponents
                F=pca.fit(X) #Fit the model with X
                com=F.components_
                T=F.fit_transform(X)
                
                E=pca.explained_variance_ratio_
                
                neig=len(com)+0.0
                if energy!=1:
                    sum=0
                    neig=0
                    while sum<energy: #energy wanted
                        sum=sum+E[neig] 
                        neig=neig+1 #number os eigencomponents
                    com=com[:(neig+1)]
                    T=T[:,:(neig+1)]
                            
                comlist.append(com)
                Tlist.append(T)
                Elist.append(E)


            # Updates variables
            self.eigen_values_list = Tlist				
            self.eigen_vectors_list = comlist
            self.train_mean_list = Xmean
            self.ev_energy_list = Elist
            
            success = True
            self.is_trained = True
        except Exception as e: 
            print e
            print traceback.format_exc()
        return success
        
    # Project test data
    def project(self, data = None ):
        """
        Input:
            data : test data, a 2 dimentional list or array [heartbeat, values] with the data of ONE individual 
            eigen_vectors : eigenvectors' matrix coming from  data training
            train_mean : mean of the training data
        Output:
            Projections : projection coeficients of each observation of the test data projected in each individual of the train set
                         *projections[testID][trainID] = set of projections of the individual testID in the individual trainID
        """
        Projections_list=[[] for i in range(len(data)) ]

        for IDtreino in range(len (self.train_mean_list)): #para todos os individuos do treino
            #---------projeccao dos dados------------------------------
            Segmentsout = (data - self.train_mean_list[IDtreino]) #check for agreeable dimentions first?
            Projections_list[IDtreino].append(np.dot(Segmentsout,self.eigen_vectors_list[IDtreino].T)) 

        return Projections_list
    
# Linear Discriminant Analysis
class lda(object):
    # Init
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.transform_matrix = None
        self.is_trained = False
    # Perform LDA on input data; determine eigenvalues, eigenvectors, transform matrix
    def train(self, data=None, label=None, energy = 1):
        """

        Perform Linear Discriminant Analysis on input data.

        Input:
            data (array): input data array of shape (number samples x number features).

            label (array): input data label array of shape (number of samples x 1). Must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples).

        Output:
            success (boolean): indicates whether training was successful (True) or not (False).

        Configurable fields:{"name": "dimreduction.lda.train", "config": {"": ""}, "inputs": ["data", "label"], "outputs": ["success"]}

        See Also:


        Notes:


        Example:


        References:
            .. [1]    ...
            .. [2]    ...
            .. [3]    ...
        """
        # Check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        if label is None:
            raise TypeError, "Please provide input data label."
        if 0 not in label:
            raise TypeError, "Label must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples)."
        success = False
        try:
            # Compute mean of each set (mi)
            m = []
            for c in set(label): m.append(scipy.mean(data[label==c], axis=0))
            m = scipy.array(m)
            # Compute Scatter Matrix of eah set (Si)
            S = []
            for c in set(label):
                S.append(scipy.cov(scipy.array(data[label==c]).T))
            # Compute Within Scatter Matrix (SW)
            SW = 0
            for s in S:
                SW += s
            # Compute Total Mean (mt)
            mt = scipy.mean(data, axis=0)
            # Compute Total Scatter Matrix (ST)
            ST = 0
            for xi in data:
                aux = scipy.matrix(xi-mt)
                ST += aux.T*aux
            # Compute Between Scatter Matrix (SB)
            SB = 0
            for c in set(label):
                aux = scipy.matrix(m[c,:]-mt)
                SB += len(pylab.find(label==c))*aux.T*aux
            # Solve (Sb - li*Sw)Wi = 0 for the eigenvectors wi
            eigenvalues,v = linalg.eig(SB,SW)
            # Get real part and sort eigenvalues
            real_sorted_eigenvalues = []
            for i in xrange(len(eigenvalues)):
                real_sorted_eigenvalues.append([scipy.real(eigenvalues[i]), i])
            real_sorted_eigenvalues.sort()
            # Get the (nclasses-1) main eigenvectors
            # Assures eigenvalue is not NaN
            nclasses = len(set(label)) - 1
            # nclasses = 5
            eigenvectors = []
            for i in xrange(-1, -len(real_sorted_eigenvalues)-1, -1):
                if not scipy.isnan(real_sorted_eigenvalues[i][0]):
                    eigenvectors.append(v[real_sorted_eigenvalues[i][1]])
                if len(eigenvectors) == nclasses: break


            # Updates variables
            self.eigen_values = real_sorted_eigenvalues
            self.eigen_vectors = eigenvectors
            self.transform_matrix = scipy.matrix(eigenvectors)

            success = True
            self.is_trained = True
        except Exception as e:
            print e
            print traceback.format_exc()
        return success
    # Project test data
    def project(self, data=None):
        """

        Transforms input data. Train method must have been performed previously.

        Input:
            data (array): input data array of shape (number samples x number features).

        Output:
            transformed_data (matrix): output data array.

        Configurable fields:{"name": "dimreduction.lda.train", "config": {"": ""}, "inputs": ["data"], "outputs": ["transformed_data"]}

        See Also:


        Notes:


        Example:


        References:
            .. [1]    ...
            .. [2]    ...
            .. [3]    ...
        """
        # Check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        if self.is_trained:
            transformed_data = scipy.matrix(data)*self.transform_matrix.T
            return scipy.array(transformed_data)
        else:
            raise TypeError, "Please perform the train method before this one."


def lda_train(data=None, label=None):

    # Check inputs
    if data is None:
        raise TypeError, "Please provide input data."
    if label is None:
        raise TypeError, "Please provide input data label."
    if 0 not in label:
        raise TypeError, "Label must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples)."
    [n, d] = data.shape
    try:
        # Compute mean of each set (mi)
        m = []
        for c in set(label):
            m.append(scipy.mean(data[label == c], axis=0))
        m = scipy.array(m)
        mt = scipy.mean(data, axis=0)
        # Compute Within Scatter Matrix of set (SW) and Between Scatter Matrix (SB)
        SW = np.zeros((d, d), dtype=np.float64)
        SB = np. zeros((d, d), dtype=np. float64)
        for c in set(label):
            SW += np.dot((scipy.array(data[label == c]) - m[c, :]).T, scipy.array(data[label == c]) - m[c, :])/(1.*n)
            aux = scipy.matrix(m[c, :]-mt)
            SB += (len(pylab.find(label==c))*aux.T*aux)/(1.*n)
        eigenvalues, v = linalg.eig(SB,SW)
        # print 'eigen:', eigenvalues, v
        # Get real part and sort eigenvalues
        real_sorted_eigenvalues = []
        for i in xrange(len(eigenvalues)):
            real_sorted_eigenvalues.append([scipy.real(eigenvalues[i]), i])
        real_sorted_eigenvalues.sort()
        # Get the (nclasses-1) main eigenvectors
        # Assures eigenvalue is not NaN
        nclasses = len(set(label)) - 1
        # nclasses = 5
        eigenvectors = []
        for i in xrange(-1, -len(real_sorted_eigenvalues)-1, -1):
            if not scipy.isnan(real_sorted_eigenvalues[i][0]):
                eigenvectors.append(v[real_sorted_eigenvalues[i][1]])
            if len(eigenvectors) == nclasses: break
        transform_matrix = scipy.matrix(eigenvectors)

    except Exception as e:
        print e
        print traceback.format_exc()

    return transform_matrix


def lda_train2(data=None, label=None):
    """

    Perform Linear Discriminant Analysis on input data.

    Input:
        data (array): input data array of shape (number samples x number features).

        label (array): input data label array of shape (number of samples x 1). Must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples).

    Output:
        success (boolean): indicates whether training was successful (True) or not (False).

    Configurable fields:{"name": "dimreduction.lda.train", "config": {"": ""}, "inputs": ["data", "label"], "outputs": ["success"]}

    See Also:


    Notes:


    Example:


    References:
        .. [1]    ...
        .. [2]    ...
        .. [3]    ...
    """
    # Check inputs
    if data is None:
        raise TypeError, "Please provide input data."
    if label is None:
        raise TypeError, "Please provide input data label."
    if 0 not in label:
        raise TypeError, "Label must start in 0 (e.g., label = [0,0,0,1,2] for 5 samples)."
    # success = False
    try:
        # Compute mean of each set (mi)
        m = []
        for c in set(label): m.append(scipy.mean(data[label==c], axis=0))
        m = scipy.array(m)
        # Compute Scatter Matrix of eah set (Si)
        S = []
        for c in set(label):
            S.append(scipy.cov(scipy.array(data[label==c]).T))
        # Compute Within Scatter Matrix (SW)
        SW = 0
        for s in S:
            SW += s
        # Compute Total Mean (mt)
        mt = scipy.mean(data, axis=0)
        # Compute Total Scatter Matrix (ST)
        ST = 0
        for xi in data:
            aux = scipy.matrix(xi-mt)
            ST += aux.T*aux
        # Compute Between Scatter Matrix (SB)
        SB = 0
        for c in set(label):
            aux = scipy.matrix(m[c,:]-mt)
            SB += len(pylab.find(label==c))*aux.T*aux
        # Solve (Sb - li*Sw)Wi = 0 for the eigenvectors wi
        eigenvalues,v = linalg.eig(SB,SW)
        # Get real part and sort eigenvalues
        real_sorted_eigenvalues = []
        for i in xrange(len(eigenvalues)):
            real_sorted_eigenvalues.append([scipy.real(eigenvalues[i]), i])
        real_sorted_eigenvalues.sort()
        # Get the (nclasses-1) main eigenvectors
        # Assures eigenvalue is not NaN
        nclasses = len(set(label)) - 1
        # nclasses = 5
        eigenvectors = []
        for i in xrange(-1, -len(real_sorted_eigenvalues)-1, -1):
            if not scipy.isnan(real_sorted_eigenvalues[i][0]):
                eigenvectors.append(v[real_sorted_eigenvalues[i][1]])
            if len(eigenvectors) == nclasses: break

        # Updates variables
        # self.eigen_values = real_sorted_eigenvalues
        # self.eigen_vectors = eigenvectors
        # self.transform_matrix = scipy.matrix(eigenvectors)
        transform_matrix = scipy.matrix(eigenvectors)

        # success = True
        # self.is_trained = True
    except Exception as e:
        print e
        print traceback.format_exc()
    # return success
    return transform_matrix

# Project test data
def lda_project(data=None, transform_matrix=None):
    """

    Transforms input data. Train method must have been performed previously.

    Input:
        data (array): input data array of shape (number samples x number features).

    Output:
        transformed_data (matrix): output data array.

    Configurable fields:{"name": "dimreduction.lda.train", "config": {"": ""}, "inputs": ["data"], "outputs": ["transformed_data"]}

    See Also:


    Notes:


    Example:


    References:
        .. [1]    ...
        .. [2]    ...
        .. [3]    ...
    """
    # Check inputs
    if data is None:
        raise TypeError, "Please provide input data."
    elif transform_matrix is None:
        raise TypeError, "Please provide transform matrix."
    transformed_data = scipy.matrix(data)*transform_matrix.T
    return scipy.array(transformed_data)

if __name__=='__main__':
    
    #===========================================================================
    # Generate Data
    data_set1 = scipy.random.normal(10, 2.5, [1000,2])
    # data_set2 = scipy.random.normal(-5, 1.0, [1000,2])
    data_set3 = scipy.random.normal(1, 2.5, [1000,2])
    data_set, label = [], []
    c = 0
    cols = 'rb'
    # for ds in [data_set1, data_set2, data_set3]:
    for ds in [data_set1, data_set3]:
        for x,y in ds:
            data_set.append([x,y])
            label.append(c)
        c += 1
    data_set, label = scipy.array(data_set), scipy.array(label)
    #===========================================================================
    # LDA
    print "Testing LDA. \n"
    fig = pylab.figure(1)
    fig.suptitle('LDA example for two classes.')
    ax = fig.add_subplot(211)
    ax.cla()
    ax.set_title('Original data')
    ax.grid()
    for lb, col in izip(xrange(c), cols):
        ax.plot(data_set[:,0][label==lb], data_set[:,1][label==lb], col+'o')
    LDA = lda()
    LDA.train(data_set, label)
    transformed_data = LDA.transform(data_set)
    
    ax = fig.add_subplot(212)
    ax.cla()
    ax.set_title('Transformed data')
    ax.grid()
    for lb, col in izip(xrange(c), cols):
        ax.plot(transformed_data[:,0][label==lb], scipy.zeros(len(transformed_data[label==lb])), col+'o')
    figname = '../temp/lda_fig1.png'
    fig.savefig(figname)
    print "Done. Results saved in %s"%figname
    
    #===========================================================================

      # PCA   -  este codigo nao foi verificado
    print "Testing PCA. \n"
    fig = pylab.figure(1)
    fig.suptitle('PCA example for two classes.')
    ax = fig.add_subplot(211)
    ax.cla()
    ax.set_title('Original data')
    ax.grid()
    for lb, col in izip(xrange(c), cols):
        ax.plot(data_set[:,0][label==lb], data_set[:,1][label==lb], col+'o')
    PCA = pca_OPop()
    PCA.train(data_set, 1)
    transformed_data = PCA.project(data_set)
    
    ax = fig.add_subplot(212)
    ax.cla()
    ax.set_title('Transformed data')
    ax.grid()
    for lb, col in izip(xrange(c), cols):
        ax.plot(transformed_data[:,0][label==lb], scipy.zeros(len(transformed_data[label==lb])), col+'o')
    figname = '../temp/pca_fig1.png'
    fig.savefig(figname)
    print "Done. Results saved in %s"%figname    
        
            