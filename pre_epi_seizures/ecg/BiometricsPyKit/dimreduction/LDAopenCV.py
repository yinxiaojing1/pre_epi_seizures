#generate data   
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

def lda (X, y, num_components =0) :
	y = np. asarray (y)
	[n,d] = X. shape
	c = np. unique (y)
	if ( num_components <= 0) or ( num_component >( len (c) -1)):
		num_components = ( len (c) -1)
	meanTotal = X. mean ( axis =0)  # igual ao do filipe
	Sw = np. zeros ((d, d), dtype =np. float32 )
	Sb = np. zeros ((d, d), dtype =np. float32 )
	
	for i in c:
		Xi = X[np. where (y==i) [0] ,:]
		meanClass = Xi. mean ( axis =0)
		#print meanClass
		Sw = Sw + np. dot ((Xi - meanClass ).T, (Xi - meanClass )) # normalizado este e o do filipe sao aprox. iguais
		aux=scipy.matrix(meanClass - meanTotal)
		Sb = Sb+ n * np.dot (( aux ).T, ( aux )) # aqui dão as 4 entradas iguais, no do filipe nao
		#aux ==( meanClass - meanTotal ) -> True
		Sb += n*aux.T*aux
	### faz toda a diferença transformar o aux em scipy.matrix, nao fazendo 4 entradas todas iguais ou n simetrica
	eigenvalues , eigenvectors = np. linalg . eig (np. linalg . inv (Sw)*Sb) #isto ou scipy.linalg.eig(SB,SW) ?
	idx = np. argsort (- eigenvalues . real )
	eigenvalues , eigenvectors = eigenvalues [idx], eigenvectors [:, idx ]
	eigenvalues = np. array ( eigenvalues [0: num_components ]. real , dtype =np. float32 , copy = True )
	eigenvectors = np. array ( eigenvectors [0: ,0: num_components ]. real , dtype =np. float32 , copy = True )
	return [ eigenvalues , eigenvectors ]
	
	
def project (W, X, mu= None ):
	if mu is None :
		return np. dot (X,W)
	return np. dot (X - mu , W)
	
	
	
	# FileII=open("./data.txt",'wb')
	# cPickle.dump(label,FileII,1)
	# FileII.close()
	# File=open("./data.txt",'rb') ###nome origem
	# label=cPickle.load(File)
	# File.close()