
from sklearn import svm


# I/O from disk -------------------------------------





# Training template *Supervised Classification ---------------------------------------

# create trainig data structure with labels
feature_training_array = X = [[0, 0], [1, 1]]
labels_nr = [0, 1]

# instance the classifier
classifier = svm.SVC()

# train the classfier
classifier.fit(feature_training_array,
                      labels_nr)

# fetch novelty data
novelty = [[-9, -9]]

# classifiy novelty
prediction = classifier.predict(novelty)
print prediction


