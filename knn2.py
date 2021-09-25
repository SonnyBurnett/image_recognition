#
# knn example with your own dataset from a csv file
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv('Social_Network_Ads.csv')
print(type(dataset))                             # The csv file is converted into a panda dataframe
print(dataset.head(10))                          # which works a lot like the R dataframes
X = dataset.iloc[:, [1, 2, 3]].values            # select only the first 3 columns
print(type(X))                                   # with values the type is converted to numpy ndarray
print(X)                                         # which we need to use it in knn
y = dataset.iloc[:, -1].values                   # select the last column
print(y)                                         # which is the value we want to predict
le = LabelEncoder()                              # knn cannot work with string variables (Male, Female)
X[:,0] = le.fit_transform(X[:,0])                # so we need to convert it to a numeric value (0,1)
print(X)
                                                 # Split data in 80% train and 20% test
                                                 # random_state with any value makes it reproducable
                                                 # with stratify=y you get approximately the same percentage
                                                 # of samples of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()                            # The input values need comparable values for knn to work
X_train = sc.fit_transform(X_train)              # so we need to rescale them
                                                 # we need the fit method once to calculate the scaling
                                                 # which can be combined with the transform method
X_test = sc.transform(X_test)                    # so we don't need the fit method again in the test set
                                                 # Now we build the model.
                                                 # The k in k nearest neighbors will be 5 (is also default)
                                                 # The calculation method is minkowski. (also default)
                                                 # p=2 means eucledian (probably also default)
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)                 # with the fit method we train the model
y_pred = classifier.predict(X_test)              # Now we can predict the test set
print(y_test)                                    # and we can see how well the model is performing
print(y_pred)                                    # by comparing the real value with the predicted value

print(confusion_matrix(y_test, y_pred))          # with a confusion matrix this can be displayed even better
print(accuracy_score(y_test,y_pred))             # in an accuracy score

