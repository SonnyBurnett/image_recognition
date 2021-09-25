#
# Goal: Predict the species from an Iris dataset with knn algorithm
#
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#    Collect data
iris = datasets.load_iris()
print(type(iris))
# print(iris.feature_names)                   # features, aka the input
# print(iris.target)                          # predicted output as numeric value
# print(iris.target_names)                    # predicted output as name

#    Prepare data
X = iris.data[:, 2:]                        # select only columns sepal length and 'sepal width (cm)
y = iris.target                             # the target is the species

#    Split data in 80% train and 20% test
testSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, stratify=y, random_state=42)

clf = KNeighborsClassifier()                # create the model
clf.fit(X_train, y_train)                   # train the model

#     Test model with one example
sepal_length = 4                            # test one value for length
sepal_width = 1                             # test one value for width
expected_species = "versicolor"             # test species for one value
print("predicted value: ", iris.target_names[clf.predict([[sepal_length, sepal_width]])[0]])
print("expected value : ", expected_species)
print('')

#     Test model with test data set
y_pred_test = clf.predict(X_test)
print(y_pred_test)
print(y_test)
print('')
print('Accuracy test data')
print(accuracy_score(y_test, y_pred_test))
