from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense

def ada_boost():
    clf = AdaBoostClassifier()
    hyperparameters = {

    }
    return clf, hyperparameters

def gradient_boost():
    clf = GradientBoostingClassifier()
    hyperparameters = {

    }
    return clf, hyperparameters

def decision_tree():
    clf = DecisionTreeClassifier()
    hyperparameters = {
        "classifier__max_depth" : [10, 50, 100],
        "classifier__leaf_size" : [5, 10, 50],
    }
    return clf, hyperparameters

def kneighbors():
    clf = KNeighborsClassifier()
    hyperparameters = {
        "classifier__n_neighbors" : [5, 10, 50],
        "classifier__leaf_size" : [5, 10],
    }
    return clf, hyperparameters

def random_forest():
    clf = RandomForestClassifier()
    hyperparameters = {
        "classifier__n_estimators": [10, 50, 100, 250, 500, 1000],
        'classifier__min_samples_leaf': [1, 3, 5],
        'classifier__max_features': ['sqrt', 'log2']
    }
    return clf, hyperparameters

def logistic_regression():
    clf = LogisticRegression()
    hyperparameters = {
        'classifier__max_iter' : [10,],
        'classifier__solver': ['liblinear', 'lbfgs'], 
        'classifier__C': np.logspace(-4, 0, 4),
    }
    return clf, hyperparameters

def nn_simple(input_dim):
    clf = Sequential()
    clf.add(Dense(108, input_dim=input_dim-3, activation='relu')) #On a supprimé 3 colonnes
    clf.add(Dense(54, activation='relu'))
    clf.add(Dense(27, activation='softmax'))
    
    clf.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    return clf, {}
