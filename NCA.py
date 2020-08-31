import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (KNeighborsClassifier,NeighborhoodComponentsAnalysis)
n_neighbors = 3
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
# Split into train/test
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.5, stratify = y, random_state = 42)
# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
X= nca.fit(X,y).transform(X)



