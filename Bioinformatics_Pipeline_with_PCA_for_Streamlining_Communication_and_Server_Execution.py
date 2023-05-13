import sys
import csv
import ibis
import argparse
import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d
from sklearn import datasets
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif


parser = argparse.ArgumentParser()
parser.add_argument('--inputFile1',default='data.csv',type=str,action='store',
                    help='input file goes here')
parser.add_argument('--inputFile2', default='metadata.csv',type=str,action='store',
                    help='input file goes here')
args = parser.parse_args()
data=args.inputFile1
metaData=args.inputFile2

with open(metaData, "r") as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        headers = row
        break

features = ["sample", "libprep"]
sample_index = headers.index(features[0])
libprep_index = headers.index(features[1])

parameters = []
libprepDict = dict()
with open(metaData, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        sample = row[sample_index]
        libprep = row[libprep_index]
        libprepDict[sample] = libprep
        values = [sample, libprep]
        parameters.append(values)
parameters = np.array(parameters)

with open(data) as csv_file:
    for s in csv.reader(csv_file, delimiter=','):
        samples = s
        break
samples.remove(samples[0])

dataInfo = pd.read_csv(data)
X = []
y = []
for sample in samples:
  column = dataInfo[sample].tolist()
  X.append(column)
  if sample in libprepDict:
    if libprepDict[sample] == "ribodepleted":
      y.append(0)
    else:
      y.append(1)
X = np.array(X)

# Adding a feature selection step using SelectKBest:
  # The `selected_feature_names` list is empty because none of the features were selected by the `SelectKBest` feature
  # selection step in the pipeline. The pipeline is using the `f_classif` function to calculate the ANOVA F-value
  # between each feature and the target variable. Then, it selects the k-best features based on the scores.
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(f_classif, k=1000)),
    ("pca", PCA(n_components=2)),
    # ("gb", GradientBoostingClassifier()),
    ("logistic", LogisticRegression(max_iter=10000, tol=0.1)),
    # ("rf", RandomForestClassifier(n_estimators=100)),
], verbose=2)

#`'pca__n_components'`: A list of integers specifying the number of principal components to keep after PCA
   # transformation.
#`'logistic__C'`: A list of positive floats specifying the inverse of regularization strength for logistic regression. Smaller values of C result in stronger regularization.
#`'logistic__penalty'`: A list of strings specifying the regularization penalty to be applied in logistic regression.
   # Possible options are `'l1'`, `'l2'`, and `'elasticnet'`.
#`'logistic__solver'`: A list of strings specifying the solver algorithm to be used in logistic regression.
   # Possible options are `'lbfgs'` and `'liblinear'`.
param_grid = {
    'pca__n_components': [1, 2],
    'logistic__C': [0.1, 1.0, 10.0],
    'logistic__penalty': ['l1', 'l2', 'elasticnet'],
    'logistic__solver': ['lbfgs', 'liblinear']
}

# You can use GridSearchCV to search over a range of hyperparameters for each step in the pipeline.
# This can help you find the best combination of hyperparameters for your specific problem.
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5)

pipe.fit(X, y)

# Get the kbest object
# Get the boolean mask indicating which features were selected
# Get the list of all feature names
# Get the list of selected feature names
kbest = pipe.named_steps["kbest"]
selected_mask = kbest.get_support()
all_feature_names = dataInfo.columns.tolist()
selected_feature_names = [all_feature_names[i] for i in range(min(len(selected_mask), len(all_feature_names)))
                          if selected_mask[i]]
print("Selected feature names:", selected_feature_names)

pca = pipe.named_steps["pca"]
variances = pca.explained_variance_ratio_

logistic = pipe.named_steps['logistic']
coefficients = logistic.coef_
intercept = logistic.intercept_

# forest = pipe.named_steps['rf']
# print(forest.feature_importances_)

# Use cross_val_score to perform cross-validation
scores = cross_val_score(pipe, X, y, cv=5) # 5-fold cross-validation
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = pipe.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Variances", variances)
print('Coefficients:', coefficients)
print('Intercept:', intercept)
print('Accuracy score:', accuracy)
