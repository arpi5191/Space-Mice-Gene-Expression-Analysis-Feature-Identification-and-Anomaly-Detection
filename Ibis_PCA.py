# Imports for the First Half of Code
import os
import sys
import csv
import ibis
import sklearn
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

# Adds the sklearn path to the code
ibis.options.interactive = True
sys.path.append('/Users/arpitha/Documents/Stuart_Lab/Ibis_Substrate/substrait-ml/sklearn_ibis')
os.environ['PYTHONPATH'] = ':'.join(sys.path)

# Command line options for the File
parser = argparse.ArgumentParser()
parser.add_argument('--inputFile1',default='data.csv',type=str,action='store',
                    help='input file goes here')
parser.add_argument('--inputFile2', default='metadata.csv',type=str,action='store',
                    help='input file goes here')
args = parser.parse_args()
data=args.inputFile1
metaData=args.inputFile2

# Read the metaData File
with open(metaData, "r") as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        headers = row
        break

# Obtain the Sample, Libprep Features & their Respective Indices
features = ["sample", "libprep"]
sample_index = headers.index(features[0])
libprep_index = headers.index(features[1])

# Obtain the libprep for Each Sample
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

# Remove First Sample
with open(data) as csv_file:
    for s in csv.reader(csv_file, delimiter=','):
        samples = s
        break
samples.remove(samples[0])

# Obtain the X and y to perform the fitting on
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
    ("logistic", LogisticRegression(max_iter=10000, tol=0.1)),
])

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
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=2)

# Peform Fitting
pipe.fit(X, y)

# Imports for the Next Part of Code
from sklearn.pipeline import Pipeline
from ibis.expr.types import Table
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_ibis.linear_model import LogisticRegressionIbis
from sklearn_ibis.decomposition import PCAIbis
from sklearn_ibis.preprocessing import StandardScalerIbis
from sklearn_ibis.ibis.ibis_transformer import IbisTransformer

# Function to Connect Pipeline to Ibis
class PipelineIbis:
    def __init__(self, wrapped: Pipeline):
        self.steps = wrapped.steps

    def to_ibis(self):
        def fn(table: Table):

            for step in self.steps:
                step = step[1]

                if isinstance(step, StandardScaler):
                    wrapper = StandardScalerIbis(step)
                elif isinstance(step, LogisticRegression):
                    wrapper = LogisticRegressionIbis(step)
                elif isinstance(step, PCA):
                    wrapper = PCAIbis(step)
                elif isinstance(step, IbisTransformer):
                    wrapper = step
                else:
                    raise Exception(f"No Ibis implemntation found for {type(step)}")

                table = wrapper.to_ibis()(table)

            return table

        return fn

# Create Memtable
t = ibis.memtable(dataInfo)
t = t.drop("gene")

# Pass the Pipeline Through the Function To Connect to Ibis
pipeline = PipelineIbis(pipe)
result = pipeline.to_ibis()(t)

# Get the kbest object
# Get the boolean mask indicating which features were selected
# Get the list of all feature names
# Get the list of selected feature names
kbest = pipe.named_steps["kbest"]
selected_mask = kbest.get_support()
all_feature_names = dataInfo.columns.tolist()
selected_feature_names = [all_feature_names[i] for i in range(min(len(selected_mask), len(all_feature_names)))
                          if selected_mask[i]]

# Obtain the PCA Variance Ratio, PCA Steps, Logistic Steps, Logistic Coefficients, Logistic Intercept
pca = pipe.named_steps["pca"]
variances = pca.explained_variance_ratio_
logistic = pipe.named_steps['logistic']
coefficients = logistic.coef_
intercept = logistic.intercept_

# Predict Y-Value and Accuracy
y_pred = pipe.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Selected feature names:", selected_feature_names)
print("Variances", variances)
print('Coefficients:', coefficients)
print('Intercept:', intercept)
print('Accuracy score:', accuracy)

# # Reference Link: https://github.com/ibis-project/ibis-examples/blob/main/examples/Substrait.ipynb

# Do this instead of dubkdb:
# https://github.com/tokoko/substrait-ml/blob/main/sklearn_ibis/pipeline/pipeline_ibis.py

# Another Reference
# https://ibis-project.org/ibis-for-sql-programmers/#non-trivial-grouping-keys

# Command
#sys.path.append(f'{os.getcwd()}/substrait-ml')

