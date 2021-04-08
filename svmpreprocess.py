import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler

"""
Read in data
"""
print("Reading data...")
# df = pd.read_csv('HTRU_2.csv')
df0 = pd.read_csv('data0.csv')
df1 = pd.read_csv('data1.csv')

df0 = df0.sample(n=len(df1), random_state = 0)

"""
Preprocess and data preparation
"""
print("Processing data...")

pre_features_0 = df0.iloc[: , :8]
pre_target_0 = df0['class']

# rescale variables to have 0 mean and unit variance
df_columns_0 = pre_features_0.columns
scaler = StandardScaler()
scaledf_0 = scaler.fit_transform(pre_features_0)

pre_features_1 = df1.iloc[: , :8]
pre_target_1 = df1['class']

# rescale variables to have 0 mean and unit variance
df_columns_1 = pre_features_1.columns
scaledf_1 = scaler.fit_transform(pre_features_1)

# restructure dataframe
df0 = pd.DataFrame(scaledf_0)
df0.columns = df_columns_0
df0['class'] = pre_target_0
df0['class'] = df0['class'].fillna(0).astype(np.int64)

df1 = pd.DataFrame(scaledf_1)
df1.columns = df_columns_1
df1['class'] = pre_target_1

# reassign feature and targets to newly scaled variables
features_0 = df0.iloc[:, :8]
target_0 = df0['class']

features_1 = df1.iloc[:, :8]
target_1 = df1['class']

# pre_features = df.iloc[: , :8]
# pre_target = df['class']

# # rescale variables to have 0 mean and unit variance
# df_columns = pre_features.columns
# scaler = StandardScaler()
# scaledf = scaler.fit_transform(pre_features)

# # restructure dataframe
# df = pd.DataFrame(scaledf)
# df.columns = df_columns
# df['class'] = pre_target

# # reassign feature and targets to newly scaled variables
# features = df.iloc[:, :8]
# target = df['class']

# split dataset into test and train split
# 80/20 split
feat_train_0, feat_test_0, target_train_0, target_test_0 = train_test_split(features_0, target_0, train_size = 0.8, random_state = 0)
feat_train_1, feat_test_1, target_train_1, target_test_1 = train_test_split(features_1, target_1, train_size = 0.8, random_state = 0)

feat_train = feat_train_0.append(feat_train_1)
feat_test = feat_test_0.append(feat_test_1)
target_train = target_train_0.append(target_train_1)
target_test = target_test_0.append(target_test_1)

# print(feat_train)
# print(feat_test)
# print(target_train)
# print(target_test)

"""
Set up different SVM models to see which performs best.

C is set to 1 to start and if different hyperplanes are needed, C is raised or 
lowered accordingly.

For Gaussian model, we set gamma to 1 to begin. If we need less or more curvature
in our decision boundary, we will adjust gamma.

We use 'ovr' as our decision function shape. This stands for one vs rest or 
one vs all. Since we only have two classes 0 or 1, this will work fine for our
purposes.
"""
# Linear
print("Preparing Linear model...")
linearSVM = svm.SVC(kernel='linear', C=1, decision_function_shape = 'ovo')
linearSVM.fit(feat_train, target_train)

# Gaussian RBF
print("Preparing Gaussian RBF model...")
rbfSVM = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape = 'ovr')
rbfSVM.fit(feat_train, target_train)

# Polynomial
print("Preparing Polynomial model...")
polySVM = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape = 'ovo')
polySVM.fit(feat_train, target_train)

# Sigmoid
print("Preparing Sigmoid model...")
sigSVM = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
sigSVM.fit(feat_train, target_train)

# Predict using the SVM models
print("Conducting predictions...")
linear_pred = linearSVM.predict(feat_test)
rbf_pred = rbfSVM.predict(feat_test)
poly_pred = polySVM.predict(feat_test)
sig_pred = sigSVM.predict(feat_test)

# Calculate the accuracies of each model
print("Calculating accuracies...")
linear_acc = accuracy_score(linear_pred, target_test)
rbf_acc = accuracy_score(rbf_pred, target_test)
poly_acc = accuracy_score(poly_pred, target_test)
sig_acc = accuracy_score(sig_pred, target_test)

# Print results
print("\n\nRESULTS:")
print("Linear model accuracy:", linear_acc)
print("Gaussian RBF model accuracy:", rbf_acc)
print("Polynomical model accuracy:", poly_acc)
print("Sigmoid model accuracy:", sig_acc)











