#Database tools
import sqlalchemy as sa
import psycopg2 as p
import pandas as pd

#Math tools
import numpy as np

#ML tools
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skrebate import ReliefF, MultiSURF

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample

#Class imbalance modules
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours,TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade, EasyEnsemble

#Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

#System tools
import time
import datetime
import os
import io
from dotenv import load_dotenv, find_dotenv
import pickle
import dill
import sys, getopt


def chi2_fs(X_df,X_train_all,X_test_all,y_train,p_val_thresh):
    '''Chi2 statistical test for feature selection'''
    #Df with only continuous variables
    cont_data = X_df.loc[:, X_df.apply(lambda x: x.nunique()) >= 1000]

    #Find column indices of continuous features
    cont_data_id =  np.where(np.isin(X_df.columns, cont_data.columns))[0]

    cont_data_colnames = cont_data.columns

    #Remove continuous features for chi-sq
    X_train_fs = np.delete(X_train_all,cont_data_id,1)

    c,p = chi2(X_train_fs,y_train)

    feature_ids = np.where(p<=p_val_thresh)[0]

    #Df with no continuous variables. This is used for chi-sq feature selection
    data_fs = X_df.drop(cont_data.columns,axis=1)

    selected_features = np.append(cont_data_colnames,np.array(data_fs.columns[feature_ids]))
    selected_feature_id = np.where(np.isin(X_df.columns, selected_features))[0]

    #New X_train and X_test matrices
    X_train = X_train_all[:,selected_feature_id]
    X_test = X_test_all[:,selected_feature_id]

    X_train_df = pd.DataFrame(X_train,columns=selected_features)

    return selected_features, X_train_df, X_train, X_test

def anova_fs(X_df,X_train_all,X_test_all,y_train,p_val_thresh):
    '''ANOVA F statistical test for feature selection'''
    f,p = f_classif(X_train_all,y_train)
    feature_ids = np.where(p<=p_val_thresh)[0]


    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    X_train_df = pd.DataFrame(X_train,columns=selected_features)

    return selected_features, X_train_df, X_train, X_test

def relieff_fs(X_df,X_train_all,X_test_all,y_train):
    '''ReliefF for feature selection'''
    fs = ReliefF(discrete_threshold = 1000, n_jobs=1)
    fs.fit(X_train_all, y_train)

    feature_scores = fs.feature_importances_
    feature_ids = np.where(feature_scores>=0)[0]
    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    return selected_features, feature_scores, X_train, X_test

def multisurf_fs(X_df,X_train_all,X_test_all,y_train):
    '''MultiSURF for feature selection'''
    fs = MultiSURF(discrete_threshold = 1000, n_jobs=1)
    fs.fit(X_train_all, y_train)

    feature_scores = fs.feature_importances_
    feature_ids = np.where(feature_scores>=0)[0]
    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    return selected_features, feature_scores, X_train, X_test
