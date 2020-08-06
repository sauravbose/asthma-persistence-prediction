#System tools
import time, datetime
import os, io
from dotenv import load_dotenv, find_dotenv
import pickle, dill
import glob
import sys, getopt, copy
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing

#Database tools
import sqlalchemy as sa
import psycopg2 as p
import pandas as pd

#Math tools
import numpy as np

#ML tools
from sklearn.preprocessing import normalize, scale, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skrebate import ReliefF, MultiSURF

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample

#Class imbalance modules
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours,TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade, EasyEnsemble

#User defined modules
currentWorkingDirectory = './'
sys.path.append(currentWorkingDirectory)
from feature_selection_methods import chi2_fs, anova_fs, relieff_fs, multisurf_fs

#Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

#Extract command line arguements
try:
    opts,args = getopt.getopt(sys.argv[1:],"b:f:p:a:",["class_balance=","feature_selection=","p_val_thresh=","ml_algo="])

except getopt.GetoptError:
    print("Invalid input arguments")
    sys.exit(2)

for opt,arg in opts:

    if opt in ["-b","--class_balance"]:
        class_balance_method = arg

    elif opt in ["-f","--feature_selection"]:
        feature_selection_method = arg

    elif opt in ["-p","--p_val_thresh"]:
        p_val_thresh = float(arg)

    elif opt in ["-a","--ml_algo"]:
        algorithm = arg

file_path = './'

#Names for storing results
if p_val_thresh:
    algorithm_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+ str(p_val_thresh)[2:]+'_'+algorithm
    trials_file_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+ str(p_val_thresh)[2:]+'_'+algorithm+'_trials' +'.pik'
    results_file_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+ str(p_val_thresh)[2:]+'_'+algorithm+'_results' +'.pik'
else:
    algorithm_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+algorithm
    trials_file_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+algorithm+'_trials'+'.pik'
    results_file_name = class_balance_method+"_cb_"+feature_selection_method+"_fs_"+algorithm+'_results'+'.pik'

#Uncomment for line by line memory profiling
# from memory_profiler import profile
# mem_data = open(results_file_name+'memory_profiler.log', 'w+')
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def load_data():
    #Read the pickle files as a list of classes
    with open(file_path+'train_data.pik', "rb") as f:
            data = dill.load(f)

    X_df = data['X_df']
    X_train_all = data['X']
    y_train = data['y']
    return X_df, X_train_all, y_train

X_df, X_train_all, y_train = load_data()

#Define objective function to optimize
#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def create_cv_data(X_train_all, y_train,r):

    num_splits = 5
    #Create CV folds
    skf = StratifiedKFold(n_splits=num_splits, random_state=r)

    def pre_process(train_index, test_index):
        train_x, test_x = X_train_all[train_index], X_train_all[test_index]
        train_y, test_y = y_train[train_index], y_train[test_index]

        #Class Balance on the training split
        if class_balance_method == 'rand_under':
            rus = RandomUnderSampler(sampling_strategy='majority',random_state = 0)
            train_x, train_y = rus.fit_sample(train_x,train_y)

        elif class_balance_method == 'enn':
            enn = EditedNearestNeighbours(n_neighbors=5, random_state=0, n_jobs=1)
            train_x, train_y = enn.fit_sample(train_x, train_y)

        elif class_balance_method == 'renn':
            renn = RepeatedEditedNearestNeighbours(n_neighbors=5, random_state=0, n_jobs=1)
            train_x, train_y = renn.fit_sample(train_x, train_y)

        elif class_balance_method == 'tomek':
            tl = TomekLinks(random_state=0)
            train_x, train_y = tl.fit_sample(train_x, train_y)

        elif class_balance_method == 'tomek_enn':
            tl = TomekLinks(random_state=0)
            train_x, train_y = tl.fit_sample(train_x, train_y)

            enn = EditedNearestNeighbours(n_neighbors=5, random_state=0, n_jobs=1)
            train_x, train_y = enn.fit_sample(train_x, train_y)

        elif class_balance_method == 'tomek_renn':
            tl = TomekLinks(random_state=0)
            train_x, train_y = tl.fit_sample(train_x, train_y)

            renn = RepeatedEditedNearestNeighbours(n_neighbors=5, random_state=0, n_jobs=1)
            train_x, train_y = renn.fit_sample(train_x, train_y)

        #Feature Selection on the training split
        #For all methods except the relief based
        feature_scores = 'N/A'

        if feature_selection_method == 'no':
            selected_features = X_df.columns

        elif feature_selection_method == 'chi2':
            selected_features, X_train_df, train_x, test_x = chi2_fs(X_df,train_x,test_x,train_y,p_val_thresh)

        elif feature_selection_method == 'anovaF':
            selected_features, X_train_df, train_x, test_x = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)

        elif feature_selection_method == 'reliefF':
            selected_features, feature_scores, train_x, test_x = relieff_fs(X_df,train_x,test_x,train_y)

        elif feature_selection_method == 'multisurf':
            selected_features, feature_scores, train_x, test_x = multisurf_fs(X_df,train_x,test_x,train_y)

        elif feature_selection_method == 'chi2_reliefF':
            selected_features_chi2, X_train_df, X_train_chi2, X_test_chi2 = chi2_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = relieff_fs(X_train_df,X_train_chi2,X_test_chi2,train_y)

        elif feature_selection_method == 'chi2_multisurf':
            selected_features_chi2, X_train_df, X_train_chi2, X_test_chi2 = chi2_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = multisurf_fs(X_train_df,X_train_chi2,X_test_chi2,train_y)

        elif feature_selection_method == 'anova_reliefF':
            selected_features_anova, X_train_df, X_train_anova, X_test_anova = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = relieff_fs(X_train_df,X_train_anova,X_test_anova,train_y)

        elif feature_selection_method == 'anova_multisurf':
            selected_features_anova, X_train_df, X_train_anova, X_test_anova = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = multisurf_fs(X_train_df,X_train_anova,X_test_anova,train_y)


        return train_x,train_y,test_x,test_y,selected_features,feature_scores

    num_cores = multiprocessing.cpu_count()
    cross_val_data = Parallel(n_jobs=5)(delayed
                        (pre_process)(train_index, test_index)
                            for train_index, test_index in skf.split(X_train_all, y_train))


    return cross_val_data

cv_rand_states = [314,42]
cross_val_data = []
for i in range(2):
    cross_val_data.extend(create_cv_data(X_train_all, y_train,cv_rand_states[i]))

#Define parameter search space for hyperopt
param_space = hp.choice('classifier_type', [
    {
        'type': 'bernaulli',
        'alpha': hp.uniform('alpha', 0.0, 2.0)
    },
    {
        'type': 'multinomial',
        'alpha': hp.uniform('alpha1', 0.0, 2.0)
    },
    {
        'type': 'gaussian'
    }
])

#Return nb classifier based on type
def build_clf(params):
    clf_name = params['type']
    del params['type']
    if clf_name == 'bernaulli':
        clf = BernoulliNB(**params)
    elif clf_name == 'multinomial':
        clf = MultinomialNB(**params)
    elif clf_name == 'gaussian':
        clf = GaussianNB(**params)
    else:
        return 0
    return clf, clf_name
#Define objective function to optimize
#Uncomment for line by line memory profiling
# from memory_profiler import profile
# mem_data = open(results_file_name+'memory_profiler.log', 'w+')
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def nb_objective(params):

    clf,clf_name = build_clf(params)

    def compute_cv_metric(split, cross_val_data):
        train_x = cross_val_data[split][0]
        train_y = cross_val_data[split][1]
        test_x = cross_val_data[split][2]
        test_y = cross_val_data[split][3]

        clf.fit(train_x,train_y)

        y_pred_prob = clf.predict_proba(test_x)

        # test_y_flipped = abs(np.array(test_y)-1)
        #
        # precision, recall, _ = precision_recall_curve(test_y_flipped, y_pred_prob[:,0])
        # aucpr = auc(recall, precision)

        aucpr = average_precision_score(test_y, y_pred_prob[:,0],pos_label=0)

        return aucpr

    eval_metric_arr = Parallel(n_jobs=10)(delayed
                        (compute_cv_metric)(split, cross_val_data)
                            for split in range(len(cross_val_data)))

    eval_metric = np.mean(eval_metric_arr)

    return {'loss':-eval_metric, 'loss_cv':eval_metric_arr, 'params':params,'clf_name':clf_name,'status':STATUS_OK}

rand_state = np.random.RandomState(314)

#Function to run and store a single trial
#Uncomment for line by line memory profiling
# from memory_profiler import profile
# mem_data = open(results_file_name+'memory_profiler.log', 'w+')
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def run_trials():
    trials_step = 1  # how many additional trials to do after loading saved trials.
    max_trials = 1  # initial max_trials.

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(file_path+'trials/'+trials_file_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    #Optimize
    best = fmin(fn=nb_objective, space=param_space, algo=tpe.suggest, max_evals=max_trials, trials=trials, rstate = rand_state)

    # save the trials object
    with open(file_path+'trials/'+trials_file_name, "wb") as f:
        pickle.dump(trials, f)

    return trials

#Run trials
def get_trials():
    for _ in range(2000):
        trials = run_trials()
    return trials

trials = get_trials()

#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def get_results(trials, X_train, y_train):

    #Sort the results in ascending order of loss
    bayes_trials_results = sorted(trials.results, key = lambda x: x['loss'])
    params = bayes_trials_results[0]['params']
    clf_name = bayes_trials_results[0]['clf_name']

    #Compute auxilary CV metrics
    def compute_cv_metric(split, cross_val_data, bayes_trials_results):

        #Create clasifier for cross validation results
        if clf_name == 'bernaulli':
            clf = BernoulliNB(**params)
        elif clf_name == 'multinomial':
            clf = MultinomialNB(**params)
        elif clf_name == 'gaussian':
            clf = GaussianNB(**params)

        train_x = cross_val_data[split][0]
        train_y = cross_val_data[split][1]
        test_x = cross_val_data[split][2]
        test_y = cross_val_data[split][3]

        clf.fit(train_x,train_y)

        y_pred_cv = clf.predict(test_x)
        y_pred_prob_cv = clf.predict_proba(test_x)

        tn = confusion_matrix(test_y, y_pred_cv)[0, 0]
        tp = confusion_matrix(test_y, y_pred_cv)[1, 1]
        fp = confusion_matrix(test_y, y_pred_cv)[0, 1]
        fn = confusion_matrix(test_y, y_pred_cv)[1, 0]

        npv = tn/(tn+fn)
        specificity = tn/(tn+fp)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        roc_auc_cv = roc_auc_score(test_y, y_pred_prob_cv[:,1])

        f1_cv = 2*(precision*recall)/(precision+recall)

        return npv, specificity, precision, recall, roc_auc_cv, f1_cv, y_pred_prob_cv

    eval_metric_arr = Parallel(n_jobs=10)(delayed
                        (compute_cv_metric)(split, cross_val_data, bayes_trials_results)
                            for split in range(len(cross_val_data)))

    cv_metrics = []
    for i in range(len(eval_metric_arr)):
        cv_metrics.append(eval_metric_arr[i][:-1])

    eval_metric = np.mean(cv_metrics,axis=0)
    #print(eval_metric)

    npv_cv = eval_metric[0]
    specificity_cv = eval_metric[1]
    precision_cv = eval_metric[2]
    recall_cv = eval_metric[3]
    roc_auc_cv = eval_metric[4]
    f1_cv = eval_metric[5]

    cv_pred_prob = []
    for i in range(len(eval_metric_arr)):
        cv_pred_prob.append(eval_metric_arr[i][-1])

    results = {'algo':algorithm_name, 'trials': trials.trials,
                'opt_param': bayes_trials_results[0]['params'],'param_sorted': bayes_trials_results,
                'specificity_CV': specificity_cv,'npv_CV':npv_cv, 'aucroc_CV':roc_auc_cv,
                'precision_CV':precision_cv, 'recall_CV':recall_cv, 'F1_CV':f1_cv,
                'cv_data':cross_val_data, 'cv_pred_prob':cv_pred_prob}

    return results

results = get_results(trials, X_train_all, y_train)

def write_results(results):
    with open(file_path+'results/'+results_file_name, "wb") as f:
        pickle.dump(results, f,protocol=pickle.HIGHEST_PROTOCOL)

write_results(results)
