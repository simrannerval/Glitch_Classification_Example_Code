'''
Functions for training random forest and classifying glitches.
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk


def split_into_training_and_test(df, train_per):

    '''
    Split pandas dataframe into training and test set.

    Input: df: pandas data frame, train_per: percent of the data for training set in decimal form
    Output: df_train: pandas dataframe of the training data, df_test: pandas dataframe of the test data, 
    inds: array of the indices of the training data in original dataframe
    '''

    train_inds = np.random.choice(a = df.shape[0], size = int(df.shape[0]*train_per), replace = False)

    df_train = df.iloc[train_inds].reset_index()

    df_test = df.drop(df.index[train_inds]).reset_index()

    return df_train, df_test, train_inds


def training_forest(df_train, cols, n_trees = 50, max_depth = 15):

    '''
    Train random forest.

    Input: df_train: pandas data frame of training data, cols: list of columns of stats to train with
    Optional: n_trees: number of trees in forest (default = 50),
    max_depth: maximum number of splits for each tree (default = 15)
    Output: forest: trained random forest
    '''

    X, Y = df_train[cols], df_train['Train_Lab']

    forest = RandomForestClassifier(criterion='entropy', n_estimators = n_trees, random_state=1, n_jobs=2, max_depth = max_depth)
    
    forest.fit(X, Y)

    return forest


def classify_data_forest(df_classify, cols, trained_forest):

    '''
    Classify glitches using a random forest.

    Input: df_classify: pandas data frame of data to classify,
    cols: list of columns of stats to use for classification (must match cols used for training),
    trained_forest: trained random forest
    Output: df_w_labs_and_stats: returns the dataframe with a column for the predicted labels - int from 
    0 - 3 corresponding to 0: Point Sources, 1: Point Sources + Other 2: Cosmic Rays, 3: Other also columns 
    with the probability that the glitch is each of the categories
    '''

    col_predictions = ['Glitch Prediction', 'Probability of being a Point Source', 'Probability of being a Point Source + Other', 'Probability of being a Cosmic Ray', 'Probability of being an Other']

    X_classify = df_classify[cols]

    y_pred_forest = trained_forest.predict(X_classify)

    y_pred_forest_probs = trained_forest.predict_proba(X_classify)

    predictions = np.zeros((X_classify.shape[0], 5))

    predictions[:, 0] = y_pred_forest
    predictions[:, 1] = y_pred_forest_probs[:, 0]
    predictions[:, 2] = y_pred_forest_probs[:, 1]
    predictions[:, 3] = y_pred_forest_probs[:, 2]
    predictions[:, 4] = y_pred_forest_probs[:, 3]

    lab_df = pd.DataFrame(predictions, columns = col_predictions)
 
    df_w_labs_and_stats = pd.concat([df_classify, lab_df], axis = 1)

    return df_w_labs_and_stats
