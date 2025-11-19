import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import Training_and_classification_functions as func
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--df", type = str, help = "Dataframe filename containing all data (training and test), they must contain labeled glitches")

parser.add_argument("--per", type = float, help = "Percentage in decimal form of data to use for training", default = 0.75)

parser.add_argument("--df_train", type = str, help = "Dataframe filename training data")

parser.add_argument("--df_classify", type = str, help = "Dataframe filename containing data to classify")

parser.add_argument("--outputdir", type = str, help = "Name of output directory", default = os.getcwd())

parser.add_argument("--datadir", type = str, help = "Name of directory with dataframes", default = os.getcwd())

parser.add_argument("--n_trees", type = float, help = "Number of trees for random forest", default = 50)

parser.add_argument("--max_depth", type = float, help = "Max depth for random forest", default = 15)

parser.add_argument("--trained", help = "Have you already trained your forest?", action = "store_true")

parser.add_argument("--trained_forest", type = str, help = "Trained forest pickle file")

parser.add_argument("--forestdir", type = str, help = "Name of directory with trained forest", default = os.getcwd())


args = parser.parse_args()

#Check that you have the correct dfs and/or trained forest
if args.trained:

  if args.trained_forest is None:
    parser.error('Need to provide the trained random forest')

  if args.df_classify is None:
      parser.error('Need to provide dataframe to classify')

else:

  if (args.df is None and args.df_train is None and args.df_classify is None) or \
  (args.df is None and args.df_train is not None and args.df_classify is None) or \
  (args.df is None and args.df_train is None and args.df_classify is not None) or \
  (args.df is not None and args.df_train is not None and args.df_classify is not None):
    parser.error('Need to either provide dataframe to split into training and test data only or separate training and classification dataframes')

args_dict = vars(args)

cols_for_training = ['Number of Detectors', 'Y and X Extent Ratio','Y Hist Max and Adjacent/Number of Detectors',
          'Within 0.1 of Y Hist Max/Number of Detectors', 'Mean abs(Correlation)', 'Mean abs(Time Lag)', 'Number of Peaks']

#if a trained forest is provided, classify the input df
if args_dict['trained']:

  forest_train = pk.load(open('{}/{}'.format(args_dict['forestdir'], args_dict['trained_forest']), 'rb'))

  df_classify = pd.read_csv('{}/{}'.format(args_dict['datadir'], args_dict['df_classify']))

  df_classified = func.classify_data_forest(df_classify, cols_for_training, forest_train)

  df_classified.to_csv('{}/predicted_labels_{}'.format(args_dict['outputdir'], args_dict['df_classify']))

#the df must have labeled glitches in order to train the forest
else:

  #if df is supplied instead then need to split the data into training and test sets as well as classify the test set
  if args.df is not None:

    df = pd.read_csv('{}/{}'.format(args_dict['datadir'], args_dict['df']))

    df_train, df_test, train_inds = func.split_into_training_and_test(df, args_dict['per'])
    
    forest_train = func.training_forest(df_train, cols_for_training, args_dict['n_trees'], args_dict['max_depth'])

    df_classified = func.classify_data_forest(df_test, cols_for_training, forest_train)

    df_classified.to_csv('{}/predicted_labels_classified_{}'.format(args_dict['outputdir'], args_dict['df']))

    df_train.to_csv('{}/{}_training_set.csv'.format(args_dict['outputdir'], args_dict['df']))

    df_test.to_csv('{}/{}_test_set.csv'.format(args_dict['outputdir'], args_dict['df']))

    np.savetxt('{}/{}_training_inds.txt'.format(args_dict['outputdir'], args_dict['df']), train_inds)

    pk.dump(forest_train, open('{}/{}_trained_forest.pkl'.format(args_dict['outputdir'], args_dict['df']), 'wb'))

  #if a training and test set are already supplied, just train then classify
  else:

    df_train = pd.read_csv('{}/{}'.format(args_dict['datadir'], args_dict['df_train']))

    df_classify = pd.read_csv('{}/{}'.format(args_dict['datadir'], args_dict['df_classify']))

    forest_train = func.training_forest(df_train, cols_for_training, args_dict['n_trees'], args_dict['max_depth'])

    df_classified = func.classify_data_forest(df_classify, cols_for_training, forest_train)

    df_classified.to_csv('{}/predicted_labels_{}'.format(args_dict['outputdir'], args_dict['df_classify']))

    pk.dump(forest_train, open('{}/{}_trained_forest.pkl'.format(args_dict['outputdir'], args_dict['df_train']), 'wb'))