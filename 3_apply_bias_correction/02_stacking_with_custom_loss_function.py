from numpy.random import seed
seed(1)
from tensorflow.keras.utils import set_random_seed
set_random_seed(2)

import os

import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import regularizers
#import keras.backend as K
from tensorflow.keras import backend as K

from sklearn.metrics import r2_score
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Base path ####################################################################

# Set paths for MX
from paths import paths_mx as paths
output_path_mx = paths["results"]
stack_path_mx = output_path_mx + '/stacking_preds/'

from paths import paths_us as paths
output_path_us = paths["results"]
stack_path_us = output_path_us + '/stacking_preds/'

# Define filing names
outcome_var_list_mx = ['yschooling', 'pbasic', 'secondary', 'primary']
outcome_var_list_us = ['yschooling', 'some_college', 'high_school', 'bachelor']

fold_datasets_mx = {}
fold_datasets_us = {}

# Get fold paths
for idx, out_var in enumerate(outcome_var_list_mx):
    fold_paths = [filename for filename in os.listdir(stack_path_mx) if filename.startswith(out_var)]
    fold_datasets_mx[out_var] = fold_paths

for idx, out_var in enumerate(outcome_var_list_us):
    fold_paths = [filename for filename in os.listdir(stack_path_us) if filename.startswith(out_var)]
    fold_datasets_us[out_var] = fold_paths


def load_and_process_data(train_file, test_file, path):

    print(path + train_file)
    X_train = pd.read_csv(path + train_file, sep=',', index_col=0, dtype={'GEOID': str})

    print(path + test_file)
    X_test = pd.read_csv(path + test_file, sep=',', index_col=0)
    geoid = X_test[['GEOID']].copy()

    X_train['const'] = 1
    X_test['const'] = 1

    y_train = X_train.iloc[:, 1].copy()
    y_test = X_test.iloc[:, 1].copy()

    model_list = ['enet', 'gb', 'svm', 'knn', 'mlp', 'const']
    X_train = X_train[model_list]
    X_test = X_test[model_list]

    y_train_percentiles = np.percentile(y_train, [20, 40, 60, 80], axis=0)

    return {'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_train_perc': y_train_percentiles,
            'GEOID': geoid[['GEOID']]}


# Ridge regression #############################################################
def custom_loss_function(q_loss, q_alpha, y_true_perc):
    def quantile_loss(y_true, y_pred):

        ## # Quantile 1
        mask_1 = tf.less(y_true, y_true_perc[0])
        q_1 = tf.math.square(K.mean(tf.boolean_mask(y_true, mask_1) - \
                                    tf.boolean_mask(y_pred, mask_1)))

        ## # Quantile 2
        mask_2 = tf.math.logical_and(tf.greater(y_true, y_true_perc[0]),
                                     tf.less(y_true, y_true_perc[1]))
        q_2 = tf.math.square(K.mean(tf.boolean_mask(y_true, mask_2) - \
                                    tf.boolean_mask(y_pred, mask_2)))

        ## # Quantile 3
        mask_3 = tf.math.logical_and(tf.greater(y_true, y_true_perc[1]),
                                     tf.less(y_true, y_true_perc[2]))
        q_3 = tf.math.square(K.mean(tf.boolean_mask(y_true, mask_3) - \
                                    tf.boolean_mask(y_pred, mask_3)))

        ## # Quantile 4
        mask_4 = tf.math.logical_and(tf.greater(y_true, y_true_perc[2]),
                                     tf.less(y_true, y_true_perc[3]))
        q_4 = tf.math.square(K.mean(tf.boolean_mask(y_true, mask_4) - \
                                    tf.boolean_mask(y_pred, mask_4)))

        ## # Quantile 5
        mask_5 = tf.greater(y_true, y_true_perc[3])
        q_5 = tf.math.square(K.mean(tf.boolean_mask(y_true, mask_5) - \
                                    tf.boolean_mask(y_pred, mask_5)))

        # print(tf.stack([q_1, q_2, q_3, q_4, q_5], axis=0, name='stack'))
        q_penalty = tf.reduce_max(tf.stack([q_1, q_2, q_3, q_4, q_5], 0))

        # The regularization losses will be added automatically by setting kernel_regularizer or bias_regularizer in each of the keras layers.
        # https://stackoverflow.com/questions/67912239/custom-loss-function-with-regularization-cost-added-in-tensorflow

        if q_loss:
            print('use MSE + Quantile loss')
            return K.mean(tf.math.square(y_true - y_pred)) + (q_alpha * q_penalty)
        else:
            print('use MSE loss')
            return K.mean(tf.math.square(y_true - y_pred))

    return quantile_loss


def evaluate_model(y_train, y_pred_train, y_test, y_pred_test):

    print('r2', r2_score(y_train, y_pred_train))
    print('r2', r2_score(y_test, y_pred_test))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_train,
                y_pred_train)

    b1, b0 = np.polyfit(y_train,
                        y_pred_train,
                        1)  # Estimate slope and intercept
    print('b:', b1)
    max_x = max(y_train) + 0.5
    min_x = min(y_train) - 0.5
    x = np.array([min_x, max_x])
    plt.plot(x, b1 * x + b0, color="black")
    plt.plot(x, x, color="black")

    plt.show()


def inst_and_run_model(X_train, y_train, X_test, y_test, y_true_perc, geoid, q_loss, q_alpha, outcome_var):

    set_random_seed(1)

    if outcome_var == 'yschooling':
        print('IS yschooling')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X_train))

        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(units=1,
                         use_bias=True,
                         kernel_regularizer=regularizers.L2(l2=0.1))
        ])

    else:
        print('NOT yschooling')

        linear_model = tf.keras.Sequential([
            layers.Dense(units=1,
                         use_bias=True,
                         kernel_regularizer=regularizers.L2(l2=0.1))
        ])

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=custom_loss_function(q_loss=q_loss,
                                  q_alpha=q_alpha,
                                  y_true_perc=y_true_perc))

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                patience=800,
                                                min_delta=0.0001,
                                                restore_best_weights=True)

    linear_model.fit(
        X_train,
        y_train,
        epochs=800,
        batch_size=int(len(X_train) * 0.15),
        verbose=0,
        validation_split=0,
        callbacks=callback)

    y_pred_train = linear_model.predict(X_train).flatten()
    y_pred_test = linear_model.predict(X_test).flatten()

    evaluate_model(y_train, y_pred_train, y_test, y_pred_test)

    y_test_pred = pd.DataFrame.from_records([y_test,
                                             y_pred_test]).T
    y_test_pred.columns = ['y_true', 'y_pred']
    y_test_pred['GEOID'] = geoid[['GEOID']]

    return y_test_pred[['GEOID', 'y_true', 'y_pred']]


def fold_processing(file_names, fold_datasets, stack_path, q_loss, q_alpha, out_path):

    fold_prediction_list = []

    print('Outcome Name: ', file_names)

    # Iterate over folds
    for fold in range(0, 5):

        print('Fold: ', fold)

        # Get train and test set of specific folds
        train_file = [s for s in fold_datasets[file_names] if 'train' + str(fold) in s][0]
        test_file = [s for s in fold_datasets[file_names] if 'test' + str(fold) in s][0]

        # Preprocess data
        data_sets = load_and_process_data(train_file, test_file, stack_path)

        # Instantiate, learn model and compute predictions
        y_pre_true = inst_and_run_model(data_sets['X_train'],
                                        data_sets['y_train'],
                                        data_sets['X_test'],
                                        data_sets['y_test'],
                                        data_sets['y_train_perc'],
                                        data_sets['GEOID'],
                                        q_loss, q_alpha,
                                        file_names)

        # Append predictions to fold_prediction_list
        fold_prediction_list.append(y_pre_true)

    # Merge and store fold prediction list
    full_true_pred_df = pd.concat(fold_prediction_list)

    print('--- Total model evaluation ---')
    evaluate_model(full_true_pred_df['y_true'],
                   full_true_pred_df['y_pred'],
                   full_true_pred_df['y_true'],
                   full_true_pred_df['y_pred'])

    full_true_pred_df.to_csv(out_path + '/bias_correction/' + \
                             file_names + '-' + \
                             str(q_loss) + '-' + str(q_alpha) + \
                             '.csv',
                             sep=';',
                             encoding='utf-8',
                             index=False)

# MX ###########################################################################
for file_names in ['yschooling']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, False, 0, output_path_mx)
# Full dataset:
# r2 0.6911339060440118
# b: 0.6684526611493973

for file_names in ['yschooling']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, True, 1, output_path_mx)
# Full dataset:
# r2 0.6537681082892379
# b: 0.8456805727826715

for file_names in ['yschooling']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, True, 3, output_path_mx)
# Full dataset:
# alpha = 3
# r2 0.5935989482027567
# b: 0.9445896775649937

for file_names in ['pbasic']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, True, 15, output_path_mx)
# Full dataset:
# alpha = 15
# r2 0.5912791534490454
# b: 0.8999294260721851
# alpha = 20
# r2 0.576195523429855
# b: 0.9251801979702536

for file_names in ['secondary']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, True, 15, output_path_mx)
# Full dataset:
# alpha = 15
# r2 0.47787030047435897
# b: 0.8988980104562855
# alpha = 20
# r2 0.45732391103458436
# b: 0.920887159476019

for file_names in ['primary']:
    print('MX: ', file_names)
    fold_processing(file_names, fold_datasets_mx, stack_path_mx, True, 15, output_path_mx)
# Full dataset:
# alpha = 15
# r2 0.5083288700388835
# b: 0.7859322159576738
# alpha = 20
# r2 0.452353696310393
# b: 0.8391579569489872


# US ###########################################################################
for file_names in ['yschooling']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, False, 0, output_path_us)
# Full dataset:
# no correction
# r2 0.6379743309160223
# b: 0.6223010000848195

for file_names in ['yschooling']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, True, 1, output_path_us)
# Full dataset:
# alpha = 1
# r2 0.5753628207005321
# b: 0.8322868824034698

for file_names in ['yschooling']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, True, 3, output_path_us)
# Full dataset:
# alpha = 3
# r2 0.5147202658014391
# b: 0.9109743124844011


for file_names in ['bachelor']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, True, 15, output_path_us)
# Full dataset:
# alpha = 15
# r2 0.5623508988303654
# b: 0.980232248651531
# alpha = 20
# r2 0.5598016839473854
# b: 0.9779858691057851

for file_names in ['some_college']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, True, 20, output_path_us)
# Full dataset:
# alpha = 15
# r2 0.29718438040053885
# b: 0.8939360111807227
# alpha = 20
# r2 r2 0.28481903174494017
# b: 0.9002814925741605

for file_names in ['high_school']:
    print('US: ', file_names)
    fold_processing(file_names, fold_datasets_us, stack_path_us, True, 20, output_path_us)
# Full dataset:
# alpha = 15
# r2 0.32209030978788367
# b: 0.5300724921917043
# alpha = 20
# r2 0.31875643895298955
# b: 0.5353237449276966
