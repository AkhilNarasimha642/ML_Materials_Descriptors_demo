import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from random import shuffle

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


parser = ArgumentParser(description='Chose descriptor, prop, set')
parser.add_argument('db', type=str, help='database name')
parser.add_argument('descriptor_name', type=str)
parser.add_argument('out_prop', type=str)
parser.add_argument('-sf', '--split_fract',default=0.6, type=float, help='split fraction')
parser.add_argument('-sn','--set_num', default='', type=str, help='set number to train/test')
args = parser.parse_args()


def get_train_test_files(df_tt_set, split_fract):
    """split given file set into train/test batches according to
    split_fraction """
    files_ls = df_tt_set['file_names'].values
    num_train = int(len(files_ls) * split_fract)
    train_files = files_ls[:num_train]
    test_files = files_ls[num_train:]
    return train_files, test_files


def get_train_test_inputs(df_descriptor, train_files, test_files, normalize):
    """Organize input matrix according to chosen train/test files
    return input training matrix, input testing matrix"""
    df_descriptor = df_descriptor.T  # as [num_samples,num_features]
    if normalize == True:
        scaler = MinMaxScaler((-1, 1))
        input_mat_norm = scaler.fit_transform(df_descriptor.as_matrix())
        df_descriptor_norm = pd.DataFrame(input_mat_norm, columns=df_descriptor.columns, index=df_descriptor.index)
        df_train = df_descriptor_norm.loc[train_files, :]
        df_test = df_descriptor_norm.loc[test_files, :]
    else:
        df_train = df_descriptor.loc[train_files, :]
        df_test = df_descriptor.loc[test_files, :]
    return df_train.as_matrix(), df_test.as_matrix()


def get_train_test_outputs(df_output_props, output_property, train_files, test_files):
    """Organize output vector of values according to chosen train/test files
    return output training vector, output testing vector"""
    output_train = np.empty(len(train_files))
    output_test = np.empty(len(test_files))
    for i in range(len(train_files)):
        file_i = train_files[i]
        output_train[i] = df_output_props[file_i][output_property]
    for i in range(len(test_files)):
        file_i = test_files[i]
        output_test[i] = df_output_props[file_i][output_property]
    return output_train, output_test


def save_path_name_ext(comp_path, database, descriptor_name, out_prop, set_name, num_trees, split_fract):
    """determing the saving directory name according to various provided parameters
    create corresponding directories"""
    set_name=set_name.split('-')[0]+'-set-'
    save_path = comp_path + '/data/' + database + '/machine_learning/results/'+str(num_trees)+'trees/'+ descriptor_name + '/' \
                + out_prop +'/'+set_name+str(split_fract)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def save_df_out(out_names_vals, predicted_train, predicted_test, save_path_name):
    """save true, train, test val outputs in df format for all files (with train/test labels)
     and actual/predicted values for each file"""
    file_names = out_names_vals[0]  # ordered as train untill cutoff then test (cutoff=len(predicted_train))
    out_vals_true = out_names_vals[1]
    num_train_f = len(predicted_train)
    num_test_f = len(predicted_test)

    df_out = pd.DataFrame(index=['type', 'out_true', 'out_pred'], columns=file_names)
    df_out.loc['out_true', :] = out_vals_true
    df_out.loc['out_pred', file_names[:num_train_f]] = predicted_train
    df_out.loc['out_pred', file_names[num_train_f:]] = predicted_test
    df_out.loc['type', file_names[:num_train_f]] = ['train' for i in range(num_train_f)]
    df_out.loc['type', file_names[num_train_f:]] = ['test' for i in range(num_test_f)]
    df_out.to_csv(save_path_name + '/ttt_out.csv')  # save out: truth train test


def train_test(train_input_mat, test_input_mat, train_out, test_out, num_trees, save_path_name, out_names_vals, pickle):
    """ML model set up
    Initialization of Random Forest, pickling options, calcualtion of rmse, cv scores"""
    rf_model = RandomForestRegressor(n_estimators=num_trees, n_jobs=1)
    rf_model.fit(train_input_mat, train_out)
    feat_imp_path_name = save_path_name + '/features_importance.csv'
    np.savetxt(feat_imp_path_name, rf_model.feature_importances_, delimiter=',')

    predicted_train = rf_model.predict(train_input_mat)
    predicted_test = rf_model.predict(test_input_mat)
    # save true, train, test outputs in df format for all files (with train/test labels)
    save_df_out(out_names_vals, predicted_train, predicted_test, save_path_name)

    # save the rf model with app params, note file is about 250mb
    if pickle == True:
        pickle_path_name = save_path_name + '/rf_model.pkl'
        joblib.dump(rf_model, pickle_path_name)

    mse_train = metrics.mean_squared_error(train_out, predicted_train)
    rmse_train = mse_train ** 0.5
    cv_rmse_train = rmse_train / np.average(train_out)

    mse_test = metrics.mean_squared_error(test_out, predicted_test)
    rmse_test = mse_test ** 0.5
    cv_rmse_test = rmse_test / np.average(test_out)
    return [rmse_train, cv_rmse_train, rmse_test, cv_rmse_test]


def data_row(comp_path, database, descriptor_name, out_prop, split_fract, set_name, num_trees, df_descriptor,
             df_output_props, df_tt_set, normalize, pickle):
    """compilation of all information about the ML training and testing"""
    train_files = get_train_test_files(df_tt_set, split_fract)[0]
    test_files = get_train_test_files(df_tt_set, split_fract)[1]

    train_input_mat = get_train_test_inputs(df_descriptor, train_files, test_files, normalize)[0]
    test_input_mat = get_train_test_inputs(df_descriptor, train_files, test_files, normalize)[1]
    train_out = get_train_test_outputs(df_output_props, out_prop, train_files, test_files)[0]
    test_out = get_train_test_outputs(df_output_props, out_prop, train_files, test_files)[1]

    out_names_vals = [np.append(train_files, test_files), np.append(train_out, test_out)]
    save_path_name = save_path_name_ext(comp_path, database, descriptor_name, out_prop, set_name, num_trees, split_fract)
    train_test_info = train_test(train_input_mat, test_input_mat, train_out, test_out, num_trees, save_path_name,
                                 out_names_vals, pickle)

    data_r = [database, descriptor_name, out_prop, len(train_files), len(test_files)]
    for v in train_test_info:
        data_r.append(round(v, 3))
    data_r.append(num_trees)
    data_r.append(str(split_fract))
    data_r.append(set_name)
    return data_r

def get_root(num_layers):
    """find the root directory of the project"""
    curren_dir = os.path.dirname(os.path.realpath(__file__))
    for i in range(num_layers):
        curren_dir = os.path.dirname(curren_dir)
    return curren_dir


def create_random_set(database):
    """create a shuffled list of files from a given database"""
    comp_path = get_root(num_layers=2)
    structures_path = comp_path + '/data/' + database + '/structure_files'
    structures_list = os.listdir(structures_path)
    list2 = structures_list.copy()
    shuffle(list2)
    df_files_sets = pd.DataFrame(list2, columns=['file_names'])
    save_path = comp_path + '/data/' + database + '/machine_learning/train_test_sets'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_name = 'set'
    for i in range(10000):
        new_file_name = str(i) + '-' + file_name + '.csv'
        if not os.path.isfile(save_path + '/' + new_file_name):
            df_files_sets.to_csv(save_path + '/' + new_file_name)
            break
    return new_file_name
    # print('increase the max index in create_random_set')


def descriptor_ml(comp_path, database, descriptor_name, out_prop, split_fract, set_name, num_trees,\
                        normalize=True, pickle=True):
    """main function
    performs ML saves predictions and statistics about the run"""
    df_tt_set = pd.DataFrame.from_csv(comp_path + '/data/' + database + '/machine_learning/train_test_sets/' + set_name)
    df_output_props = pd.DataFrame.from_csv(comp_path + '/data/' + database + '/output_props.csv')
    if descriptor_name not in ('xrd', 'cm_es', 'ofm'):
        print('provide a valid descriptor name')
    else:
        df_descriptor = pd.DataFrame.from_csv(comp_path + '/data/' + database + '/descriptor_dfs/' \
            + descriptor_name + '/' + descriptor_name + '.csv')
        data_r = data_row(comp_path, database, descriptor_name, out_prop, split_fract,set_name, num_trees, df_descriptor,
                      df_output_props, df_tt_set, normalize, pickle)
        print('trained')

        data_cols = ['Database', 'Descriptor_Name', 'Output_Prop', 'Training_Size', 'Testing_Size',
                     'rmse_train', 'cv_rmse_train','rmse_test', 'cv_rmse_test',
                     'Num_Trees','split_fract','set_name']
        df = pd.DataFrame([data_r], columns=data_cols)
        save_path_name = save_path_name_ext(comp_path, database, descriptor_name, out_prop, set_name, num_trees, \
                                                    split_fract)
        df.to_csv(save_path_name+'/ml_stats.csv')
        print('data saved')


if __name__ == "__main__":
    database = args.db
    descriptor_name = args.descriptor_name
    out_prop = args.out_prop
    split_fract = args.split_fract
    set_num = args.set_num
    num_trees = 300
    comp_path = get_root(num_layers=2)
    if set_num == '':
        set_name = create_random_set(database)
    else:
        set_name = set_num + '-set.csv'
    descriptor_ml(comp_path, database, descriptor_name, out_prop, split_fract, set_name, num_trees)
