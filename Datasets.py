import pandas as pd
import os
import math


def get_dataset(data_file, label_file=None, csv=True, remote=True):
    """
    function to retrieve dataset from a file location

    :param data_file: data file location
    :param label_file: label file location. default to None
    :param csv: True if data is in csv format
    :param remote: False if data is local, True if data is on google drive
    :return: a tuple (data, label)

    usage example:
        # bioresponse, redundant = get_dataset('Bioresponse')
        # print(bioresponse.shape)

        # cancer , red = get_dataset('gene expression cancer RNA-Seq/data', label_file='gene expression cancer RNA-Seq/labels')
        # print(cancer.shape)
        # print(red.shape)
    """

    data = None
    label = None
    file_extention = 'csv' if csv else 'dat'
    if remote:
        data = pd.read_csv(f'/content/gdrive/My Drive/Final Project/Code/DataSets/{data_file}.{file_extention}')
        if label_file is not None:
            label = pd.read_csv(f'/content/gdrive/My Drive/Final Project/Code/DataSets/{label_file}.{file_extention}')
    else:
        path = os.path.abspath(os.path.dirname(__file__))
        data = pd.read_csv(os.path.join(path, 'data', f'{data_file}.{file_extention}'))
        if label_file is not None:
            label = pd.read_csv(os.path.join(path, 'data', f'{label_file}.{file_extention}'))

    return data, label


# normalization
def drop_columns_with_one_value(df):
    """
    function to drop columns with only 1 value

    :param df: the dataframe
    """
    df_copy = df.copy()
    for column in df.columns:
        if len(df[column].unique()) == 1:
            df_copy.drop(column, inplace=True, axis=1)
    return df_copy


def normalize(df, method):
    """
    function to normalize the values with a given method

    :param df: dataframe to be normalized
    :param method: either 'min-max' or 'norm'
    """
    if method == 'min-max':
        return (df - df.min()) / (df.max() - df.min())
    elif method == 'norm':
        return (df - df.mean()) / df.std()
    else:
        raise NotImplementedError('Other normalization not implemented yet')


def split_data(df_data, df_label, data_name):
    """
    method to split data from a given dataframe

    :param df_data: data dataframe
    :param df_label: label dataframe
    :param data_name: dataset name
    :return: an object with train,validation,test data processed
    """

    if data_name == 'Bioresponse':
        Bioresponse = drop_columns_with_one_value(df_data)
        Bioresponse = normalize(Bioresponse, 'min-max')
        Bioresponse_taret_0 = Bioresponse.loc[Bioresponse['target'] == 0]
        Bioresponse_taret_1 = Bioresponse.loc[Bioresponse['target'] == 1]
        Bioresponse_taret_0.shape, Bioresponse_taret_1.shape

        X_train_taret_0 = Bioresponse_taret_0.iloc[0:math.trunc(Bioresponse_taret_0.shape[0] * 0.6)]
        X_train_taret_1 = Bioresponse_taret_1.iloc[0:math.trunc(Bioresponse_taret_1.shape[0] * 0.6)]
        X_train = pd.concat([X_train_taret_0, X_train_taret_1], ignore_index=False, sort=False)
        X_train.shape

        X_validation_taret_0 = Bioresponse_taret_0.iloc[math.trunc(Bioresponse_taret_0.shape[0] * 0.6):math.trunc(
            Bioresponse_taret_0.shape[0] * 0.8)]
        X_validation_taret_1 = Bioresponse_taret_1.iloc[math.trunc(Bioresponse_taret_1.shape[0] * 0.6):math.trunc(
            Bioresponse_taret_1.shape[0] * 0.8)]
        X_validation = pd.concat([X_validation_taret_0, X_validation_taret_1], ignore_index=False, sort=False)
        X_validation.shape

        X_test_taret_0 = Bioresponse_taret_0.iloc[math.trunc(Bioresponse_taret_0.shape[0] * 0.8):]
        X_test_taret_1 = Bioresponse_taret_1.iloc[math.trunc(Bioresponse_taret_1.shape[0] * 0.8):]
        X_test = pd.concat([X_test_taret_0, X_test_taret_1], ignore_index=False, sort=False)

        y_train = X_train['target']
        y_validation = X_validation['target']
        y_test = X_test['target']

        X_train = X_train.drop('target', axis=1)
        X_validation = X_validation.drop('target', axis=1)
        X_test = X_test.drop('target', axis=1)

        X_train_transpose = X_train.T
        X_validation_transpose = X_validation.T
        X_test_transpose = X_test.T

        return {
            'train': {
                'original': X_train,
                'transpose': X_train.T,
                'labels': y_train
            },
            'validation': {
                'original': X_validation,
                'transpose': X_validation.T,
                'labels': y_validation
            },
            'test': {
                'original': X_test,
                'transpose': X_test.T,
                'labels': y_test
            }
        }

    elif data_name == 'gene_expression_cancer_data':
        gene_expression_cancer_data = df_data
        gene_expression_cancer_labels = df_label
        gene_expression_cancer_data.drop('Unnamed: 0', inplace=True, axis=1)
        gene_expression_cancer_data = drop_columns_with_one_value(gene_expression_cancer_data)
        gene_expression_cancer_data = normalize(gene_expression_cancer_data, 'min-max')
        gene_expression_cancer_data['target'] = gene_expression_cancer_labels['Class']
        classDictionary = {
            "PRAD": 0,
            "LUAD": 1,
            "BRCA": 2,
            "KIRC": 3,
            "COAD": 4
        }
        gene_expression_cancer_data = gene_expression_cancer_data.applymap(
            lambda s: classDictionary.get(s) if s in classDictionary else s)

        gene_expression_cancer_data_0 = gene_expression_cancer_data.loc[gene_expression_cancer_data['target'] == 0]
        gene_expression_cancer_data_1 = gene_expression_cancer_data.loc[gene_expression_cancer_data['target'] == 1]
        gene_expression_cancer_data_2 = gene_expression_cancer_data.loc[gene_expression_cancer_data['target'] == 2]
        gene_expression_cancer_data_3 = gene_expression_cancer_data.loc[gene_expression_cancer_data['target'] == 3]
        gene_expression_cancer_data_4 = gene_expression_cancer_data.loc[gene_expression_cancer_data['target'] == 4]

        X_train_target_0 = gene_expression_cancer_data_0.iloc[
                           0:math.trunc(gene_expression_cancer_data_0.shape[0] * 0.6)]
        X_train_target_1 = gene_expression_cancer_data_1.iloc[
                           0:math.trunc(gene_expression_cancer_data_1.shape[0] * 0.6)]
        X_train_target_2 = gene_expression_cancer_data_2.iloc[
                           0:math.trunc(gene_expression_cancer_data_2.shape[0] * 0.6)]
        X_train_target_3 = gene_expression_cancer_data_3.iloc[
                           0:math.trunc(gene_expression_cancer_data_3.shape[0] * 0.6)]
        X_train_target_4 = gene_expression_cancer_data_4.iloc[
                           0:math.trunc(gene_expression_cancer_data_4.shape[0] * 0.6)]
        X_train = pd.concat([X_train_target_0, X_train_target_1, X_train_target_2, X_train_target_3, X_train_target_4],
                            ignore_index=False, sort=False)

        X_validation_target_0 = gene_expression_cancer_data_0.iloc[
                                math.trunc(gene_expression_cancer_data_0.shape[0] * 0.6):math.trunc(
                                    gene_expression_cancer_data_0.shape[0] * 0.8)]
        X_validation_target_1 = gene_expression_cancer_data_1.iloc[
                                math.trunc(gene_expression_cancer_data_1.shape[0] * 0.6):math.trunc(
                                    gene_expression_cancer_data_1.shape[0] * 0.8)]
        X_validation_target_2 = gene_expression_cancer_data_2.iloc[
                                math.trunc(gene_expression_cancer_data_2.shape[0] * 0.6):math.trunc(
                                    gene_expression_cancer_data_2.shape[0] * 0.8)]
        X_validation_target_3 = gene_expression_cancer_data_3.iloc[
                                math.trunc(gene_expression_cancer_data_3.shape[0] * 0.6):math.trunc(
                                    gene_expression_cancer_data_3.shape[0] * 0.8)]
        X_validation_target_4 = gene_expression_cancer_data_4.iloc[
                                math.trunc(gene_expression_cancer_data_4.shape[0] * 0.6):math.trunc(
                                    gene_expression_cancer_data_4.shape[0] * 0.8)]
        X_validation = pd.concat(
            [X_validation_target_0, X_validation_target_1, X_validation_target_2, X_validation_target_3,
             X_validation_target_4], ignore_index=False, sort=False)

        X_test_target_0 = gene_expression_cancer_data_0.iloc[math.trunc(gene_expression_cancer_data_0.shape[0] * 0.8):]
        X_test_target_1 = gene_expression_cancer_data_1.iloc[math.trunc(gene_expression_cancer_data_1.shape[0] * 0.8):]
        X_test_target_2 = gene_expression_cancer_data_2.iloc[math.trunc(gene_expression_cancer_data_2.shape[0] * 0.8):]
        X_test_target_3 = gene_expression_cancer_data_3.iloc[math.trunc(gene_expression_cancer_data_3.shape[0] * 0.8):]
        X_test_target_4 = gene_expression_cancer_data_4.iloc[math.trunc(gene_expression_cancer_data_4.shape[0] * 0.8):]
        X_test = pd.concat([X_test_target_0, X_test_target_1, X_test_target_2, X_test_target_3, X_test_target_4],
                           ignore_index=False, sort=False)

        y_train = X_train['target']
        y_validation = X_validation['target']
        y_test = X_test['target']

        X_train = X_train.drop('target', axis=1).reset_index().drop('index', axis=1)
        X_validation = X_validation.drop('target', axis=1).reset_index().drop('index', axis=1)
        X_test = X_test.drop('target', axis=1).reset_index().drop('index', axis=1)

        X_train_transpose = X_train.T
        X_validation_transpose = X_validation.T
        X_test_transpose = X_test.T

        return {
            'train': {
                'original': X_train,
                'transpose': X_train.T,
                'labels': y_train
            },
            'validation': {
                'original': X_validation,
                'transpose': X_validation.T,
                'labels': y_validation
            },
            'test': {
                'original': X_test,
                'transpose': X_test.T,
                'labels': y_test
            }
        }
