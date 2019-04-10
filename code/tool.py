# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score


def regression_score(y_true, y_pred):
    return np.mean(np.exp(-100 * np.abs(y_true - y_pred) / (np.max([np.abs(y_true), [1e-15] * len(y_true)], axis=0))))


# lgb eval_metric
def lgb_metric(y_pred, train_data):
    y_true = train_data.get_label()
    loss = np.mean(np.exp(-100 * np.abs(y_true - y_pred) / (np.max([np.abs(y_true), [1e-15] * len(y_true)], axis=0))))
    return "regression_loss", loss, True


# eval_metric
def xgb_metric(y_pred, train_data):
    y_true = train_data.get_label()
    loss = np.mean(np.exp(-100 * np.abs(y_true - y_pred) / (np.max([np.abs(y_true), [1e-15] * len(y_true)], axis=0))))
    return "regression_loss", loss


def label_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    print(df.dtypes)
    return df


types = [['var001', 'var004', 'var005', 'var038', 'var042', 'var046', 'var050', 'var060'],
         ['var002', 'var003', 'var006', 'var007', 'var011', 'var012', 'var014',
          'var018', 'var021', 'var022', 'var024', 'var028', 'var029', 'var030',
          'var033', 'var035', 'var036', 'var037', 'var040', 'var045', 'var051',
          'var052', 'var055', 'var057', 'var062', 'var067'],
         ['var008', 'var009', 'var010', 'var017', 'var019', 'var023', 'var025',
          'var026', 'var039', 'var044', 'var048', 'var059', 'var063', 'var064',
          'var065'],
         ['var013', 'var015', 'var016', 'var020', 'var027', 'var031', 'var032',
          'var034', 'var041', 'var043', 'var047', 'var049', 'var053', 'var054',
          'var056', 'var058', 'var061', 'var066', 'var068']
         ]
