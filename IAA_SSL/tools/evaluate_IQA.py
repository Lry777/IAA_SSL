import pandas as pd
import numpy as np
from sklearn import metrics
from scipy import stats
import copy




def ava_quality(path):
    df = pd.read_csv(path)
    new_col = np.zeros(len(df['image_id']))
    col = list(df.columns)
    if 'quality' not in col:
        df.insert(len(df.columns), 'quality', new_col)

    for i in range(len(df['image_id'])):
        row = df.iloc[i]
        # print('row',row, len(row))
        y = row[1:11].values.astype("int")
        # print('y', y)
        p = y / y.sum()
        # print(p)
        quality = quality_sum(p)
        # print(quality)
        df.at[i, 'quality'] = quality
    df.to_csv(path, index=False)


def ava_add_eval_quality(images_id, quality_eval, df):
    new_col = np.zeros(len(df['image_id']))
    col = list(df.columns)
    if 'quality_eval' not in col:
        df.insert(len(df.columns), 'quality_eval', new_col)

    for i in range(len(df['image_id'])):
        if df['image_id'][i] == images_id:
            df.at[i, 'quality_eval'] = quality_eval.cpu().numpy()
            break

    return df


def quality_sum(data):
    sum = 0
    for i, j in enumerate(data):
        sum = sum + (i+1)*j
    return sum

def LCC_ps(y_true, y_pre):
    # print(y_true.shape, y_pre.shape)
    return stats.pearsonr(y_true, y_pre)

def SRCC_sm(y_true, y_pre):
    return stats.spearmanr(y_true, y_pre)


def compute_emd(y_true, y_pre):
    cdf_diff = y_pre - y_true
    # samplewise_emd = np.sqrt(np.mean(pow(abs(cdf_diff), 1)))
    samplewise_emd = np.mean(pow(abs(cdf_diff), 1))
    return samplewise_emd.mean()

def MAE(y_true, y_pre):
    return metrics.mean_absolute_error(y_true, y_pre)

def RMSE(y_true, y_pre):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pre))

def cl_trans(l, cut_off=5):
    for i, n in enumerate(l):
        if n >= cut_off:
            l[i] = 1
        else:
            l[i] = 0
    return l
def IAA_ACC(y_true, y_pre, cut_off=5):
    y_true = copy.deepcopy(y_true)
    y_pre = copy.deepcopy(y_pre)
    p_label = cl_trans(y_true, cut_off=cut_off)
    p_pres = cl_trans(y_pre, cut_off=cut_off)
    return metrics.accuracy_score(p_label, p_pres)