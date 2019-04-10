# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
import tool
import datetime
import warnings


warnings.filterwarnings('ignore')
data_path = './data/train_test.h5'
var_col = ["var" + str(i).zfill(3) for i in range(1, 69)]
category_col = ["var016", "var020", "var047", "var053", "var066"]
head_col = ["ts", "wtid"]
head_col.extend(var_col)
threshold = 0.01


# 不进行数据处理的情况下，进行插值
def interpolate_predict_base(method="index"):
    def get_data(i):
        temp_data = pd.read_csv('./data/dataset/' + str(i).zfill(3) + '/201807.csv', parse_dates=[0])
        global res
        temp_res = res[res['wtid'] == i]
        temp_res['flag'] = 1
        temp_data = temp_res.merge(temp_data, on=['ts', 'wtid'], how='outer')
        temp_data = temp_data.sort_values(['ts', 'wtid']).reset_index(drop=True)
        return temp_data

    res = pd.read_csv('./data/template_submit_result.csv', parse_dates=[0])[['ts', 'wtid']]
    df = pd.DataFrame()
    for i in tqdm(range(1, 34)):
        data = get_data(i)
        fe = [i for i in data.columns if 'var' in i]
        for j in fe:
            data[j] = data[j].interpolate(method=method)  # nearest index
        df = pd.concat([df, data], axis=0)

    sub = df[df.flag == 1].copy().reset_index(drop=True)
    del sub['flag']
    # 变为类别
    for var in category_col:
        sub[var] = sub[var].astype(int)

    sub.to_csv('./result/{}_result.csv'.format(method), index=False, float_format='%.2f')


# 插值预测
def interpolate_predict(method="index"):
    start = datetime.datetime.now()
    data = pd.read_hdf(data_path)

    final_result = pd.DataFrame()
    score_df = pd.DataFrame()
    score_df["var"] = var_col
    for i in tqdm(range(1, 34)):
        sub = data[data["wtid"] == i]
        score_temp = []
        for var in var_col:
            sub1 = sub[pd.notna(sub[var])].reset_index(drop=True)
            index = 0
            for index, t in enumerate(tool.types):
                if var in t:
                    break
            col_name = str(index) + "_test"
            sub2 = sub1[[var]].copy()
            sub1.loc[sub1[col_name] == 1, var] = np.nan
            sub1[var] = sub1[var].interpolate(method=method)

            true_value = sub2[sub1[col_name] == 1][var]
            predict_value = sub1[sub1[col_name] == 1][var]
            if_round = False
            if var in category_col:
                predict_value = np.array(predict_value).astype(int)
                true_value = np.array(true_value).astype(int)
                score = tool.label_score(true_value, predict_value)
            else:
                score = tool.regression_score(true_value, predict_value)
                predict_value2 = np.round(predict_value, 2)
                score2 = tool.regression_score(true_value, predict_value2)
                if score < score2 - threshold:
                    score = score2
                    if_round = 2
                predict_value2 = np.round(predict_value, 1)
                score2 = tool.regression_score(true_value, predict_value2)
                if score < score2 - threshold:
                    score = score2
                    if_round = 1
            score_temp.append(score)

            # 预测结果
            sub[var] = sub[var].interpolate(method=method)
            if if_round:
                sub[var] = np.round(sub[var], if_round)

        final_result = pd.concat((final_result, sub), axis=0, ignore_index=True)
        score_df[str(i)] = score_temp

    score_df.set_index("var", inplace=True)
    score_df = score_df.T
    score_df.reset_index(inplace=True)
    score_df.rename(columns={"index": "wtid"}, inplace=True)
    score_df.to_csv("./result/{}_score.csv".format(method), encoding="utf8", index=False, float_format='%.4f')

    final_result = final_result[final_result["count_miss"] > 0]
    final_result = final_result[head_col]
    final_result.sort_values(["wtid", "ts"], inplace=True)
    for var in category_col:
        final_result[var] = final_result[var].astype(int)
    final_result.to_csv("./result/{}_result.csv".format(method), encoding="utf8", index=False, float_format='%.2f')
    end = datetime.datetime.now()
    print("finish", method, "interpolate_predict time: ", end - start)


# 取top结果预测
def top_predict():
    data = pd.read_hdf(data_path)

    score_df = pd.DataFrame()
    score_df["var"] = [i for i in var_col]
    final_result = pd.DataFrame()
    start = datetime.datetime.now()

    for wtid in tqdm(range(1, 34)):
        use_data = data[data["wtid"] == wtid]
        test_scores = []

        for var in var_col:
            train_data = use_data[pd.notna(use_data[var])]
            predict_data = use_data[pd.isna(use_data[var])]

            index = 0
            for index, t in enumerate(tool.types):
                if var in t:
                    break
            test_label_col = str(index) + "_test"

            train_feature = train_data[train_data[test_label_col] == 0]
            top_values = train_feature[var].value_counts().index
            test_feature = train_data[train_data[test_label_col] == 1]
            test_y = np.array(test_feature[var])

            # 用出现次数最多的数值
            test_pred = np.array([top_values[0]] * len(test_y))
            predict_y = np.array([top_values[0]] * len(predict_data))
            if var in category_col:
                test_score = tool.label_score(test_y, test_pred)
            else:
                test_score = tool.regression_score(test_y, test_pred)

            # 检验第二多的数值
            if test_score > 0.1 and len(top_values) > 1:
                test_pred2 = [top_values[1]] * len(test_y)
                if var in category_col:
                    test_score2 = tool.label_score(test_y, test_pred2)
                else:
                    test_score2 = tool.regression_score(test_y, test_pred2)
                if test_score2 > test_score:
                    test_score = test_score2
                    predict_y = np.array([top_values[1]] * len(predict_data))

            test_scores.append(test_score)
            use_data.loc[predict_data.index, var] = predict_y

        score_df[str(wtid)] = test_scores
        final_result = pd.concat((final_result, use_data[use_data["count_miss"] > 0]), axis=0, ignore_index=True)

    final_result = final_result[head_col]
    final_result.sort_values(["wtid", "ts"], inplace=True)
    final_result.to_csv("./result/top_result.csv", encoding="utf8", index=False, float_format='%.2f')

    score_df.set_index("var", inplace=True)
    score_df = score_df.T
    score_df.reset_index(inplace=True)
    score_df.rename(columns={"index": "wtid"}, inplace=True)
    score_df.to_csv("./result/top_score.csv", encoding="utf8", index=False, float_format='%.4f')
    end = datetime.datetime.now()
    print("finish top_predict time: ", end - start, "\n")


if __name__ == "__main__":
    interpolate_predict("index")
    interpolate_predict("nearest")
    top_predict()
