# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import pickle
import gc
path = './data'
var_col = ["var" + str(i).zfill(3) for i in range(1, 69)]
category_col = ["var016", "var020", "var047", "var053", "var066"]
numeric_col = [i for i in var_col if i not in category_col]
data_path = path + '/train.h5'


def merge_data():
    start = datetime.datetime.now()
    sub = pd.read_csv(path + '/template_submit_result.csv')
    sub.drop_duplicates()
    train = pd.DataFrame()
    for i in range(1, 34):
        file = path + '/dataset/' + str(i).zfill(3) + '/201807.csv'
        print(train.shape, file)
        temp_train = pd.read_csv(file)
        train = pd.concat([train, temp_train], ignore_index=True)

    # 合并sub到train中
    train = pd.merge(train, sub[["ts", "wtid"]], "outer", on=["ts", "wtid"])
    train.sort_values(by=["wtid", "ts"], inplace=True)
    train.reset_index(drop=True, inplace=True)

    # 处理异常数据
    train = train[((train["var008"] >= -4) & (train["var008"] <= 5)) | pd.isna(train["var008"])]
    train = train[(train["var009"] == 0) | (train["var009"] >= 20) | pd.isna(train["var009"])]
    train = train[(train["var017"] == 0) | (train["var017"] >= 20) | pd.isna(train["var017"])]
    train = train[((train["var025"] >= -5) & (train["var025"] <= 5)) | pd.isna(train["var025"])]
    train = train[((train["var026"] >= -5) & (train["var026"] <= 5)) | pd.isna(train["var026"])]
    train = train[(train["var028"] >= 0) | pd.isna(train["var028"])]
    train = train[((train["var054"] >= -25) & (train["var054"] <= 0)) | pd.isna(train["var054"])]
    train = train[(train["var068"] >= -1) | pd.isna(train["var068"])]

    train["count_miss"] = np.sum(pd.isna(train), axis=1)
    train.reset_index(drop=True, inplace=True)
    # reduce_mem_usage会降低精度，在内存允许的情况下，不使用
    # import tool
    # train = tool.reduce_mem_usage(train)

    # 存储F5文件
    train.to_hdf(path + '/train.h5', 'train', mode="w")
    end = datetime.datetime.now()
    print("merge data time:", end - start)


def construct_test_label():
    start = datetime.datetime.now()
    data = pd.read_hdf(data_path)
    # 按十秒聚合
    data["time"] = data["ts"].apply(lambda x: x[:18] + "0")
    data["time"] = pd.to_datetime(data["time"])
    # 数值型的用均值
    group_data1 = data.groupby(["wtid", "time"], as_index=False)[numeric_col].agg("mean")
    temp = ["wtid", "time"]
    temp.extend(category_col)
    # 类别型的直接取第一个值
    group_data2 = data[temp].drop_duplicates(["wtid", "time"], keep="first", inplace=False)
    group_data = pd.merge(group_data1, group_data2, on=["wtid", "time"])
    del group_data1, group_data2
    gc.collect()

    for index, var in enumerate(["var001", "var002", "var008", "var013"]):
        print("============", var, "===========")
        use_data = group_data[["time", "wtid", var]]
        use_col = ["time", var]
        sub_data = use_data[use_data["wtid"] == 1][use_col]
        sub_data.rename(columns={var: "1" + var}, inplace=True)
        for wtid in range(2, 34):
            temp = use_data[use_data["wtid"] == wtid][use_col]
            temp.rename(columns={var: str(wtid) + var}, inplace=True)
            sub_data = pd.merge(sub_data, temp, "outer", on=["time"])
        sub_data = sub_data.sort_values(by=["time"]).reset_index(drop=True)

        sub_data["day"] = sub_data["time"].map(lambda x: x.day)

        time_df = pd.DataFrame()
        wtid = 1
        for day in range(1, 32):
            temp = sub_data[sub_data["day"] == day]
            # wtid 从1往34循环，选择缺失值多于200的
            while len(temp[pd.isna(temp[str(wtid) + var])]) < 200:
                wtid += 1
                if wtid > 33:
                    wtid = 1
            temp = temp[pd.isna(temp[str(wtid) + var])]
            time_df = pd.concat((time_df, temp[["time"]]), axis=0, ignore_index=True)
        col_name = str(index)+"_test"
        time_df[col_name] = 1
        print(index, len(time_df))
        data = pd.merge(data, time_df, "left", on=["time"])
        data.loc[pd.isna(data[col_name]), col_name] = 0
        del use_data, sub_data
        gc.collect()

    data.to_hdf(path + "/train_test.h5", "train", mode="w")
    end = datetime.datetime.now()
    print("construct test label time:", end - start)


def construct_count_miss():
    # 用于拼接横向模型的结果
    data = pd.read_hdf(data_path)
    count_miss = data[data["count_miss"] > 0][["ts", "wtid", "count_miss"]]
    count_miss.sort_values(["wtid", "ts"], inplace=True)
    count_miss.to_csv("./result/count_miss.csv", index=False)
    print("finish construct count_miss.csv")


def group_data_10_second():
    data = pd.read_hdf(path + '/train_test.h5')
    data["time"] = data["ts"].apply(lambda x: x[:18] + "0")
    data["time"] = pd.to_datetime(data["time"])
    # 数值型的用均值
    group_data1 = data.groupby(["wtid", "time"], as_index=False)[numeric_col].agg("mean")
    temp = ["wtid", "time"]
    temp.extend(category_col)
    # 类别型的直接取第一个值
    group_data2 = data[temp].drop_duplicates(["wtid", "time"], keep="first", inplace=False)
    group_data = pd.merge(group_data1, group_data2, on=["wtid", "time"])
    # 对var排序
    temp = ["wtid", "time"]
    temp.extend(var_col)
    group_data = group_data[temp]
    del group_data1
    del group_data2

    start = datetime.datetime.now()
    sub_data = group_data[group_data["wtid"] == 1].drop(["wtid"], axis=1)
    change_col = {var: "1" + var for var in var_col}
    sub_data.rename(columns=change_col, inplace=True)
    for wtid in range(2, 34):
        temp = group_data[group_data["wtid"] == wtid].drop(["wtid"], axis=1)
        change_col = {var: str(wtid) + var for var in var_col}
        temp.rename(columns=change_col, inplace=True)
        sub_data = pd.merge(sub_data, temp, "outer", on=["time"])
    sub_data = sub_data.sort_values(by=["time"]).reset_index(drop=True)

    sub_data["day"] = sub_data["time"].map(lambda x: x.day)
    sub_data["hour"] = sub_data["time"].map(lambda x: x.hour)
    sub_data["minute"] = sub_data["time"].map(lambda x: x.minute)
    sub_data["weekday"] = sub_data["time"].map(lambda x: x.weekday())

    sub_data.to_hdf(path + '/group_data.h5', 'group_data', mode="w")
    end = datetime.datetime.now()
    print("group data use time:", end - start)


def compute_corr():
    start = datetime.datetime.now()
    sub_data = pd.read_hdf(path + '/group_data.h5')
    print(sub_data.shape)
    corr = sub_data.corr()
    corr.to_hdf(path + '/group_data_corr.h5', 'group_data_corr', mode="w")
    end = datetime.datetime.now()
    print("compute corr use time:", end - start)


def compute_dict(num1=25, num2=2, num3=40):
    def select_col(df_corr, wtid, var, dictory):
        target_value = str(wtid) + var
        # 同一个变量不同风场，保存前15个
        sam_var_column = [str(i) + var for i in range(1, 34)]
        sam_var_column.remove(target_value)

        target_related_var_list = df_corr[target_value]
        own_list = target_related_var_list[sam_var_column].sort_values(ascending=False).dropna()[0:num1]  # 自己这一部分的变量
        target_related_var_list = target_related_var_list.drop(index=own_list.index)

        # 其它风场相关性高的每个取两个
        wt_list = list(range(1, 34))
        # 不选择和自己同一风场的变量
        wt_list.remove(wtid)

        cols_list = []
        for wt in wt_list:
            wtid_column = [str(wt) + 'var' + str(i).zfill(3) for i in range(1, 69)]
            wtid_be_select = str(wt) + var
            # 取得时候不能包括之前被选中的15个变量
            if wtid_be_select in own_list.index.tolist():
                wtid_column.remove(wtid_be_select)
            wtid_list = target_related_var_list[wtid_column].sort_values(ascending=False).dropna()[0:num2]
            target_related_var_list = target_related_var_list.drop(index=wtid_column)
            cols_list.append(wtid_list)

        concat_col = pd.concat(cols_list)
        concat_index = concat_col.sort_values(ascending=False)[0:num3].index.tolist()
        concat_index.extend(own_list.index.tolist())
        dictory[target_value] = concat_index

    corr = pd.read_hdf('./data/group_data_corr.h5')
    columns = ['var' + str(i).zfill(3) for i in range(1, 69)]
    feature_dict = dict()
    for wtid in range(1, 34):
        for var in columns:
            select_col(corr, wtid, var, feature_dict)
    pickle.dump(feature_dict, open("./data/feature_dict_{}_{}_{}.pkl".format(num1, num2, num3), "wb"))
    print("finish feature_dict_{}_{}_{}.pkl".format(num1, num2, num3))


def compute_relate():
    def select_col(df_corr, wtid, var, dictory):
        target_value = str(wtid) + var
        sam_var_column = [str(i) + var for i in range(1, 34)]
        sam_var_column.remove(target_value)

        target_related_var_list = df_corr[target_value]
        own_list = target_related_var_list[sam_var_column].sort_values(ascending=False).dropna()  # 自己这一部分的变量
        dictory[target_value] = own_list.index.tolist()

    corr = pd.read_hdf('./data/group_data_corr.h5')
    columns = ['var' + str(i).zfill(3) for i in range(1, 69)]
    feature_dict = dict()
    for wtid in range(1, 34):
        for var in columns:
            select_col(corr, wtid, var, feature_dict)
    pickle.dump(feature_dict, open("./data/feature_relate_dict.pkl", "wb"))
    print("finish feature_relate_dict.pkl")


if __name__ == "__main__":
    # 合并数据
    merge_data()

    # 构造验证集标签
    construct_test_label()

    # 保存count_miss，用于横向模型的结果合并
    construct_count_miss()

    # 按10秒聚合数据
    group_data_10_second()

    # 计算特征相关性
    compute_corr()

    # 计算各个风场各个变量相关性最高的一组变量
    compute_dict(25, 2, 40)
    compute_dict(25, 3, 60)

    # 计算各个风场各个变量相关性最高的一个变量
    compute_relate()
