# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
import gc

var_col = ["var" + str(i).zfill(3) for i in range(1, 69)]
category_col = ["var016", "var020", "var047", "var053", "var066"]
threshold = 0.01


def merge_new_result_unfair(best_result_path="index_nearest_ver.csv", best_score_path="index_nearest_ver_score.csv",
                            new_result_path="top_result.csv", new_score_path="top_score.csv",
                            final_result_path="index_nearest_ver_top.csv", final_score_path="index_nearest_ver_top_score.csv"):
    best_result = pd.read_csv("./result/"+best_result_path)
    best_score = pd.read_csv("./result/"+best_score_path)
    new_result = pd.read_csv("./result/"+new_result_path)
    new_score = pd.read_csv("./result/"+new_score_path)

    change_col = set()
    change_count = 0
    for wtid in tqdm(range(1, 34)):
        best_row = best_score[best_score["wtid"] == wtid].iloc[0]
        new_row = new_score[new_score["wtid"] == wtid].iloc[0]
        for var in var_col:
            if new_row[var] > best_row[var] + threshold:
                best_result.loc[best_result["wtid"] == wtid, var] = new_result.loc[new_result["wtid"] == wtid, var]
                best_score.loc[best_score["wtid"] == wtid, var] = new_row[var]
                print("wtid:", wtid, var, new_row[var], best_row[var])
                change_col.add(var)
                change_count += 1

    print("change_col:", sorted(list(change_col)))
    print("change_count:", change_count)
    for var in category_col:
        best_result[var] = best_result[var].astype(int)
    best_result.to_csv("./result/"+final_result_path, encoding="utf8", index=False, float_format='%.2f')
    best_score.to_csv("./result/"+final_score_path, encoding="utf8", index=False, float_format='%.4f')


def merge_vertical():
    index_result = pd.read_csv("./result/index_result.csv")
    nearest_result = pd.read_csv("./result/nearest_result.csv")
    vertical_result = pd.read_csv("./result/vertical_result_merge.csv")

    index_score = pd.read_csv("./result/index_score.csv")
    nearest_score = pd.read_csv("./result/nearest_score.csv")
    vertical_score = pd.read_csv("./result/vertical_score_merge.csv")

    index_mean = index_score.mean()
    nearest_mean = nearest_score.mean()
    vertical_mean = vertical_score.mean()
    index_col = []
    nearest_col = []
    vertical_col = []
    for var in tqdm(var_col):
        max_score = max(index_mean[var], nearest_mean[var], vertical_mean[var])
        if max_score == nearest_mean[var]:
            nearest_col.append(var)
        elif max_score == vertical_mean[var]:
            vertical_col.append(var)
        else:
            index_col.append(var)
    change_count = 0
    for wtid in tqdm(range(1, 34)):
        index_row = index_score[index_score["wtid"] == wtid].iloc[0]
        nearest_row = nearest_score[nearest_score["wtid"] == wtid].iloc[0]
        vertical_row = vertical_score[vertical_score["wtid"] == wtid].iloc[0]
        for var in var_col:
            max_score = max(index_row[var], nearest_row[var], vertical_row[var])
            if max_score == nearest_row[var]:
                if var in index_col:
                    if nearest_row[var] > index_row[var] + threshold:
                        print(var, "nearest:", nearest_row[var], "index:", index_row[var])
                        change_count += 1
                        index_result.loc[index_result["wtid"] == wtid, var] = nearest_result.loc[nearest_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = nearest_row[var]
                elif var in vertical_col:
                    if nearest_row[var] > vertical_row[var] + threshold:
                        print(var, "nearest:", nearest_row[var], "vertical:", vertical_row[var])
                        change_count += 1
                        index_result.loc[index_result["wtid"] == wtid, var] = nearest_result.loc[nearest_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = nearest_row[var]
                    else:
                        index_result.loc[index_result["wtid"] == wtid, var] = vertical_result.loc[vertical_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = vertical_row[var]
                else:
                    index_result.loc[index_result["wtid"] == wtid, var] = nearest_result.loc[nearest_result["wtid"] == wtid, var]
                    index_score.loc[index_score["wtid"] == wtid, var] = nearest_row[var]
            elif max_score == vertical_row[var]:
                if var in index_col:
                    if vertical_row[var] > index_row[var] + threshold:
                        print(var, "vertical:", vertical_row[var], "index:", index_row[var])
                        change_count += 1
                        index_result.loc[index_result["wtid"] == wtid, var] = vertical_result.loc[vertical_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = vertical_row[var]
                elif var in nearest_col:
                    if vertical_row[var] > nearest_row[var] + threshold:
                        print(var, "vertical:", vertical_row[var], "nearest:", nearest_row[var])
                        change_count += 1
                        index_result.loc[index_result["wtid"] == wtid, var] = vertical_result.loc[vertical_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = vertical_row[var]
                    else:
                        index_result.loc[index_result["wtid"] == wtid, var] = nearest_result.loc[nearest_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = nearest_row[var]
                else:
                    index_result.loc[index_result["wtid"] == wtid, var] = vertical_result.loc[vertical_result["wtid"] == wtid, var]
                    index_score.loc[index_score["wtid"] == wtid, var] = vertical_row[var]
            else:
                if var in nearest_col:
                    if index_row[var] > nearest_row[var] + threshold:
                        print(var, "index:", index_row[var], "nearest:", nearest_row[var])
                        change_count += 1
                    else:
                        index_result.loc[index_result["wtid"] == wtid, var] = nearest_result.loc[nearest_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = nearest_row[var]
                elif var in vertical_col:
                    if index_row[var] > vertical_row[var] + threshold:
                        print(var, "index:", index_row[var], "vertical:", vertical_row[var])
                        change_count += 1
                    else:
                        index_result.loc[index_result["wtid"] == wtid, var] = vertical_result.loc[vertical_result["wtid"] == wtid, var]
                        index_score.loc[index_score["wtid"] == wtid, var] = vertical_row[var]

    for var in category_col:
        index_result[var] = index_result[var].astype(int)

    print("index_col:", index_col)
    print("nearest_col:", nearest_col)
    print("vertical_col:", vertical_col)
    print("change_count:", change_count)
    index_result.to_csv("./result/index_nearest_ver.csv", encoding="utf8", index=False, float_format='%.2f')
    index_score.to_csv("./result/index_nearest_ver_score.csv", encoding="utf8", index=False, float_format='%.4f')


def merge_new_result(best_result_path="index_nearest_ver.csv", best_score_path="index_nearest_ver_score.csv",
                     new_result_path="top_result.csv", new_score_path="top_score.csv",
                     final_result_path="index_nearest_ver_top.csv", final_score_path="index_nearest_ver_top_score.csv"):
    best_result = pd.read_csv("./result/" + best_result_path)
    best_score = pd.read_csv("./result/" + best_score_path)
    new_result = pd.read_csv("./result/"+new_result_path)
    new_score = pd.read_csv("./result/"+new_score_path)

    best_mean = best_score.mean()
    new_mean = new_score.mean()

    new_col = set()
    change_count = 0
    for wtid in tqdm(range(1, 34)):
        best_row = best_score[best_score["wtid"] == wtid].iloc[0]
        new_row = new_score[new_score["wtid"] == wtid].iloc[0]
        for var in var_col:
            max_score = max(best_mean[var], new_mean[var])
            if max_score == new_mean[var]:
                if best_row[var] > new_row[var] + threshold:
                    print(var, "best:", best_row[var], "new:", new_row[var])
                    change_count += 1
                else:
                    best_result.loc[best_result["wtid"] == wtid, var] = new_result.loc[new_result["wtid"] == wtid, var]
                    best_score.loc[best_score["wtid"] == wtid, var] = new_row[var]
                new_col.add(var)
            else:
                if new_row[var] > best_row[var] + threshold:
                    print(var, "new:", new_row[var], "best:", best_row[var])
                    best_result.loc[best_result["wtid"] == wtid, var] = new_result.loc[new_result["wtid"] == wtid, var]
                    best_score.loc[best_score["wtid"] == wtid, var] = new_row[var]
                    change_count += 1

    for var in category_col:
        best_result[var] = best_result[var].astype(int)

    print("new_col:", sorted(list(new_col)))
    print("change_count:", change_count)
    best_result.to_csv("./result/"+final_result_path, encoding="utf8", index=False, float_format='%.2f')
    best_score.to_csv("./result/"+final_score_path, encoding="utf8", index=False, float_format='%.4f')


def merge_horizontal():
    horizontal_result = pd.read_csv("./result/horizontal_result_merge.csv")
    horizontal_score = pd.read_csv("./result/horizontal_score_merge.csv")
    count_miss = pd.read_csv("./result/count_miss.csv")

    best_result = pd.read_csv("./result/index_nearest_ver_top.csv")
    best_score = pd.read_csv("./result/index_nearest_ver_top_score.csv")

    best_mean = best_score.mean()
    horizontal_mean = horizontal_score.mean()

    horizontal_col = set()
    change_count = 0
    for wtid in tqdm(range(1, 34)):
        best_row = best_score[best_score["wtid"] == wtid].iloc[0]
        horizontal_row = horizontal_score[horizontal_score["wtid"] == wtid].iloc[0]
        for var in var_col:
            max_score = max(best_mean[var], horizontal_mean[var])
            if max_score == horizontal_mean[var]:
                if best_row[var] > horizontal_row[var] + threshold:
                    print(var, "best:", best_row[var], "horizontal:", horizontal_row[var])
                    change_count += 1
                else:
                    best_result.loc[(count_miss["count_miss"] < 68) & (best_result["wtid"] == wtid), var] = \
                        horizontal_result.loc[(count_miss["count_miss"] < 68) & (horizontal_result["wtid"] == wtid), var]
                    best_score.loc[best_score["wtid"] == wtid, var] = horizontal_row[var]
                horizontal_col.add(var)
            else:
                if horizontal_row[var] > best_row[var] + threshold:
                    print(var, "horizontal:", horizontal_row[var], "best:", best_row[var])
                    best_result.loc[(count_miss["count_miss"] < 68) & (best_result["wtid"] == wtid), var] = \
                        horizontal_result.loc[(count_miss["count_miss"] < 68) & (horizontal_result["wtid"] == wtid), var]
                    best_score.loc[best_score["wtid"] == wtid, var] = horizontal_row[var]
                    change_count += 1

    for var in category_col:
        best_result[var] = best_result[var].astype(int)

    print("horizontal_col:", sorted(list(horizontal_col)))
    print("change_count:", change_count)
    best_result.to_csv("./result/index_nearest_ver_top_hor.csv", encoding="utf8", index=False, float_format='%.2f')
    best_score.to_csv("./result/index_nearest_ver_top_hor_score.csv", encoding="utf8", index=False, float_format='%.4f')


if __name__ == "__main__":
    # 合并纵向模型结果与mape结果
    merge_new_result_unfair(best_result_path="vertical_result.csv", best_score_path="vertical_score.csv", new_result_path="vertical_result_mape.csv", new_score_path="vertical_score_mape.csv", final_result_path="vertical_result.csv", final_score_path="vertical_score.csv")
    # 合并横向模型结果与mape结果
    merge_new_result_unfair(best_result_path="horizontal_result.csv", best_score_path="horizontal_score.csv", new_result_path="horizontal_result_mape.csv", new_score_path="horizontal_score_mape.csv", final_result_path="horizontal_result.csv", final_score_path="horizontal_score.csv")
    # # 合并横向（加纵向特征）模型结果与mape结果
    merge_new_result_unfair(best_result_path="horizontal_result_ver.csv", best_score_path="horizontal_score_ver.csv", new_result_path="horizontal_result_ver_mape.csv", new_score_path="horizontal_score_ver_mape.csv", final_result_path="horizontal_result_ver.csv", final_score_path="horizontal_score_ver.csv")
    # 合并横向模型结果与横向（加纵向特征）模型结果
    merge_new_result(best_result_path="horizontal_result.csv", best_score_path="horizontal_score.csv", new_result_path="horizontal_result_ver.csv", new_score_path="horizontal_score_ver.csv", final_result_path="horizontal_result_merge.csv", final_score_path="horizontal_score_merge.csv")
    gc.collect()

    # 合并xgb纵向模型结果
    merge_new_result(best_result_path="xgb_vertical_result_hor1.csv", best_score_path="xgb_vertical_score_hor1.csv", new_result_path="xgb_vertical_result_hor2.csv", new_score_path="xgb_vertical_score_hor2.csv", final_result_path="xgb_vertical_result_hor.csv", final_score_path="xgb_vertical_score_hor.csv")
    merge_new_result_unfair(best_result_path="xgb_vertical_result_hor.csv", best_score_path="xgb_vertical_score_hor.csv", new_result_path="xgb_vertical_result.csv", new_score_path="xgb_vertical_score.csv", final_result_path="xgb_vertical_result_merge.csv", final_score_path="xgb_vertical_score_merge.csv")
    # 合并xgb横向模型结果
    merge_new_result_unfair(best_result_path="xgb_horizontal_result_ver.csv", best_score_path="xgb_horizontal_score_ver.csv", new_result_path="xgb_horizontal_result.csv", new_score_path="xgb_horizontal_score.csv", final_result_path="xgb_horizontal_result_merge.csv", final_score_path="xgb_horizontal_score_merge.csv")
    # 合并xgb横向relate模型结果
    merge_new_result(best_result_path="horizontal_result_relate.csv", best_score_path="horizontal_score_relate.csv", new_result_path="horizontal_result_relate_mape.csv", new_score_path="horizontal_score_relate_mape.csv", final_result_path="horizontal_result_relate_merge.csv", final_score_path="horizontal_score_relate_merge.csv")
    merge_new_result(best_result_path="horizontal_result_relate_merge.csv", best_score_path="horizontal_score_relate_merge.csv", new_result_path="xgb_horizontal_result_relate.csv", new_score_path="xgb_horizontal_score_relate.csv", final_result_path="horizontal_result_relate_merge.csv", final_score_path="horizontal_score_relate_merge.csv")
    gc.collect()

    # 合并纵向模型结果与xgb纵向模型结果
    merge_new_result(best_result_path="vertical_result.csv", best_score_path="vertical_score.csv", new_result_path="xgb_vertical_result_merge.csv", new_score_path="xgb_vertical_score_merge.csv", final_result_path="vertical_result_merge.csv", final_score_path="vertical_score_merge.csv")
    # 合并横向模型结果与xgb横向模型结果
    merge_new_result(best_result_path="horizontal_result_merge.csv", best_score_path="horizontal_score_merge.csv", new_result_path="horizontal_result_relate_merge.csv", new_score_path="horizontal_score_relate_merge.csv", final_result_path="horizontal_result_merge.csv", final_score_path="horizontal_score_merge.csv")
    merge_new_result(best_result_path="horizontal_result_merge.csv", best_score_path="horizontal_score_merge.csv", new_result_path="xgb_horizontal_result_merge.csv", new_score_path="xgb_horizontal_score_merge.csv", final_result_path="horizontal_result_merge.csv", final_score_path="horizontal_score_merge.csv")
    # 合并横向模型rf结果
    merge_new_result_unfair(best_result_path="horizontal_result_merge.csv", best_score_path="horizontal_score_merge.csv", new_result_path="horizontal_result_ver_rf.csv", new_score_path="horizontal_score_ver_rf.csv", final_result_path="horizontal_result_merge.csv", final_score_path="horizontal_score_merge.csv")
    gc.collect()

    # 合并插值结果与纵向模型结果
    merge_vertical()
    # 合并top模型结果
    merge_new_result()
    # 合并纵向模型rf结果
    merge_new_result_unfair(best_result_path="index_nearest_ver_top.csv", best_score_path="index_nearest_ver_top_score.csv", new_result_path="vertical_result_rf.csv", new_score_path="vertical_score_rf.csv", final_result_path="index_nearest_ver_top.csv", final_score_path="index_nearest_ver_top_score.csv")
    # 合并横向模型结果
    merge_horizontal()
    gc.collect()
