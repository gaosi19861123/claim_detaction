#!/usr/bin/env python
# -*- conding: utf-8 -*-

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("preprocess_logger")

def rm_outliers(df_ts, dict_config):
    '''
    異常値処理
    - １つの軸の加速度の90%分位-10%分位がconfig.jsonでの閾値より大きければ、該当の事故を除外
    
    Parameters
    ----------
    df_ts : pandas.DataFrame
        加速度時系列データ
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_ok : pandas.DataFrame
        異常値処理済み加速度時系列データ

    df_summary : pandas.DataFrame
        各事故の加速度分位数、フラグ
    '''

    if len(df_ts)==0:
        return pd.DataFrame(), pd.DataFrame()

    # 時系列データをIDでgroupbyしたときに加速度のパーセンタイルを求める関数
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    # 時系列データをIDでgroupbyしたときに加速度の分位数を求める関数
    def diff_90_10():
        def diff_90_10_(x):
            return np.percentile(x,90)-np.percentile(x,10)
        diff_90_10_.__name__ = "percentile_90-10"
        return diff_90_10_

    # groupbyしながら分位数計算
    df_ptile = df_ts.groupby("id").agg([percentile(10), percentile(90), diff_90_10()])

    # 異常値フラグ付与
    logger.info("config.jsonで設定された90パーセント分位-10パーセント分位の閾値")
    if "acceleration_x" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("lim_dx_90_10= %g"%dict_config["data_condition"]["lim_dx_90_10"])
    if "acceleration_y" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("lim_dy_90_10= %g"%dict_config["data_condition"]["lim_dy_90_10"])
    if "acceleration_z" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("lim_dz_90_10= %g"%dict_config["data_condition"]["lim_dz_90_10"])
    logger.info("")

    # フラグ付与関数
    def flag_outlier(x):
        cond = (("acceleration_x" in dict_config["input"]["time_series"]["column_types"] \
                and x["acceleration_x"]["percentile_90-10"] > dict_config["data_condition"]["lim_dx_90_10"]) \
                or ("acceleration_y" in dict_config["input"]["time_series"]["column_types"] \
                    and x["acceleration_y"]["percentile_90-10"] > dict_config["data_condition"]["lim_dy_90_10"]) \
                or ("acceleration_z" in dict_config["input"]["time_series"]["column_types"] \
                    and x["acceleration_z"]["percentile_90-10"] > dict_config["data_condition"]["lim_dz_90_10"]))
        if cond:
            return "huge_variation"
        else:
            return "OK"

    df_summary = df_ptile.copy()
    del df_ptile
    df_summary["outlier_flag"] = df_summary.apply(flag_outlier, axis=1)

    logger.info("フラグ別事故件数")
    logger.info(df_summary["outlier_flag"].value_counts())
    logger.info("")
    logger.info("OK: 異常なし")
    logger.info("huge_variation: いずれかの軸で加速度の90パーセント分位-10パーセント分位が閾値より大きい")
    logger.info("")
    logger.info("【各事故ID、各軸での加速度分位数、異常値フラグのデータフレーム冒頭】")
    logger.info(df_summary.head())
    logger.info("")

    # pandas.mergeで、インデックス階層の数が異なるデータフレーム同士を結合すると警告が出る
    # --> df_summaryのインデックス階層を１つに減らす
    df_flag = pd.DataFrame({"outlier_flag": df_summary["outlier_flag"]})
    logger.info("【時系列データとの結合に使うフラグのデータフレーム冒頭】")
    logger.info(df_flag.head())

    logger.info("huge_variationフラグのついた事故の除外実行")
    ## mergeを使う方法
    '''
    df_ok = pd.merge(df_ts, df_flag[df_flag["outlier_flag"]=="OK"],
                    left_on="id", right_index=True, how="inner")[df_ts.columns]
    '''

    ## mergeを使わない方法
    list_id = df_flag[df_flag["outlier_flag"]=="OK"].index
    df_ok = df_ts[df_ts["id"].apply(lambda x: x in list_id)]

    ## インデックス振り直し
    df_ok.reset_index(drop=True, inplace=True)

    # 結合前データ削除
    del df_ts

    logger.info("除外後レコード数：%g"%len(df_ok))
    logger.info("除外後事故IDユニーク数：")
    logger.info(df_ok["id"].nunique())
    logger.info("")

    return df_ok, df_summary