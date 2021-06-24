#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import pickle
import sys
import logging
import dask.dataframe as dd
import multiprocessing

logger = logging.getLogger("preprocess_logger")

# daskのオプション
scheduler = "threads"

def median_shift(df_ts, dict_config):
    '''
    各事故各軸において、
    1. 加速度の中央値を計算
    2. 加速度列の値を「加速度-中央値」に変換

    Parameters
    ----------
    df_ts : pandas.DataFrame
        加速度時系列データ
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_mshifted : pandas.DataFrame
        加速度補正済時系列データ
    '''

    # 各事故各軸の加速度中央値を事故IDでgroupbyしながら計算
    df_sum_med = df_ts.groupby("id").agg(["median"])

    logger.info("")
    logger.info("【各事故各軸の加速度中央値計算後,時系列データをIDでgroupbyしたデータフレームの冒頭】")
    logger.info(df_sum_med.head())

    # pandas.mergeで、インデックス階層の数が異なるデータフレーム同士を結合すると警告が出る
    # --> df_sum_medのインデックス階層を１つに減らす
    dict_sum_median=dict()
    if "acceleration_x" in dict_config["input"]["time_series"]["column_types"]:
        dict_sum_median["med_acc_x"]=df_sum_med[("acceleration_x","median")]
    if "acceleration_y" in dict_config["input"]["time_series"]["column_types"]:
        dict_sum_median["med_acc_y"]=df_sum_med[("acceleration_y","median")]
    if "acceleration_z" in dict_config["input"]["time_series"]["column_types"]:
        dict_sum_median["med_acc_z"]=df_sum_med[("acceleration_z","median")]
    df_sum_median = pd.DataFrame(dict_sum_median)

    logger.info("【時系列データに追加する中央値データフレーム冒頭】")
    logger.info(df_sum_median.head())

    # 時系列データに中央値を追加
    df_med_added = pd.merge(df_ts, df_sum_median,
                    left_on="id", right_index=True, how="inner")
    df_med_added.reset_index(drop=True, inplace=True)

    # もとのデータフレームの列名リスト保持
    cols_df_ts = df_ts.columns

    # もとのデータフレーム削除
    del df_ts
    logger.info("【中央値列追加後データフレーム冒頭】")
    logger.info(df_med_added.head())

    # 並列処理のためdask.dataframeへ変換
    fac_cpu = dict_config["computation"]["factor_cpu_count"]
    df_med_added = dd.from_pandas(df_med_added, npartitions=int(fac_cpu*multiprocessing.cpu_count()))

    # 各軸加速度の中央値を0とするシフト補正実行して上書き
    if "acceleration_x" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("x軸加速度の中央値補正値計算中")
        df_med_added["acceleration_x"] \
            = df_med_added.apply(lambda x: x["acceleration_x"]-x["med_acc_x"], \
                axis=1, meta=("float")).compute(scheduler=scheduler)

    if "acceleration_y" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("y軸加速度の中央値補正値計算中")
        df_med_added["acceleration_y"] \
            = df_med_added.apply(lambda x: x["acceleration_y"]-x["med_acc_y"], \
                axis=1, meta=("float")).compute(scheduler=scheduler)

    if "acceleration_z" in dict_config["input"]["time_series"]["column_types"]:
        logger.info("z軸加速度の中央値補正値計算中")
        df_med_added["acceleration_z"] \
            = df_med_added.apply(lambda x: x["acceleration_z"]-x["med_acc_z"], \
                axis=1, meta=("float")).compute(scheduler=scheduler)

    # 中央値列はreturnしない
    logger.info("加速度中央値列削除中")
    df_mshifted = df_med_added[cols_df_ts].compute(scheduler=scheduler)

    logger.info("中央値補正後レコード数：%g"%len(df_mshifted))

    del df_med_added

    return df_mshifted


def normalize_with_rscaler(df_ts_train, df_ts_test, dict_config):
    '''
    sk-learnのrobustscalerを用いて加速度の値の標準化を行う
    - 各軸で全事故一括して（ただし、学習用、検証用で分けて）行う
    - 学習用データのみを使って決めた統計量で学習用データと検証用データを標準化する

    Parameters
    ----------
    df_ts_train : pandas.DataFrame
        学習用加速度時系列データ
    df_ts_test : pandas.DataFrame
        検証用加速度時系列データ
    dict_config : dict
        configファイル設定値

    Returns
    -------
    df_norm_train : pandas.DataFrame
        標準化後学習用加速度時系列データ
    df_norm_test : pandas.DataFrame
        標準化後学習用加速度時系列データ
    
    Outputs
    -------
    robustscaler.pkl : sklearn.preprocessing.RobustScaler
        学習データで作成した加速度標準化のクラス
    '''

    logger.info("各軸全事故一括のsk-learn robustscalerによる標準化")

    # 加速度のみのndarray作成
    cols_mat = []
    if "acceleration_x" in dict_config["input"]["time_series"]["column_types"]:
        cols_mat.append("acceleration_x")
    if "acceleration_y" in dict_config["input"]["time_series"]["column_types"]:
        cols_mat.append("acceleration_y")
    if "acceleration_z" in dict_config["input"]["time_series"]["column_types"]:
        cols_mat.append("acceleration_z")
    
    logger.info("標準化対象列:")
    logger.info(cols_mat)

    mat_train = df_ts_train[cols_mat].values
    mat_test = df_ts_test[cols_mat].values

    # 標準化
    ## 学習データを用いた標準化クラス(transformer)作成
    rscaler = preprocessing.RobustScaler()
    transformer = rscaler.fit(mat_train)

    ## 学習、検証用データの標準化
    mat_scaled_train = transformer.transform(mat_train)
    mat_scaled_test = transformer.transform(mat_test)

    # 標準化済みndarrayをデータフレーム化
    df_scaled_train = pd.DataFrame(mat_scaled_train)
    df_scaled_train.columns = cols_mat
    df_scaled_test = pd.DataFrame(mat_scaled_test)
    df_scaled_test.columns = cols_mat

    # 加速度列以外の列名リスト
    cols_except_acc = []
    for x in df_ts_train.columns:
        if x not in cols_mat:
            cols_except_acc.append(x)
    logger.info("標準化対象外列：")
    logger.info(cols_except_acc)

    # 加速度以外の列との結合
    df_norm_train = pd.concat([df_ts_train[cols_except_acc], df_scaled_train], axis=1)
    df_norm_test = pd.concat([df_ts_test[cols_except_acc], df_scaled_test], axis=1)

    logger.info("標準化後学習用データレコード数：%g"%len(df_norm_train))
    logger.info("標準化後検証用データレコード数：%g"%len(df_norm_test))

    # もとの時系列データ削除
    del df_ts_train, df_scaled_train, df_ts_test, df_scaled_test

    # 加速度値標準化クラスのpickle出力
    outpath = dict_config["output"]["folder_path"]["train_data"]\
        +"/robustscaler.pkl"
    pickle.dump(transformer, open(outpath, "wb"))
    logger.info("標準化クラス出力完了：%s"%outpath)

    return df_norm_train, df_norm_test


def regularize(df_ts_train, df_ts_test, dict_config):
    '''
    加速度値の正規化を行う。
    以下の1,2を選択的に実行。
    - 1のみ行う
    - 2のみ行う
    - 1⇒2の順で行う
    - 2⇒1の順で行う
    - 両方行わない
    
    1. 標準化
        - sk-learnのrobustScalerを使う
        - 軸ごとに全事故一括で行う
    2. 各軸各事故において中央値を0とするシフト補正
        
    Parameters
    ----------
    df_ts_train : pandas.DataFrame
        学習用加速度時系列データ
    df_ts_test : pandas.DataFrame
        検証用加速度時系列データ
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_reg_train : pandas.DataFrame
        正規化済み学習用加速度時系列データ
    df_reg_test : pandas.DataFrame
        正規化済み検証用加速度時系列データ
    '''

    if dict_config["regularization"]["type"]=="rscaler":
        logger.info("\n 標準化のみ実行（中央値補正は行わない） \n")
        df_reg_train, df_reg_test \
            = normalize_with_rscaler(df_ts_train, df_ts_test, dict_config)
        del df_ts_train, df_ts_test

    elif dict_config["regularization"]["type"]=="mshift":
        logger.info("\n 中央値補正のみ実行（標準化は行わない）\n")
        df_reg_train = median_shift(df_ts_train, dict_config)
        df_reg_test = median_shift(df_ts_test, dict_config)
        del df_ts_train, df_ts_test

    elif dict_config["regularization"]["type"]=="rscaler_to_mshift":
        logger.info("\n 標準化->中央値補正の順で実行 \n")
        logger.info("標準化開始 \n")
        df_norm_train, df_norm_test \
            = normalize_with_rscaler(df_ts_train, df_ts_test, dict_config)
        del df_ts_train, df_ts_test

        logger.info("\n 学習用データ中央値補正開始 \n")
        df_reg_train = median_shift(df_norm_train, dict_config) 
        logger.info("\n 検証用データ中央値補正開始 \n")
        df_reg_test = median_shift(df_norm_test, dict_config)
        del df_norm_train, df_norm_test

    elif dict_config["regularization"]["type"]=="mshift_to_rscaler":
        logger.info("\n 中央値補正->標準化の順で実行")
        logger.info("学習用データ中央値補正開始 \n")
        df_mshift_train = median_shift(df_ts_train, dict_config)
        logger.info("\n 検証用データ中央値補正開始 \n")
        df_mshift_test = median_shift(df_ts_test, dict_config)
        del df_ts_train, df_ts_test

        logger.info("\n 標準化開始 \n")
        df_reg_train, df_reg_test \
            = normalize_with_rscaler(df_mshift_train, df_mshift_test, dict_config)
        del df_mshift_train, df_mshift_test

    elif dict_config["regularization"]["type"]=="skip":
        logger.info("\n 正規化を行わない \n")
        return df_ts_train, df_ts_test
    else:
        logger.info("config.jsonのregularization/typeの値が正しく設定されていません")
        sys.exit()

    return df_reg_train, df_reg_test