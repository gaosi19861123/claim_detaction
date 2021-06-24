#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("preprocess_logger")

def split_data(df_ts, df_sum, dict_config):
    '''
    データ分割
    - 学習データと検証データに分割
    - 事故ラベルをサマリーファイルデータから取得

    Parameters
    ----------
    df_ts: pandas.DataFrame
        加速度時系列データ
    df_sum: pandas.DataFrame
        サマリーデータ
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_ts_train : pandas.DataFrame
        学習用時系列データ
    df_ts_test : pandas.DataFrame
        検証用時系列データ
    df_sum_train : pandas.DataFrame
        学習用サマリーデータ
    df_sum_test : pandas.DataFrame
        検証用サマリーデータ
    '''

    # 時系列データをIDがサマリーデータにあるものに限定
    # （通常はすべてサマリーデータにあるはずなので、ここでレコード数は減らないはず）
    if len(df_sum)>0:
        df_ts_join = pd.merge(df_ts, df_sum, on="id", how="inner")
        logger.info("時系列データをIDがサマリーデータにあるものに限定")
        logger.info("限定前レコード数：%g"%len(df_ts))
        logger.info("限定後レコード数：%g"%len(df_ts_join))
        logger.info("")

        # 結合後のデータフレームに対応するサマリー情報のデータフレーム作成
        df_sum_join = df_ts_join[df_sum.columns].drop_duplicates()
        
    else:
        logger.info("サマリーデータがないので、時系列データとサマリーデータの結合は行わない\n")
        df_ts_join=df_ts.copy()

        # 結合後のデータフレームに対応するサマリー情報(事故IDのみ)のデータフレーム作成
        df_sum_join = df_ts_join[["id"]].drop_duplicates()

    # インデックス再設定
    df_sum_join.reset_index(drop=True, inplace=True)

    # もとの時系列データの削除
    del df_ts
    
    # 学習、検証データ分割
    ## サマリーデータフレームの分割
    if dict_config["data_split"]["test_size"]==0.0:
        df_sum_train = df_sum_join.copy()
        df_sum_test = pd.DataFrame({})
    elif dict_config["data_split"]["test_size"]==1.0:
        df_sum_train = pd.DataFrame({})
        df_sum_test = df_sum_join.copy()
    elif "claim_flag" in df_sum_join.columns:
        df_sum_train, df_sum_test \
            = train_test_split(df_sum_join, 
                test_size=dict_config["data_split"]["test_size"],
                random_state=dict_config["data_split"]["random_state"],
                stratify=df_sum_join["claim_flag"])
    else:
        df_sum_train, df_sum_test \
            = train_test_split(df_sum_join, 
                test_size=dict_config["data_split"]["test_size"],
                random_state=dict_config["data_split"]["random_state"])
    
    logger.info("学習データ事故件数：%g"%len(df_sum_train))
    logger.info("検証データ事故件数：%g"%len(df_sum_test))

    if "claim_flag" in df_sum_train.columns:
        logger.info("【学習データの事故ラベルの値ごとの件数") 
        logger.info(df_sum_train["claim_flag"].value_counts())

    if "claim_flag" in df_sum_test.columns:
        logger.info("【検証データの事故ラベルの値ごとの件数") 
        logger.info(df_sum_test["claim_flag"].value_counts())

    ## 時系列データの分割
    if len(df_sum_train)==0:
        df_ts_train = pd.DataFrame({})
    else:
        ## この部分ではmergeの方が早かった
        df_ts_train = pd.merge(df_ts_join, df_sum_train["id"], on="id", how="inner")
    
        ## mergeを使わない方法
        '''
        id_list = df_sum_train["id"].values
        df_ts_train = df_ts_join[df_ts_join["id"].apply(lambda x: x in id_list)]
        '''

    if len(df_sum_test)==0:
        df_ts_test = pd.DataFrame({})
    else:
        ## この部分ではmergeの方が早かった
        df_ts_test = pd.merge(df_ts_join, df_sum_test["id"], on="id", how="inner")
        
        ## mergeを使わない方法
        '''
        id_list = df_sum_test["id"].values
        df_ts_test = df_ts_join[df_ts_join["id"].apply(lambda x: x in id_list)]
        '''

    logger.info("学習用時系列データ総レコード数：%g"%len(df_ts_train))
    logger.info("検証用時系列データ総レコード数：%g"%len(df_ts_test))

    # もとの時系列データの削除
    del df_ts_join

    return df_ts_train, df_ts_test