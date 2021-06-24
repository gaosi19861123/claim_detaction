#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
import multiprocessing
from submodules.myfunc_multiprocessing import processing_multi_core
from submodules.interp_null import interp_null
from submodules.downsampling import downsampling
from submodules.sampling_by_interp import sampling_by_interp

logger = logging.getLogger("preprocess_logger")

def interp_null_and_do_sampling(df_ts, df_sum, dict_config):
    '''時系列データに対して欠損処理、サンプリングを実行
    - 欠損処理
        - 事故ID, または時刻列がnullの行を除外
        - それ以外の列のnull箇所を線形補間で値を補充
    - サンプリング
        基準時刻から等間隔にデータ点を以下のいずれかの方法でサンプリング
        1. ダウンサンプリング（もとからあるデータ点を抽出）
            - サンプリングの時間間隔はナノ秒単位で設定可能だが、もとのデータの間隔の倍数でないと設定どおりの間隔で出力はされない
        2. 線形補間（ダウン・アップサンプリング両方可）
            - サンプリングの時間間隔はナノ秒単位で設定可能
        
        - configファイルでサンプリングを複数回かけるように設定することも可能
            - ダウンサンプリングしてから再度アップサンプリングする場合は、どちらも上記の「2. 線形補間」にすることを推奨
                - 「1. ダウンサンプリング」を用いると、設定の時間間隔によっては開始、終了の時刻点が削られ、再度アップサンプリングをする際、その部分の時刻点を生成することができない

    Parameters
    ----------
    df_ts: pandas.DataFrame
        時系列データ
    df_sum: pandas.DataFrame
        サマリーデータ
    dict_config: dict
        config設定値

    Returns
    -------
    df_sampled : pandas.DataFrame
        欠損処理、サンプリング後時系列データ
    '''

    logger.info("入力データレコード数: %g"%len(df_ts))
    logger.info("入力データ欠損数：")
    logger.info(df_ts.isnull().sum())
    logger.info("")

    # レコードがない場合空のデータフレームを返す
    if len(df_ts)==0:
        return pd.DataFrame()

    logger.info("id, またはtimestampが欠損しているレコードを除外")
    df_no_tnull = df_ts.dropna(subset=["id", "timestamp"])
    logger.info("除外後件数：%g"%len(df_no_tnull))

    logger.info("事故ID、時刻でソート（事故ID優先）")
    df_sort = df_no_tnull.sort_values(["id", "timestamp"])
    logger.info("【ソート後データフレーム冒頭】")
    logger.info(df_sort.head())
    logger.info("【ソート後データフレーム末尾】")
    logger.info(df_sort.tail())
    logger.info("")

    fac_cpu = dict_config["computation"]["factor_cpu_count"]

    # 事故IDリスト
    id_crash_list = np.sort(df_sort["id"].unique())

    # 欠損補間
    logger.info("id, timestamp以外の欠損箇所を線形補間")

    ## 時系列データ全体でnullの個数を数えて、１つでもあれば並列処理で欠損補間
    ## - interp_null函数の中で各事故IDでnull個数を再度数えるので二度手間ではあるが、
    ##   df_sort.isnull().sum().sum()はそれほどかからない(5億レコードに対して10秒ほど)。
    ## - 一方で、並列処理に入ってしまうとある程度時間がかかる。 
    ## - よって、入力データに欠損が全くない場合はこの方法が時間の節約になる
    if df_sort.isnull().sum().sum()>0:
        ## 並列処理
        dict_list = processing_multi_core(interp_null, id_crash_list, 100, \
                        retain_order=True, \
                        n_jobs=int(fac_cpu*multiprocessing.cpu_count()), \
                        verbose=1, df_sort=df_sort)
        df_list = [pd.DataFrame(x) for x in dict_list]

        ## 全事故データユニオン
        df_modna = pd.concat(df_list, ignore_index=True)

        ## 元のデータフレームリスト削除
        del df_list
    else:
        df_modna = df_sort.copy()
    
    # 元のデータフレーム削除
    del df_sort

    logger.info("欠損線形補間後レコード数: %g"%len(df_modna))
    logger.info("欠損線形補間後欠損数：")
    logger.info(df_modna.isnull().sum())
    logger.info("")

    # サンプリング
    ## config設定値
    reference_time = dict_config["sampling"]["reference_time"]
    n_data_each_crash = dict_config["data_condition"]["n_data_each_crash"]
    sampling_types = dict_config["sampling"]["types"]
    interval_secs = dict_config["sampling"]["interval_secs"]

    adjust_n_data = False

    df_sampled = df_modna.copy()
    del df_modna

    do_multiproc = not ("skip" in set(sampling_types) and len(set(sampling_types))==1)
    logger.info("並列処理を行う: %r"%do_multiproc)

    if do_multiproc:
        ## idを数値に置き換える
        logger.info("並列処理速度向上のためにIDを文字列から数値へ変換")

        ### 時系列、サマリーデータのデータフレームのidを数値に
        dict_num_id = dict(zip(np.arange(len(df_sum)), df_sum["id"]))
        dict_id_num = {v: k for k, v in dict_num_id.items()}

        logger.info("変換のための辞書のサンプル")
        logger.info(str(id_crash_list[0])+" -> %g"%dict_id_num[id_crash_list[0]])
        logger.info("")

        df_sampled["id"] = df_sampled["id"].apply(lambda x: dict_id_num[x])
        df_sum["id"] = df_sum["id"].apply(lambda x: dict_id_num[x])

        ### 並列処理のfor用のidリストを数値に
        id_num_list = [dict_id_num[x] for x in id_crash_list]

    for i in range(len(sampling_types)):

        ## データ数調整はループの最後でのみ行う
        if i==len(sampling_types)-1:
            adjust_n_data=True        

        if sampling_types[i]=="down":
            logger.info("ダウンサンプリング実行。間隔秒数：%f"%interval_secs[i])

            ## ダウンサンプリング並列処理
            dict_list = processing_multi_core(downsampling, id_num_list, 1, \
                            retain_order=True, \
                            n_jobs=int(fac_cpu*multiprocessing.cpu_count()), \
                            verbose=1, df_ts=df_sampled, df_sum=df_sum, \
                            reference_time = reference_time, 
                            interval_sec= interval_secs[i], \
                            n_data_each_crash=n_data_each_crash, \
                            adjust_n_data=adjust_n_data)
            df_list = [pd.DataFrame(x) for x in dict_list]

            ### 全事故データユニオン
            df_sampled = pd.concat(df_list, ignore_index=True)

        elif sampling_types[i]=="interp":
            logger.info("線形補間によるサンプリング実行。間隔秒数：%f"%interval_secs[i])

            ## 線形補間並列処理
            dict_list = processing_multi_core(sampling_by_interp, id_num_list, 1, \
                            retain_order=True, \
                            n_jobs=int(fac_cpu*multiprocessing.cpu_count()), \
                            verbose=1, df_ts=df_sampled, df_sum=df_sum, \
                            reference_time = reference_time, 
                            interval_sec= interval_secs[i], \
                            n_data_each_crash=n_data_each_crash, \
                            adjust_n_data=adjust_n_data)
            df_list = [pd.DataFrame(x) for x in dict_list]

            ### 全事故データユニオン
            df_sampled = pd.concat(df_list, ignore_index=True)

        elif sampling_types[i]=="skip":
            logger.info("サンプリング実行せず")

        logger.info("【サンプリング後データフレーム統計量】")
        logger.info(df_sampled.describe())
        logger.info("")
        logger.info("【1事故サンプル】")
        logger.info(df_sampled[df_sampled["id"]==df_sampled["id"][0]])
        logger.info("")

    logger.info("サンプリング終了")
    
    # idを数値からもとの文字列に戻す
    if do_multiproc:
        df_sampled["id"] = df_sampled["id"].apply(lambda x: dict_num_id[x])
        df_sum["id"] = df_sum["id"].apply(lambda x: dict_num_id[x])
        logger.info("idを数値にしていたのをもとの文字列に再変換")

    # config指定のデータ数の事故のみ抽出
    logger.info("データ数が%g"%n_data_each_crash+"の事故のみ抽出\n")

    ## 各事故IDのデータ数集計
    ser_id_cnt= df_sampled["id"].value_counts()
    ser_id_cnt.name = "n_data"
    logger.info("総事故数: %g"%len(ser_id_cnt))

    ser_id_ok = ser_id_cnt[ser_id_cnt==n_data_each_crash]
    logger.info("抽出事故数：%g"%len(ser_id_ok))

    ## 該当IDのみ抽出
    set_id_ok = set(ser_id_ok.index)
    df_sampled = df_sampled[ df_sampled["id"].apply(lambda x: x in set_id_ok) ]

    logger.info("抽出後レコード数：%g"%len(df_sampled))

    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム冒頭】")
    logger.info(df_sampled.head())
    logger.info("")
    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム末尾】")
    logger.info(df_sampled.tail())
    logger.info("")
    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム統計量】")
    logger.info(df_sampled.describe())
    logger.info("")

    return df_sampled