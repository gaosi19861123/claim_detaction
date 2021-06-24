#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
import sys
import math
import numpy as np
import logging
from python.myfunc_multiprocessing import processing_multi_core
import multiprocessing

logger = logging.getLogger("preprocess_logger")

def downsampling(id_crash, **kwargs):
    '''
    時系列データのダウンサンプリングを実行
    - 既存のデータからピックアップするのみ（線形補間等で新たにデータ点を作ることはしない）
        - ダウンサンプリング後の時刻点のうちで、もともと欠損していた点はそのまま欠損のまま
    - 時系列データ(df_ts)は時刻順にソートしてあること前提
    - 読み取りデータの開始～終了時刻でconfig指定の時間間隔(interval_sec)でサンプリングする
        - データ個数がconfig指定値(n_data_each_crash)に合わない事故は除外

    Parameters
    ----------
    id_crash: str or int
        事故ID

    **kwargs
        df_ts: pandas.DataFrame
            時系列データ
        df_sum: pandas.DataFrame
            サマリーデータ
        dict_config: dict
            config設定値

    Returns
    -------
    dict_sampled : dict
        サンプリング済み加速度時系列データ
    '''

    df_ts = kwargs["df_ts"]
    df_sum = kwargs["df_sum"]
    dict_config = kwargs["dict_config"]

    # １事故分のデータ抽出
    df_ts_id = df_ts[df_ts["id"]==id_crash].reset_index(drop=True)

    # 読み取りデータ開始終了時刻
    start_time = df_ts_id["timestamp"].min()
    end_time = df_ts_id["timestamp"].max()
    #logger.debug("開始：%s"%str(start_time)+" 終了：%s"%str(end_time))

    # 基準時刻設定
    if dict_config["sampling"]["reference_time"]=="timestamp_summary" and len(df_sum)>0:
        ref_time = df_sum[df_sum["id"]==id_crash]["timestamp_summary"].values[0]
        #logger.debug("基準時刻（サマリーファイル）: %s"%str(ref_time))
    elif dict_config["sampling"]["reference_time"]=="midtime":
        ref_time = df_ts_id["timestamp"][int(len(df_ts_id)/2)]
        #logger.debug("基準時刻（中間時刻）: %s"%str(ref_time))
    else:
        #logger.info("config.jsonのsampling/reference_timeの値が正しくありません")
        sys.exit()

    # 時刻リスト作成
    ## 時間間隔引数設定
    dt_sampling = float(dict_config["data_condition"]["interval_sec"])
    freq = str(dt_sampling)+"S"

    ## 基準時刻後
    ts_list_after = pd.date_range(ref_time, end_time, freq=freq)

    ## 基準時刻前
    dur_before = ref_time - start_time
    n_before = math.floor(dur_before.total_seconds()/dt_sampling)
    time_start_before = ref_time - pd.to_timedelta(dt_sampling*n_before, unit="s")
    time_end_before = ref_time - pd.to_timedelta(dt_sampling, unit="s")
    ts_list_before = pd.date_range(time_start_before, time_end_before, freq=freq)

    ## 基準前後結合
    ts_list = list(ts_list_before) + list(ts_list_after)

    ## データ数調整
    ## - config指定のデータ数より多い場合は先頭と末尾の双方からデータ点を減らす
    ## - config指定のデータ数より少ない場合はinterp_null_and_do_sampling関数内で除外される
    n_rm = len(ts_list) - dict_config["data_condition"]["n_data_each_crash"]
    if n_rm>0:
        if n_rm%2==0:
            ts_list = ts_list[:-int(n_rm/2)]
            ts_list = ts_list[int(n_rm/2):]
        else:
            ts_list = ts_list[:-math.floor(n_rm/2)]
            ts_list = ts_list[math.ceil(n_rm/2):]

    ## ダウンサンプリング実施
    ### mergeを用いる場合
    '''
    ## pandas.Seriesに変換        
    ser_ts_list = pd.Series(ts_list)
    ser_ts_list.name = "time_sampled"

    # タイムスタンプ列で内部結合することによるダウンサンプリング
    df_sampled = pd.merge(df_ts_id, ser_ts_list, 
                            left_on="timestamp", right_on="time_sampled", 
                            how="inner")[df_ts_id.columns]
    '''
    
    ### mergeを用いない場合
    df_sampled = df_ts_id[df_ts_id["timestamp"].apply(lambda x: x in ts_list)]

    ## インデックス振り直し
    df_sampled.reset_index(drop=True, inplace=True)

    dict_sampled = df_sampled.to_dict()

    return dict_sampled


def downsampling_id_group(id_crash_list, df_ts, df_sum, dict_config):
    '''
    ダウンサンプリングの並列処理を複数の事故IDのグループごとに行う関数

    Parameters
    ----------
    id_crash_list: list of str or int
        事故IDリスト
    df_ts: pandas.DataFrame
        時系列データ
    df_sum: pandas.DataFrame
        サマリーデータ
    dict_config: dict
        config設定値

    Returns
    -------
    pandas.DataFrame
        ダウンサンプリング済み時系列データ
    '''

    df_list = [downsampling(id_crash, df_ts, df_sum, dict_config) \
                for id_crash in id_crash_list]

    return pd.concat(df_list, ignore_index=True)


def interp_null(id_crash, **kwargs):
    '''
    加速度列の欠損箇所を線形補間で値補充

    Parameters
    ----------
    id_crash: str or int
        事故ID

    **kwargs
        df_sort: pandas.DataFrame
            時刻でソートされている時系列データ
    
    Returns
    -------
    df_each_id.to_dict(): dict
        処理後の１事故のデータ
    '''

    df_sort = kwargs["df_sort"]

    # id_crashに対応する事故のデータのみ抽出
    df_each_id = df_sort[df_sort["id"]==id_crash]

    if df_each_id.isnull().sum().sum()>0:
        # timestamp列をインデックスに
        df_each_id = df_each_id.set_index("timestamp")

        # 各軸加速度の欠損箇所線形補間    
        if "acceleration_x" in df_sort.columns:
            df_each_id["acceleration_x"] = df_each_id["acceleration_x"].interpolate(axis=0)
        if "acceleration_y" in df_sort.columns:
            df_each_id["acceleration_y"] = df_each_id["acceleration_y"].interpolate(axis=0)
        if "acceleration_z" in df_sort.columns:
            df_each_id["acceleration_z"] = df_each_id["acceleration_z"].interpolate(axis=0)

        # timestampをインデックスから列へ
        df_each_id = df_each_id.reset_index()
    
    return df_each_id.to_dict()


def interp_null_and_do_sampling(df_ts, df_sum, dict_config):
    '''
    時系列データの
    - 欠損処理
        - 事故ID, または時刻列がnullの行を除外
        - それ以外の列のnull箇所を線形補間で値を補充
    - ダウンサンプリングを実施
    - 将来的にアップサンプリング（線形補間）も実装予定

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

    # 事故IDリスト
    id_crash_list = np.sort(df_sort["id"].unique())

    # 欠損補間
    logger.info("id, timestamp以外の欠損箇所を線形補間")

    ## 時系列データ全体でnullの個数を数えて、１つでもあれば並列処理で欠損補間
    ## - interp_null函数の中で各事故IDでnull個数を再度数えるので二度手間ではあるが、
    ##   下記の並列処理に入るとある程度時間がかかる。
    ## - 一方で、df_sort.isnull().sum().sum()はそれほどかからない(5億レコードに対して10秒ほど)。
    ##   よって、入力データに欠損が全くない場合はこの方が圧倒的に時間の節約になる
    if df_sort.isnull().sum().sum()>0:
        ## 並列処理
        dict_list = processing_multi_core(interp_null, id_crash_list, 100, \
                        retain_order=True, n_jobs=int(multiprocessing.cpu_count()), \
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
    if dict_config["sampling"]["type"]=="down":
        logger.info("ダウンサンプリング実行")
        
        ## ダウンサンプリング並列処理
        dict_list = processing_multi_core(downsampling, id_crash_list, 1, \
                        retain_order=True, n_jobs=int(multiprocessing.cpu_count()), \
                        verbose=1, df_ts=df_modna, df_sum=df_sum, \
                        dict_config=dict_config)
        df_list = [pd.DataFrame(x) for x in dict_list]

        ### 全事故データユニオン
        df_sampled = pd.concat(df_list, ignore_index=True)

    elif dict_config["sampling"]["type"]=="skip":
        logger.info("サンプリング実行せず")
        df_sampled = df_modna.copy()

    # 元のデータフレーム削除
    del df_modna

    n_data = dict_config["data_condition"]["n_data_each_crash"]
    logger.info("データ数が%g"%n_data+"の事故のみ抽出\n")
    
    # 各事故IDのデータ数集計
    ser_id_cnt= df_sampled["id"].value_counts()
    ser_id_cnt.name = "n_data"
    logger.info("総事故数: %g"%len(ser_id_cnt))

    ser_id_ok = ser_id_cnt[ser_id_cnt==n_data]
    logger.info("抽出事故数：%g"%len(ser_id_ok))

    # 該当IDのみ抽出
    list_id_ok = ser_id_ok.index
    df_sampled = df_sampled[ df_sampled["id"].apply(lambda x: x in list_id_ok) ]

    logger.info("抽出後レコード数：%g"%len(df_sampled))

    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム冒頭】")
    logger.info(df_sampled.head())
    logger.info("")
    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム末尾】")
    logger.info(df_sampled.tail())
    logger.info("")
    logger.info("【サンプリング・指定データ数の事故抽出後データフレーム統計量】")
    logger.info(df_sampled.info())
    logger.info("")

    return df_sampled