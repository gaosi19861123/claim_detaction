#!/usr/bin/env python
# -*- conding: utf-8 -*-

import numpy as np
import math
import sys
import logging
import datetime

logger = logging.getLogger("preprocess_logger")

def sampling_by_interp(id_crash, **kwargs):
    '''線形補間によるサンプリングを実行
    - 基準時刻(ref_time)から設定の時間間隔(interval_sec)で時刻点をサンプリング
    - 読み込みデータの開始から終了時刻の範囲内において、設定の時間間隔(interval_sec)によっては新たにデータ点を生成
        - downsampling.pyのdownsampling函数とは違い、開始から終了時刻の範囲外にinterval_sec以下のはみ出しは許す
            - 最後のサンプリングでない場合はinterval_sec１つ分はみ出させて出力（最後の場合はデータ点の数がconfig設定値(n_data_each_crash)に調整される）
            - 許さないと、大きい時間間隔でダウンサンプリングした場合時刻範囲がかなり狭まり、そこから再度アップサンプリングする場合データ点をかなり失ってしまう可能性がある
    - 時系列データ(df_ts)は時刻順にソートしてあること前提
    - サンプリングの時間間隔はナノ秒単位で設定可能だが、もとのデータの間隔の倍数でないと設定どおりの間隔で出力はされない

    Parameters
    ----------
    id_crash: str or int
        事故ID

    **kwargs
        df_ts: pandas.DataFrame
            時系列データ
        df_sum: pandas.DataFrame
            サマリーデータ
        reference_time: str
            サンプリング基準時刻のタイプ
        interval_sec: float
            サンプリング間隔秒数
        adjust_n_data: bool
            True: データ数調整実施
            False: 実施しない

    Returns
    -------
    dict_sampled : dict
        サンプリング済み加速度時系列データ
    '''

    df_ts = kwargs["df_ts"]
    df_sum = kwargs["df_sum"]

    # １事故分のデータ抽出
    nd_id = np.array(df_ts["id"])
    nd_select_id = nd_id == id_crash
    df_ts_id = df_ts[nd_select_id].reset_index(drop=True)

    # 読み取りデータ開始終了時刻
    # - .valuesをかけているのはpandasのdatetime型ではなく、numpyのdatetime型にするため
    #   - これらを用いて後続の処理でサンプリング時刻列をnumpyのdatetime型で作成したいため
    start_time = df_ts_id["timestamp"].values.min()
    end_time = df_ts_id["timestamp"].values.max()

    # 基準時刻設定
    if kwargs["reference_time"]=="timestamp_summary" and len(df_sum)>0:
        ref_time = df_sum[df_sum["id"]==id_crash]["timestamp_summary"].values[0]
    elif kwargs["reference_time"]=="midtime":
        ref_time = df_ts_id["timestamp"].values[int(len(df_ts_id)/2)]
    else:
        sys.exit()

    # 時刻列作成
    ## 一旦開始～終了時刻範囲から指定時間間隔１つ分はみ出して時刻列を作成

    ### 時間間隔引数設定
    dt_ns_sampling = np.timedelta64(int(kwargs["interval_sec"]*1e9),"ns")

    ts_list = []

    ### 基準時刻前
    t = ref_time
    while t >= start_time-dt_ns_sampling:
        ts_list.append(t)
        t -= dt_ns_sampling

    ### 時刻順にするため順序逆転
    ts_list.reverse()

    ### 基準時刻後
    t = ref_time + dt_ns_sampling
    while t <= end_time+dt_ns_sampling:
        ts_list.append(t)
        t += dt_ns_sampling

    ## データ数調整
    ## 最後のサンプリングである場合のみ実施
    ## - config指定のデータ数より多い場合は先頭と末尾の双方からデータ点を減らす
    ## - config指定のデータ数より少ない場合はinterp_null_and_do_sampling関数内で事故ごと除外される
    if kwargs["adjust_n_data"]:
        n_rm = len(ts_list) - kwargs["n_data_each_crash"]
        if n_rm>0:
            if n_rm%2==0:
                ts_list = ts_list[:-int(n_rm/2)]
                ts_list = ts_list[int(n_rm/2):]
            else:
                ts_list = ts_list[:-math.ceil(n_rm/2)]
                ts_list = ts_list[math.floor(n_rm/2):]

    ## unix時間に
    t_new = [x.astype(datetime.datetime) for x in ts_list]
    
    # もとの時刻列をunix時間に
    t_old = [x.astype(datetime.datetime) for x in df_ts_id["timestamp"].values]

    # 時刻列をreturnする辞書に格納
    dict_sampled = {"timestamp": ts_list}
    
    # 時系列データではない（事故IDなど全行同じ値の）列を辞書に格納
    for col in df_ts_id.columns:
        if col not in ["timestamp", "acceleration_x", "acceleration_y", "acceleration_z"]:
            dict_sampled[col] = [df_ts_id[col].values[0]]*len(ts_list) 

    # 加速度の線形補間実行＆辞書に格納
    if "acceleration_x" in df_ts_id.columns:
        dict_sampled["acceleration_x"] = np.interp(t_new, t_old, df_ts_id["acceleration_x"].values)
    if "acceleration_y" in df_ts_id.columns:
        dict_sampled["acceleration_y"] = np.interp(t_new, t_old, df_ts_id["acceleration_y"].values)
    if "acceleration_z" in df_ts_id.columns:
        dict_sampled["acceleration_z"] = np.interp(t_new, t_old, df_ts_id["acceleration_z"].values)

    return dict_sampled