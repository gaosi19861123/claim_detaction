#!/usr/bin/env python
# -*- conding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import sys
import logging

logger = logging.getLogger("preprocess_logger")

def downsampling(id_crash, **kwargs):
    '''時系列データのダウンサンプリングを実行
    - 基準時刻(ref_time)から設定の時間間隔(interval_sec)で時刻点をサンプリング
    - 読み取りデータの開始～終了時刻でconfig指定の時間間隔(interval_sec)でサンプリングする
    - 既存のレコードからピックアップするのみ（線形補間等で新たにレコードを作ることはしない）
        - したがって、時刻範囲がサンプリング前より広がることはない
        - もともと欠損していた時刻はそのまま欠損のまま
    - 時系列データ(df_ts)は時刻順にソートしてあること前提
    - データ個数がconfig指定値(n_data_each_crash)より多ければ、開始、終了時刻の双方から削っていく
        - 少ない場合は、呼び出し元の函数interp_null_and_do_samplingで除外される

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
        n_data_each_crash: int
            各事故での時系列データ数
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
    #   - 後続の処理でサンプリング時刻列をnumpyのdatetime型で作成
    start_time = df_ts_id["timestamp"].values.min()
    end_time = df_ts_id["timestamp"].values.max()

    # 基準時刻設定
    if kwargs["reference_time"]=="timestamp_summary" and len(df_sum)>0:
        ref_time = df_sum[df_sum["id"]==id_crash]["timestamp_summary"].values[0]
    elif kwargs["reference_time"]=="midtime":
        ref_time = df_ts_id["timestamp"].values[int(len(df_ts_id)/2)]
    else:
        sys.exit()

    # 時刻リスト作成
    ## 時間間隔引数設定
    dt_ns_sampling = np.timedelta64(int(kwargs["interval_sec"]*1e9),"ns")

    ts_list = []

    ## 基準時刻前
    t = ref_time
    while t >= start_time:
        ts_list.append(t)
        t -= dt_ns_sampling

    ## 時刻順にするため順序逆転
    ts_list.reverse()

    ## 基準時刻後
    t = ref_time+dt_ns_sampling
    while t <= end_time:
        ts_list.append(t)
        t += dt_ns_sampling

    ## データ数調整
    ## - config指定のデータ数より多い場合は先頭と末尾の双方からデータ点を減らす
    ## - config指定のデータ数より少ない場合はinterp_null_and_do_sampling関数内で除外される
    if kwargs["adjust_n_data"]:
        n_rm = len(ts_list) - kwargs["n_data_each_crash"]
        if n_rm>0:
            if n_rm%2==0:
                ts_list = ts_list[:-int(n_rm/2)]
                ts_list = ts_list[int(n_rm/2):]
            else:
                ts_list = ts_list[:-math.ceil(n_rm/2)]
                ts_list = ts_list[math.floor(n_rm/2):]

    # 処理速度向上のためset化
    ts_set = set(ts_list)

    ## ダウンサンプリング実施
    ## - to_datetime64をつけないとnumpyのdatetime型にならず、ts_setの中身と比較できない
    df_sampled = df_ts_id[df_ts_id["timestamp"].apply(lambda x: x.to_datetime64() in ts_set)]

    ## インデックス振り直し
    df_sampled.reset_index(drop=True, inplace=True)

    dict_sampled = df_sampled.to_dict()

    return dict_sampled
