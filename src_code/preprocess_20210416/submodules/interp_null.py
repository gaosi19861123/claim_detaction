#!/usr/bin/env python
# -*- conding: utf-8 -*-

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
