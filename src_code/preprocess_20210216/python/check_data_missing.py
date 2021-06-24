import pandas as pd
import datetime

def check_data_missing(df_ts, dict_config):
    '''
    欠損レコードの除外
    1. 1つの列でもnullがあるレコードは除外（事故ごとは除外しない）
    2. データ数、総時間がconfig.jsonで設定した上限下限の範囲外ならばその事故を除外

    memo
    ----
    - 将来的には、タイムスタンプ、加速度列のどちらか一方の欠損の場合は線形補間で補充するようにしたい）

    Parameters
    ----------
    df_ts : pandas.DataFrame
        読み取りデータ
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_ok : pandas.DataFrame
        欠損処理後データ
    df_summary : pandas.DataFrame
        各事故のデータ数、総時間、欠損フラグ
    '''

    print("入力データレコード数: ", len(df_ts))
    print("入力データ欠損数：")
    print(df_ts.isnull().sum())
    print()

    # 1つの列でもnullがあるレコードは除外
    print("1つの列でもnullがあるレコードの除外実行")
    print()
    df_dropna = df_ts.dropna(how="any")
    del df_ts

    print("欠損処理後データレコード数: ", len(df_dropna))
    print("欠損処理後データ欠損数：")
    print(df_dropna.isnull().sum())
    print()

    # 各事故IDのデータ数, 総時間の集計
    ## IDで時系列データをgroup by
    df_ts_grpby_id = df_dropna[["id", "timestamp"]].groupby("id").agg(["count", "min", "max"])
    print("事故ID数: ", len(df_ts_grpby_id))
    print()

    ## 各IDの総時間リスト
    def dt(x):
        return pd.to_datetime(x["max"])-pd.to_datetime(x["min"])
    df_sec = df_ts_grpby_id["timestamp"].apply(dt, axis=1)
    df_sec.name="total_sec"

    ## 各IDのデータ数、総時間リストの横結合
    df_cnt_sec = pd.concat([df_ts_grpby_id["timestamp"]["count"],\
                            df_sec], axis=1)

    # 欠損フラグ付与
    # OK: 欠損なし 
    # bad_cnt_dur : データ数、総時間がconfig.jsonの上限下限の範囲外
    def flag_cnt_sec(x):
        cond = (x["total_sec"] < datetime.timedelta(seconds=dict_config["data_condition"]["total_sec_low"]) \
            or x["total_sec"] > datetime.timedelta(seconds=dict_config["data_condition"]["total_sec_up"]) \
            or x["count"] != dict_config["data_condition"]["n_data_each_crash"])
        if cond:
            return "bad_cnt_sec"
        else:
            return "OK"

    df_summary = df_cnt_sec.copy()
    del df_cnt_sec
    df_summary["data_miss_flag"] = df_summary.apply(flag_cnt_sec, axis=1)
    print("【フラグ別事故件数】")
    print(df_summary["data_miss_flag"].value_counts())
    print("\n OK: 欠損なし")
    print("bad_cnt_sec: データ数、総時間がconfig.jsonでの設定の範囲外 \n")
    print("【各IDデータ数、総時間統計量】")
    print(df_summary.describe(), "\n")
    print("【時系列データ数別件数】")
    print(df_summary["count"].value_counts(),"\n")

    print("【欠損フラグのデータフレーム冒頭】")
    print(df_summary.head(), "\n")

    print("bad_cnt_secフラグが付いた事故を除外")
    # 欠損除去後データとフラグテーブルの内部結合
    df_ok = pd.merge(df_dropna, df_summary[df_summary["data_miss_flag"]=="OK"],
                    left_on="id", right_index=True,
                    how="inner")[df_dropna.columns]
    df_ok.reset_index(drop=True, inplace=True)

    # 結合前データ削除
    del df_dropna

    print("除外後総レコード数：", len(df_ok))
    print("除外後事故IDユニーク数：", df_ok["id"].nunique())
    print()

    return df_ok, df_summary