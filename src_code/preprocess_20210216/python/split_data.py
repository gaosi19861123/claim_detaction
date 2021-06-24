import pandas as pd
from sklearn.model_selection import train_test_split

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
        print("時系列データをIDがサマリーデータにあるものに限定")
        print("限定前レコード数：", len(df_ts))
        print("限定後レコード数：", len(df_ts_join))
        print()
    else:
        print("サマリーデータがないので、時系列データとサマリーデータの結合は行わない\n")
        df_ts_join=df_ts.copy()

    # もとの時系列データの削除
    del df_ts

    # 結合後のデータフレームに対応するサマリー情報のデータフレーム作成
    df_sum_join = df_ts_join[df_sum.columns].drop_duplicates()
    df_sum_join.reset_index(drop=True, inplace=True)
    
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
    
    print("学習データ事故件数：", len(df_sum_train))
    print("検証データ事故件数：", len(df_sum_test), "\n")

    if "claim_flag" in df_sum_train.columns:
        print("【学習データの事故ラベルの値ごとの件数】\n", 
            df_sum_train["claim_flag"].value_counts(), "\n")

    if "claim_flag" in df_sum_test.columns:
        print("【検証データの事故ラベルの値ごとの件数】\n", 
            df_sum_test["claim_flag"].value_counts(), "\n")

    ## 時系列データの分割
    if len(df_sum_train)==0:
        df_ts_train = pd.DataFrame({})
    else:
        df_ts_train = pd.merge(df_ts_join, df_sum_train["id"], on="id", how="inner")

    if len(df_sum_test)==0:
        df_ts_test = pd.DataFrame({})
    else:
        df_ts_test = pd.merge(df_ts_join, df_sum_test["id"], on="id", how="inner")
    
    print("学習用時系列データ総レコード数：", len(df_ts_train))
    print("検証用時系列データ総レコード数：", len(df_ts_test))

    # もとの時系列データの削除
    del df_ts_join

    return df_ts_train, df_ts_test