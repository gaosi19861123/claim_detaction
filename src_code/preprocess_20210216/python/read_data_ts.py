import pandas as pd

def read_data_ts(dict_config):
    '''
    時系列データファイルを読み込み、データフレームに変換
    - 入力ファイルパスはsettings.pyで設定

    Parameters
    ----------
    dict_config : dict
        configファイル設定値
    '''

    dict_cnfg = dict_config["input"]["time_series"]

    print("【read_csv引数】")
    print("header= ", dict_cnfg["header"])
    print("delmiter = ", dict_cnfg["delimiter"])
    print()
    print("読み込み時系列データファイル名リスト：")
    print(dict_cnfg["file_names"])

    # 使用列名(ファイル上)リスト作成
    colnames_ts = []
    colnames_ts.append(dict_cnfg["column_name"]["id"])
    colnames_ts.append(dict_cnfg["column_name"]["timestamp"])
    if "acceleration_x" in dict_cnfg["column_types"]:
        colnames_ts.append(dict_cnfg["column_name"]["acceleration_x"])

    if "acceleration_y" in dict_cnfg["column_types"]:
        colnames_ts.append(dict_cnfg["column_name"]["acceleration_y"])

    if "acceleration_z" in dict_cnfg["column_types"]:
        colnames_ts.append(dict_cnfg["column_name"]["acceleration_z"])
    
    print()
    print("使用列名（読み込みファイル上）リスト：", colnames_ts)
    print()

    # gz圧縮されたcsvをデータフレームに読み込み
    for i in range(len(dict_cnfg["file_names"])):
        filepath = dict_cnfg["folder_path"]\
                    +"/"+dict_cnfg["file_names"][i]

        print("読み込み中：", filepath)

        header = dict_cnfg["header"]
        if header=="None":
            header = None
        
        df_read = pd.read_csv(filepath, header=header, \
                    delimiter=dict_cnfg["delimiter"],\
                    usecols=colnames_ts)[colnames_ts]

        if i==0:
            df_ts = df_read.copy()
        else:
            df_pre = df_ts.copy()
            df_ts = df_pre.append(df_read, ignore_index=True)

        del df_read
    
    # 列名付与
    df_ts.rename(columns={dict_cnfg["column_name"]["id"]: "id", 
                        dict_cnfg["column_name"]["timestamp"]: "timestamp",
                        dict_cnfg["column_name"]["acceleration_x"]: "acceleration_x",
                        dict_cnfg["column_name"]["acceleration_y"]: "acceleration_y",
                        dict_cnfg["column_name"]["acceleration_z"]: "acceleration_z"}, 
                        inplace=True)

    print()
    print("レコード数：", len(df_ts))
    print("【読み取りデータ冒頭】")
    print(df_ts.head())
    print()
    print("【データ統計量】")
    print(df_ts.describe())
    print()
    print("【データフレーム情報】")
    print(df_ts.info())
    print()

    return df_ts