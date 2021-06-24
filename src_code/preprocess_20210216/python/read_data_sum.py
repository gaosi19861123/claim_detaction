import pandas as pd

def read_data_sum(dict_config):
    '''
    サマリーデータ（１行１事故）ファイルを読み込み、データフレームに変換
    - 入力ファイルパスはsettings.pyで設定

    Parameters
    ----------
    dict_config: dict
        configファイル設定値

    Returns
    -------
    df_sum: pandas.DataFrame
        サマリーファイル内容
    '''

    dict_cnfg = dict_config["input"]["summary"]

    print("read_sum: ", dict_cnfg["read_file"])
    print()
    if not dict_cnfg["read_file"]:
        print("サマリーデータファイルは読み込まず、空のデータフレームを返します")
        return pd.DataFrame()

    print("【read_csv引数】")
    print("header= ", dict_cnfg["header"])
    print("delmiter = ", dict_cnfg["delimiter"])
    print()
    print("読み込みサマリーデータファイル名:", dict_cnfg["file_path"])

    # 使用列名(ファイル上)リスト作成
    colnames_sum = []
    colnames_sum.append(dict_cnfg["column_name"]["id"])
    if "claim_flag" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["claim_flag"])
    if "timestamp_summary" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["timestamp_summary"])
    if "category" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["category"])

    print()
    print("使用列名（読み込みファイル上）リスト：", colnames_sum)
    print()

    # gz圧縮されたcsvをデータフレームに読み込み
    print("読み込み中：")
    print(dict_cnfg["file_path"])
    print()

    header = dict_cnfg["header"]
    if header=="None":
        header = None
        
    df_sum = pd.read_csv(dict_cnfg["file_path"], header=header, \
                delimiter=dict_cnfg["delimiter"], 
                usecols=colnames_sum)[colnames_sum]


    # 列名付与
    df_sum.rename(columns={dict_cnfg["column_name"]["id"]: "id", 
                           dict_cnfg["column_name"]["claim_flag"]: "claim_flag",
                           dict_cnfg["column_name"]["timestamp_summary"]: "timestamp_summary",
                           dict_cnfg["column_name"]["category"]: "category"}, 
                  inplace=True)

    print()
    print("レコード数：", len(df_sum))
    print("【読み取りデータ冒頭】")
    print(df_sum.head())
    print()
    print("【データ統計量】")
    print(df_sum.describe())
    print()
    print("【データフレーム情報】")
    print(df_sum.info())
    print()

    return df_sum
