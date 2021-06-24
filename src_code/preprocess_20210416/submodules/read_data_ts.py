#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
import logging

logger = logging.getLogger("preprocess_logger")

def read_data_ts(dict_config):
    '''
    時系列データファイルを読み込み、データフレームに変換
    - 入力ファイルパスはsettings.pyで設定

    Parameters
    ----------
    dict_config : dict
        configファイル設定値

    Returns
    -------
    df_ts : pandas.DataFrame
        読み取った時系列データ
    '''

    dict_cnfg = dict_config["input"]["time_series"]

    logger.info("【read_csv引数】")
    logger.info("header= %s"%dict_cnfg["header"])
    logger.info("delmiter = %s"%dict_cnfg["delimiter"]+"\n")
    logger.info("読み込み時系列データファイル名リスト：")
    logger.info(dict_cnfg["file_names"])
    logger.info("")

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
    
    logger.info("使用列名（読み込みファイル上）リスト：")
    logger.info(colnames_ts)

    # gz圧縮されたcsvをデータフレームに読み込み
    df_list = []
    for i in range(len(dict_cnfg["file_names"])):
        filepath = dict_cnfg["folder_path"]\
                    +"/"+dict_cnfg["file_names"][i]

        logger.info("読み込み中：%s"%filepath)

        header = dict_cnfg["header"]
        if header=="None":
            header = None
        
        df_read = pd.read_csv(filepath, header=header, \
                    delimiter=dict_cnfg["delimiter"],\
                    usecols=colnames_ts)[colnames_ts]

        df_list.append(df_read)

    # 上のループで読み込んだ各データフレームをユニオン
    df_ts = pd.concat(df_list, ignore_index=True)
    
    # 列名付与
    df_ts.rename(columns={dict_cnfg["column_name"]["id"]: "id", 
                        dict_cnfg["column_name"]["timestamp"]: "timestamp"}, inplace=True)
    if "acceleration_x" in dict_cnfg["column_types"]:
        df_ts.rename(columns={dict_cnfg["column_name"]["acceleration_x"]: "acceleration_x"}, 
                        inplace=True)
    if "acceleration_y" in dict_cnfg["column_types"]:
        df_ts.rename(columns={dict_cnfg["column_name"]["acceleration_y"]: "acceleration_y"},
                        inplace=True)
    if "acceleration_z" in dict_cnfg["column_types"]:
        df_ts.rename(columns={dict_cnfg["column_name"]["acceleration_z"]: "acceleration_z"}, 
                        inplace=True)

    # timestamp列をdatetime64[ns]型へ
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])

    logger.info("")
    logger.info("レコード数：%g"%len(df_ts))
    logger.info("【読み取りデータ冒頭】")
    logger.info(df_ts.head())
    logger.info("")
    logger.info("【データ統計量】")
    logger.info(df_ts.describe())
    logger.info("")

    return df_ts