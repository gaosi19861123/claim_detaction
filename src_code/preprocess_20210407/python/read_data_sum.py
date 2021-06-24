#!/usr/bin/env python
# -*- conding: utf-8 -*-

import pandas as pd
import logging

logger = logging.getLogger("preprocess_logger")

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

    logger.info("read_sum: %g"%dict_cnfg["read_file"])
    logger.info("")
    if not dict_cnfg["read_file"]:
        print("サマリーデータファイルは読み込まず、空のデータフレームを返します")
        return pd.DataFrame()

    logger.info("【read_csv引数】")
    logger.info("header= "+dict_cnfg["header"])
    logger.info("delmiter = "+dict_cnfg["delimiter"]+"\n")
    logger.info("読み込みサマリーデータファイル名: "+dict_cnfg["file_path"]+"\n")

    # 使用列名(ファイル上)リスト作成
    colnames_sum = []
    colnames_sum.append(dict_cnfg["column_name"]["id"])
    if "claim_flag" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["claim_flag"])
    if "timestamp_summary" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["timestamp_summary"])
    if "category" in dict_cnfg["column_types"]:
        colnames_sum.append(dict_cnfg["column_name"]["category"])

    logger.info("使用列名（読み込みファイル上）リスト：")
    logger.info(colnames_sum)
    logger.info("")

    # gz圧縮されたcsvをデータフレームに読み込み
    logger.info("読み込み中：")
    logger.info(dict_cnfg["file_path"]+"\n")

    header = dict_cnfg["header"]
    if header=="None":
        header = None
        
    df_sum = pd.read_csv(dict_cnfg["file_path"], header=header, \
                delimiter=dict_cnfg["delimiter"], 
                usecols=colnames_sum)[colnames_sum]

    # 列名付与
    df_sum.rename(columns={dict_cnfg["column_name"]["id"]: "id"}, inplace=True)
    if "claim_flag" in dict_cnfg["column_types"]:
        df_sum.rename(columns={dict_cnfg["column_name"]["claim_flag"]: "claim_flag"}, 
                        inplace=True)
    if "timestamp_summary" in dict_cnfg["column_types"]:
        df_sum.rename(columns={dict_cnfg["column_name"]["timestamp_summary"]: "timestamp_summary"},
                        inplace=True)
    if "category" in dict_cnfg["column_types"]:
        df_sum.rename(columns={dict_cnfg["column_name"]["category"]: "category"}, 
                        inplace=True)

    # timestamp列をdatetime64[ns]型へ
    df_sum["timestamp_summary"] = pd.to_datetime(df_sum["timestamp_summary"])

    logger.info("レコード数：%g"%len(df_sum))
    logger.info("【読み取りデータ冒頭】")
    logger.info(df_sum.head())
    logger.info("")
    logger.info("【データ統計量】")
    logger.info(df_sum.describe())
    logger.info("")
    logger.info("【データフレーム情報】")
    logger.info(df_sum.info())
    logger.info("")

    return df_sum
