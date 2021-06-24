#!/usr/bin/env python
# -*- conding: utf-8 -*-

import json
from submodules.read_data_ts import read_data_ts
from submodules.read_data_sum import read_data_sum
from submodules.split_data import split_data
from submodules.rm_outliers import rm_outliers
from submodules.regularize import regularize
from submodules.interp_null_and_do_sampling import interp_null_and_do_sampling
from submodules.output import output
import logging
import io
import sys
from datetime import datetime
import argparse
import subprocess
import os

def get_args(args):
    '''
    コマンドライン引数を定義

    Parameters
    ----------
    args: sys.argv
        コマンドライン引数入力値
    
    Returns
    -------
    parser.parse_args(args)
        定義された引数値
    '''

    parser = argparse.ArgumentParser(
        description="事故検知モデルにかける時系列データの前処理を行う")
    parser.add_argument(
        '--config',
        '-c',
        help="config設定jsonファイルのパス。デフォルトは./config.json",
        type=str,
        default="config.json"
    )

    return parser.parse_args(args)


def main(dict_config):
    '''
    事故検知モデル前処理コードメイン関数

    Parameters
    ----------
    dict_config : dict
        config設定値
    '''

    # 時系列データ読み込み
    logger.info("=== 時系列データファイル読み込み開始 ===")
    df_ts = read_data_ts(dict_config)
    logger.info("=== 時系列データファイル読み込み終了 ===\n")

    # サマリーデータ読み込み
    logger.info("=== サマリーデータ読み込み開始 ===")
    df_sum = read_data_sum(dict_config)
    logger.info("=== サマリーデータ読み込み終了 ===\n")

    # データ分割
    logger.info("=== データ分割開始 ===")
    df_ts_train, df_ts_test = split_data(df_ts, df_sum, dict_config)
    logger.info("=== データ分割終了 ===\n")

    # 元のデータフレーム削除
    del df_ts

    # 欠損処理
    logger.info("=== 学習用データ欠損処理開始 ===")
    df_missing_checked_train \
        = interp_null_and_do_sampling(df_ts_train, df_sum, dict_config)
    logger.info("=== 学習用欠損処理終了 ===\n")
    logger.info("=== 検証用データ欠損処理開始 ===")
    df_missing_checked_test \
        = interp_null_and_do_sampling(df_ts_test, df_sum, dict_config)
    logger.info("=== 検証用欠損処理終了 ===\n")

    # 元のデータフレーム削除
    del df_ts_train, df_ts_test

    # 異常値処理
    logger.info("=== 学習用データ異常値処理開始 ===")
    df_no_outliers_train, df_sum_outliers_train \
        = rm_outliers(df_missing_checked_train, dict_config)
    logger.info("=== 学習用データ異常値終了 ===\n")
    logger.info("=== 検証用データ異常値処理開始 ===")
    df_no_outliers_test, df_sum_outliers_test \
        = rm_outliers(df_missing_checked_test, dict_config)
    logger.info("=== 検証用データ異常値処理終了 ===\n")

    # 元のデータフレーム削除
    del df_missing_checked_train, df_missing_checked_test

    # 加速度値の正規化
    logger.info("=== 正規化開始 ===")
    df_reg_train, df_reg_test \
        = regularize(df_no_outliers_train, df_no_outliers_test, dict_config)
    logger.info("=== 正規化終了 ===\n")

    # 元のデータフレーム削除
    del df_no_outliers_train, df_no_outliers_test

    # 出力
    logger.info("=== 出力開始 ===")
    output(df_reg_train, df_reg_test, dict_config)
    logger.info("=== 出力終了 ===")
    logger.info("===== 前処理正常終了 ======")


if __name__=="__main__":

    # コマンドライン引数受け取り
    args = get_args(sys.argv[1:])

    # configファイル読み取り
    print("=== configファイル読み取り開始 ===")
    dict_config = json.load(open(args.config, "r"))
    print("【config設定値】")
    print(dict_config)
    print("\n=== configファイル読み取り終了 ===\n")

    # configファイルをログ出力フォルダにコピー
    log_dir = dict_config["logger"]["folder_path"]
    if os.path.exists(log_dir):
        subprocess.call(["cp", args.config, log_dir])
    else:
        print("以下のログ出力フォルダが存在しません。フォルダを作ってください")
        print(log_dir)
        sys.exit()

    # ログ設定
    ## 1. logger設定
    ### loggerオブジェクト宣言
    logger = logging.getLogger("preprocess_logger")

    ### handlerに渡すエラーメッセージのレベル    
    logger.setLevel("DEBUG")

    ## 2. handler設定
    ### ログのフォーマットを定義
    handler_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ### 2-1. 標準出力のhandler
    #### handler生成
    stream_handler = logging.StreamHandler()

    #### ログレベル設定
    stream_handler.setLevel("INFO")

    #### 出力フォーマット設定
    stream_handler.setFormatter(handler_format)

    ### 2-2. テキスト出力のhandler
    #### ファイルパス設定
    now_time = datetime.now()
    logfile_name = 'log_preprocess_{time}.log'.format(
        time=now_time.strftime('%Y%m%d%H%M')
    )
    logfile_path = log_dir+"/"+logfile_name

    #### handlerの生成
    file_handler = logging.FileHandler(logfile_path)

    #### ログレベル設定
    file_handler.setLevel("DEBUG")    

    #### 出力フォーマット設定
    file_handler.setFormatter(handler_format)

    ### 3. loggerにhandlerをセット
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


    # config値ログ表示
    logger.debug("【config設定値】")
    logger.debug(dict_config)

    main(dict_config)

