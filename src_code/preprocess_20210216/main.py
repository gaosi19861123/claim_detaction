import json
from python.read_data_ts import read_data_ts
from python.read_data_sum import read_data_sum
from python.split_data import split_data
from python.check_data_missing import check_data_missing
from python.rm_outliers import rm_outliers
from python.regularize import regularize
from python.output import output

def main():

    # configファイル読み取り
    print("=== configファイル読み取り開始 ===")
    dict_config = json.load(open("config.json", "r"))
    print("【config設定値】")
    print(dict_config)
    print()
    print("=== configファイル読み取り終了 ===")
    print()

    # 時系列データ読み込み
    print("=== 時系列データファイル読み込み開始 ===")
    df_ts = read_data_ts(dict_config)
    print("=== 時系列データファイル読み込み終了 ===\n")

    # サマリーデータ読み込み
    print("=== サマリーデータ読み込み開始 ===")
    df_sum = read_data_sum(dict_config)
    print("=== サマリーデータ読み込み終了 ===\n")

    # データ分割
    print("=== データ分割開始 ===")
    df_ts_train, df_ts_test = split_data(df_ts, df_sum, dict_config)
    print("=== データ分割終了 ===\n")

    # 元のデータフレーム削除
    del df_ts

    # 欠損処理
    print("=== 学習用データ欠損処理開始 ===")
    df_missing_checked_train, df_sum_missing_train \
        = check_data_missing(df_ts_train, dict_config)
    print("=== 学習用欠損処理終了 ===\n")
    print("\n === 検証用データ欠損処理開始 ===")
    df_missing_checked_test, df_sum_missing_test \
        = check_data_missing(df_ts_test, dict_config)
    print("=== 検証用欠損処理終了 ===\n")

    # 元のデータフレーム削除
    del df_ts_train, df_ts_test

    # 異常値処理
    print("=== 学習用データ異常値処理開始 ===")
    df_no_outliers_train, df_sum_outliers_train \
        = rm_outliers(df_missing_checked_train, dict_config)
    print("=== 学習用データ異常値終了 ===\n")
    print("\n === 検証用データ異常値処理開始 ===\n")
    df_no_outliers_test, df_sum_outliers_test \
        = rm_outliers(df_missing_checked_test, dict_config)
    print("=== 検証用データ異常値処理終了 ===\n")
    print()

    # 元のデータフレーム削除
    del df_missing_checked_train, df_missing_checked_test

    # 加速度値の正規化
    print("=== 正規化開始 ===")
    df_reg_train, df_reg_test \
        = regularize(df_no_outliers_train, df_no_outliers_test, dict_config)
    print("=== 正規化終了 ===")
    print()

    # 元のデータフレーム削除
    del df_no_outliers_train, df_no_outliers_test

    # 出力
    print("=== 出力開始 ===")
    output(df_reg_train, df_reg_test, dict_config)
    print("=== 出力終了 ===")
    print("===== 前処理正常終了 ======")


if __name__=="__main__":
    main()

