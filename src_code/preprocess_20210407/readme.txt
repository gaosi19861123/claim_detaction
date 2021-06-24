# 事故検知モデル前処理コード

## 概要
- このREADMEでは、事故検知モデル前処理コードの使用方法を説明する。
- このコードは、事故検知モデルに使うデータに対して、以下の処理を行う
    - 学習、検証用データへの分割
    - 欠損処理
    - ダウンサンプリング
    - 異常値処理
    - 正規化（中央値シフト/標準化）


## 目次
A. フォルダ構成と主要ファイル
B. 入力データ要件
C. 実行手順
D. 出力形式
E. config.jsonの設定値定義


## A. 主要なフォルダとファイル
preprocess
├── Dockerfile
├── readme.txt: このファイル
├── config.json: config設定ファイル
├── main.py: メイン関数
└── python: main.pyから呼び出されるサブモジュール群
 

## B. 入力データ要件
- 入力ファイル
    - 加速度時系列ファイル（必須）
    - サマリーデータ（１行が１事故に対応する）ファイル（任意）
- 時系列ファイルには以下の列が必須
    - 各事故に付与されたID(int or string型)
    - タイムスタンプ(string型)
    - 少なくとも1~3つの軸の加速度
- （読み込む場合）サマリーファイルには以下の列が必須
    - 各事故に付与されたID(string型)


## C. 実行手順
1. Dockerfileがあるフォルダに移動
  $ cd /home/ubuntu/accident_detection/src_code/preprocess_20210407
2. dockerイメージを作成（ここではイメージ名を"prepro_acc_detect", タグを"20210330"とする）
  $ docker build -t prepro_acc_detect:20210330 . 
3. dockerコンテナを作成してログイン
  $ docker run -it -v /home/ubuntu/accident_detection:/workspace prepro_acc_detect:20210330
4. config.jsonを編集（E.参照）
5. ログインしたコンテナ上で"/workspace/src_code/preprocess"に移り、main.pyを実行
  # cd /workspace/src_code/preprocess
  # python3.7 main.py
  - オプション
    - "-c": config設定のJSONファイルのパス指定（デフォルト：main.pyと同じフォルダのconfig.json）


## D. 出力
### 出力フォルダ
- config.jsonの以下で設定
    - "output"/"folder_path"/"train_data": 学習用データフォルダ（robustscaler標準化クラスも出力）
    - "output"/"folder_path"/"test_data": 検証用データフォルダ
    - "logger"/"folder_path": ログ出力フォルダ（使用したconfig.jsonファイルのコピーも出力）

### 出力データリスト
- 加速度時系列データ({train/test}_X.pkl)
    - shapeが(事故数, 時系列データ数, 軸の数)のndarray
- 事故フラグ({train/test}_y.pkl)
    - shapeが(事故数, 1)のndarray
- 加速度時系列データに対応するタイムスタンプ({train/test}_ts.pkl)
    - shapeが(事故数, 時系列データ数, 1)のndarray
- 事故IDとconfig.jsonの"input"/"summary"/"column_types"で指定した事故単位の情報({train/test}_others.{csv/pkl})
    - csv形式
        行数：事故数、列数：情報の種類の数
    - pickle形式
        shapeが(事故数, 情報の種類の数)のndarray
- robustscaler標準化統計量(robustscaler.pkl)
    - sklearn.preprocess.RobustScaler
    - 学習モデルの標準化に用いた統計量を格納
    - 学習データのフォルダ(config.jsonの"output"/"output"/"train_data"で指定)のみに出力
- ログファイル(log_preprocess_YYYYMMDDHHMM.log)
    - コードの実行中の記録


## E. config.jsonの設定値定義
### input
#### summary
- read_file(bool)
    - true：サマリーファイルを読み込む
    - false: 読み込まない
- file_path(str)
    - 入力ファイルのパス
- header(None or int)
    - pandas.read_csvのheader引数
        - "None": 入力ファイルにヘッダーがない
        - 0: 1行目がヘッダー
- delimiter(str)
    - pandas.read_csvのdelimiter引数
        - 区切り記号
- column_types(list of str)
    - 使用する値の種類をリストで指定。
      - "claim_flag": 事故ラベル
      - "timestamp": 事故発生時刻
      - "category": 事故カテゴリー
- column_name(str or int)
    - 各値の読み込みファイルでの列名
    - 読み込みファイルにヘッダーがない場合は左から何番目か(一番左の列を0番目とする)の整数値を入れる

#### time_series
- folder_path(str)
    入力ファイルが配置されているフォルダパス
- file_names(list of str)
    入力ファイル名のリスト
- header(None or int)
    - pandas.read_csvのheader引数
        - "None": 入力ファイルにヘッダーがない
        - 0: 1行目がヘッダー
- delimiter(str)
    - pandas.read_csvのdelimiter引数
        - 区切り記号
- column_types(list of str)
    - 使用する値の種類をリストで指定
        - "acceleration_x": x軸加速度
        - "acceleration_y": y軸加速度
        - "acceleration_z": z軸加速度
- column_name(str or int)
    - 各値の読み込みファイルでの列名
    - 読み込みファイルにヘッダーがない場合は左から何番目か(一番左端を0とする)の整数値を入れる

### data_condition
- n_data_each_crash(int)
    1事故あたりの時系列データ数
- interval_sec(float)
    出力する間隔秒数（サンプリングを実行する場合のみ考慮される）
- lim_d{x,y,z}_90_10(float)
    各事故における{x,y,z}軸加速度の「90%分位-10%分位」の上限（値の単位は入力ファイルと同一）

### sampling
- type(str)
    "skip": サンプリングを行わない
    "down": ダウンサンプリング実行
- reference_time(str)
    "timestamp_summary": サマリーファイルの"timestamp_summary"列を基準にinterval_secの秒数間隔で出力
    "midtime": 各事故の中央の時刻を基準にinterval_secの秒数間隔で出力

### data_split
- test_size(float: 0~1)
    検証データにするデータ数の割合
- random_state(int)
    分割シード(int)

### regularization
- type(str)
    "skip": 正規化を行わない
    "rscaler": sk-learn robustscalerで標準化
    "mshift": 中央値シフト補正
    "rscaler_to_mshift": robustscaler標準化⇒中央値補正の順で行う
    "mshift_to_rscaler": 中央値補正⇒robustscaler標準化の順で行う

### output
#### folder_path
- train_data(str): 学習用データ出力フォルダパス
- test_data(str): 検証用データ出力フォルダパス

#### other_infos
- file_type(str)
    - 事故ID、事故フラグに加えて、その他事故単位の情報を出力する形式
        - "csv": csvで出力
        - "pickle": pickleで出力
- csv_max_lines(int)
    - （file_type="csv"の場合）１つのcsvファイルでの出力行数の上限

### logger
- folder_path(str)
    - ログファイルの出力フォルダパス