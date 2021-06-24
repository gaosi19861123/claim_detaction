# 事故検知モデル推論コード

## 概要
- このREADMEでは、事故検知モデル推論コードの使用方法を説明する。
- このコードは、事故検知モデルに使うデータに対して、以下の処理を行う
    - 推論データのロード
    - モデル読み込み
    - 推論
    - 推論結果の出力

## 目次
A. フォルダ構成と主要ファイル
B. 入力データ要件
C. 実行手順
D. 出力形式
E. config.iniの設定値定義

## A. フォルダ構成と主要ファイル
inference
├── model/
|     └── multi_scale_ori.py：推論用ResNetのスクリプト
├── readme.txt: このファイル
├── config.ini: config設定ファイル
└── inference.py: メインスクリプト

## B. 入力データ要件
- 入力ファイル
    - 入力波形データ
    - フォーマット：pickle
    - データ型：float64
    - object：numpy array object
    - shape：[データ列、系列長=750, センサー次元=3]

## C. 実行手順
1. inference.pyがあるフォルダに移動
    $ cd /home/ubuntu/accident_detection/src_code/inference
2. pytorch_p36環境に入る ※Dockerを使う場合はここの仕様を変更
    $ conda activate pytorch_p36
3. config.iniを編集(E.参照)
4. inference.pyを実行
    $ (pytorch_p36) python inference.py

## D. 出力データ
- 出力ファイル
    - 出力ラベル
    - フォーマット:pickle
    - データ型：float64
    - object：numpy array object
    - shape：[データ数、ラベル=1]
    - 命名規則：<inference.pyの実行時刻>_result.pkl
    - 0:事故なし判定、1:事故あり判定

## E. config.iniの設定値定義
[PATH]
infer_data：推論対象のデータのパスを指定
infer_model_param：推論に用いるモデルのパスを指定

[PARAMETER]
infer_batch_size：dataloaderから読み込む際のバッチサイズの指定。デフォルトでは1で正直ここの値は変更する必要はない。

