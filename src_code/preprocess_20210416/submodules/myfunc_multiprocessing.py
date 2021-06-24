#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################
# 自作並列処理函数 ver 0.08
# create date: 2020/09/28
# Author: Koki Takasue
# Partially copied by hand 2021/04/05 Takafumi Sonoi
#############################

# 並列処理用
from multiprocessing import Process, Manager, Lock, Value, Array, cpu_count

import logging
import math
import time

logger = logging.getLogger("preprocess_logger")

def print_progress_bar(num_row, row_per_job, proc_endno, start_time):
    '''
    progress bar表示用函数

    Parameters
    ----------
    num_row: int
        処理件数
    row_per_job: int
        1process当たりの処理件数
    proc_endno: int
        処理が完了したprocess数
    start_time: time
        処理開始時間
    '''

    # 単位時間の平均処理数
    process_per_time = row_per_job*(proc_endno+1)/(time.time()-start_time)

    # 残時間（大雑把に）
    retain_time = (num_row - row_per_job*(proc_endno+1))/process_per_time
    if retain_time < 0:
        retain_time = 0

    # 表示テンプレート
    bar_template = "\r[{0}] {1:.2f}% @{2}, {3:.2f}/sec"

    # バーの長さ
    progress_bar_range = 50

    # 進捗率
    z = (proc_endno+1)/math.ceil(num_row/row_per_job)*100
    # #の数
    if z>=100:
        step_add = progress_bar_range
    else:
        step_add = progress_bar_range/math.ceil(num_row/row_per_job)*(proc_endno+1)
    # []内の"#"と空白を調整する。
    bar = "#" * int(step_add) + " " * (progress_bar_range-int(step_add))

    print(bar_template.format(bar, z, time.strftime("%H:%M:%S", time.gmtime(retain_time)), process_per_time), end="")


def sub_proc( function, input_data, result_data, num_row, row_per_job, lock, val, end_val, start_time, verbose, kwargs):
    '''
    個々のcoreで走る処理
    １処理が終わる毎に残った演算対象のindexを取得して、処理を開始。

    Parameters
    ----------
    function: function
        並列化する函数
    input_data: list
        並列化for文でこのlistの値をfunctionに突っ込んでいく
    result_data: dict
        結果保存用のデータ。正確には共有メモリ（Manager.dict()）である
    num_row: int
        input_dataの長さ。要はループ回数
    row_per_job: int
        1処理毎の処理件数
    lock: Lock
        フラグ管理用
    val: int
        フラグ管理用
    end_val: int
        フラグ管理用。プログレスバー表示用
    start_time: list
        演算開始時間。残り時間計算用
    verbose: int
        表示レベル
    kwargs:
        函数（function）に引き渡すその他のデータ

    Returns
    -------
    (result_data): dict
        明示的なreturn文で戻すわけではない

    Notes
    -----
    functionの書き方は呼び出し元函数（processing_multi_core）を参照
    '''

    while True:
        lock.acquire()
        proc_no = val.value
        val.value = val.value + 1
        lock.release()

        # 開始行が総件数を超えた場合にはデータを保存して動作終了
        if proc_no*row_per_job >= num_row:
            break
            
        # 計算対象を抽出
        start_index = proc_no * row_per_job
        end_index = (proc_no+1) * row_per_job

        for i in range( len( input_data[start_index:end_index] ) ):
            result_data[start_index+i] = function(input_data[start_index+i], **kwargs)

        # progress_barの表示
        if verbose!=0:
            lock.acquire()
            # 終了したproc数
            proc_endno = end_val.value
            end_val.value = end_val.value + 1
            lock.release()

            print_progress_bar(num_row, row_per_job, proc_endno, start_time)


def processing_multi_core( function, input_data, row_per_job, retain_order=False, return_objects_exist=True,
                            n_jobs = int(0.8*cpu_count()), verbose=0, **kwargs):
    '''
    自作汎用並列化函数

    Parameters
    ----------
    function: function
        並列化する函数（隠ぺい函数は非推奨）
    input_data: list
        並列化for文でこのlistの値をfunctionに突っ込んでいく
    row_per_job: int
        1処理毎の処理件数
    retain_order: bool, default False
        入力結果の順序（行）を出力結果を維持するか否か
        なお、並び替えが発生するのでTrueにすると若干遅くなる
    return_objects_exist: bool, default True
        函数に戻り値が存在するか否か。
        ファイル出力など関数に戻り値が無い場合にはFalseにした方がコードが綺麗になる
    n_jobs: int, default 0.8*cpu_count
        並列コア数
    verbose: int, default 0
        progress barと残り時間、単位時間の平均処理件数の表示有無。(0: 表示なし、0以外:表示)
    kwargs: 
        函数(function)に引き渡すその他のデータ

    Returns
    -------
    result_data: list or dict
        functionの出力値依存。dictで返さない限りは、list型で出力。

    Notes
    -----
    並列化対象函数は二重アンダーバーの隠ぺい(__func)とすると、Winでは並列化できないので注意！
    Winの子プロセス開始メソッド"swapn"の仕様が原因。
    なお一重アンダーバー(_func)なら並列化は可能なので、隠ぺいしたい場合にはそちらで。

    並列化対象の函数は次の表記を守らないとバグる。
    function( loop値, kwargs)
    第一引数にloop値(input_dataの中身)、第二引数以降にその他の引数を書く。

    また、row_per_jobが１処理の件数だが、あまり細かすぎると一瞬で１処理が終わるのでCPUを使い切れず、
    逆に多すぎると並列化度が下がり処理の後半になると使用するcore数が減ることがある。
    '''
    
    # 注意
    if not return_objects_exist:
        logger.info("Warning: Now the multi-processed function is set not to return objects.")

    lock = Lock()
    val = Value("i", 0)
    end_val = Value("i", 0)

    num_row = len(input_data)

    # 共有メモリ：managerを用意（順序保存用にkey=index, value=出力データで保存したので辞書型）
    manager = Manager()
    result_data = manager.dict()
    # 残り時間計算用
    ## 時間計測開始
    start_time = time.time()

    # processをcore数分だけ用意
    jobs = []
    for num in range(n_jobs):
        # jobの定義
        # 函数(target)と引数(args)を指定する
        jobs.append( Process( target=sub_proc, 
                            args=(function, input_data, result_data, num_row, row_per_job, \
                                  lock, val, end_val, start_time, verbose, kwargs) ) )
    
    # 並列処理機構（ここでBrokenPipeErrorが出ることがある）
    # jobの並列処理スタート
    single_flg = 0
    try:
        for j in jobs:
            j.start()
    except BrokenPipeError:
        logger.info("INFO: Multiprocessing cannot be used in this enviroment. Now single process:%s"%function)
        single_flg = 1
        #逐次実行
        temp_data = {}
        for i in range(len(input_data)):
            temp_data.update( {i: function(input_data[i], **kwargs )} )
        result_data.update(temp_data)

    # jobの終了
    err_flg = False
    if single_flg == 0:
        for j in jobs:
            j.join()
            if j.exitcode !=0:
                err_flg = True
    
    if err_flg == True:
        logger.error("Error: multiprocessing errors")
        raise Exception("Error: multiprocessing errors")

    # 出力制御
    if return_objects_exist:
        # 出力順序の制御有無
        # ここをもう少し高速化したい
        if retain_order:
            return [ result_data[i] for i in range(len(result_data)) ]
        else:
            if type( result_data[0] )== dict:
                dict_result = {}
                for i in range(len(result_data)):
                    dict_result.update( result_data[i] )
                return dict_result
            return result_data.values()