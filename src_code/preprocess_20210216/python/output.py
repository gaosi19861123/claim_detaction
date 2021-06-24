import pickle
import math

def output_csv(df, path_prefix, dict_config):
    '''
    データを分割してcsv出力

    Parameters
    ----------
    df : pandas.DataFrame
        出力データ
    path_prefix : str
        出力パス末尾を除いた部分
    dict_config
        config.jsonでの設定値
    '''

    n_lines = dict_config["output"]["other_infos"]["csv_max_lines"]
    n_files = math.ceil(len(df)/n_lines)
    
    print("\n 出力中：", path_prefix+"_*.csv (", n_files, "個のファイル)\n")

    print(df.head())

    for i in range(n_files):
        df[n_lines*i:min(n_lines*(i+1),len(df))].to_csv(path_prefix+"_%i"%i+".csv", index=False)
    

def output_pickle(obj, outpath):
    '''
    オブジェクトをpickleファイルとして出力

    Parameters
    ----------
    obj : 
        出力したいオブジェクト
    outpath : str
        出力パス
    '''

    print("出力中:", outpath)
    pickle.dump(obj, open(outpath, "wb"))
        

def output(df_ts_train, df_ts_test, dict_config):
    '''
    前処理済みデータをpickle形式で出力

    Parameters
    ----------
    df_ts_train : pandas.DataFrame
        学習用時系列データ(claim_flagなどのサマリーデータも含む)
    df_ts_test : pandas.DataFrame
        検証用時系列データ(claim_flagなどのサマリーデータも含む)
    dict_config : dict
        config設定値

    Outputs
    -------
    train_X.pkl : numpy.array
        学習用時系列データ加速度
    test_X.pkl : numpy.array
        検証用時系列データ加速度
    train_y.pkl : numpy.array
        学習用事故ラベル
    test_y.pkl : numpy.array
        検証用事故ラベル
    train_ts.pkl : numpy.array
        学習用時系列データタイムスタンプ
    test_ts.pkl : numpy.array
        検証用時系列データタイムスタンプ
    train_others.pkl : numpy.array
        学習用データの事故単位での情報
    test_others.pkl : numpy.array
        検証用データの事故単位での情報    
    '''
    
    print("学習用データレコード数：", len(df_ts_train))
    print("検証用データレコード数：", len(df_ts_test))

    # numpy化に備えてソート
    df_ts_train_sorted = df_ts_train.sort_values(["id", "timestamp"])
    df_ts_train_sorted.reset_index(drop=True, inplace=True)

    df_ts_test_sorted = df_ts_test.sort_values(["id", "timestamp"])
    df_ts_test_sorted.reset_index(drop=True, inplace=True)

    # 元のデータフレーム削除
    del df_ts_train, df_ts_test

    # 加速度データ
    ## 加速度列名リスト作成
    cols_mat=[]
    if "acceleration_x" in df_ts_train_sorted.columns:
        cols_mat.append("acceleration_x")
    if "acceleration_y" in df_ts_train_sorted.columns:
        cols_mat.append("acceleration_y")
    if "acceleration_z" in df_ts_train_sorted.columns:
        cols_mat.append("acceleration_z")
    print("加速度列名リスト\n", cols_mat, "\n")

    ## 加速度のみをデータフレームから抽出
    mat_train = df_ts_train_sorted[cols_mat].values
    mat_test = df_ts_test_sorted[cols_mat].values

    ## １事故当たりの時系列データ数
    n_each_crash = dict_config["data_condition"]["n_data_each_crash"]

    ## 出力ディレクトリ
    dir_train = dict_config["output"]["folder_path"]["train_data"]
    dir_test = dict_config["output"]["folder_path"]["test_data"]    

    ## 事故ごとに区切るためにndarrayを変形
    ## 変形後のshape: (事故数, 時系列データ数, 加速度軸数)
    train_X = mat_train.reshape(-1, n_each_crash, len(cols_mat))
    test_X = mat_test.reshape(-1, n_each_crash, len(cols_mat))
    print("学習用加速度データndarrayのshape= ", train_X.shape)
    print("検証用加速度データndarrayのshape= ", test_X.shape)

    ## pickle出力
    output_pickle(train_X, dir_train+"/train_X.pkl")
    output_pickle(test_X, dir_test+"/test_X.pkl")
    print()


    # 事故ラベル
    if "claim_flag" in df_ts_train_sorted.columns:
        ## ndarray作成
        ## shape: (事故数, 1)
        train_y = df_ts_train_sorted[["claim_flag"]].values.reshape(-1, n_each_crash, 1).astype(int).max(axis=1)
        test_y = df_ts_test_sorted[["claim_flag"]].values.reshape(-1, n_each_crash, 1).astype(int).max(axis=1)
        print("学習用事故ラベルndarrayのshape= ", train_y.shape)
        print("検証用事故ラベルndarrayのshape= ", test_y.shape)

        ## pickle出力
        output_pickle(train_y, dir_train+"/train_y.pkl")
        output_pickle(test_y, dir_test+"/test_y.pkl")
        print()


    # 時系列データタイムスタンプ
    ## ndarray作成
    ## shape: (事故数, 時系列データ数, 1)
    train_ts = df_ts_train_sorted[["timestamp"]].values.reshape(-1,n_each_crash,1)
    test_ts = df_ts_test_sorted[["timestamp"]].values.reshape(-1,n_each_crash,1)
    print("学習用時系列タイムスタンプndarrayのshape= ", train_ts.shape)
    print("検証用時系列タイムスタンプndarrayのshape= ", test_ts.shape)

    ## pickle出力    
    output_pickle(train_ts, dir_train+"/train_ts.pkl")
    output_pickle(test_ts, dir_test+"/test_ts.pkl")
    print()

    # 事故IDなどの事故単位情報
    if dict_config["input"]["summary"]["read_file"]:
        ## その他情報列リスト
        cols = ["id"]+dict_config["input"]["summary"]["column_types"]
        dict_config["input"]["summary"]["column_types"]
        print("その他情報列リスト", cols, "\n")

        if dict_config["output"]["other_infos"]["file_type"]=="csv":
            # csv出力
            output_csv(df_ts_train_sorted[cols].drop_duplicates(), dir_train+"/train_others", dict_config)
            output_csv(df_ts_test_sorted[cols].drop_duplicates(), dir_test+"/test_others", dict_config)

        elif dict_config["output"]["other_infos"]["file_type"]=="pickle":
            # pickle出力

            ## ndarray作成
            ## shape: (事故数, 情報の種類の数)
            ## 情報の種類の数 = 1(id) + config.jsonの"summary"/"column_types"の要素数
            train_others = df_ts_train_sorted[cols].values.reshape(-1,n_each_crash,len(cols))[:,0,:]
            test_others = df_ts_test_sorted[cols].values.reshape(-1,n_each_crash,len(cols))[:,0,:]
            print("学習用事故単位情報ndarrayのshape= ", train_others.shape)
            print("検証用事故単位情報ndarrayのshape= ", test_others.shape)

            output_pickle(train_others, dir_train+"/train_others.pkl")
            output_pickle(test_others, dir_train+"/test_others.pkl")
        
        else:
            print("config.jsonでoutput/other_infos/file_typeの値が正しく設定されていません")
            sys.exit()
