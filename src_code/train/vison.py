def visulization(vis, ptype, X, Y, win_name):
    """
    vis:visdom instance
    ptype:plot type, "line", "scattor"
    X:X aixs label
    Y:Y aixs label
    win_name: name of pane
    """
    if ptype == "line":
        vis.line(
            X=np.array([X]), 
            Y=np.array([Y]), 
            win='loss_' + win_name, 
            update="append",
            opts={
                'title':win_name,
                'xlabel':'epoch',
                'ylabel':'loss', 
                } 
            )