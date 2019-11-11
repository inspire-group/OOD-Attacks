#from detectors import detector_1

#
def get_roc_data(in_stats, ood_stats):
    '''
    in_stats (float): List/Array of in-distribution output (such as output confidence)
    ood_stats (float): List/Array of out-of-distribution output
    '''
    #ToDo: Favor custom iterator over np.percentile.
    fpr = []
    tpr = []
    for i in np.linspace(100, 0, 1000):
        th = np.percentile(tpr, i)
        fpr.append(np.sum(ood_stats > th))/len(ood_stats)
        tpr.append(i) 
    return tpr, fpr 


def get_fpr(tpr, fpr, th):
    '''
    tpr/fpr (float): List/Array of corresponding statistics
    th (int): Calculate fpr when equals to threshold th
    '''
    assert isinstance(th, int)
    assert len(tpr) == len(fpr)

    return np.mean([fpr[i] for i, _ in enumerate(tpr) if int(tpr[i]) == th])


################# Specialized training for OOD detector #################
def ood_trainer(args, model, loss_fn, optimizer, loader_train, verbose=True):
    if args.ood_detector == 'detector_1':
        mean_loss = detector_1()

    return mean_loss


################# Specialized testing for OOD detector #################
def robust_ood_eval(args, net, loader_test, loader_ood, n_batches=10):
    """
    ToDo.
    """
    if args.ood_detector == 'odin':
        p, _ = get_stats_odin(img, label, t)

        pass

    return acc, acc_adv

