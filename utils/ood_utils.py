from detectors import detector_1


################# Specialized training for OOD detector #################
def ood_trainer(args, model, loss_fn, optimizer, loader_train, verbose=True):
  if args.ood_detector == 'detector_1':
    mean_loss = detector_1()

  return mean_loss

################# Specialized testing for OOD detector #################
def robust_ood_eval(args, net, loader_test, loader_ood, n_batches=10):
	return acc, acc_adv
