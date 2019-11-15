import torch
import torch.nn as nn
import numpy as np


def get_stats_odin(img, model, args):
	with torch.no_grad():
		input_var = torch.autograd.Variable(img).cuda()
		out = nn.Softmax(dim=-1)(model(input_var)/args.temp)
		p, index = torch.max(out, dim=-1)
	return p.data.cpu().numpy(), index.data.cpu().numpy()


################# Specialized training for OOD detector #################
def ood_trainer(args, model, loss_fn, optimizer, loader_train, verbose=True):
  if args.ood_detector == 'detector_1':
    mean_loss = detector_1()

  return mean_loss

################# Specialized testing for OOD detector #################
def robust_ood_eval(args, net, loader_test, loader_ood, n_batches=10, adv_ood=None):
	fpr = []
	if args.ood_detector == 'odin':
		# collect benign stats
	    in_stats = []
	    for i, (img, label) in enumerate(loader_test):
	        p, _ = get_stats_odin(img, net, args)
	        in_stats += list(p)
	        if i == 20:
	            break

	    ood_stats = []
	    for i, (img, label) in enumerate(loader_ood):
	        p, _ = get_stats_odin(img, net, args)
	        ood_stats += list(p)

	    ood_adv_stats = []
	    for i, (img, label) in enumerate(adv_ood):
	        p, _ = get_stats_odin(img, net, args)
	        ood_adv_stats += list(p)
	    
	    for i in [0,5,10]:
	        th = np.percentile(in_stats, i)
	        print("for TPR = {}, OOD FPR = {}".format(100-i, np.sum(ood_stats > th)/len(ood_stats)))
	        print("for TPR = {}, OOD Adv FPR = {}".format(100-i, np.sum(ood_adv_stats > th)/len(ood_adv_stats)))
	        # fpr.append(np.sum(ood_stats > th)/len(ood_stats))

	# print(fpr)
	return fpr
