import os

def init_dirs(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	if args.is_adv:
		model_name += '_robust' + '_eps' + str(args.epsilon) + '_k' + str(args.attack_iter) + '_delta' + str(args.eps_step)
	# if args.is_adv:
	# 	model_name += '_robust' + '_eps' + str(args.epsilon)
	if args.rand_init:
		model_name += 'rand'
	if args.n_classes != 10:
		model_name += '_cl' + str(args.n_classes)
	model_dir_name = args.checkpoint_path + '_' + args.dataset_in + '/' + model_name
	log_dir_name = 'logs_' + args.dataset_in + '/' +model_name
	if not os.path.exists(model_dir_name):
		os.makedirs(model_dir_name)
	if not os.path.exists(log_dir_name):
		os.makedirs(log_dir_name)
	model_dir_name += '/'
	log_dir_name += '/'
	return model_dir_name, log_dir_name