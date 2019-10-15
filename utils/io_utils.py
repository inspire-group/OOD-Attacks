import os

def init_dirs(args):
	model_name = args.model
	if 'wrn' in args.model:
		model_name += '-' + str(args.depth) + '-' + str(args.width)
	if args.is_adv:
		model_name += '_robust' + '_eps' + str(args.epsilon)  
	model_dir_name = args.checkpoint_path + '_' + args.dataset_in + '/' + model_name
	if not os.path.exists(model_dir_name):
		os.makedirs(model_dir_name)
	model_dir_name += '/'
	return model_dir_name
		
