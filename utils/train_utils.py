from torch.autograd import Variable

import numpy as np
import time 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack, pgd_l2_attack

def update_hyparam(epoch, args):
    #args.learning_rate = args.learning_rate * (0.6 ** ((max((epoch-args.schedule_length), 0) // 5)))
    lr_steps = [100, 150, 200]
    lr = args.learning_rate
    for i in lr_steps:
        if epoch<i:
            break
        lr /= 10
    return lr 

# Checking which training to perform
def train_one_epoch_all(args, model, loss_fn, optimizer, loader_train, verbose=True):
  if not args.ood_training:
    if args.is_adv:
      mean_loss = robust_train_one_epoch(args, model, loss_fn, optimizer, loader_train, verbose)
    else:
      mean_loss = train_one_epoch(model, loss_fn, optimizer, loader_train, verbose)
  else:
    mean_loss = ood_trainer(args, model, loss_fn, optimizer, loader_train)
  return mean_loss


########################################  Natural training ########################################
def train_one_epoch(model, loss_fn, optimizer, loader_train, verbose=True):
    losses = []
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        scores = model(x_var)
        loss = loss_fn(scores, y_var)
        losses.append(loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if verbose:
        print('loss = %.8f' % (loss.data))
    return np.mean(losses)      


########################################  Adversarial training ########################################
def robust_train_one_epoch(args, model, loss_fn, optimizer, loader_train, verbose=True):
    losses = []
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if args.targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        if 'PGD_linf' in args.attack:
            adv_x = pgd_attack(model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           args.epsilon,
                           args.eps_step,
                           args.clip_min,
                           args.clip_max, 
                           args.targeted,
                           args.rand_init)
        elif 'PGD_l2' in args.attack:
            adv_x = pgd_l2_attack(model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           args.epsilon,
                           args.eps_step,
                           args.clip_min,
                           args.clip_max, 
                           args.targeted,
                           args.rand_init)
        
        scores = model(adv_x)
        loss = loss_fn(scores, y)
        losses.append(loss.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print('loss = %.8f' % (loss.data))
    return np.mean(losses)