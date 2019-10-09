from torch.autograd import Variable

import numpy as np
import time 

from .attack_utils import cal_loss, generate_target_label_tensor, pgd_attack 


def test(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    return acc



def robust_test(model, loader, args, n_batches=0, targeted=True):
    """
    n_batches (int): Number of batches for evaluation.
    """
    model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        if 'PGD' in args.attack:
            adv_x = pgd_attack(model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           args.epsilon,
                           args.eps_step,
                           args.clip_min,
                           args.clip_max, 
                           targeted)
        scores = model(x.cuda()) 
        _, preds = scores.data.max(1)
        scores_adv = model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        num_correct += (preds == y).sum()
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds)
        if n_batches > 0 and steps==n_batches:
            break
        steps += 1

    acc = float(num_correct) / num_samples
    acc_adv = float(num_correct_adv) / num_samples
    print('Clean accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
    ))
    print('Adversarial accuracy: {:.2f}% ({}/{})'.format(
        100.*acc_adv,
        num_correct_adv,
        num_samples,
    ))

    return acc, acc_adv