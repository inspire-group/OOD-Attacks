import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import time 

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


########################################  Natural training ########################################
def train_one_epoch(model, loss_fn, optimizer, loader_train, verbose=True):
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x_var, y_var = to_var(x), to_var(y.long())
        scores = model(x_var)
        loss = loss_fn(scores, y_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if verbose:
        print('loss = %.8f' % (loss.data))


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


########################################  Adversarial training ########################################
def cal_loss(y_out, y_true, targeted):
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(y_out, y_true)
    if targeted:
        return loss_cal
    else:
        return -1*loss_cal


def generate_target_label_tensor(true_label, args):
    t = torch.floor(10*torch.rand(true_label.shape)).type(torch.int64)
    m = t == true_label
    t[m] = (t[m]+ torch.ceil(9*torch.rand(t[m].shape)).type(torch.int64)) % args.n_classes
    return t


def pgd_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        x_grad = -1 * eps_step * torch.sign(img_variable.grad.data)
        adv_temp = img_variable.data + x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad
        x_adv = torch.clamp(torch.clamp(
            x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv
    #print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor)))))
    return img_variable


def madry_train_one_epoch(model, loss_fn, optimizer, loader_train, args, targeted=True, verbose=True):
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
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
        
        scores = model(adv_x)
        loss = loss_fn(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print('loss = %.8f' % (loss.data))
        
        
def test_madry(model, loader, args, n_steps=0, targeted=True):
    """
    n_steps (int): Number of batches for evaluation.
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
        if n_steps > 0 and steps==n_steps:
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