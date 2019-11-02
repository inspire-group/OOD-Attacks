import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np


def rand_init_l2(img_variable, eps_max):
    random_vec = torch.FloatTensor(*img_variable.shape).normal_(0, 1).cuda()
    random_vec_norm = torch.max(
               random_vec.view(random_vec.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
    random_dir = random_vec/random_vec_norm.view(random_vec.size(0),1,1,1)
    random_scale = torch.FloatTensor(img_variable.size(0)).uniform_(0, eps_max).cuda()
    random_noise = random_scale.view(random_vec.size(0),1,1,1)*random_dir
    img_variable = Variable(img_variable.data + random_noise, requires_grad=True)

    return img_variable

def rand_init_linf(img_variable, eps_max):
    random_noise = torch.FloatTensor(*img_variable.shape).uniform_(-eps_max, eps_max).to(device)
    img_variable = Variable(img_variable.data + random_noise, requires_grad=True)

    return img_variable

def track_best(blosses, b_adv_x, curr_losses, curr_adv_x):
    if blosses is None:
        b_adv_x = curr_adv_x.clone().detach()
        blosses = curr_losses.clone().detach()
    else:
        replace = curr_losses < blosses
        b_adv_x[replace] = curr_adv_x[replace].clone().detach()
        blosses[replace] = curr_losses[replace]

    return blosses, b_adv_x


def cal_loss(y_out, y_true, targeted):
    losses = torch.nn.CrossEntropyLoss(reduction='none')
    losses_cal = losses(y_out, y_true)
    loss_cal = torch.mean(losses_cal)
    if targeted:
        return loss_cal, losses_cal
    else:
        return -1*loss_cal, -1*losses_cal

def generate_target_label_tensor(true_label, args):
    t = torch.floor(args.n_classes*torch.rand(true_label.shape)).type(torch.int64)
    m = t == true_label
    t[m] = (t[m]+ torch.ceil((args.n_classes-1)*torch.rand(t[m].shape)).type(torch.int64)) % args.n_classes
    return t

def pgd_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted, rand_init):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    
    best_losses = None
    best_adv_x = None

    if rand_init:
        img_variable = rand_init_linf(img_variable, eps_max)

    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted)
        best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)

        loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        x_grad = -1 * eps_step * torch.sign(img_variable.grad.data)
        adv_temp = img_variable.data + x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad
        x_adv = torch.clamp(torch.clamp(
            x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv

    best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)
    #print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor)))))
    return best_adv_x

def pgd_l2_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted, rand_init):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    best_losses = None
    best_adv_x = None

    if rand_init:
        img_variable = rand_init_l2(img_variable, eps_max)

    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted)
        best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)
        loss_cal.backward()
        raw_grad = img_variable.grad.data
        grad_norm = torch.max(
               raw_grad.view(raw_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
        grad_dir = raw_grad/grad_norm.view(raw_grad.size(0),1,1,1)
        adv_temp = img_variable.data +  -1 * eps_step * grad_dir
        # Clipping total perturbation
        total_grad = adv_temp - image_tensor
        total_grad_norm = torch.max(
               total_grad.view(total_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
        total_grad_dir = total_grad/total_grad_norm.view(total_grad.size(0),1,1,1)
        total_grad_norm_rescale = torch.min(total_grad_norm, torch.tensor(eps_max).cuda())
        clipped_grad = total_grad_norm_rescale.view(total_grad.size(0),1,1,1) * total_grad_dir
        x_adv = image_tensor + clipped_grad
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
        img_variable.data = x_adv

    best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)

    diff_array = np.array(x_adv.cpu())-np.array(image_tensor.data.cpu())
    diff_array = diff_array.reshape(len(diff_array),-1)
    # print("peturbation= {}".format(
    #    np.max(np.linalg.norm(diff_array,axis=1))))
    return best_adv_x