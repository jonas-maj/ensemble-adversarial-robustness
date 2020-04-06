from convex_adversarial.dual_network import robust_loss, RobustBounds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time
import copy
import os

DEBUG = False

## standard training
def train_baseline(loader, model, opt, epoch, log1, log2, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()
    print('==================== training ====================')

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        print(epoch, i, '{0:.4f}'.format(err.item()), '{0:.4f}'.format(ce.item()), file=log1)
        if verbose and i % verbose == 0: 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.4f} ({errors.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors))
        log1.flush()

    print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(losses.avg), file=log2)
    log2.flush()  

def evaluate_baseline(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()
    print('==================== validating ====================')

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()
    
    print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(losses.avg), file=log)
    log.flush()  
    print(' * Error: {error.avg:.2%}'.format(error=errors))
    return errors.avg

## robust training for overall robustness
def train_robust(loader, model, opt, epsilon, epoch, log1, log2, verbose, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.train()
    print('==================== training ====================')
    print('epsilon:', '{:.4f}'.format(epsilon))
    
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        # data_time.update(time.time() - end)

        with torch.no_grad(): 
            ce = nn.CrossEntropyLoss()(model(X), y).item()
            err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        opt.zero_grad()
        robust_ce.backward()

        if clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose and i % verbose == 0: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.4f} ({rerrors.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.4f} ({errors.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors, rloss = robust_losses, 
                   rerrors = robust_errors), end=endline)
        print(epoch, i, '{0:.4f}'.format(err), '{0:.4f}'.format(robust_err),
             '{0:.4f}'.format(ce), '{0:.4f}'.format(robust_ce.detach().item()), file=log1)
        log1.flush()

        del X, y, robust_ce, ce, err, robust_err

        if DEBUG and i == 10: 
            break

    print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(robust_errors.avg), 
            '{:.4f}'.format(robust_losses.avg), file=log2)
    log2.flush()  
    torch.cuda.empty_cache()

def evaluate_robust(loader, model, epsilon, epoch, log, verbose, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()
    print('==================== validating ====================')
    print('epsilon:', '{:.4f}'.format(epsilon))
    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        ce = nn.CrossEntropyLoss()(model(X), y).item()
        err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.4f} ({rerrors.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                      i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        
        del X, y, robust_ce, ce, err, robust_err

        if DEBUG and i == 10: 
            break
            
    print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(robust_errors.avg), 
            '{0:.4f}'.format(robust_losses.avg), file=log)
    log.flush()
    print('')
    print(' * Error: {error.avg:.2%}\n'
          ' * Robust error: {rerror.avg:.2%}'.format(
              error=errors, rerror=robust_errors))
    
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    return errors.avg, robust_errors.avg

## joint robust training for overall robustness
def train_joint_robust(loader, model1, model2, opt1, opt2, epsilon, epoch, log1, log2, verbose, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    errors1 = AverageMeter()
    errors2 = AverageMeter()
    robust_losses1 = AverageMeter()
    robust_losses2 = AverageMeter()
    robust_errors1 = AverageMeter()
    robust_errors2 = AverageMeter()

    model1.train()
    model2.train()
    print('==================== training ====================')
    print('epsilon:', '{:.4f}'.format(epsilon))
    
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        # data_time.update(time.time() - end)

        ce1 = nn.CrossEntropyLoss()(model1(X), y).item()
        ce2 = nn.CrossEntropyLoss()(model2(X), y).item()
        with torch.no_grad(): 
            err1 = (model1(X).max(1)[1] != y).float().sum().item() / X.size(0)
            err2 = (model2(X).max(1)[1] != y).float().sum().item() / X.size(0)

        robust_ce1, robust_err1 = robust_loss(model1, epsilon, X, y, **kwargs)
        robust_ce2, robust_err2 = robust_loss(model2, epsilon, X, y, **kwargs)
        robust_ce = robust_ce1 + robust_ce2

        opt1.zero_grad()
        opt2.zero_grad()
        robust_ce.backward()

        if clip_grad: 
            nn.utils.clip_grad_norm_(model1.parameters(), clip_grad)
            nn.utils.clip_grad_norm_(model2.parameters(), clip_grad)

        opt1.step()
        opt2.step()

        # measure accuracy and record loss
        losses1.update(ce1, X.size(0))
        losses2.update(ce2, X.size(0))
        errors1.update(err1, X.size(0))
        errors2.update(err2, X.size(0))
        robust_losses1.update(robust_ce1.detach().item(), X.size(0))
        robust_losses2.update(robust_ce2.detach().item(), X.size(0))
        robust_errors1.update(robust_err1, X.size(0))
        robust_errors2.update(robust_err2, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose and i % verbose == 0: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.4f} ({rerrors.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.4f} ({errors.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses1, errors=errors1, rloss = robust_losses1, 
                   rerrors = robust_errors1), end=endline)
        print(epoch, i, '{0:.4f}'.format(err1), '{0:.4f}'.format(robust_err1),
             '{0:.4f}'.format(ce1), '{0:.4f}'.format(robust_ce1.detach().item()), file=log1)
        log1.flush()

        del X, y, robust_ce1, robust_ce2, ce1, ce2, err1, err2, robust_err1, robust_err2

        if DEBUG and i == 10: 
            break

    print(epoch, '{:.4f}'.format(errors1.avg), '{:.4f}'.format(robust_errors1.avg), 
            '{:.4f}'.format(robust_losses1.avg), file=log2)
    log2.flush()  
    torch.cuda.empty_cache()

## robsut training for cost-sensitive robustness
def train_robust_task_spec(loader, model, opt, epsilon, epoch, log1, log2, verbose, 
                           input_mat, mat_type, alpha, clip_grad=None, **kwargs):
    model.train()
    print('==================== training ====================')
    print('epsilon:', '{:.4f}'.format(epsilon)) 
    batch_time = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    cost_adv_exps_total = 0
    num_exps_total = 0

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        clas_err, robust_ce, cost_adv_exps, num_exps = robust_loss_task_spec(model, epsilon, 
                                                                    Variable(X), Variable(y), 
                                                                    input_mat, mat_type, 
                                                                    alpha, **kwargs)
        opt.zero_grad()
        robust_ce.backward()
        if clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()

        if num_exps != 0:
            robust_cost = cost_adv_exps/num_exps
        else:
            robust_cost = 0.0 

        # measure accuracy and record loss
        errors.update(clas_err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        cost_adv_exps_total += cost_adv_exps
        num_exps_total += num_exps

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, '{:.4f}'.format(clas_err), '{:.4f}'.format(robust_cost), 
                    '{:.4f}'.format(robust_ce.item()), file=log1)

        if verbose and i % verbose == 0: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})\t'
                  'Robust cost {rcost:.4f}\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time, error=errors, 
                   rcost = robust_cost, rloss = robust_losses), end=endline)
        log1.flush()
        del X, y, robust_ce, clas_err, cost_adv_exps, num_exps, robust_cost

        if DEBUG and i ==10: 
            break 

    if num_exps_total != 0:
        robust_cost_avg = cost_adv_exps_total/num_exps_total
    else:
        robust_cost_avg = 0.0    

    print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(robust_cost_avg), 
            '{:.4f}'.format(robust_losses.avg), file=log2)
    log2.flush()  
    torch.cuda.empty_cache()

def evaluate_robust_task_spec(loader, model, epsilon, epoch, log, verbose,
                              input_mat, mat_type, alpha, **kwargs):
    model.eval()
    print('==================== validating ====================')
    print('epsilon:', '{:.4f}'.format(epsilon))
    batch_time = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    cost_adv_exps_total = 0  
    num_exps_total = 0

    end = time.time()
    torch.set_grad_enabled(False)
    tic = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        clas_err, robust_ce, cost_adv_exps, num_exps = robust_loss_task_spec(model, epsilon, X, y, 
                                                                        input_mat, mat_type, alpha, **kwargs)
        
        if num_exps != 0:
            robust_cost = cost_adv_exps/num_exps
        else:
            robust_cost = 0.0 

        # measure accuracy and record loss
        errors.update(clas_err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        cost_adv_exps_total += cost_adv_exps
        num_exps_total += num_exps

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})\t'
                  'Robust cost {rcost:.4f}\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})'.format(
                      i, len(loader), batch_time=batch_time, error=errors, 
                      rcost=robust_cost, rloss=robust_losses), end=endline)
        del X, y, robust_ce, clas_err, cost_adv_exps, num_exps, robust_cost

        if DEBUG and i ==10: 
            break

    if num_exps_total != 0:     # for binary case, same as the portion of cost-sensitive adv exps
        robust_cost_avg = cost_adv_exps_total/num_exps_total
    else:
        robust_cost_avg = 0.0

    print('')
    if mat_type == 'binary':
        print(' * Classification error: {error.avg:.2%}\n'
              ' * Cost-sensitive robust error: {rerror:.2%}'.format(
              error=errors, rerror=robust_cost_avg))
        print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(robust_cost_avg), 
                '{0:.4f}'.format(robust_losses.avg), file=log)
    else:
        print(' * Classication Error: {error.avg:.2%}\n'
              ' * Average cost: {rcost:.3f}'.format(
                error=errors, rcost=robust_cost_avg))
        print(epoch, '{:.4f}'.format(errors.avg), '{:.4f}'.format(robust_cost_avg), 
                '{0:.4f}'.format(robust_losses.avg), file=log)
    log.flush()
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    return errors.avg, robust_cost_avg

## compute the pairwise test robust error w.r.t. a given classifier
def evaluate_test_clas_spec(loader, model, epsilon, path, verbose, **kwargs): 
    print('==================== testing ====================')
    model.eval()

    num_classes = model[-1].out_features
    # define the class-specific error matrices for aggregation
    clas_err_mats = torch.FloatTensor(num_classes+1, num_classes+1).zero_().cuda()
    robust_err_mats = torch.FloatTensor(num_classes+1, num_classes+1).zero_().cuda()
    num_exps_vecs = torch.FloatTensor(num_classes+1).zero_().cuda()

    torch.set_grad_enabled(False)
    # aggregate the error matrices for the whole dataloader
    for i, (X,y) in enumerate(loader):

        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        clas_err_mat, robust_err_mat, \
                    num_exps_vec = calc_err_clas_spec(model, epsilon, X, y, **kwargs)

        clas_err_mats += clas_err_mat
        robust_err_mats += robust_err_mat
        num_exps_vecs += num_exps_vec

        if verbose: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]'.format(i, len(loader)), end=endline)

        del X, y, clas_err_mat, robust_err_mat, num_exps_vec

    clas_err_mats = clas_err_mats.cpu().numpy().astype(int)
    robust_err_mats = robust_err_mats.cpu().numpy().astype(int)
    num_exps_vecs = num_exps_vecs.cpu().numpy()
    clas_err_overall = clas_err_mats[-1, -1] / num_exps_vecs[-1]
    robust_err_overall = robust_err_mats[-1, -1] / num_exps_vecs[-1]
    print('overall classification error: ', '{:.2%}'.format(clas_err_overall)) 
    print('overall robust error: ', '{:.2%}'.format(robust_err_overall))

    # compute the robust error probabilities for each pair of class
    robust_prob_mats = np.zeros((num_classes+1, num_classes+1))
    for i in range(num_classes):
        class_size_col = copy.deepcopy(num_exps_vecs)
        class_size_col[num_classes] -= num_exps_vecs[i]
        robust_prob_mats[:,i] = robust_err_mats[:,i]/class_size_col		
    robust_prob_mats[:,num_classes] = robust_err_mats[:,num_classes]/num_exps_vecs

    print('')
    print('pairwise robust test error:')
    print(robust_prob_mats)
    print(robust_err_mats)
    print('')
    print('overall classification error: ', '{:.2%}'.format(clas_err_overall)) 
    print('overall robust error: ', '{:.2%}'.format(robust_err_overall))
    return

    clas_err_mats.to_csv(path+'_clasErrs.csv', sep='\t')
    robust_err_mats.to_csv(path+'_robustErrs.csv', sep='\t')
    robust_prob_mats.to_csv(path+'_robustProbs.csv', sep='\t')

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

## compute the overall cost-weighted error on test dataset
def evaluate_test(loader, model, epsilon, input_mat, mat_type, verbose, **kwargs):
    model.eval()
    # print('==================== testing ====================')

    errors = AverageMeter()
    cost_adv_exps_total = 0
    num_exps_total = 0
    
    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)         

        clas_err, robust_ce, cost_adv_exps, num_exps = robust_loss_task_spec(model, epsilon, X, y, 
                                                        input_mat, mat_type, **kwargs)

        if num_exps != 0:
            robust_cost = cost_adv_exps/num_exps
        else:
            robust_cost = 0.0 

        errors.update(clas_err, X.size(0))
        cost_adv_exps_total += cost_adv_exps
        num_exps_total += num_exps

        if verbose == len(loader):
            print('Test: [{0}/{1}]\t\t'
                  'Robust cost {rcost:.4f}\t\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                    i, len(loader), error=errors, rcost=robust_cost), end='\r') 
        elif verbose:
            endline = '\n' if i % verbose == 0 else '\r'
            print('Test: [{0}/{1}]\t\t'
                  'Robust error {rcost:.4f}\t\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                    i, len(loader), error=errors, 
                    rcost=robust_cost), end=endline)

        del X, y, robust_ce, clas_err, cost_adv_exps, num_exps, robust_cost
       
    if num_exps_total != 0:
        robust_cost_avg = cost_adv_exps_total/num_exps_total
    else:
        robust_cost_avg = 0.0

    print('')
    if mat_type == 'binary':
        print(' * Classification error: {error.avg:.2%}\n'
              ' * Cost-sensitive robust error: {rerror:.2%}'.format(
                  error=errors, rerror=robust_cost_avg))   
    else:   # real-valued
        print(' * Classification error: {error.avg:.2%}\n'
              ' * Average cost: {rcost:.3f}'.format(
                  error=errors, rcost=robust_cost_avg))                  
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    return errors.avg, robust_cost_avg


## define the metric and training loss function for cost-sensitive robustness
def robust_loss_task_spec(net, epsilon, X, y, input_mat, mat_type, alpha=1.0, **kwargs):
    num_classes = net[-1].out_features
    # loss function for standard classification
    out = net(X)
    clas_err = (out.max(1)[1] != y).float().sum().item() / X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduction='elementwise_mean')(out, y)

    # regularization term for cost-sensitive robustness
    cost_adv_exps = 0.0    
    num_exps = 0

    for k in range(num_classes):
        if np.all(input_mat[k, :] == 0):
            continue    
        else:
            targ_clas = np.nonzero(input_mat[k, :])[0]    # extract the corresponding output classes
            ind = (y == k).nonzero()   # extract the considered input example indices   

            if len(ind) != 0:
                ind = ind.squeeze(1)
                X_sub = X[ind, ...]
                y_sub = y[ind, ...]

                # robust score matrix
                f = RobustBounds(net, epsilon, **kwargs)(X_sub,y_sub)[:,targ_clas]
                zero_col = torch.FloatTensor(np.zeros(len(ind), dtype=float)).cuda()
                weight_vec = torch.FloatTensor(input_mat[k, targ_clas]).repeat(len(ind),1).cuda() 

                # cost-weighted robust score matrix
                f_weighted = torch.cat((f + torch.log(weight_vec), zero_col.unsqueeze(1)), dim=1)
                target = torch.LongTensor(len(targ_clas)*np.ones(len(ind), dtype=float)).cuda()
                # aggregate the training loss function (including the robust regularizer)
                ce_loss = ce_loss + alpha*nn.CrossEntropyLoss(reduction='elementwise_mean')(f_weighted, target)

                zero_tensor = torch.FloatTensor(np.zeros(f.size())).cuda()
                err_mat = (f > zero_tensor).cpu().numpy()

                if mat_type == 'binary':    # same as the number of cost-sensitive adversarial exps
                    cost_adv_exps += err_mat.max(1).sum().item()               
                else:   # real-valued case
                    # use the total costs as the measure
                    cost_adv_exps += np.dot(np.sum(err_mat, axis=0), input_mat[k,targ_clas])
                num_exps += len(ind)
    return clas_err, ce_loss, cost_adv_exps, num_exps


## compute the pairwise classification and robust error
def calc_err_clas_spec(net, epsilon, X, y, **kwargs):
    num_classes = net[-1].out_features
    targ_clas = range(num_classes)
    zero_mat = torch.FloatTensor(X.size(0), num_classes).zero_()

    # aggregate the class-specific classification and robust error counts
    clas_err_mat = torch.FloatTensor(num_classes+1, num_classes+1).zero_()
    robust_err_mat = torch.FloatTensor(num_classes+1, num_classes+1).zero_()
    # aggregate the number of examples for each class
    num_exps_vec = torch.FloatTensor(num_classes+1).zero_()
        
    if X.is_cuda:
        zero_mat = zero_mat.cuda()
        clas_err_mat = clas_err_mat.cuda()
        robust_err_mat = robust_err_mat.cuda()
        num_exps_vec = num_exps_vec.cuda()

    # compute the class-specific classification error matrix
    val, idx = torch.max(net(X), dim=1)
    for j in range(len(y)):
        row_ind = y[j]
        col_ind = idx[j].item()
        if row_ind != col_ind:
            clas_err_mat[row_ind, col_ind] += 1
            clas_err_mat[row_ind, num_classes] += 1
    clas_err_mat[num_classes, ] = torch.sum(clas_err_mat[:num_classes, ], dim=0)

    f = RobustBounds(net, epsilon, **kwargs)(X,y)[:,targ_clas]
    # robust error counts for each example
    err_mat = (f > zero_mat).float()    # class-specific robust error counts
    err = (f.max(1)[1] != y).float()    # overall robust error counts

    # compute pairwise robust error matrix
    for i in range(num_classes):
        ind = (y == i).nonzero()    # indices of examples in class i
        if len(ind) != 0:
            ind = ind.squeeze(1)  
            robust_err_mat[i, :num_classes] += torch.sum(err_mat[ind, ].squeeze(1), dim=0)
            robust_err_mat[i, num_classes] += torch.sum(err[ind])
        num_exps_vec[i] += len(ind)

    # compute the weighted average for each target class 
    robust_err_mat[num_classes, ] = torch.sum(robust_err_mat[:num_classes, ], dim=0)
    num_exps_vec[num_classes] = torch.sum(num_exps_vec[:num_classes])
    return clas_err_mat, robust_err_mat, num_exps_vec


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def merge_models2(nn1, nn2, nn3, n_layers):
    c = 0
    for i, layer in enumerate(nn3.modules()):
        if isinstance(layer, nn.Linear):
            if c < n_layers:
                layer1 = list(nn1.modules())[i]
                layer2 = list(nn2.modules())[i]
            layer3 = list(nn3.modules())[i]

            if c == 0:
                size_w = layer3.weight.data.size()[0] // 2;
                size_b = layer3.bias.data.size()[0] // 2;
                layer3.weight.data[:size_w , :] = layer1.weight.data[:, :].cuda()
                layer3.weight.data[size_w: , :] = layer2.weight.data[:, :].cuda()

                layer3.bias.data[:size_b] = layer1.bias.data[:].cuda()
                layer3.bias.data[size_b:] = layer2.bias.data[:].cuda()

                layer3.weight.data.cuda()
                layer3.bias.data.cuda()
            elif c == n_layers:
                size_w = layer3.weight.data.size()[0] // 2;
                size_b = layer3.bias.data.size()[0] // 2;
                out_size = layer3.weight.data.size()[1]
                diag = torch.diag(0.5 * torch.ones(out_size // 2).cuda()).cuda()
                layer3.weight.data = torch.cat((diag, diag), 1).cuda()
                layer3.bias.data = torch.zeros(out_size // 2).cuda()
                layer3.weight.data.cuda()
                layer3.bias.data.cuda()
            else:
                size_w0 = layer3.weight.data.size()[0] // 2;
                size_w1 = layer3.weight.data.size()[1] // 2;
                size_b = layer3.bias.data.size()[0] // 2;

                layer3.weight.data[:size_w0, :size_w1] = layer1.weight.data[:, :].cuda()
                layer3.weight.data[size_w0:, size_w1:] = layer2.weight.data[:, :].cuda()
                layer3.weight.data[:size_w0, size_w1:] = torch.zeros(size_w0, size_w1).cuda()
                layer3.weight.data[size_w0:, :size_w1] = torch.zeros(size_w0, size_w1).cuda()

                layer3.bias.data[:size_b] = layer1.bias.data[:].cuda()
                layer3.bias.data[size_b:] = layer2.bias.data[:].cuda()

                layer3.weight.data.cuda()
                layer3.bias.data.cuda()
            c += 1 
    return nn3
