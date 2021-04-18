import sys
import os

import torch
import numpy as np
import time
import torch.onnx
from torch.autograd import Variable

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))

import logging
logger = logging.getLogger('mainlogger')


class classification_script(object):
    """docstring for classification_script"""
    def __init__(self, model, loss_function, train_loader, eval_loader,
                 optimizer, lr_scheduler, device='cpu'):
        super(classification_script, self).__init__()
        self.model = model
        self.criterion = loss_function

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # print('model.id',id(model))
        # print('self.model.id',id(self.model))
        # print('optimizer.id',id(optimizer))
        # print('self.optimizer.id',id(self.optimizer))

        self.device = device
        self.accumulate = 1 # work for multi batch then backforward
        self.best_result = 0

        self.activation_step = {'print': 50, 'save': 500, 'eval': 500}

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        self.state = 'train'
        self.set_train()
        self._show_img = False
        self.eval_class_list = [1, 2]
        self.only_encoder = False

    def train(self, epoch):


        self.epoch = epoch
        self.reset_Metets()
        self.set_train()

        end = time.time()
        # loss = torch.tensor(0.).to(self.device)

        for i, (inputs, target) in enumerate(self.train_loader):
            self.data_time.update(time.time() - end)

            # target = target.to(self.device)
            # input_var = torch.autograd.Variable(inputs).to(self.device)
            # target_var = torch.autograd.Variable(target).to(self.device)

            target = target.long().to(self.device)
            input_var = inputs.to(self.device)
            target_var = target.to(self.device)

            output = self.model(input_var)

            loss = self.criterion(output, target_var.long())

            self.losses.update(loss.data.item(), inputs.size(0))

            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            self.top1.update(prec1.item(), inputs.size(0))
            self.top5.update(prec5.item(), inputs.size(0))


            # if i%self.accumulate==0:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # loss = torch.tensor(0.).to(self.device)

            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % self.activation_step['print'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.epoch, i, len(self.train_loader), len(self.train_loader),
                                                                      batch_time=self.batch_time, data_time=self.data_time,
                                                                      loss=self.losses, top1=self.top1, top5=self.top5))


            if i % self.activation_step['save'] == 0 and i!=0:
                self.save_model('./runs/classification/mnist/experiment0_last.pth')

            # log
            if i % self.activation_step['eval'] == 0 and i!=0:
                self.set_eval()
                # self.eval()
                top1 = self.eval()
                if top1 > self.best_result:
                    self.save_model('./runs/classification/mnist/experiment0_best.pth')
                self.set_train()
        self.lr_scheduler.step()
        print(self.lr_scheduler.last_epoch)

    def eval(self):
        _batch_time = AverageMeter()
        _data_time = AverageMeter()
        _losses = AverageMeter()
        _top1 = AverageMeter()
        _top5 = AverageMeter()

        # switch to evaluate mode
        self.set_eval()

        print(' **start eval')

        end = time.time()
        for i, (inputs, target) in enumerate(self.eval_loader):
            _data_time.update(time.time() - end)
            target = target.long().to(self.device)

            input_var = torch.autograd.Variable(inputs).to(self.device)
            target_var = torch.autograd.Variable(target).to(self.device)

            # compute output
            output = self.model(input_var)

            if self._show_img is True:
                pass
                # inputs_out = input_var.detach().cpu().numpy()
                # outputs_out = output.detach().cpu().numpy()
                # target_out = target_var.detach().cpu().numpy()

            # target_var = target_var.argmax(dim=1)
            # print('output.shape', output.shape)
            # print('target_var.shape', target_var.shape)
            loss = self.criterion(output, target_var)

            _losses.update(loss.data.item(), inputs.size(0))

            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            _top1.update(prec1.data.item(), inputs.size(0))
            _top5.update(prec5.data.item(), inputs.size(0))

            # # measure elapsed time
            _batch_time.update(time.time() - end)
            end = time.time()

        print('Eval result on {} datas:\t'
              'Time {batch_time.avg:.3f}\t'
              'Data {data_time.avg:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(len(self.eval_loader),
                                                              batch_time=_batch_time,
                                                              data_time=_data_time,
                                                              loss=_losses,
                                                              class_list=self.eval_class_list,
                                                              top1=_top1,
                                                              top5=_top5,
                                                              ))

        return _top1.avg

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        # self.model = load_my_state_dict(self.model, torch.load(path))

    def set_train(self):
        self.model.train()
        self.state = 'train'

    def set_eval(self):
        self.model.eval()
        self.state = 'eval'

    def _to_device(self):
        self.model.to(self.device)

    def reset_Metets(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def export_onnx(self, path):
        virtual_input = Variable(torch.randn(1, 3, 28, 28), requires_grad=True).to(self.device)
        torch.onnx.export(self.model, virtual_input, path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    values, index = output.topk(maxk, dim=1, largest=True, sorted=True)

    index = index.t().long()
    correct = index.eq(target.view(1, -1).expand_as(index))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


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


def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
    own_state = model.state_dict()
    print(own_state.keys())
    print('=='*15)
    print(state_dict['state_dict'].keys())
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    for name, param in state_dict.items():

        name_element = name.split('.')
        # print(name_element)
        name = '.'.join(name_element[1:])

        if name not in own_state:
            print('not in ', name)
            continue

        if param.shape != own_state[name].shape:
            print('tarined weight shape:', param.shape)
            print('wanted weight shape:', own_state[name].shape)
            print('name:', name)

            if param.shape[-1] != own_state[name].shape[-1]:
                _temp_param = torch.zeros(own_state[name].shape)
                _temp_param[:,:,:, 1:2] = param
                own_state[name].copy_(_temp_param)

            if param.shape[-2] != own_state[name].shape[-2]:
                _temp_param = torch.zeros(own_state[name].shape)
                _temp_param[:,:, 1:2, :] = param
                own_state[name].copy_(_temp_param)

        else:
            # print('param shape:',param.shape)
            own_state[name].copy_(param)
    return model