import torch


def _get_model_param(weight_decay_list, *args):
    # args contain should contrain all the model, which wanted to be train
    param_list = []
    if len(weight_decay_list) == 1 and len(args) != 1:
        weight_decay_list = [weight_decay_list[0] for i in range(len(args))]

    assert len(weight_decay_list) == len(args), 'length of weight_decay_list is not the same as models'

    for i in range(len(args)):
        param_list.append({'params': args[i].parameters(), 'weight_decay': weight_decay_list[i]})

    return param_list


def _get_model_param_without_weight_decay(*args):
    param_list = []
    for i in range(len(args)):
        param_list.append({'params':args[i].parameters()})
    return param_list
