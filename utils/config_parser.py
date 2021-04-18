import importlib
import yaml
import logging
###
# module.__code__.varname
# 可以得到所有参数key
# class 与  function 的 __call__ 无法混用，会有bug
# 全局格式


def _sub_dict(dicts, key_list, default=None):
    return dict([(key, dicts.get(key, default)) for key in key_list])


def _reorder_argument(function, dicts, implicit_parameters_list=None):
    varnames = function.__code__.co_varnames
    argument = varnames[0:function.__code__.co_argcount]
    # print('argument',argument)
    ordered_input_list = []
    for name in argument:
        assert name in dicts, '{} is require, but not exist in parameters dicts'.format(name)
        ordered_input_list.append(dicts[name])

    if ('args' in varnames) and (implicit_parameters_list is not None):
        ordered_input_list.extend(implicit_parameters_list)
    return ordered_input_list


def _parser_source(source_dict, global_dict):
    module = importlib.import_module(source_dict['parser_rule']['define_file_path'])
    function = getattr(module, source_dict['parser_rule']['run_key'])

    transmit_input_dict = source_dict['parameters']

    if 'implicit_parameters' in source_dict: 
        if source_dict['implicit_parameters'] is not None:
            transmit_input_dict.update(_sub_dict(global_dict, source_dict['implicit_parameters']))

    transmit_input_args = None
    if 'implicit_parameters_args' in source_dict:
        transmit_input_args= []
        if source_dict['implicit_parameters_args'] is not None:
            for name in source_dict['implicit_parameters_args']:
                transmit_input_args.append(global_dict[name])


    if transmit_input_args is None:
        if transmit_input_dict is None:
            output = function()
        else:
            output = function(**transmit_input_dict)
    else:
        input_list = _reorder_argument(function, transmit_input_dict, transmit_input_args)
        output = function(*input_list)


    if len(source_dict['parser_element']) == 1:

        # if source_dict['parser_element'][0] == 'params':
        #     print('!!!!!params:', output)

        global_dict[source_dict['parser_element'][0]] = output
    else:
        for i in range(len(source_dict['parser_element'])):
            global_dict[source_dict['parser_element'][i]] = output[i]


def parser_n_load(yaml_file, print_ID=False):

    global_dict = {}

    with open(yaml_file, 'r') as F:
        config = yaml.load(F)
    # parser_module = 'input_data'

    dynamic_parsing_part = config['dynamic_parsing_part']

    for name in dynamic_parsing_part:
        print('='*15)
        print('now parsing dynamic_parsing_part [{}]'.format(name))
        for key in dynamic_parsing_part[name].keys():
            print(key, ':', dynamic_parsing_part[name][key])

        source_config_dict = dynamic_parsing_part[name]

        _parser_source(source_config_dict, global_dict)

    if print_ID:
        for name in global_dict:
            print(name, ' ID:', id(global_dict[name]))

    return global_dict


if __name__ == '__main__':
    yaml_file = './yaml_config/face_embedding_v2.yml'
    parser_n_load(yaml_file)