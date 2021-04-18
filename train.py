import logging
from utils.config_parser import parser_n_load


def main(opt):

    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%a, %d %b %Y %H:%M:%S '

    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    logging.basicConfig(level=logging.INFO,
                        format=BASIC_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename='test.log')

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    logger = logging.getLogger('mainlogger')
    logger.addHandler(chlr)


    yaml_file_path = './luanch_script/config/classification_mnist.yaml'
    module_dict = parser_n_load(yaml_file_path)
    logger.info('parsing file: [{}]'.format(yaml_file_path))

    print_ID = False
    if print_ID is True:
        for name in module_dict:
            print(name, ' ID:', id(module_dict[name]))

    model_script = module_dict['model_script']
    model_script._to_device()

    for i in range(opt.epoch):
        model_script.train(i)


class opt_container(object):
    """docstring for opt"""
    def __init__(self, name=None):
        super(opt_container, self).__init__()
        self.name = 'opt'
        self.epoch = 50


if __name__ == '__main__':
    opt = opt_container(name='opt')
    main(opt)


