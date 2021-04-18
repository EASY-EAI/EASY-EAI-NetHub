import os
import random
import sys

from boundingbox_tools import Boundingbox_container, mosaic_augumentation
from keypoint_tools import Keypoint_container
from pixel_label_tools import Pixellabel_container


container_type_list = {'box': Boundingbox_container,
                       'keypoint': Keypoint_container,
                       'pixellabel': Pixellabel_container}


'''
读取适配：box, keypoint, pixel_label_tools

生成方式： 随机剪切， 水平翻转， 竖直翻转， 旋转， 仿射变换， 透射变换， 马赛克变换

输出：原路返回格式/新路径无子文件夹/新路径带原子文件夹
'''


def search_file_in_folder(path, file_format_list, container):
    file_list = os.listdir(path)
    for file in file_list:
        file_true_path = os.path.join(path, file)
        if os.path.isdir(file_true_path):
            search_file_in_folder(file_true_path, file_format_list, container)
        else:
            _format = file.split('.')[-1]
            if _format in file_format_list:
                container.append(file_true_path)


def scan_file_from_folder(path_list, file_format_list):
    path_container = []
    for path in path_list:
        search_file_in_folder(path, file_format_list, path_container)

    return path_container


def create_init_container(container_type):
    if container_type == 'box':
        container = Boundingbox_container(None, 0, 'box', 0, False)
    elif container_type == 'keypoint':
        container = Keypoint_container(None, None, None)
    elif container_type == 'pixellabel':
        container = None

    return container


def create_folder(path, file=True):
    # if path container file, set file to be True. others set to be False
    # path like './test_save/1/2/3'
    #      or   'D:/workspace/LMO/Zen_torch/data_process/test_save/1/2/3'
    #      or   './test_save/1/2/3/img.jpg'
    #      or   'D:/workspace/LMO/Zen_torch/data_process/test_save/1/2/3/img.jpg'
    path = path.replace('\\', '/')
    segment = path.split('/')

    for i in range(len(segment) - 1*file):
        _temp_path = '/'.join(segment[:i + 1])
        if not os.path.exists(_temp_path):
            os.mkdir(_temp_path)


def path_convert(path, method, name_add, new_path=None, init_path=None):
    # method chose to be ['origin_path',
    #                     'new_path_without_subfolder',
    #                     'new_path_with_same_subfolder']
    path = path.replace('\\', '/')
    if new_path is not None: new_path = new_path.replace('\\', '/')
    if init_path is not None: init_path = init_path.replace('\\', '/')

    file_format = path.split('.')[-1]
    if method == 'origin_path':
        output_path = path.replace('.' + file_format, '{}.{}'.format(name_add, file_format))

    elif method == 'new_path_without_subfolder' \
                   and (new_path is not None):
        file_name = path.split('/')[-1].rstrip('.' + file_format)
        output_path = os.path.join(new_path, '{}{}.{}'.format(file_name, name_add, file_format))

    elif method == 'new_path_with_same_subfolder' \
                   and (init_path is not None) \
                   and (new_path is not None):
        path = path.lstrip(init_path)
        output_path = path.replace('.' + file_format, '{}.{}'.format(name_add, file_format))
        output_path = os.path.join(new_path, output_path)
    else:
        print('Not support for method:{}'.format(method))

    output_path = output_path.replace('\\', '/')
    return output_path


def main(path_list, file_format_list, container_type, target_path):
    target_path = target_path.replace('\\', '/')

    file_path_container = scan_file_from_folder(path_list, file_format_list)
    print('all file number:', len(file_path_container))

    TRANSFORMER_TIME = [5, 5]
    MOSAIC_NUMBER = len(file_path_container)

    container = create_init_container(container_type)

    for path in file_path_container:
        try:
            for i in range(TRANSFORMER_TIME[0]):
                container._load_file(path, 'json')

                container.padding_image_cv2((max(container.img.shape[0], container.img.shape[1]), max(container.img.shape[0], container.img.shape[1])))
                # container._show_img()

                container.random_transform()
                # container.randomom_warpperspective()

                # container._show_img()
                container.random_crop((int(container.img.shape[0]*0.8), int(container.img.shape[1]*0.8)))
                container.resize_cv2((416, 416))
                # container._show_img()

                # save_path = path_convert(path, 'origin_path', '_{}'.format(i))
                # save_path = path_convert(path, 'new_path_without_subfolder', '_{}'.format(i),
                #                          r'D:\workspace\LMO\Zen_torch\data_process\test_save',
                #                          path_list[0])
                save_path = path_convert(path, 'new_path_with_same_subfolder', '_{}'.format(i),
                                         target_path,
                                         path_list[0])

                create_folder(save_path)
                container._write_down(save_path, ['json', 'txt'])


            for j in range(TRANSFORMER_TIME[1]):
                container._load_file(path, 'json')
                # container._show_img()

                container.padding_image_cv2((max(container.img.shape[0], container.img.shape[1]), max(container.img.shape[0], container.img.shape[1])))
                # container._show_img()

                container.random_transform()
                # container.random_warpperspective()

                # container._show_img()
                container.random_crop((int(container.img.shape[0]*0.8), int(container.img.shape[1]*0.8)))
                container.resize_cv2((416, 416))
                # container._show_img()

                # save_path = path_convert(path, 'origin_path', '_{}'.format(i))
                # save_path = path_convert(path, 'new_path_without_subfolder', '_{}'.format(i),
                #                          r'D:\workspace\LMO\Zen_torch\data_process\test_save',
                #                          path_list[0])
                save_path = path_convert(path, 'new_path_with_same_subfolder', '_{}'.format(i+j+1),
                                         target_path,
                                         path_list[0])

                create_folder(save_path)
                container._write_down(save_path, ['json', 'txt'])
        except Exception as err:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

    container_1 = create_init_container(container_type)
    container_2 = create_init_container(container_type)
    container_3 = create_init_container(container_type)
    container_4 = create_init_container(container_type)

    container_list = [container_1, container_2, container_3, container_4]

    create_folder(os.path.join(target_path, 'masaic_data'), file=False)

    out_count = 0
    print('MOSAIC_NUMBER:', MOSAIC_NUMBER)
    for k in range(MOSAIC_NUMBER):
        try:
            for _count in range(4):
                select_index = random.randint(0, len(file_path_container)-1)
                container_list[_count]._load_file(file_path_container[select_index], 'json')

                container_list[_count].padding_image_cv2((max(container_list[_count].img.shape[0], container_list[_count].img.shape[1]), max(container_list[_count].img.shape[0], container_list[_count].img.shape[1])))
                container_list[_count].random_transform()

                # container._show_img()
                container_list[_count].random_crop((int(container_list[_count].img.shape[0]*0.8), int(container_list[_count].img.shape[1]*0.8)))
                container_list[_count].resize_cv2((416, 416))
                # container._show_img()

            new_container = mosaic_augumentation(container_list, (416, 416))
            # new_container._show_img()

            save_path = os.path.join(target_path, 'masaic_data/{}.jpg'.format(out_count))
            while os.path.exists(save_path):
                out_count +=1
                save_path = os.path.join(target_path, 'masaic_data/{}.jpg'.format(out_count))

            new_container._write_down(save_path, ['json', 'txt'])

        except Exception as err:
            s=sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

if __name__ == '__main__':
    file_format_list = ['jpg']
    # path_list = [r'D:\data\anti_spoofing\NIR_ZEN']
    # path = r'G:\data\drive-download-20200527T141059Z-001\test_sample'
    # sub_path = path

    # path_list = ['./sample_load/dianpingche']
    # target_path = r'./test/cheche'

    path_list = ['G:/data/water_bump/label_data_0']
    target_path = 'G:/data/water_bump/reinforced'
    container_type = 'box'

    main(path_list, file_format_list, container_type, target_path)


def add_item(self, target_path, class_type, depth_level=None, limit_sample=None, shuffle=True):
    start_len = len(self.img_file_list)
    for root, dirs, files in os.walk(target_path):
        if depth_level is not None:
            # start next loop if depth not match
            if (len(root.split('\\')) - len(target_path.split('\\'))) != depth_level:
                continue

        temp_file_list = []
        for name in files:
            if name.split('.')[-1] in self.img_format_list:
                temp_file_list.append(os.path.join(root, name))
                if limit_sample is not None and shuffle is False:
                    # limit and not shuffle, then its allowed to break early
                    if len(temp_file_list) >= limit_sample:
                        break

        if shuffle is True:
            random.shuffle(temp_file_list)
        if limit_sample is not None:
            temp_file_list = temp_file_list[0: limit_sample]

        if isinstance(class_type, int):
            temp_label_list = [class_type for i in range(len(temp_file_list))]
        elif isinstance(class_type, str):
            temp_label_list = [self.class_name.index(class_type) for i in range(len(temp_file_list))]

        self.img_file_list = self.img_file_list + temp_file_list
        self.label_file_list = self.label_file_list + temp_label_list
