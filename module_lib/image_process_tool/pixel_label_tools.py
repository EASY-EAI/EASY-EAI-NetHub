import os
import sys
import random

import cv2
import numpy as np
import imgviz

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))

from boundingbox_tools import Boundingbox_container
from keypoint_tools import Keypoint_container

COLOR_POOL = imgviz.label_colormap()[1:]


def possibilitymap_to_heatmap(label, class_dim):
    assert (class_dim+1) <= len(label.shape), 'class_dim[{}] exceed label shape[{}]'.format(class_dim, label.shape)

    if class_dim != len(label.shape):
        _shape_list = [i for i in range(len(label.shape))]
        _shape_list[class_dim] = _shape_list[-1]
        _shape_list[-1] = class_dim

    label = label.transpose(*_shape_list)
    _inverse_onehot = label.argmax(axis=-1)
    onehot = np.eye(label.shape[-1])[_inverse_onehot]
    onehot = onehot.transpose(*_shape_list)

    return onehot


def get_color_class(label):
    # work for color map to distill class color.
    # used it when you dont have the class_color list.
    assert len(label.shape) == 3, 'label should be 3 dimension'
    _color_pool = [(0, 0, 0),]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            _color = label[i, j, :]
            _color = tuple(_color)
            if _color not in _color_pool:
                _color_pool.append(_color)
    return _color_pool


def segment_PA(label, gt_label, class_dim):
    # count pixel_label accuracy
    score = segment_pixel_accuracy(label, gt_label, class_dim)
    return score


def segment_pixel_accuracy(label, gt_label, class_dim):
    # also name PA
    score = np.array(segment_CPA(label, gt_label, class_dim)).mean()
    return score


def segment_CPA(label, gt_label, class_dim):
    # count class_pixel_accuracy
    CPA = segment_class_pixel_accuracy(label, gt_label, class_dim)
    return CPA


def segment_class_pixel_accuracy(label, gt_label, class_dim):
    # count class_pixel_accuracy, valid on heatmap
    assert label.shape == gt_label.shape, 'label.shape[{}] is not equal to gt_label.shape[{}]'.format(label.shape, gt_label.shape)
    assert (class_dim+1) <= len(label.shape), 'class_dim[{}] exceed label shape[{}]'.format(class_dim, label.shape)
    assert len(label.shape) <= 4, 'label shape is limitd under 4 dimension'

    channal = label.shape[class_dim]

    if class_dim != len(label.shape):
        _shape_list = [i for i in range(len(label.shape))]
        _shape_list[class_dim] = _shape_list[-1]
        _shape_list[-1] = class_dim
        label = label.transpose(*_shape_list)
        gt_label = gt_label.transpose(*_shape_list)

    CPA = []
    _bool_metrix = (label == gt_label)
    # valid_point = (gt_label == 1) ## count right only
    valid_point = ((label + gt_label) > 0)
    _bool_metrix = _bool_metrix * valid_point
    for i in range(channal):
        if len(label.shape) == 4:
            # 4D
            if valid_point[:, :, :, i].sum() == 0:
                score = 0
            else:
                score = (_bool_metrix[:, :, :, i] == 1).sum() / valid_point[:, :, :, i].sum()

        elif len(label.shape) == 3:
            # 3D
            if valid_point[:, :, i].sum() == 0:
                score = 0
            else:
                score = (_bool_metrix[:, :, i] == 1).sum() / valid_point[:, :, i].sum()
        CPA.append(score)
    return CPA


def segment_MPA(label, gt_label, class_dim=None):
    # Mean of class_pixel_accuracy
    CPA = segment_class_pixel_accuracy(label, gt_label, class_dim)
    CPA = np.array(CPA)
    return CPA.mean()


# def segment_heat_map_MIOU(label, gt_label):
#     # count MIOU with heatmap
#     assert label.shape == gt_label.shape, 'both pixel label should have same shape'
#     class_dim = 2
#     if len(label.shape) == 2:
#         channal = 1
#     elif len(label.shape) == 3:
#         channal = label.shape[2]
#     else:
#         assert False, 'label shape is not allowed, plz check'

#     label = np.reshape(label, (-1, channal))
#     gt_label = np.reshape(gt_label, (-1, channal))


#     _bool_target_label = (label == 1)
#     _bool_gt_label = (gt_label == 1)
#     _bool_cross = _bool_target_label*_bool_gt_label

#     iou_list = _bool_cross.sum(0) / (_bool_target_label.sum(0) + _bool_gt_label.sum(0) - _bool_cross.sum(0)+ 1)

#     return iou_list


def bounddingbox_container_to_pixellabel_container(bb_container,
                                                   total_class_number=None,
                                                   class_color=None,):
    # assert 0 not in bb_container.class_id, 'id 0 is prohibited in Boundingbox_container, \
    #         using Boundingbox_container.abalienate_zeros classes to solve this problem.'
    assert (total_class_number is not None or class_color is not None), \
        'total_class_number or class_color shoud not be eighter None'

    if total_class_number is not None and class_color is not None:
        assert total_class_number == len(class_color), 'total_class_number \
            and lenth of class_color should have same valid, if all asigned'
    elif total_class_number is not None:
        assert max(bb_container.class_id) < total_class_number, \
            'id exceed classes range'
        class_color = COLOR_POOL[:total_class_number]
    elif class_color is not None:
        assert max(bb_container.class_id) < len(class_color), \
            'id exceed color range'
        total_class_number = len(class_color)

    height, width, channal = bb_container.img.shape
    boxes_label = bb_container.label
    box_class_id = bb_container.class_id
    pixel_label = np.zeros((height, width, total_class_number)).astype(np.int)

    for i in range(len(boxes_label)):
        pixel_label[boxes_label[i][1]: boxes_label[i][3],
                    boxes_label[i][0]: boxes_label[i][2],
                    box_class_id[i]] = 1

    new_class_id = [i for i in range(total_class_number)]
    pl_container = Pixellabel_container(img=bb_container.img.copy(),
                                        pixel_label=pixel_label,
                                        label_type='heat_map',
                                        class_color=class_color,
                                        class_id=new_class_id,
                                        total_class_number=total_class_number)
    # pl_container.force_merge_ignoring_overlap()
    # for i in range(len(box_class_id)):
    #     print(class_color[box_class_id[i]])
    return pl_container


def generate_guassian_heat_label_on_point(target_layer_shape, keypoint, sigma=4):
    # keypoint format (x, y), sigma is a parameter of label
    _temp_layer = np.zeros(target_layer_shape).astype(np.uint8)
    miu_x = int(keypoint[0])
    miu_y = int(keypoint[1])

    x_lower_bound = max(0, miu_x - sigma*3)
    x_upper_bound = min(target_layer_shape[1], miu_x + sigma*3 + 1)
    y_lower_bound = max(0, miu_y - sigma*3)
    y_upper_bound = min(target_layer_shape[0], miu_y + sigma*3 + 1)

    x_range = np.arange(x_lower_bound, x_upper_bound, 1)
    y_range = np.arange(y_lower_bound, y_upper_bound, 1)

    x_map = np.reshape(x_range, (*x_range.shape, 1)).repeat(len(y_range), axis=1)
    y_map = np.reshape(y_range, (*y_range.shape, 1)).repeat(len(x_range), axis=1).T

    guassian_map = np.exp(- (pow((x_map-miu_x), 2) + pow((y_map-miu_y), 2)) / (2 * pow(sigma, 2)))
    _temp_layer[y_lower_bound: y_upper_bound,
                x_lower_bound: x_upper_bound] = (guassian_map*255).astype(np.uint8).T

    return _temp_layer


def keypoint_contain_to_pixellabel_container(kp_container, total_class_number, sigma=5):
    # guassian
    assert len(kp_container.keypoint_list) <= total_class_number, \
        'keypoint number exceed total_class_number'

    height, width, channal = kp_container.img.shape
    keypoint_list = kp_container.keypoint_list
    keypoint_index_list = kp_container.keypoint_index_list
    pixel_label = np.zeros((height, width, total_class_number)).astype(np.int)

    for i in range(len(keypoint_list)):
        pixel_label[:, :, keypoint_index_list[i]] \
            += generate_guassian_heat_label_on_point((height, width), keypoint_list[i], sigma)

    pixel_label[pixel_label > 255] = 255
    new_class_id = [i for i in range(total_class_number)]

    pl_container = Pixellabel_container(img=kp_container.img.copy(),
                                        pixel_label=pixel_label,
                                        label_type='gaussian_heat_map',
                                        class_color=kp_container,
                                        class_id=new_class_id,
                                        total_class_number=total_class_number)
    return pl_container


class Pixellabel_container(object):
    '''
    Discription:
        The Keypoint_container collect the image and its keypoint_labels,
    Then do the transform as padding/resize/crop/affine/perspective.

    INITIALIZE_INPUT:
        img                         ->  The image, cv2 fromat
        pixel_label                 ->  pixel_label. Sharing the same size with img
        label_type                  ->  'heat_map' or 'color_map' or 'gaussian_heat_map'
        class_id                    ->  class_id
        total_class_number          ->  maximum class number
        class_color                 ->  list of color set(RGB). default will get color from lib-imgviz

    TIPS:
        cv2_img.shape is [height, width, channal]
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)
        cv2_resize(img, (width, height))

        all the target_size in this code is [height, width]
        all the box label in this code is [left, up, right, down]
        all the point in this code is (x, y)

    FUNCTION:
        copy                    -> create an identity new container
        image_shape             -> return img.shape
        check_overlap           -> check_overlap of label
        force_merge_ignoring_overlap -> fix the ooverlap of label. May used it everytime after transformer.
        heat_2_color            -> convert heat_label to color_label
        color_2_heat            -> convert color_label to heat_label
        padding_image_cv2       -> padding and resize.
        add_blank_edge          -> adding blank_edge.
        resize                  -> resize
        crop                    -> crop with the box
        random_crop             -> random crop.
        flip                    -> flip
        rotate                  -> rotate
        warpaffine              -> warpaffine on img and label
        warpperspective         -> warpperspective on img and label.
        random_warpperspective  -> do random warpperspective transform
        random_rotate           -> do random rotate transform
        random_drift            -> do random drift transform
        random_scale            -> do random scale transform
        random_shear            -> do random shear transform
        random_flip             -> do random flip transform
        random_transform        -> do random transform transform
        _show_img               -> show img via CV2
        _write_down             -> write down img and label. label could be npy or png
        _load_file              -> load img and label. label could be npy or png
        _switch_label_name      -> switch label name, always can be used to switch label_name from str_type to number_type.
        _abalienate_unused_id   -> abalienate the unused label. work based on heatmap
        onehot_lable_with_background    -> return onehot label with background
        onehot_label_with_nobackground  -> return onehot label without background

    '''
    def __init__(self, img, pixel_label, label_type, class_id,
                 total_class_number, class_color=None):
        super(Pixellabel_container, self).__init__()
        self.img = img
        self.pixel_label = pixel_label
        assert label_type == 'heat_map' \
            or label_type == 'color_map' \
            or label_type == 'gaussian_heat_map', \
            'The type should be heat_map/color_map'
        self.type = label_type
        self.class_id = class_id
        self.total_class_number = total_class_number
        if class_color is None:
            self.class_color = COLOR_POOL[:total_class_number]
        else:
            self.class_color = class_color

    def copy(self):
        # create a new identity container.
        return Pixellabel_container(self.img.copy(),
                                    self.pixel_label.copy(),
                                    self.type,
                                    self.class_id.copy(),
                                    self.total_class_number,
                                    self.class_color.copy())

    def image_shape(self):
        return self.img.shape

    def check_overlap(self):
        '''
        Discription:
            after transformers, heat_map label may get overlap problem. This function will
        check whether this is happen
        RETURN:
            1               -> overlap occur
            0               -> no overlap problem
            -1              -> label_map_type error
        '''
        if self.type == 'heat_map':
            overlap_area = self.pixel_label.sum(axis=2) > 1
            if overlap_area.sum() > 0:
                return 1
            else:
                return 0
        else:
            print('color_map would not have overlap always')
            return -1

    def heat_2_color(self, info=False):
        if self.type == 'heat_map':
            _valid = self.check_overlap()
            assert _valid < 1, 'heat_map is overlaped, could not converted to color_map'
            _color_img = np.zeros(self.img.shape)
            for i in range(self.pixel_label.shape[2]):
                _color_img += (self.pixel_label[:, :, i] == 1).repeat(3).reshape(self.img.shape)* self.class_color[i]

            self.pixel_label = _color_img
            self.type = 'color_map'
        elif self.type == 'color_map':
            if info is True: print('pixel_label is alreay heat_map')
        elif self.type == 'gaussian_heat_map':
            print('WARNING : pixel_label with gaussian_heat_map cant not convert to color_map, invalid operater')

    def color_2_heat(self, info=False):
        if self.type == 'color_map':
            _heat_img = np.zeros((*self.img.shape[:2], self.total_class_number))
            for i in range(self.total_class_number):
                ## if color not in the self.class_color, then it will be ignore
                _class_map = np.array([self.pixel_label[:, :, j] == self.class_color[i][j]
                                       for j in range(len(self.class_color[i]))]).sum(axis=0)
                _heat_img[:, :, i] = ((_class_map == len(self.class_color[i]))*1)
            self.pixel_label = _heat_img
            self.type = 'heat_map'
        else:
            if info is True: print('pixel_label is alreay heat_map')

    def force_merge_ignoring_overlap(self):
        # The smaller_area map will always stay on the bigger one.
        _converted = False
        if self.type == 'color_map':
            self.color_2_heat()
            _converted = True

        _temp_label = self.pixel_label.copy()
        overlap_area = _temp_label.sum(axis=2) > 1

        if overlap_area.sum() > 0:
            # start convert
            _area_of_label = self.pixel_label.sum(axis=0).sum(axis=0)
            _sort_area_index = np.argsort(_area_of_label) # sort from smaller to bigger

            _overlaped_label = _temp_label[overlap_area].copy()

            for i in range(len(_sort_area_index)-1):
                if _area_of_label[_sort_area_index[-(i+1)]] > 0:
                    _area_single = _temp_label[overlap_area][:, _sort_area_index[-(i+1)]].copy()
                    _area_left = _temp_label[overlap_area][:, _sort_area_index[:-(i+1)]].copy().sum(axis=-1)

                    _area_single[_area_left > 0] = 0
                    _overlaped_label[:, _sort_area_index[-(i+1)]] = _area_single.copy()

                else:
                    break

            _temp_label[overlap_area] = _overlaped_label
            self.pixel_label = _temp_label.copy()

        if _converted is True:
            self.heat_2_color()

    def merge_heat_map(self):
        # merge heat map. Used for imshow.
        if self.type != 'color_map':
            _merged_img = np.zeros(self.img.shape[:2]).astype(np.uint32)
            for i in range(self.pixel_label.shape[2]):
                _merged_img += self.pixel_label[:, :, i].astype(np.uint32)
            _merged_img[_merged_img >= 255] = 255
            return _merged_img.astype(np.uint8)
        else:
            return None

    def padding_image_cv2(self, target_size):
        # This operation will keep the object scale, but not warping them.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        image_scale = self.image_shape()[0] / self.image_shape()[1]
        target_size_scale = target_size[0] / target_size[1]

        padding_pixel = [0, 0]  # [horizontal, vetical]
        if image_scale < target_size_scale:
            padding_pixel[0] = int(self.image_shape()[1]*(target_size_scale - image_scale))
        else:
            padding_pixel[1] = int(self.image_shape()[0]/target_size_scale
                                   - self.image_shape()[0]/image_scale)

        # paddding img
        blank_image = np.zeros((self.image_shape()[0] + padding_pixel[0],
                                self.image_shape()[1] + padding_pixel[1],
                                self.image_shape()[2])).astype(np.uint8)
        blank_image[int(padding_pixel[0]/2): self.image_shape()[0] + int(padding_pixel[0]/2),
                    int(padding_pixel[1]/2): self.image_shape()[1] + int(padding_pixel[1]/2),
                    :] = self.img
        new_image = blank_image

        # padding pixel label
        blank_pixel_label = np.zeros((self.pixel_label.shape[0] + padding_pixel[0],
                                      self.pixel_label.shape[1] + padding_pixel[1],
                                      self.pixel_label.shape[2])).astype(np.uint8)
        blank_pixel_label[int(padding_pixel[0]/2): self.pixel_label.shape[0] + int(padding_pixel[0]/2),
                          int(padding_pixel[1]/2): self.pixel_label.shape[1] + int(padding_pixel[1]/2),
                          :] = self.pixel_label

        self.img = cv2.resize(new_image, (target_size[1], target_size[0]))
        self.pixel_label = cv2.resize(blank_pixel_label, (target_size[1], target_size[0]))

    def add_blank_edge(self, pixels):
        # adding blank edge by target_pixel
        assert isinstance(pixels, int) or isinstance(pixels, tuple), 'adding pixels must be int or tuple'
        if isinstance(pixels, int):
            pixels = (pixels, pixels)

        image_shape = self.image_shape()
        new_image = np.zeros((image_shape[0] + pixels[0], image_shape[1] + pixels[1], image_shape[2]))
        new_image_shape = new_image.shape

        new_image[int(pixels[0]/2): int(new_image_shape[0] - pixels[0]/2),
                  int(pixels[1]/2): int(new_image_shape[1] - pixels[1]/2),
                  :] = self.img.copy()

        pixel_label_shape = self.pixel_label.shape
        new_pixel_label = np.zeros((pixel_label_shape[0] + pixels[0], pixel_label_shape[1] + pixels[1], pixel_label_shape[2]))
        new_pixel_label_shape = new_pixel_label.shape

        new_pixel_label[int(pixels[0]/2): int(new_pixel_label_shape[0] - pixels[0]/2),
                        int(pixels[1]/2): int(new_pixel_label_shape[1] - pixels[1]/2),
                        :] = self.pixel_label.copy()

        self.img = new_image.astype('uint8')
        self.pixel_label = new_pixel_label

    def resize_cv2(self, target_size):
        # resize the image. This transform may wrap the object.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.img = cv2.resize(self.img, (target_size[1], target_size[0]))
        self.pixel_label = cv2.resize(self.pixel_label, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    def flip(self, method):
        # method = horizontal/vertical
        if method == 'horizontal' or method == 1:
            self.img = cv2.flip(self.img, 1)
            self.pixel_label = cv2.flip(self.pixel_label, 1)
        elif method == 'vertical' or method == 0:
            self.img = cv2.flip(self.img, 0)
            self.pixel_label = cv2.flip(self.pixel_label, 0)

    def crop(self, crop_box):
        '''
            crop the image and label with crop_box.
        INPUT:
                crop_box            -> [left, up, right, down]
        RETURN:
                null
        '''

        crop_box = [int(value) for value in crop_box]
        _temp_img = np.zeros((crop_box[3] - crop_box[1], crop_box[2] - crop_box[0], self.image_shape()[2]))
        _temp_img = self.img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :].copy()

        _temp_pixel_label = np.zeros((crop_box[3] - crop_box[1], crop_box[2] - crop_box[0], self.pixel_label.shape[2]))
        _temp_pixel_label = self.pixel_label[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :].copy()

        self.img = _temp_img
        self.pixel_label = _temp_pixel_label

    def check_iou_remain(self, pick_box, keep_iou_rate, keep_id_list):
        _label_shape = self.pixel_label.shape
        _temp_label = np.zeros(_label_shape)
        _temp_label[pick_box[1]:pick_box[3], pick_box[0]:pick_box[2], :] = \
            self.pixel_label.copy()[pick_box[1]:pick_box[3], pick_box[0]:pick_box[2], :]

        status = 'valid'
        # iou_list = segment_heat_map_MIOU(_temp_label, self.pixel_label.copy())
        # segment_CPA
        iou_list = segment_CPA(_temp_label, self.pixel_label.copy(), 2)
        iou_list = iou_list[keep_id_list]
        if (iou_list < keep_iou_rate).sum() > 0:
            status = 'invalid'
        return status

    def random_crop(self, target_size, padding=False, keep_iou_rate=0.6,
                    keep_id_list=None, tolerant_step=10):
        assert isinstance(target_size, int) or isinstance(target_size, tuple),\
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        assert (target_size[0] <= self.img.shape[0] and target_size[1] <= self.img.shape[1]),\
            'the target_size exceed the range of image, {}'.format(self.img.shape)

        assert keep_iou_rate <= 1, 'keep_iou_rate must in range(0, 1)'

        image_shape = self.img.shape
        crop_size = target_size

        if padding:
            padding_size = (crop_size[0] + image_shape[0], crop_size[1] + image_shape[1])
            self.add_blank_edge(tuple(crop_size))
        else:
            padding_size = (image_shape[0], image_shape[1])

        _step = 0
        status = 'valid'
        while _step < tolerant_step:
            _step += 1

            pick_point_up = random.randint(0, padding_size[0] - crop_size[0])
            pick_point_left = random.randint(0, padding_size[1] - crop_size[1])
            pick_point_down = pick_point_up + crop_size[0]
            pick_point_right = pick_point_left + crop_size[1]
            pick_box = (pick_point_left, pick_point_up, pick_point_right, pick_point_down)
            if keep_id_list is not None:
                status = self.check_iou_remain(pick_box, keep_iou_rate, keep_id_list)
            if status == 'valid':
                break
        if status == 'invalid':
            pass
        self.crop(pick_box)

    def get_valid_segment_layer(self):
        _temp_pixel_label = self.pixel_label.copy()
        _temp_pixel_label = np.reshape(_temp_pixel_label, (-1, self.pixel_label.shape[2]))
        _temp_pixel_label = (_temp_pixel_label == 1).sum(0)
        non_segment_layer_list = []
        for i in range(len(_temp_pixel_label)):
            if _temp_pixel_label[i] != 0:
                non_segment_layer_list.append(i)
        return non_segment_layer_list

    def rotate(self, angle, center=None, scale=1):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if center is None:
            center = (width*0.5, height*0.5)
        # if scale >= 1:
        #     self.add_blank_edge(int(height*scale*0.2), int(width*scale*0.2))

        matRotate = cv2.getRotationMatrix2D(center, angle, scale)
        dst = cv2.warpAffine(self.img.copy(), matRotate, (width, height))
        pixel_label_dst = cv2.warpAffine(self.pixel_label.astype(np.uint8).copy(), matRotate, (width, height))

        self.img = dst
        self.pixel_label = pixel_label_dst

    def warpaffine(self, matRotate=None):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if matRotate is None:
            pts1 = np.float32([[0, 0],
                               [width - 1, 0],
                               [0, height - 1]])
            pts2 = np.float32([[width*0.2, height*0.1],
                               [width*0.9, height*0.2],
                               [width*0.1, height*0.9]])
            matRotate = cv2.getAffineTransform(pts1, pts2)

        dst = cv2.warpAffine(self.img.copy(), matRotate, (width, height))
        pixel_label_dst = cv2.warpAffine(self.pixel_label.astype(np.uint8).copy(), matRotate, (width, height))

        self.img = dst
        self.pixel_label = pixel_label_dst

    def warpperspective(self, P_metrix=None):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if P_metrix is None:
            pts1 = np.float32([[0, 0],
                               [width - 1, 0],
                               [0, height - 1],
                               [width - 1, height - 1]])
            pts2 = np.float32([[width*0.1, height*0.1],
                               [width - 1, 0],
                               [width*0.1, height*0.8],
                               [width - 1, height - 1]])
            P_metrix = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(self.img.copy(), P_metrix, (width, height))
        pixel_label_dst = cv2.warpPerspective(self.pixel_label.astype(np.uint8).copy(), P_metrix, (width, height))

        self.img = dst
        self.pixel_label = pixel_label_dst

    def random_warpperspective(self, drift_range=0.08):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        # random.randint(-1*drift_range*width, drift_range*width)
        # random.randint(-1*drift_range*height, drift_range*height)

        pts1 = np.float32([[0, 0],
                           [width - 1, 0],
                           [0, height - 1],
                           [width - 1, height - 1]])

        pts2 = np.float32([[0 + random.randint(-1*int(drift_range*width), int(drift_range*width)), 0 + random.randint(-1*int(drift_range*height), int(drift_range*height))],
                           [width - 1 + random.randint(-1*int(drift_range*width), int(drift_range*width)), 0 + random.randint(-1*int(drift_range*height), int(drift_range*height))],
                           [0 + random.randint(-1*int(drift_range*width), int(drift_range*width)), height - 1 + random.randint(-1*int(drift_range*height), int(drift_range*height))],
                           [width - 1 + random.randint(-1*int(drift_range*width), int(drift_range*width)), height - 1 + random.randint(-1*int(drift_range*height), int(drift_range*height))]])

        P_metrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.warpperspective(P_metrix)
        # self.check_and_fix_outsider_box()

    def random_rotate(self, rotate_range=5):
        angle = random.randint(-1*rotate_range, rotate_range)
        m = np.float32([[np.cos(angle/180*np.pi), np.sin(angle/180*np.pi), 0],
                        [-np.sin(angle/180*np.pi), np.cos(angle/180*np.pi), 0]])
        self.warpaffine(m)

    def random_drift(self, drift_range=0.1):
        x_shift = (random.random()*2 - 1) * drift_range * self.img.shape[1]
        y_shift = (random.random()*2 - 1) * drift_range * self.img.shape[0]
        m = np.float32([[1, 0, x_shift],
                        [0, 1, y_shift]])
        self.warpaffine(m)

    def random_scale(self, scale_range=0.05):
        x_scale = 1 + (random.random()*2 - 1) * scale_range
        y_scale = 1 + (random.random()*2 - 1) * scale_range
        m = np.float32([[x_scale, 0, 0],
                        [0, y_scale, 0]])
        self.warpaffine(m)

    def random_shear(self, shear_range=0.05):
        x_shear = (random.random()*2 - 1) * shear_range
        y_shear = (random.random()*2 - 1) * shear_range
        m = np.float32([[1, x_shear, 0],
                        [y_shear, 1, 0]])
        self.warpaffine(m)

    def random_flip(self):
        if random.random() > 0.5:
            self.flip(1) # random x_flip
        if random.random() > 0.5:
            self.flip(0) # random f_flip

        # got bug
        # x_flip = 1 if random.random()>0.5 else -1
        # y_flip = 1 if random.random()>0.5 else -1
        # m = np.float32([[x_flip, 0, 0],
        #                 [0, y_flip, 0]])
        # self.warpaffine(m)

    def random_transform(self, T_rate=0.5):
        self.color_2_heat(info=False)
        # use heat_map to do transform is better than color_map, which may cause edge problem 

        if random.random() < T_rate:
            self.flip(1)
        if random.random() < T_rate:
            self.random_shear()
        if random.random() < T_rate:
            self.random_rotate()
        if random.random() < T_rate:
            self.random_drift()
        if random.random() < T_rate:
            self.random_scale()
        # self.check_and_fix_outsider_box()
        self.force_merge_ignoring_overlap()

    def _show_img(self, name='imshow', show_time=0):
        if self.type == 'heat_map':
            self.force_merge_ignoring_overlap()
            self.heat_2_color()
        if self.type == 'color_map':
            cv2.imshow(name + ' label', self.pixel_label)
            cv2.imshow(name, self.img)
            cv2.imshow(name + ' merge', np.array(self.pixel_label*0.3 + self.img*0.7).astype('uint8'))
            cv2.waitKey(show_time)
        elif self.type == 'gaussian_heat_map':
            merged_img = self.merge_heat_map()
            cv2.imshow(name + ' label', merged_img)
            merged_img = merged_img.reshape((*merged_img.shape, 1)).repeat(3, axis=2)
            cv2.imshow(name, self.img)
            cv2.imshow(name + ' merge', np.array(merged_img*0.3 + self.img*0.7).astype('uint8'))
            cv2.waitKey(show_time)

    def _write_down(self, output_img_path, method_list, ignore_overlap=False):
        # method could be choose from [npy, label_png]
        if 'npy' in method_list:
            if self.type == 'gaussian_heat_map':
                pic_format = output_img_path.split('.')[-1]
                npy_file_path = output_img_path.replace('.{}'.format(pic_format), '_GHM.npy')

                # cv2.imwrite(output_img_path, self.img)
                cv2.imencode('.jpg', self.img)[1].tofile(output_img_path)
                np.save(npy_file_path, self.pixel_label)
            else:
                self.color_2_heat()
                if ignore_overlap: self.force_merge_ignoring_overlap()

                _valid = self.check_overlap()
                assert _valid < 1, 'heat_map is overlaped, could not converted to color_map, enable ignore_overlap first'

                pic_format = output_img_path.split('.')[-1]
                npy_file_path = output_img_path.replace('.{}'.format(pic_format), '.npy')
                # cv2.imwrite(output_img_path, self.img)
                cv2.imencode('.jpg', self.img)[1].tofile(output_img_path)
                np.save(npy_file_path, self.onehot_label_with_background().argmax(axis=-1).astype(np.int8))

        if 'label_png' in method_list and self.type != 'gaussian_heat_map':
            self.heat_2_color()
            pic_format = output_img_path.split('.')[-1]
            label_file_path = output_img_path.replace('.{}'.format(pic_format), '_label.png')
            # cv2.imwrite(output_img_path, self.img)
            cv2.imencode('.jpg', self.img)[1].tofile(output_img_path)
            # cv2.imwrite(label_file_path, self.pixel_label)
            cv2.imencode('.jpg', self.pixel_label)[1].tofile(label_file_path)

    def _load_file(self, img_path, method='npy', annotation_file_path=None, total_class_number=None, class_color=None, gaussian_heat_map=False):
        # method could be choose from [npy, label_png]
        if method == 'npy':
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
            if gaussian_heat_map is False:
                if annotation_file_path is None:
                    pic_format = img_path.split('.')[-1]
                    npy_file_path = img_path.replace('.{}'.format(pic_format), '.npy')
                    annotation_file_path = npy_file_path

                label = np.load(annotation_file_path)
                if total_class_number is not None:
                    assert total_class_number >= (label.max+1), 'label classes exceed total_class_number'
                    _onehot_label = np.eye(total_class_number)[label]
                else:
                    _onehot_label = np.eye(label.max()+1)[label]
                # drop background label.
                onehot_label = _onehot_label[:, :, 1:]
                self.type = 'heat_map'
            else:
                if annotation_file_path is None:
                    pic_format = img_path.split('.')[-1]
                    npy_file_path = img_path.replace('.{}'.format(pic_format), '_GHM.npy')
                    annotation_file_path = npy_file_path

                label = np.load(annotation_file_path)
                onehot_label = label
                self.type = 'gaussian_heat_map'

            self.img = img
            self.pixel_label = onehot_label
            self.total_class_number = onehot_label.shape[-1]
            self.class_id = None

        if method == 'label_png':
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
            if annotation_file_path is None:
                pic_format = img_path.split('.')[-1]
                annotation_file_path = img_path.replace('.{}'.format(pic_format), '_label.png')
            if class_color is None:
                if total_class_number is None:
                    class_color = COLOR_POOL
                else:
                    class_color = COLOR_POOL[:, :total_class_number]
            # label = cv2.imread(annotation_file_path)
            label = cv2.imdecode(np.fromfile(annotation_file_path, dtype=np.uint8),-1)

            self.img = img
            self.pixel_label = label.astype(np.float64)
            # self.pixel_label = label will not cause any transform problem. but the img show will be a little bit different. Try and see what happen if you are interest.

            self.type = 'color_map'
            self.total_class_number = class_color.shape[-1]
            self.class_id = None

    def _switch_label_name(self, elder_label_list, new_label_list):
        _temp_class_id = []
        for i in range(len(self.class_id)):
            if self.class_id[i] in elder_label_list:
                _temp_class_id.append(new_label_list[elder_label_list.index(self.class_id[i])])
            else:
                _temp_class_id.append(self.class_id[i])

        self.class_id = _temp_class_id

    def _abalienate_unused_id(self):
        self.color_2_heat()
        _inverse_onehot = self.pixel_label.argmax(axis=-1)
        _back_ground = self.pixel_label.sum(axis=-1) == 0

        # due to the background is not a class in this container. Onehot may cause error.
        onehot = np.eye(_inverse_onehot.max()+1)[_inverse_onehot]
        onehot[_back_ground] = 0 # fix the background problem.

        self.pixel_label = onehot.copy()
        self.total_class_number = _inverse_onehot.max()+1
        self.class_id = self.class_id[0: _inverse_onehot.max()+1]

    def onehot_label_with_background(self):
        self.color_2_heat()
        _back_ground = self.pixel_label.sum(axis=-1) == 0
        _back_ground_label = np.ones(_back_ground.shape) * _back_ground
        _back_ground_label = _back_ground_label.reshape(*_back_ground_label.shape, 1)

        return np.concatenate((_back_ground_label, self.pixel_label), axis=-1)

    def onehot_label_with_nobackground(self):
        self.color_2_heat()
        return self.pixel_label


def _segment_4_part(target_size):
    height, width = target_size

    segment_point = (random.uniform(0.3, 0.7)*width, random.uniform(0.3, 0.7)*height)  # format (x, y)
    segment_point = [int(value) for value in segment_point]
    segment_list = [[0, 0, segment_point[0], segment_point[1]],
                    [segment_point[0], 0, width-1, segment_point[1]],
                    [0, segment_point[1], segment_point[0], height-1],
                    [segment_point[0], segment_point[1], width-1, height-1]]           # format [x_left, y_up, x_right, y_down]

    return segment_list


def mosaic_augumentation(pl_container_list, target_size):
    assert len(pl_container_list) == 4, 'now only 4 image mosaic available'
    assert isinstance(target_size, int) or isinstance(target_size, tuple),\
        'target_size should be a single int or tuple'
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    output_img = np.zeros((*target_size, pl_container_list[0].img.shape[2])).astype(np.uint8)
    output_pixel_label = np.zeros((*target_size, pl_container_list[0].pixel_label.shape[2]))

    segment_list = _segment_4_part(target_size)
    assert len(segment_list) == len(pl_container_list), \
        'segment part and prepare boundingbox should have same number'

    for i in range(len(segment_list)):
        _crop_size = (segment_list[i][3] - segment_list[i][1],
                      segment_list[i][2] - segment_list[i][0])
        try:
            crop_pl_container = pl_container_list[i]
            crop_pl_container.random_crop(_crop_size)
        except:
            crop_pl_container = pl_container_list[i].copy()
            crop_pl_container.resize(target_size)
            crop_pl_container.random_crop(_crop_size)

        # crop_boundingbox = boundingbox_container_list[i].random_crop(_crop_size)
        output_img[segment_list[i][1]: segment_list[i][3],
                   segment_list[i][0]: segment_list[i][2], :] = crop_pl_container.img.copy()

        output_pixel_label[segment_list[i][1]: segment_list[i][3],
                           segment_list[i][0]: segment_list[i][2], :] = crop_pl_container.pixel_label.copy()

    output_pl_container = Pixellabel_container(output_img,
                                               output_pixel_label,
                                               label_type=pl_container_list[0].type,
                                               class_id=crop_pl_container.class_id,
                                               total_class_number=crop_pl_container.total_class_number)

    return output_pl_container


def get_mask_with_expand(pixel_label, expand_range = 1, label_format='HWC'):
    # pixel_label shape shoulbe be 'HWC' or 'CHW'
    assert label_format == 'HWC' or label_format == 'CHW'
    import torch
    exist_map = pixel_label > 0
    expand_filter = torch.nn.MaxPool2d(2*expand_range+1, stride=1, padding= expand_range)
    if label_format == 'HWC':
        exist_map = exist_map.transpose(2, 0, 1)
    exist_map = torch.tensor(exist_map)
    expand_map = expand_filter(exist_map.view(1, *exist_map.shape).float())
    expand_map = expand_map.view(exist_map.shape).bool().numpy()
    if label_format == 'HWC':
        expand_map = expand_map.transpose(1, 2, 0)
    return expand_map


def _testing(testing_type):

    if testing_type == 'boundingbox':
        bb_container = Boundingbox_container(None, 0, 'box', 0, False)
        bb_container._load_file('./data/sample/bounding_box/captain.jpg')

        elder_name_list = ['cat_box', 'box', 'screwdriver_box']
        new_name_list = [0, 1, 2]
        bb_container._switch_label_name(elder_name_list, new_name_list)

        bb_container.centor2box()
        pl_container = bounddingbox_container_to_pixellabel_container(bb_container, 10)

    elif testing_type == 'keypoint':
        kp_container = Keypoint_container(None, None, None)
        kp_container._load_file('./data/sample/keypoint/40.png')
        kp_container.resize_cv2((400, 400))

        elder_name_list = ['eye', 'nose', 'mouth']
        new_name_list = [0, 1, 2]
        kp_container._switch_label_name(elder_name_list, new_name_list)
        pl_container = keypoint_contain_to_pixellabel_container(kp_container, 5, sigma=10)

    print('- showing resized image')
    pl_container.resize_cv2((int(pl_container.img.shape[0]*0.5), int(pl_container.img.shape[1]*0.5)))
    pl_container._show_img()

    print('- showing padded image')
    test_pl_container = pl_container.copy()
    test_pl_container.padding_image_cv2((412, 412))
    test_pl_container._show_img()

    print('- showing added blank_edge image')
    test_pl_container = pl_container.copy()
    test_pl_container.add_blank_edge((50, 0))
    test_pl_container._show_img()

    print('- showing rotated image')
    test_pl_container = pl_container.copy()
    test_pl_container.rotate(15)
    test_pl_container._show_img()

    print('- showing warpaffined image')
    test_pl_container = pl_container.copy()
    test_pl_container.warpaffine()
    test_pl_container._show_img()

    print('- showing warpperspective image')
    test_pl_container = pl_container.copy()
    test_pl_container.warpperspective()
    test_pl_container.force_merge_ignoring_overlap()
    test_pl_container._show_img()

    print('- showing random croped image')
    test_pl_container.random_crop(200, 200)
    test_pl_container._show_img('2')
    test_pl_container.color_2_heat()

    random_transform_time = 10
    for i in range(10):
        print('- showing 10 times random transform {}/{}'.format(i+1, random_transform_time), end='\r')
        test_pl_container = pl_container.copy()
        test_pl_container.random_transform()
        test_pl_container.force_merge_ignoring_overlap() # after transform better use force_merge_ignoring_overlap
        test_pl_container._show_img()

    if testing_type == 'boundingbox':
        # print('testing _switch_label_name')
        new_name_list = ['cat_box', 'box', 'screwdriver_box']
        elder_name_list = [0, 1, 2]
        test_pl_container = pl_container.copy()
        test_pl_container._abalienate_unused_id()
        # print('elder name:', test_pl_container.class_id)
        test_pl_container._switch_label_name(elder_name_list, new_name_list)
        # print('new name:', test_pl_container.class_id)

        write_down_path = './data/test_save/boundingbox2segment_image.jpg'
        write_down_format = ['npy', 'label_png']
        print('\n- testing write down to path: {}, in format: {}'.format(write_down_path, write_down_format))
        test_pl_container._write_down(write_down_path, write_down_format, ignore_overlap=True)
        test_pl_container._load_file(write_down_path, 'npy', gaussian_heat_map=False)
        test_pl_container._show_img()

    elif testing_type == 'keypoint':
        test_pl_container = pl_container.copy()
        write_down_path = './data/test_save/keypoint2segment_image.jpg'
        write_down_format = ['npy']
        print('\n- testing write down to path: {}, in format: {}'.format(write_down_path, write_down_format))
        test_pl_container._write_down(write_down_path, write_down_format)
        test_pl_container._load_file(write_down_path, 'npy', gaussian_heat_map=True)
        test_pl_container._show_img()

    test_pl_container = pl_container.copy()
    test_pl_container.random_warpperspective()
    # test_pl_container.force_merge_ignoring_overlap()
    test_pl_container.color_2_heat()
    pl_container.force_merge_ignoring_overlap()
    pl_container.color_2_heat()

    _PA = segment_PA(pl_container.pixel_label, test_pl_container.pixel_label, 2)
    print('- PA( pixel_label accuracy )', _PA)

    _CPA = segment_CPA(pl_container.pixel_label, test_pl_container.pixel_label, 2)
    print('- CPA( class pixel_label accuracy )', _CPA)

    _MPA = segment_MPA(pl_container.pixel_label, test_pl_container.pixel_label, 2)
    print('- MPA( mean pixel_label accuracy )', _MPA)


if __name__ == '__main__':
    testing_type = 'keypoint'
    # testing_type = 'boundingbox'
    _testing(testing_type)
