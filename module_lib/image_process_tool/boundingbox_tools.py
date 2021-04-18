import os
import sys
import cv2
import numpy as np
import math
import random
import json
import codecs


realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))


def _caculate_area(label_box):
    '''
    Description:
        caculate the area of the box.

    INPUT:
        label_box               -> [left, up, right, down]
    RETURN:
        area                    -> float
    '''
    w = max(0, label_box[2] - label_box[0])
    h = max(0, label_box[3] - label_box[1])
    area = w*h
    return area


def _caculate_overlapping_area(label_box1, label_box2, box_need=False):
    '''
    Discription:
        caculate the overlapping area between two boxes.

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]
        box_need                -> bool

    RETURN:
        if box_need is ture:
            overlapping_area,       -> float
            overlapping_box_label   -> [left, up, right, down]
        else:
            overlapping_area,       -> float
    '''

    cross_area_left = max(label_box1[0], label_box2[0])
    cross_area_up = max(label_box1[1], label_box2[1])
    cross_area_right = min(label_box1[2], label_box2[2])
    cross_area_down = min(label_box1[3], label_box2[3])

    overlapping_area = _caculate_area([cross_area_left, cross_area_up, cross_area_right, cross_area_down])
    if box_need is True:
        if (cross_area_left < cross_area_right) and (cross_area_up < cross_area_down):
            return overlapping_area, (cross_area_left, cross_area_up, cross_area_right, cross_area_down)
        else:
            return overlapping_area, (0, 0, 0, 0)
    else:
        return overlapping_area


def _caculate_smallest_enclosing_area(label_box1, label_box2, box_need=False):
    '''
    Discription:
        caculate the smallest enclosing area of two boxes.

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]
        box_need                -> bool

    RETURN:
        if box_need is ture:
            enclosing_area,       -> float
            enclosing_box         -> [left, up, right, down]
        else:
            enclosing_area,       -> float
    '''

    enclosing_area_left = min(label_box1[0], label_box2[0])
    enclosing_area_up = min(label_box1[1], label_box2[1])
    enclosing_area_right = max(label_box1[2], label_box2[2])
    enclosing_area_down = max(label_box1[3], label_box2[3])

    enclosing_box = [enclosing_area_left, enclosing_area_up, enclosing_area_right, enclosing_area_down]
    enclosing_area = _caculate_area(enclosing_box)
    if box_need is True:
        return enclosing_area, enclosing_box
    else:
        return enclosing_area


def l2_distance(point1, point2):
    return np.square(point1[0]-point2[0]) + np.square(point1[1]-point2[1])


def IoU(label_box1, label_box2):
    '''
    Discription:
        caculate the Intersection over Union(IoU)

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]

    RETURN:
        iou                     -> float, range from 0 ~ 1.
    '''
    overlapping_area = _caculate_overlapping_area(label_box1, label_box2)
    iou = overlapping_area/(_caculate_area(label_box1) + _caculate_area(label_box2) - overlapping_area + 1e-6)
    return iou


def GIoU(label_box1, label_box2):
    '''
    Discription:
        caculate the Generalized Intersection over Union(GIoU)

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]

    RETURN:
        giou                     -> float, range from 0 ~ 1.
    '''
    enclosing_area = _caculate_smallest_enclosing_area(label_box1, label_box2)
    overlapping_area = _caculate_overlapping_area(label_box1, label_box2)

    iou = overlapping_area/(_caculate_area(label_box1) + _caculate_area(label_box2) - overlapping_area + 1e-6)
    giou = iou - (enclosing_area - (_caculate_area(label_box1) + _caculate_area(label_box2) - overlapping_area))/enclosing_area
    return giou


def DIoU(label_box1, label_box2):
    '''
    Discription:
        caculate the Distance-IoU(DIoU)

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]

    RETURN:
        diou                     -> float, range from -1 ~ 1.
    '''
    enclosing_area_left = min(label_box1[0], label_box2[0])
    enclosing_area_up = min(label_box1[1], label_box2[1])
    enclosing_area_right = max(label_box1[2], label_box2[2])
    enclosing_area_down = max(label_box1[3], label_box2[3])

    box_1_centor = ((label_box1[0]+label_box1[2])/2, (label_box1[1]+label_box1[3])/2)
    box_2_centor = ((label_box2[0]+label_box2[2])/2, (label_box2[1]+label_box2[3])/2)

    diou = IoU(label_box1, label_box2) - \
        l2_distance(box_1_centor, box_2_centor) / \
        l2_distance((enclosing_area_left, enclosing_area_up), (enclosing_area_right, enclosing_area_down))
    return diou


def CIoU(label_box1, label_box2):
    '''
    Discription:
        caculate the Complete-IoU(CIoU)

    INPUT:
        label_box1              -> [left, up, right, down]
        label_box2              -> [left, up, right, down]

    RETURN:
        ciou                     -> float
    '''
    v_scale = 4/np.square(math.pi) \
              * np.square(np.arctan(abs(label_box1[0] - label_box1[2])/abs(label_box1[1] - label_box1[3])) \
                          - np.arctan(abs(label_box2[0] - label_box2[2])/abs(label_box2[1] - label_box2[3])))
    alpha = v_scale/(1 - IoU(label_box1, label_box2) + v_scale + 1e-6)

    ciou = DIoU(label_box1, label_box2) + v_scale*alpha
    return ciou


def _box_rotate(label, angle, matRotate, fix=False):
    '''
    Discription:
        Rotate the box with angle and matRotate(affine_metrix). The
    matRotate metrix must be generate by angle. If 'fix' is True,
    Then the rotated box position will be fix more accurate.

    INPUT:    angle                   ->int, range from 0 ~ 360
              matRotation             ->metrix, shape (2,3)
                                        generate by cv2.getAffineTransform
    RETURN:   new_box                 ->[left, up, right, down]
              rotated_porint_list     ->[leftup_point_rotated(x ,y),
                                         rightup_point_rotated(x ,y),
                                         leftdown_point_rotated(x ,y)
                                         rightdown_point_rotated(x ,y)]

    Tips:
        The 'fix' operator is a experimental function. Change the max_fix_scale
    variable to adjust the effect.
    '''

    angle_pi = angle/180 * math.pi

    point_1 = (label[0], label[1], 1)
    point_2 = (label[2], label[1], 1)
    point_3 = (label[0], label[3], 1)
    point_4 = (label[2], label[3], 1)

    point_1 = matRotate.dot(point_1)[0:2]
    point_2 = matRotate.dot(point_2)[0:2]
    point_3 = matRotate.dot(point_3)[0:2]
    point_4 = matRotate.dot(point_4)[0:2]

    ultra_left = min(point_1[0], point_2[0], point_3[0], point_4[0])
    ultra_up = min(point_1[1], point_2[1], point_3[1], point_4[1])
    ultra_right = max(point_1[0], point_2[0], point_3[0], point_4[0])
    ultra_down = max(point_1[1], point_2[1], point_3[1], point_4[1])

    if fix is True:
        diff_hor = ultra_right - ultra_left
        diff_ver = ultra_down - ultra_up
        max_fix_scale = 1/6

        fix_scale = min(abs(math.cos(angle_pi)), abs(math.sin(angle_pi))) \
            / max(abs(math.cos(angle_pi)), abs(math.sin(angle_pi)))

        ultra_left = ultra_left + max_fix_scale * fix_scale * diff_hor
        ultra_up = ultra_up + max_fix_scale * fix_scale * diff_ver
        ultra_right = ultra_right - max_fix_scale * fix_scale * diff_hor
        ultra_down = ultra_down - max_fix_scale * fix_scale * diff_ver

    return [ultra_left, ultra_up, ultra_right, ultra_down], [point_1, point_2, point_3, point_4]


def _point_perspective(point, P_metrix):
    '''
    Discription:
        perspective transform for point.
    INPUT:        point       -> (x, y)
                  P_metrix    -> perspective metirx
                                 get from cv2.getPerspectiveTransform
    OUTPUT:       point       -> (x, y), the data type of xy is always float.
    '''

    P_metrix = P_metrix.T
    _temp_metrix = point.dot(P_metrix)
    x_u = _temp_metrix[0] / _temp_metrix[2]
    y_u = _temp_metrix[1] / _temp_metrix[2]
    return (x_u, y_u)


def _box_perspective(label, P_metrix, fix=False):
    '''
        perspective transfrom with the box and P_metrix(perspective
    metirx).If 'fix' is True, then the rotated box position will be
    fix more accurate.(fix function is still not implement.)

    INPUT:    P_metrix                ->metrix, shape (3,3)
                                        generate by cv2.getPerspectiveTransform
    RETURN:   new_box                 ->[left, up, right, down]
              rotated_porint_list     ->[leftup_point_perspectived,
                                         rightup_point_perspectived,
                                         leftdown_point_perspectived
                                         rightdown_point_perspectived]
                                        The point format is (x, y)
    '''

    point_1 = np.array((label[0], label[1], 1))
    point_2 = np.array((label[2], label[1], 1))
    point_3 = np.array((label[0], label[3], 1))
    point_4 = np.array((label[2], label[3], 1))

    point_1 = _point_perspective(point_1, P_metrix)
    point_2 = _point_perspective(point_2, P_metrix)
    point_3 = _point_perspective(point_3, P_metrix)
    point_4 = _point_perspective(point_4, P_metrix)

    ultra_left = min(point_1[0], point_2[0], point_3[0], point_4[0])
    ultra_up = min(point_1[1], point_2[1], point_3[1], point_4[1])
    ultra_right = max(point_1[0], point_2[0], point_3[0], point_4[0])
    ultra_down = max(point_1[1], point_2[1], point_3[1], point_4[1])

    return (ultra_left, ultra_up, ultra_right, ultra_down), [point_1, point_2, point_3, point_4]


class Boundingbox_container(object):
    """
    Discription:
        The Boundingbox_container collect the image and its box_labels,
    Then do the transform as padding/resize/crop/affine/perspective.

    INITIALIZE_INPUT:
        img         ->  The image, cv2 fromat
        label       ->  The box label.The label can be two type. [yolo / xml]
                        The label parameter should be a list/array/tuple
                        'center' is like the yolo label.
                            - label[i][0] = center_point_x (scale, range(0,1))
                            - label[i][1] = center_point_y (scale, range(0,1))
                            - label[i][2] = box_width (scale, range(0,1))
                            - label[i][3] = box_height (scale, range(0,1))
                        'box' is the normal label.
                            - label[i][0] = left pixel (pixel position)
                            - label[i][1] = upper pixel (pixel position)
                            - label[i][2] = right pixel (pixel position)
                            - label[i][3] = downer pixel (pixel position)
        class_id    -> class_id should have same lenth as label.

    TIPS:
        cv2_img.shape is [height, width, channal]
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)
        cv2_resize(img, (width, height))

        all the target_size in this code is [height, width]
        all the box label in this code is [left, up, right, down]
        all the point in this code is (x, y)

    FUNCTION:
        abalienate_zero_id      -> abalienate id-zero
        copy                    -> create an identity new container with same content.
        image_shape             -> return img.shape
        check_value_position    -> check the label format to be [left, up, right, dowm]
        centor2box              -> convert centor_label to box_label
        box2centor              -> convert box_label to centor_label
        padding_image_cv2       -> padding and resize.
        add_blank_edge          -> adding blank_edge.
        resize_cv2              -> resize
        box_cover_rate_with_target_label             -> Goto the function for detail
        crop                    -> crop with the box
        random_crop             -> random crop.
        center_crop             -> centor crop
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
        check_and_fix_outsider_box  -> after transform box may go outsider of the img, this can fix it
        _show_img               -> show img via CV2
        _write_down             -> write down img and label. label could be txt(darknet) or json(labelme)
        _load_file              -> load img and label. label could be txt(darknet) or json(labelme)
        _switch_label_name      -> switch label name, always can be used to switch label_name from str_type to number_type.

    OTHERS:
        box_label to pixel_label -> see 'pixel_label_tools' for more detail.
    """
    def __init__(self, img, label, label_type, class_id, check_value=True):
        super(Boundingbox_container, self).__init__()
        self.img = img
        self.label = np.array(label)
        assert label_type == 'center' or label_type == 'box', 'label_type must be center or box'
        self.label_type = label_type
        self.class_id = class_id
        if check_value is True:
            self.check_value_position()

    def abalienate_zero_id(self):
        _temp_id = np.array(self.class_id)
        _temp_id = _temp_id + 1
        self.class_id = _temp_id.astype('int8')

    def copy(self):
        return Boundingbox_container(self.img.copy(), self.label.copy(), self.label_type, self.class_id.copy())

    def image_shape(self):
        return self.img.shape

    def check_value_position(self):
        '''
        In some case right/left, up/down may be fused. Use this function to fix it.
        Fix the order to be 'box' format.
            - label[i][0] = left pixel (pixel position)
            - label[i][1] = upper pixel (pixel position)
            - label[i][2] = right pixel (pixel position)
            - label[i][3] = downer pixel (pixel position)
        '''
        _trans_mark = False
        if self.label_type == 'center':
            self.centor2box()
            _trans_mark = True

        _temp_label = self.label.copy()
        for i in range(len(self.label)):
            if self.label[i][0] > self.label[i][2]:
                _temp_label[i][0] = self.label[i][2]
                _temp_label[i][2] = self.label[i][0]
            if self.label[i][1] > self.label[i][3]:
                _temp_label[i][1] = self.label[i][3]
                _temp_label[i][3] = self.label[i][1]
            self.label[i] = _temp_label[i].copy()

        if _trans_mark is True:
            self.box2centor()

    def centor2box(self):
        # change 'centor format' to 'box format'
        if self.label_type == 'center':
            image_shape = self.image_shape()

            for i in range(len(self.label)):
                new_label = self.label[i].copy()
                new_label = new_label*np.array([image_shape[1], image_shape[0],
                                                image_shape[1], image_shape[0]])
                self.label[i][0:2] = new_label[0:2] - new_label[2:4]/2
                self.label[i][2:4] = new_label[0:2] + new_label[2:4]/2

            self.label = self.label.astype(np.int)
            self.label_type = 'box'
        # else:
        #     print('the Bounddingbox label is already box type')

    def box2centor(self):
        # change 'box format' to 'centor format'
        if self.label_type == 'box':
            image_shape = self.image_shape()
            self.label = np.array(self.label)
            self.label = self.label.astype(np.float)

            for i in range(len(self.label)):
                new_label = self.label[i].copy()
                new_label[0:2] = (self.label[i][2:4] + self.label[i][0:2])/2
                new_label[2:4] = (self.label[i][2:4] - self.label[i][0:2])
                self.label[i] = new_label/np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
            self.label_type = 'center'
        # else:
        #     print('the Bounddingbox label is already center type')

    def padding_image_cv2(self, target_size):
        # This operation will keep the object scale, but not warping them.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.centor2box()
        image_scale = self.image_shape()[0] / self.image_shape()[1]
        target_size_scale = target_size[0] / target_size[1]

        padding_pixel = [0, 0]  # [horizontal, vetical]
        if image_scale < target_size_scale:
            padding_pixel[0] = int(self.image_shape()[1]*(target_size_scale - image_scale))
        else:
            padding_pixel[1] = int(self.image_shape()[0]/target_size_scale
                                   - self.image_shape()[0]/image_scale)

        blank_image = np.zeros((self.image_shape()[0] + padding_pixel[0],
                                self.image_shape()[1] + padding_pixel[1],
                                self.image_shape()[2])).astype(np.uint8)
        blank_image_shape = blank_image.shape
        blank_image[int(padding_pixel[0]/2): self.image_shape()[0] + int(padding_pixel[0]/2),
                    int(padding_pixel[1]/2): self.image_shape()[1] + int(padding_pixel[1]/2),
                    :] = self.img
        new_image = blank_image

        if len(self.label) != 0:
            self.label = self.label + (np.array([padding_pixel[1],
                                                 padding_pixel[0],
                                                 padding_pixel[1],
                                                 padding_pixel[0]]) / 2).astype(np.int)

            self.label = (self.label
                          / np.array([blank_image_shape[1]/target_size[1],
                                      blank_image_shape[0]/target_size[0],
                                      blank_image_shape[1]/target_size[1],
                                      blank_image_shape[0]/target_size[0]])).astype(np.int)

        self.img = cv2.resize(new_image, (target_size[1], target_size[0]))

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

        new_label = self.label.copy()
        new_label = new_label + np.array([pixels[1]/2,
                                          pixels[0]/2,
                                          pixels[1]/2,
                                          pixels[0]/2]).astype(np.int)

        self.img = new_image.astype('uint8')
        self.label = new_label

    def resize_cv2(self, target_size):
        # resize the image. This transform may wrap the object.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.centor2box()
        self.label = (self.label
                      / np.array((self.image_shape()[1]/target_size[1],
                                  self.image_shape()[0]/target_size[0],
                                  self.image_shape()[1]/target_size[1],
                                  self.image_shape()[0]/target_size[0]))
                      ).astype(np.int)
        self.img = cv2.resize(self.img, (target_size[1], target_size[0]))

    def box_cover_rate_with_target_label(self, target_label, keep_object_rate, number_at_least=1):
        '''
        Discription:
           calculate the rate of the cover_area for every label box.

        INPUT:
            target_label            -> [left, up, right, down]
            keep_object_rate        -> range from 0~1
            number_at_least         -> number_at_least >= 0
        RETURN:
            overlapping_area_rate_list      -> list,lenth of which equal to the number of label
            accept_label_index              -> list,contain indexes of boxes,covering bigger area than keep_object_rate
            status                          -> 'valid'/ 'invalid'

        Tips:
            if the number(box_cover_area > keep_object_rate) > number_at_least.
            Then the status if 'valid'.
        '''
        overlapping_area_rate_list = [_caculate_overlapping_area(target_label, box)/_caculate_area(box) for box in self.label]
        accept_label_index = np.argwhere(np.array(overlapping_area_rate_list) > keep_object_rate)

        if len(accept_label_index) >= number_at_least:
            status = 'valid'
        else:
            status = 'invalid'
        return overlapping_area_rate_list, accept_label_index, status

    def crop(self, crop_box, accept_label_index=None, keep_object_threshold=0.6):
        '''
        Discription:
            crop the image and label with crop_box. 
        INPUT:
                crop_box            -> [left, up, right, down]
                accept_label_index  -> a list record which label_box is keep.
                                       If accept_label_index is None, then
                                       it will be caculate by value of
                                       keep_object_threshold.
                keep_object_threshold   -> float, set value to retain the box.
        RETURN:
                null
        '''

        crop_box = [int(value) for value in crop_box]
        _temp_img = np.zeros((crop_box[3] - crop_box[1], crop_box[2] - crop_box[0], self.image_shape()[2]))
        _temp_img = self.img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :].copy()
        _temp_label = self.label.copy()

        if accept_label_index is None:
            __, accept_label_index, __ = self.box_cover_rate_with_target_label(crop_box, keep_object_threshold)

        accept_label_index = np.reshape(accept_label_index, -1)
        _temp_label = [_temp_label[index] for index in accept_label_index]

        for i in range(len(_temp_label)):
            drift_box = _caculate_overlapping_area(crop_box, _temp_label[i], box_need=True)[1]
            _temp_label[i] = [drift_box[0] - crop_box[0],
                              drift_box[1] - crop_box[1],
                              drift_box[2] - crop_box[0],
                              drift_box[3] - crop_box[1]]

        _temp_id = [self.class_id[index] for index in accept_label_index]

        self.img = _temp_img.copy()
        self.label = _temp_label
        self.class_id = _temp_id

    def random_crop(self, target_size, padding=False, keep_object_rate=0.6,
                    keep_object_number=1, tolerant_step=10):
        '''
        Discription:
            random_crop
        INPUT:
                target_size             -> (height, width) or side_width
                padding                 -> bool, if you want to get the edge pixel
                                           on the middle img. Set True. 
                keep_object_rate        -> float, set to keep the box with
                                           higher iou than this value.
                keep_object_number      -> at least how many objects is keep.
                tolerant_step           -> times to try keep object. Exceed the
                                           value will force to generate the crop
                                           img, ignoring the object limited
                                           parameters.
        RETURN:
                null
        '''
        assert isinstance(target_size, int) or isinstance(target_size, tuple),\
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        assert (target_size[0] <= self.img.shape[0] and target_size[1] <= self.img.shape[1]),\
            'the target_size exceed the range of image, {}'.format(self.img.shape)

        assert keep_object_rate <= 1, 'keep_object_rate must in range(0, 1)'
        if keep_object_number > len(self.label):
            keep_object_number == len(self.label)
        self.centor2box()

        image_shape = self.img.shape
        crop_size = target_size

        if padding:
            padding_size = (crop_size[0] + image_shape[0], crop_size[1] + image_shape[1])
            self.add_blank_edge(tuple(crop_size))
        else:
            padding_size = (image_shape[0], image_shape[1])

        _step = 0

        while _step < tolerant_step:
            _step += 1

            pick_point_up = random.randint(0, padding_size[0] - crop_size[0])
            pick_point_left = random.randint(0, padding_size[1] - crop_size[1])
            pick_point_down = pick_point_up + crop_size[0]
            pick_point_right = pick_point_left + crop_size[1]
            pick_box = (pick_point_left, pick_point_up, pick_point_right, pick_point_down)
            overlapping_area_rate_list, accept_label_index, status = self.box_cover_rate_with_target_label(pick_box, keep_object_rate, keep_object_number)
            if status == 'valid':
                break

        if status == 'invalid':
            print('force to crop')
        self.crop(pick_box, accept_label_index)
        self.padding_image_cv2(target_size)

    def center_crop(self, keep_rate=0.5):
        '''
        Discription:
            crop and remain the center area

        INPUT:
            keep_rate               -> range from 0~1, based on lenth ratio
        RETURN:
            null
        '''
        image_shape = self.image_shape()
        height, width, channal = image_shape
        drop_rate = 1 - keep_rate
        half_drop_rate = drop_rate/2
        pick_box = (int(width*half_drop_rate), int(height*half_drop_rate), int(width*(1 - half_drop_rate)), int(height*(1 - half_drop_rate)))

        self.crop(pick_box)

    def flip(self, method):
        # flip the label and image
        # method = horizontal/vertical
        image_shape = self.image_shape()
        height, width, channal = image_shape

        _temp_label = np.array(self.label)
        _temp_img = self.img

        if method == 'horizontal' or method == 1:
            _temp_label[:, 0] = width - _temp_label[:, 0]
            _temp_label[:, 2] = width - _temp_label[:, 2]
            _temp_img = cv2.flip(self.img, 1)
        elif method == 'vertical' or method == 0:
            _temp_label[:, 1] = height - _temp_label[:, 1]
            _temp_label[:, 3] = height - _temp_label[:, 3]
            _temp_img = cv2.flip(self.img, 0)

        self.img = _temp_img
        self.label = _temp_label
        self.check_value_position()

    def rotate(self, angle, center=None, scale=1):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if center is None:
            center = (width*0.5, height*0.5)
        # if scale >= 1:
        #     self.add_blank_edge(int(height*scale*0.2), int(width*scale*0.2))

        _temp_img = self.img.copy()
        _temp_label = self.label

        matRotate = cv2.getRotationMatrix2D(center, angle, scale)
        dst = cv2.warpAffine(_temp_img, matRotate, (width, height))

        self.img = dst
        self.point_list = []
        for i in range(len(_temp_label)):
            _new_label, point_list = _box_rotate(_temp_label[i], angle, matRotate, fix=True)
            self.label[i] = _new_label
            self.point_list.append(point_list)

    def warpaffine(self, M=None):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if M is None:
            pts1 = np.float32([[0, 0],
                               [width - 1, 0],
                               [0, height - 1]])
            pts2 = np.float32([[width*0.2, height*0.1],
                               [width*0.9, height*0.2],
                               [width*0.1, height*0.9]])
            M = cv2.getAffineTransform(pts1, pts2)

        _temp_img = self.img.copy()
        _temp_label = self.label
        angle = 0
        dst = cv2.warpAffine(_temp_img, M, (width, height))
        self.img = dst

        self.point_list = []
        for i in range(len(_temp_label)):
            # print('_temp_label[{}]'.format(i), _temp_label[i])
            _new_label, point_list = _box_rotate(_temp_label[i], angle, M)
            self.label[i] = _new_label
            self.point_list.append(point_list)

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

        _temp_img = self.img.copy()
        _temp_label = self.label

        dst = cv2.warpPerspective(_temp_img, P_metrix, (width, height))
        self.img = dst
        self.point_list = []

        for i in range(len(_temp_label)):
            # print('_temp_label[{}]'.format(i), _temp_label[i])
            _new_label, point_list = _box_perspective(_temp_label[i], P_metrix)
            self.label[i] = _new_label
            self.point_list.append(point_list)

    def random_warpperspective(self, drift_range=0.05):
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
        self.check_and_fix_outsider_box()

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

        # x_flip = 1 if random.random()>0.5 else -1
        # y_flip = 1 if random.random()>0.5 else -1
        # m = np.float32([[x_flip, 0, 0],
        #                 [0, y_flip, 0]])
        # self.warpaffine(m)

    def random_transform(self, T_rate=0.5, limit_op=3):
        op_count = limit_op
        if random.random() < T_rate and op_count > 0:
            self.flip(1)
            op_count -= 1
        if random.random() < T_rate and op_count > 0:
            self.random_shear()
            op_count -= 1
        if random.random() < T_rate and op_count > 0:
            self.random_rotate()
            op_count -= 1
        if random.random() < T_rate and op_count > 0:
            self.random_drift()
            op_count -= 1
        if random.random() < T_rate and op_count > 0:
            self.random_scale()
            op_count -= 1
        self.check_and_fix_outsider_box()

    def check_and_fix_outsider_box(self):
        invalid_fix = 0
        self.centor2box()
        _temp_label = self.label.copy() 
        if type(self.label) == np.ndarray: self.label = self.label.tolist()
        if type(self.class_id) == np.ndarray: self.class_id = self.class_id.tolist()
        # _temp_label = self.label.copy()

        for i in range(len(self.label)):
            _temp_label[i][0] = max(self.label[i - invalid_fix][0], 0)
            _temp_label[i][1] = max(self.label[i - invalid_fix][1], 0)
            _temp_label[i][2] = min(self.label[i - invalid_fix][2], self.img.shape[1])
            _temp_label[i][3] = min(self.label[i - invalid_fix][3], self.img.shape[0])

            _new_box_iou = IoU(_temp_label[i], self.label[i - invalid_fix])
            # print(_new_box_iou)
            if _new_box_iou < 0.55:
                # print('self.label', self.label)
                self.label.pop(i - invalid_fix)
                self.class_id.pop(i - invalid_fix)
                invalid_fix += 1
            else:
                # print('self.label', self.label)
                self.label[i - invalid_fix] = _temp_label[i]

    def _show_img(self, name='imshow', show_time=0):
        self.centor2box()
        _img = self.img.copy()
        for i in range(len(self.label)):
            cv2.rectangle(_img,
                          (self.label[i][0], self.label[i][1]),
                          (self.label[i][2], self.label[i][3]),
                          (0, 255, 0),
                          2)
        cv2.imshow(name, _img)
        cv2.waitKey(show_time)

    def _write_down(self, output_img_path, method_list):
        '''
        Discription:
            write_down image and label. support ['.json', '.txt'] label format.

        INPUT:
            output_img_path             -> str
            method_list                 -> ['.json'] or ['.txt'] or ['.json', '.txt']
        RETURN:
            Null
        '''
        # cv2.imwrite(output_img_path, self.img)
        cv2.imencode('.jpg', self.img)[1].tofile(output_img_path)

        if 'json' in method_list:
            self.centor2box()
            pic_format = output_img_path.split('.')[-1]
            output_json_path = output_img_path.replace('.{}'.format(pic_format), '.json')

            baisc_json_format = {"signed": "zen_V1.0.0",
                                 "flags": {},
                                 "imagePath": output_img_path,
                                 "imageData": None,
                                 "imageHeight": float(self.img.shape[0]),
                                 "imageWidth": float(self.img.shape[1])
                                 }
            box_list = []
            for i in range(len(self.label)):
                label = self.label[i]
                box = {"label": self.class_id[i],
                       "points": [[float(label[0]), float(label[1])],
                                  [float(label[2]), float(label[3])]],
                       "group_id": None,
                       "shape_type": "rectangle",
                       "flags": {}}
                box_list.append(box.copy())

            baisc_json_format.update({"shapes": box_list})
            with open(output_json_path, "w") as fjson:
                fjson.write(json.dumps(baisc_json_format, indent=2))

        if 'txt' in method_list:
            self.box2centor()
            pic_format = output_img_path.split('.')[-1]
            output_txt_path = output_img_path.replace('.{}'.format(pic_format), '.txt')

            with open(output_txt_path, 'w') as F:
                for i in range(len(self.label)):
                    label_index = self.class_id[i]
                    center_x = self.label[i][0]
                    center_y = self.label[i][1]
                    side_x = self.label[i][2]
                    side_y = self.label[i][3]
                    F.write('{} {} {} {} {}\n'.format(label_index, abs(center_x), abs(center_y), abs(side_x), abs(side_y)))

    def _load_file(self, img_path, method='json', annotation_file_path=None):
        '''
        Discription:
            load image with label. support ['.json', '.txt'] label format.

        INPUT:
            output_img_path             -> str
            method_list                 -> ['.json'] or ['.txt'] or ['.json', '.txt']
            annotation_file_path        -> str.
        RETURN:
            Null

        Tips:
            if annotation_file and the img under same folder path, annotation_file_path
        can be set to None.
        '''
        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)

        _temp_label = []
        _class_id = []

        if method == 'json':
            if annotation_file_path is None:
                pic_format = img_path.split('.')[-1]
                annotation_file_path = img_path.replace('.{}'.format(pic_format), '.json')

            with codecs.open(annotation_file_path, 'r', encoding='UTF-8') as F:
                text = F.read()
            # annotation = json.loads(text, encoding="GB2312")
            annotation = json.loads(text)

            for i in range(len(annotation['shapes'])):
                if annotation['shapes'][i]['shape_type'] == 'rectangle':
                    _temp_label.append([int(annotation['shapes'][i]['points'][0][0]),
                                        int(annotation['shapes'][i]['points'][0][1]),
                                        int(annotation['shapes'][i]['points'][1][0]),
                                        int(annotation['shapes'][i]['points'][1][1]),
                                        ])
                    _class_id.append(annotation['shapes'][i]['label'])
            label_type = 'box'

        if method == 'txt':
            if annotation_file_path is None:
                pic_format = img_path.split('.')[-1]
                annotation_file_path = img_path.replace('.{}'.format(pic_format), '.txt')
            with open(annotation_file_path, 'r') as F:
                content = F.readlines()
                for i in range(len(content)):
                    segment = content[i].split(' ')
                    _temp_label.append([float(segment[1]),
                                        float(segment[2]),
                                        float(segment[3]),
                                        float(segment[4]),
                                        ])
                    _class_id.append(int(segment[0]))
            label_type = 'center'

        self.img = img
        self.label = np.array(_temp_label)
        self.class_id = _class_id
        self.label_type = label_type
        self.check_value_position()

    def _switch_label_name(self, elder_label_list, new_label_list):
        # take care of the order.
        _temp_class_id = []
        for i in range(len(self.class_id)):
            if self.class_id[i] in elder_label_list:
                _temp_class_id.append(new_label_list[elder_label_list.index(self.class_id[i])])
            else:
                _temp_class_id.append(self.class_id[i])

        self.class_id = _temp_class_id

def _segment_4_part(target_size):
    # get random img size of 4 part. Then you can merge them to one img.
    height, width = target_size

    segment_point = (random.uniform(0.3, 0.7)*width, random.uniform(0.3, 0.7)*height)  # format (x, y)
    segment_point = [int(value) for value in segment_point]
    segment_list = [[0, 0, segment_point[0], segment_point[1]],
                    [segment_point[0], 0, width-1, segment_point[1]],
                    [0, segment_point[1], segment_point[0], height-1],
                    [segment_point[0], segment_point[1], width-1, height-1]]           # format [x_left, y_up, x_right, y_down]

    return segment_list


def mosaic_augumentation(boundingbox_container_list, target_size):
    # doing mosaic augumentation on 4 boundingbox_container.
    # return a new container.
    assert len(boundingbox_container_list) == 4, 'now only 4 image mosaic available'
    assert isinstance(target_size, int) or isinstance(target_size, tuple),\
        'target_size should be a single int or tuple'
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    output_img = np.zeros((*target_size, boundingbox_container_list[0].img.shape[2])).astype(np.uint8)
    output_label = []
    output_class_id = []

    segment_list = _segment_4_part(target_size)

    assert len(segment_list) == len(boundingbox_container_list), \
        'segment part and prepare boundingbox should have same number'

    for i in range(len(segment_list)):
        _crop_size = (segment_list[i][3] - segment_list[i][1],
                      segment_list[i][2] - segment_list[i][0])
        try:
            crop_boundingbox = boundingbox_container_list[i].copy()
            crop_boundingbox.random_crop(_crop_size)
        except:
            crop_boundingbox = boundingbox_container_list[i].copy()
            crop_boundingbox.resize_cv2(target_size)
            crop_boundingbox.random_crop(_crop_size)

        crop_boundingbox.centor2box()

        output_img[segment_list[i][1]: segment_list[i][3],
                   segment_list[i][0]: segment_list[i][2], :] = crop_boundingbox.img.copy()
        _temp_label = list(crop_boundingbox.label)
        for j in range(len(_temp_label)):
            _temp_label[j] = [_temp_label[j][0] + segment_list[i][0],
                              _temp_label[j][1] + segment_list[i][1],
                              _temp_label[j][2] + segment_list[i][0],
                              _temp_label[j][3] + segment_list[i][1]]
        output_label.extend(_temp_label)
        output_class_id.extend(list(crop_boundingbox.class_id))

    output_bounding_box = Boundingbox_container(output_img, output_label, 'box', output_class_id)

    return output_bounding_box


def _testing():

    bb_container = Boundingbox_container(None, 0, 'box', 0, False)
    bb_container._load_file('./data/sample/bounding_box/captain.jpg', 'txt')

    print('- showing original image')
    bb_container._show_img()

    print('- showing resized image')
    bb_container.resize_cv2((int(bb_container.img.shape[0]*0.5), int(bb_container.img.shape[1]*0.5)))
    bb_container._show_img()

    print('- showing padded image')
    test_container = bb_container.copy()
    test_container.padding_image_cv2((512, 512))
    test_container._show_img('padding_image')

    print('- showing added blank_edge image')
    test_container = bb_container.copy()
    test_container.add_blank_edge((30, 100))
    test_container._show_img('add_blank_edge')

    print('- showing random croped image')
    test_container = bb_container.copy()
    test_container.random_crop((400, 300))
    test_container._show_img('random_crop')

    print('- showing center croped image')
    test_container = bb_container.copy()
    test_container.center_crop()
    test_container._show_img('center_crop')

    print('- showing fliped image')
    test_container = bb_container.copy()
    test_container.flip('vertical')
    test_container._show_img('flip')

    print('- showing rotated image')
    test_container = bb_container.copy()
    test_container.rotate(15, scale=0.7)
    test_container._show_img('rotate')

    print('- showing warpaffined image')
    test_container = bb_container.copy()
    test_container.warpaffine()
    test_container._show_img('warpaffine')

    print('- showing warpperspective image')
    test_container = bb_container.copy()
    test_container.warpperspective()
    test_container._show_img('warpperspective')

    random_transform_time = 10
    for i in range(10):
        print('- showing 10 times random transform {}/{}'.format(i+1, random_transform_time), end='\r')
        test_container = bb_container.copy()
        test_container.random_transform()
        test_container._show_img('random_transform')

    print('\n- showing mosaic_augumentation')
    bb_container_list = [bb_container.copy(), bb_container.copy(),
                         bb_container.copy(), bb_container.copy()]
    test_container = mosaic_augumentation(bb_container_list, (512, 512))
    test_container._show_img('mosaic_augumentation')

    write_down_path = './data/test_save/boundingbox_image.jpg'
    write_down_format = ['json', 'txt']
    print('- test write_down image to path: {}, in format: {}.'.format(write_down_path, write_down_format))
    test_container._write_down(write_down_path, write_down_format)


if __name__ == '__main__':
    _testing()
