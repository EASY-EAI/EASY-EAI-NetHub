import cv2
import PIL
import numpy as np
import math
import random
import json

class Keypoint_container(object):
    '''
    Discription:
        The Keypoint_container collect the image and its keypoint_labels,
    Then do the transform as padding/resize/crop/affine/perspective.

    INITIALIZE_INPUT:
        img                     ->  The image, cv2 fromat
        keypoint_list           ->  Keypoint list. Every point should follow point format
        keypoint_index_list     ->  Keypoint index list. Also can be seen as keypoint_name_list.
                                    Sharing the same order with keypoint_list.

    TIPS:
        cv2_img.shape is [height, width, channal]
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)
        cv2_resize(img, (width, height))

        all the target_size in this code is [height, width]
        all the box label in this code is [left, up, right, down]
        all the point in this code is (x, y)

    FUNCTION:
        copy                    -> create an identity new container with same content
        image_shape             -> return img.shape
        padding_image_cv2       -> padding and resize
        add_blank_edge          -> adding blank_edge
        resize_cv2              -> resize
        check_point_list_in_box -> check how many point in the input_box
        crop                    -> crop with the box
        random_crop             -> random crop
        flip                    -> flip
        rotate                  -> rotate
        warpaffine              -> warpaffine
        warpperspective         -> warpperspective
        random_warpperspective  -> do random warpperspective transform
        random_rotate           -> do random rotate transform
        random_drift            -> do random drift transform
        random_scale            -> do random scale transform
        random_shear            -> do random shear transform
        random_flip             -> do random flip transform
        random_transform        -> do random transform transform
        check_and_fix_outsider_point  -> after transform box may go outsider of the img, this can fix it.
        _show_img               -> show img via CV2
        _write_down             -> write down img and label. label could be json(labelme)
        _load_file              -> load img and label. label could be json(labelme)
        _switch_label_name      -> switch label name, also can be used to switch label_name from str_type to number_type.

    OTHERS:
        keypoint label to pixel label -> see 'pixel_label_tools' for more detail.


    '''
    def __init__(self, img, keypoint_list, keypoint_index_list):
        super(Keypoint_container, self).__init__()
        self.img = img
        self.keypoint_list = np.array(keypoint_list)
        self.keypoint_index_list = np.array(keypoint_index_list)

    def copy(self):
        return Keypoint_container(self.img.copy(),
                                  self.keypoint_list.copy(),
                                  self.keypoint_index_list.copy())

    def image_shape(self):
        # img shape in cv2 [height, width, channal]
        return self.img.shape

    def resize_cv2(self, target_size):
        # resize the image. This transform may wrap the object.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.keypoint_list = np.array(self.keypoint_list)
        self.keypoint_list = self.keypoint_list / [self.image_shape()[1]/target_size[1],
                                                   self.image_shape()[0]/target_size[0]]

        self.img = cv2.resize(self.img, (target_size[1], target_size[0]))

    def padding_image_cv2(self, target_size):
        # This operation will keep the object scale, but not warping them.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        height, width, channal = self.image_shape()
        image_scale = height / width
        target_size_scale = target_size[0] / target_size[1]

        padding_pixel = [0, 0]  # [vetical, horizontal]
        if image_scale < target_size_scale:
            padding_pixel[0] = int(self.image_shape()[1]*(target_size_scale - image_scale))
        else:
            padding_pixel[1] = int(self.image_shape()[0]/target_size_scale
                                   - self.image_shape()[0]/image_scale)

        blank_image = np.zeros((height + padding_pixel[0],
                                width + padding_pixel[1],
                                channal)).astype(np.uint8)
        blank_image_shape = blank_image.shape

        blank_image[int(padding_pixel[0]/2): height + int(padding_pixel[0]/2),
                    int(padding_pixel[1]/2): width + int(padding_pixel[1]/2),
                    :] = self.img.copy()
        new_image = blank_image

        self.keypoint_list = self.keypoint_list + np.array(padding_pixel[::-1])/2
        self.keypoint_list = self.keypoint_list / [blank_image_shape[1]/target_size[1],
                                                   blank_image_shape[0]/target_size[0]]

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

        new_label = self.keypoint_list.copy()
        new_label = new_label + [pixels[1]/2, pixels[0]/2]

        self.img = new_image.astype('uint8')
        self.keypoint_list = new_label

    def flip(self, method):
        # method = horizontal/vertical
        image_shape = self.image_shape()
        height, width, channal = image_shape

        _temp_label = np.array(self.keypoint_list)
        _temp_img = self.img

        if method == 'horizontal' or method == 1:
            _temp_label[:, 0] = width - _temp_label[:, 0]
            _temp_img = cv2.flip(self.img, 1)
        elif method == 'vertical' or method == 0:
            _temp_label[:, 1] = height - _temp_label[:, 1]
            _temp_img = cv2.flip(self.img, 0)

        self.img = _temp_img
        self.keypoint_list = _temp_label

    def check_point_list_in_box(self, pick_box, keep_point_rate=0.6, keep_point_index_list=None):
        '''
        Discription:
            Check and maintain the wanted point. Points could be seleceted by parameter-keep_point_index_list.
            If the remain point number ratio > 0.6, return 'valid' status
            If the remain point number ratio < 0.6, return 'invalid' status

        INPUT:
            pick_box                        -> [left, up, right, down]
            keep_point_rate                 -> float, range from 0~1.
            keep_point_index_list           -> list of index. max(list)< len(self.keypoint_list)
        RETURN:
            remain_point_index_list         -> list of index
            status                          -> 'valid' / 'invalid'
        '''
        if keep_point_index_list is None:
            keep_point_index_list = self.keypoint_index_list

        _temp_keypoint_index_list = np.array(self.keypoint_index_list.copy()).tolist()

        valid_index_list = [_temp_keypoint_index_list.index(value) for value in keep_point_index_list]
        valid_point = np.array(self.keypoint_list[valid_index_list])
        # valid_index = self.keep_point_index_list[valid_index_list]

        x_check = (valid_point[:, 0] > pick_box[0]) * (valid_point[:, 0] < pick_box[2])
        y_check = (valid_point[:, 1] > pick_box[1]) * (valid_point[:, 1] < pick_box[3])
        remain_list = x_check * y_check

        if remain_list.sum() > (keep_point_rate * len(valid_index_list)):
            status = 'valid'
        else:
            status = 'invalid'

        return np.array(valid_index_list)[remain_list], status

    def crop(self, crop_box, remain_index_list=None):
        '''
        Discription:
            crop the image and label with crop_box. 
        INPUT:
                crop_box            -> [left, up, right, down]
                remain_index_list   -> a list record which keypoint is keep.
                                       If remain_index_list is None, then
                                       all the point will be caculate.
        RETURN:
                null
        '''

        crop_box = [int(value) for value in crop_box]
        _temp_img = np.zeros((crop_box[3] - crop_box[1], crop_box[2] - crop_box[0], self.image_shape()[2]))
        _temp_img = self.img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :].copy()

        if remain_index_list is None:
            remain_index_list, __ = self.check_point_list_in_box(crop_box, 1)

        _temp_keypoint_list = np.array(self.keypoint_list.copy())
        _temp_keypoint_index_list = np.array(self.keypoint_index_list.copy())

        _temp_keypoint_index_list = _temp_keypoint_index_list[remain_index_list]
        _temp_keypoint_list = _temp_keypoint_list[remain_index_list]
        _temp_keypoint_list = _temp_keypoint_list - [crop_box[0], crop_box[1]]

        self.img = _temp_img.copy()
        self.keypoint_list = _temp_keypoint_list
        self.keypoint_index_list = _temp_keypoint_index_list

    def random_crop(self, target_size, padding=False, keep_point_rate=0.6,
                    keep_point_index_list=None, tolerant_step=10):
        '''
        Discription:
            random_crop.
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

        assert keep_point_rate <= 1, 'keep_object_rate must in range(0, 1)'
        if keep_point_index_list is None:
            keep_point_index_list = self.keypoint_index_list
        else:
            # check_index_exists
            exists_list = [(value in self.keypoint_index_list) for value in keep_point_index_list]
            keep_point_index_list = np.array(keep_point_index_list)[exists_list]

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
            remain_index_list, status = self.check_point_list_in_box(pick_box, keep_point_rate, keep_point_index_list)
            if status == 'valid':
                break
        # if status == 'invalid':
        #     print('force to crop')

        self.crop(pick_box, remain_index_list)
        self.padding_image_cv2(target_size)

    def rotate(self, angle, center=None, scale=1):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if center is None:
            center = (width*0.5, height*0.5)
        # if scale >= 1:
        #     self.add_blank_edge(int(height*scale*0.2), int(width*scale*0.2))

        _temp_img = self.img.copy()
        _temp_keypoint_list = np.array(self.keypoint_list)

        matRotate = cv2.getRotationMatrix2D(center, angle, scale)
        dst = cv2.warpAffine(_temp_img, matRotate, (width, height))

        self.img = dst

        _temp_keypoint_list = np.hstack((_temp_keypoint_list, np.ones((len(_temp_keypoint_list), 1))))
        _temp_keypoint_list = matRotate.dot(_temp_keypoint_list.T).T
        # _temp_keypoint_list = [matRotate.dot(point)[0:2] for point in _temp_keypoint_list]
        self.keypoint_list = _temp_keypoint_list

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

        _temp_img = self.img.copy()
        _temp_keypoint_list = np.array(self.keypoint_list)

        dst = cv2.warpAffine(_temp_img, matRotate, (width, height))

        self.img = dst

        _temp_keypoint_list = np.hstack((_temp_keypoint_list, np.ones((len(_temp_keypoint_list), 1))))
        _temp_keypoint_list = matRotate.dot(_temp_keypoint_list.T).T
        # _temp_keypoint_list = [matRotate.dot(point)[0:2] for point in _temp_keypoint_list]
        self.keypoint_list = _temp_keypoint_list

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
        _temp_keypoint_list = np.array(self.keypoint_list)

        dst = cv2.warpPerspective(_temp_img, P_metrix, (width, height))
        self.img = dst

        _temp_keypoint_list = np.hstack((_temp_keypoint_list, np.ones((len(_temp_keypoint_list), 1))))
        _temp_keypoint_list = _temp_keypoint_list.dot(P_metrix.T)
        self.keypoint_list[:, 0] = _temp_keypoint_list[:, 0] / _temp_keypoint_list[:, 2]
        self.keypoint_list[:, 1] = _temp_keypoint_list[:, 1] / _temp_keypoint_list[:, 2]

    def random_rotate(self, rotate_range=15):
        angle = random.randint(-1*rotate_range, rotate_range)
        m = np.float32([[np.cos(angle/180*np.pi), np.sin(angle/180*np.pi), 0],
                        [-np.sin(angle/180*np.pi), np.cos(angle/180*np.pi), 0]])
        self.warpaffine(m)

    def random_drift(self, drift_range=0.1):
        x_shift = (random.random()*2 -1)* drift_range* self.img.shape[1]
        y_shift = (random.random()*2 -1)* drift_range* self.img.shape[0]
        m = np.float32([[1, 0, x_shift],
                        [0, 1, y_shift]])
        self.warpaffine(m)

    def random_scale(self, scale_range=0.05):
        x_scale = 1 + (random.random()*2 -1)* scale_range
        y_scale = 1 + (random.random()*2 -1)* scale_range
        m = np.float32([[x_scale, 0, 0],
                        [0, y_scale, 0]])
        self.warpaffine(m)

    def random_shear(self, shear_range=0.05):
        x_shear = (random.random()*2 -1)* shear_range
        y_shear = (random.random()*2 -1)* shear_range
        m = np.float32([[1, x_shear, 0],
                        [y_shear, 1, 0]])
        self.warpaffine(m)

    def random_flip(self):
        if random.random() > 0.5:
            self.flip(1) # random x_flip
        if random.random() > 0.5:
            self.flip(0) # random f_flip


    def random_transform(self, T_rate=0.5):
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
        self.check_and_fix_outsider_point()

    def check_and_fix_outsider_point(self):
        invalid_fix = 0
        # 转为 list 是为了利用 pop 的功能。
        if type(self.keypoint_list) == np.ndarray: self.keypoint_list = self.keypoint_list.tolist()
        for i in range(len(self.keypoint_list)):
            if min(self.keypoint_list[i - invalid_fix]) < 0 \
                or self.keypoint_list[i - invalid_fix][0] > self.img.shape[1] \
                    or self.keypoint_list[i - invalid_fix][1] > self.img.shape[0]:
                self.keypoint_list.pop(i - invalid_fix)
                invalid_fix += 1
        self.keypoint_list = np.array(self.keypoint_list)

    def _show_img(self, name='imshow', show_time=0):
        _img = self.img.copy()
        for i in range(len(self.keypoint_list)):
            point = (int(self.keypoint_list[i][0]), int(self.keypoint_list[i][1]))
            cv2.circle(_img, point, 4, (0, 0, 255), 4)
        cv2.imshow(name, _img)
        cv2.waitKey(show_time)

    def _switch_label_name(self, elder_label_list, new_label_list):
        _temp_keypoint_index_list = []
        for i in range(len(self.keypoint_index_list)):
            if self.keypoint_index_list[i] in elder_label_list:
                _temp_keypoint_index_list.append(new_label_list[elder_label_list.index(self.keypoint_index_list[i])])
            else:
                _temp_keypoint_index_list.append(self.keypoint_index_list[i])

        self.keypoint_index_list = _temp_keypoint_index_list

    def _write_down(self, output_img_path, method_list):
        '''
        Discription:
            write_down image and label. support ['.json', '.txt'] label format.

        INPUT:
            output_img_path             -> str
            method_list                 -> ['.json']
        RETURN:
            Null
        '''
        # cv2.imwrite(output_img_path, self.img)
        cv2.imencode('.jpg', self.img)[1].tofile(output_img_path)

        if 'json' in method_list:
            pic_format = output_img_path.split('.')[-1]
            output_json_path = output_img_path.replace('.{}'.format(pic_format), '.json')

            baisc_json_format = {"signed": "zen_V1.0.0",
                                 "flags": {},
                                 "imagePath": output_img_path,
                                 "imageData": None,
                                 "imageHeight": float(self.img.shape[0]),
                                 "imageWidth": float(self.img.shape[1])
                                 }
            point_list = []
            for i in range(len(self.keypoint_index_list)):
                label = self.keypoint_list[i]
                box = {"label": self.keypoint_index_list[i],
                       "points": [[float(label[0]), float(label[1])]],
                       "group_id": None,
                       "shape_type": "point",
                       "flags": {}}
                point_list.append(box.copy())

            baisc_json_format.update({"shapes": point_list})
            with open(output_json_path, "w") as fjson:
                fjson.write(json.dumps(baisc_json_format, indent=2))

    def _load_file(self, img_path, method='json', annotation_file_path=None):
        '''
        Discription:
            load image with label. support ['.json', '.txt'] label format.

        INPUT:
            output_img_path             -> str
            method_list                 -> ['.json']
            annotation_file_path        -> str.
        RETURN:
            Null

        Tips:
            if annotation_file and the img under same folder path, annotation_file_path
        can be set to None.
        '''
        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)

        keypoint_list = []
        keypoint_index_list = []

        if method == 'json':
            if annotation_file_path is not None:
                annotation = json.load(open(annotation_file_path))
            else:
                pic_format = img_path.split('.')[-1]
                json_file_path = img_path.replace('.{}'.format(pic_format), '.json')
                annotation = json.load(open(json_file_path))

            for i in range(len(annotation['shapes'])):
                if annotation['shapes'][i]['shape_type'] == 'point':
                    keypoint_list.append([float(annotation['shapes'][i]['points'][0][0]),
                                          float(annotation['shapes'][i]['points'][0][1]),
                                          ])
                    keypoint_index_list.append(annotation['shapes'][i]['label'])

        self.img = img
        self.keypoint_list = np.array(keypoint_list)
        self.keypoint_index_list = keypoint_index_list


def _segment_4_part(target_size):
    height, width = target_size

    segment_point = (random.uniform(0.3, 0.7)*width, random.uniform(0.3, 0.7)*height)  # format (x, y)
    segment_point = [int(value) for value in segment_point]
    segment_list = [[0, 0, segment_point[0], segment_point[1]],
                    [segment_point[0], 0, width-1, segment_point[1]],
                    [0, segment_point[1], segment_point[0], height-1],
                    [segment_point[0], segment_point[1], width-1, height-1]]           # format [x_left, y_up, x_right, y_down]

    return segment_list


def mosaic_augumentation(kp_container_list, target_size):
    assert len(kp_container_list) == 4, 'now only 4 image mosaic available'
    assert isinstance(target_size, int) or isinstance(target_size, tuple),\
        'target_size should be a single int or tuple'
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    output_img = np.zeros((*target_size, kp_container_list[0].img.shape[2])).astype(np.uint8)
    output_keypoint_list = []
    output_keypoint_index_list = []

    segment_list = _segment_4_part(target_size)

    assert len(segment_list) == len(kp_container_list), \
        'segment part and prepare boundingbox should have same number'

    for i in range(len(segment_list)):
        _crop_size = (segment_list[i][3] - segment_list[i][1],
                      segment_list[i][2] - segment_list[i][0])
        try:
            crop_kp_container = kp_container_list[i]
            crop_kp_container.random_crop(_crop_size)
        except:
            crop_kp_container = kp_container_list[i].copy()
            crop_kp_container.resize_cv2(target_size)
            crop_kp_container.random_crop(_crop_size)

        # crop_boundingbox = boundingbox_container_list[i].random_crop(_crop_size)
        output_img[segment_list[i][1]: segment_list[i][3],
                   segment_list[i][0]: segment_list[i][2], :] = crop_kp_container.img.copy()

        _temp_keypoint_list = np.array(crop_kp_container.keypoint_list)
        _temp_keypoint_list = _temp_keypoint_list + [segment_list[i][0], segment_list[i][1]]
        output_keypoint_list.extend(_temp_keypoint_list.tolist())
        output_keypoint_index_list.extend(np.array(crop_kp_container.keypoint_index_list).tolist())

    output_kp_container = Keypoint_container(output_img, output_keypoint_list, output_keypoint_index_list)

    return output_kp_container


def _testing():
    kp_container = Keypoint_container(None, None, None)
    kp_container._load_file('./data/sample/keypoint/40.png')

    print('- showing original image')
    kp_container._show_img()

    print('- showing resized image')
    kp_container = kp_container.copy()
    kp_container.resize_cv2((int(kp_container.img.shape[0]*2), int(kp_container.img.shape[1]*2)))
    kp_container._show_img('origin')

    print('- showing padded image')
    test_container = kp_container.copy()
    test_container.padding_image_cv2((512, 512))
    test_container._show_img('padding_image_cv2')

    print('- showing added blank_edge image')
    test_container = kp_container.copy()
    test_container.add_blank_edge((30, 100))
    test_container._show_img('add_blank_edge')

    print('- showing fliped image')
    test_container = kp_container.copy()
    test_container.flip(0)
    test_container._show_img('flip')

    print('- showing random croped image')
    test_container = kp_container.copy()
    test_container.random_crop((100, 120), keep_point_rate=0.5, padding=True)
    test_container._show_img('random_crop')

    print('- showing rotated image')
    test_container = kp_container.copy()
    test_container.rotate(30, scale=0.7)
    test_container._show_img('rotate')

    print('- showing warpaffined image')
    test_container = kp_container.copy()
    test_container.warpaffine()
    test_container._show_img('warpaffine')

    print('- showing warpperspective image')
    test_container = kp_container.copy()
    test_container.warpperspective()
    test_container._show_img('warpperspective')

    random_transform_time = 10
    for i in range(10):
        print('- showing 10 times random transform {}/{}'.format(i+1, random_transform_time), end='\r')
        test_container = kp_container.copy()
        test_container.random_transform()
        test_container._show_img('random_transform')

    print('- showing mosaic_augumentation')
    kp_container_list = [kp_container.copy(), kp_container.copy(),
                         kp_container.copy(), kp_container.copy()]
    test_container = mosaic_augumentation(kp_container_list, (512, 512))
    test_container._show_img('mosaic_augumentation')

    write_down_path = './data/test_save/keypoint_image.jpg'
    write_down_format = ['json']
    print('- test write_down image to path: {}, in format: {}.'.format(write_down_path, write_down_format))
    test_container._write_down(write_down_path, write_down_format)


if __name__ == '__main__':
    _testing()
