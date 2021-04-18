import os
import sys
import numpy as np
import random
import cv2


realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))


class nonlabel_img_container(object):
    '''
    FUNCTION:
        copy                    -> create an identity new container
        image_shape             -> return img.shape
        padding_image_cv2       -> padding and resize.
        add_blank_edge          -> adding blank_edge.
        resize_cv2              -> resize
        check_point_list_in_box -> check how many point in the input_box
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
        _write_down             -> write down img and label. label could be json(labelme)
        _load_file              -> load img and label. label could be json(labelme)

    OTHERS:
        keypoint label to pixel label -> see 'pixel_label_tools' for more detail.


    '''
    def __init__(self, img):
        super(nonlabel_img_container, self).__init__()
        self.img = img

    def copy(self):
        return nonlabel_img_container(self.img.copy())

    def image_shape(self):
        # img shape in cv2 [height, width, channal]
        return self.img.shape

    def resize_cv2(self, target_size):
        # resize the image. This transform may wrap the object.
        assert isinstance(target_size, int) or isinstance(target_size, tuple), \
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

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

        blank_image[int(padding_pixel[0]/2): height + int(padding_pixel[0]/2),
                    int(padding_pixel[1]/2): width + int(padding_pixel[1]/2),
                    :] = self.img.copy()
        new_image = blank_image

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

        self.img = new_image.astype('uint8')

    def flip(self, method):
        # method = horizontal/vertical
        image_shape = self.image_shape()
        height, width, channal = image_shape

        _temp_img = self.img

        if method == 'horizontal' or method == 1:
            _temp_img = cv2.flip(self.img, 1)
        elif method == 'vertical' or method == 0:
            _temp_img = cv2.flip(self.img, 0)

        self.img = _temp_img


    def crop(self, crop_box):
        '''
            crop the image and label with crop_box.
        INPUT:
                crop_box            -> [left, up, right, down]
        RETURN:

        '''

        crop_box = [int(value) for value in crop_box]
        _temp_img = np.zeros((crop_box[3] - crop_box[1], crop_box[2] - crop_box[0], self.image_shape()[2]))
        _temp_img = self.img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :].copy()

        self.img = _temp_img.copy()


    def random_crop(self, target_size, padding=False):
        '''
            random_crop and return a new container.
        INPUT:

        RETURN:

        '''
        assert isinstance(target_size, int) or isinstance(target_size, tuple),\
            'target_size should be a single int or tuple'
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        assert (target_size[0] <= self.img.shape[0] and target_size[1] <= self.img.shape[1]),\
            'the target_size exceed the range of image, {}'.format(self.img.shape)


        image_shape = self.img.shape
        crop_size = target_size

        if padding:
            padding_size = (crop_size[0] + image_shape[0], crop_size[1] + image_shape[1])
            self.add_blank_edge(tuple(crop_size))
        else:
            padding_size = (image_shape[0], image_shape[1])



        pick_point_up = random.randint(0, padding_size[0] - crop_size[0])
        pick_point_left = random.randint(0, padding_size[1] - crop_size[1])
        pick_point_down = pick_point_up + crop_size[0]
        pick_point_right = pick_point_left + crop_size[1]
        pick_box = (pick_point_left, pick_point_up, pick_point_right, pick_point_down)
 

        self.crop(pick_box)
        self.padding_image_cv2(target_size)

    def rotate(self, angle, center=None, scale=1):
        image_shape = self.image_shape()
        height, width, channal = image_shape

        if center is None:
            center = (width*0.5, height*0.5)
        # if scale >= 1:
        #     self.add_blank_edge(int(height*scale*0.2), int(width*scale*0.2))

        _temp_img = self.img.copy()

        matRotate = cv2.getRotationMatrix2D(center, angle, scale)
        dst = cv2.warpAffine(_temp_img, matRotate, (width, height))

        self.img = dst

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

        dst = cv2.warpAffine(_temp_img, matRotate, (width, height))

        self.img = dst

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

        dst = cv2.warpPerspective(_temp_img, P_metrix, (width, height))
        self.img = dst

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

        # got bug
        # x_flip = 1 if random.random()>0.5 else -1
        # y_flip = 1 if random.random()>0.5 else -1
        # m = np.float32([[x_flip, 0, 0],
        #                 [0, y_flip, 0]])
        # self.warpaffine(m)

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

    def _show_img(self, name='imshow', show_time=0):
        _img = self.img.copy()
        cv2.imshow(name, _img)
        cv2.waitKey(show_time)

    def _write_down(self, output_img_path):
        cv2.imwrite(output_img_path, self.img)

    def _load_file(self, img_path, method='json', annotation_file_path=None):
        img = cv2.imread(img_path)
        self.img = img

def _segment_4_part(target_size):
    height, width = target_size

    segment_point = (random.uniform(0.3, 0.7)*width, random.uniform(0.3, 0.7)*height)  # format (x, y)
    segment_point = [int(value) for value in segment_point]
    segment_list = [[0, 0, segment_point[0], segment_point[1]],
                    [segment_point[0], 0, width-1, segment_point[1]],
                    [0, segment_point[1], segment_point[0], height-1],
                    [segment_point[0], segment_point[1], width-1, height-1]]           # format [x_left, y_up, x_right, y_down]

    return segment_list


def _testing():
    nonlb_container = nonlabel_img_container(None)
    nonlb_container._load_file('./data/sample/bounding_box/captain.jpg')

    print('- showing original image')
    nonlb_container._show_img()

    print('- showing resized image')
    nonlb_container.resize_cv2((int(nonlb_container.img.shape[0]*0.5), int(nonlb_container.img.shape[1]*0.5)))
    nonlb_container._show_img('origin')

    print('- showing padded image')
    test_container = nonlb_container.copy()
    test_container.padding_image_cv2((512, 512))
    test_container._show_img('padding_image_cv2')

    print('- showing added blank_edge image')
    test_container = nonlb_container.copy()
    test_container.add_blank_edge((30, 100))
    test_container._show_img('add_blank_edge')

    print('- showing fliped image')
    test_container = nonlb_container.copy()
    test_container.flip(0)
    test_container._show_img('flip')

    print('- showing random croped image')
    test_container = nonlb_container.copy()
    test_container.random_crop((300, 400), padding=True)
    test_container._show_img('random_crop')

    print('- showing rotated image')
    test_container = nonlb_container.copy()
    test_container.rotate(30, scale=0.7)
    test_container._show_img('rotate')

    print('- showing warpaffined image')
    test_container = nonlb_container.copy()
    test_container.warpaffine()
    test_container._show_img('warpaffine')

    print('- showing warpperspective image')
    test_container = nonlb_container.copy()
    test_container.warpperspective()
    test_container._show_img('warpperspective')

    random_transform_time = 10
    for i in range(10):
        print('- showing 10 times random transform {}/{}'.format(i+1, random_transform_time), end='\r')
        test_container = nonlb_container.copy()
        test_container.random_transform()
        test_container._show_img('random_transform')

    write_down_path = './data/test_save/nonlabel_image.jpg'
    print('- test write_down image to path: {}.'.format(write_down_path))
    test_container._write_down(write_down_path)


if __name__ == '__main__':
    _testing()
