#!/usr/bin/env python2.7

import pydicom, cv2, re, math, shutil
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from itertools import izip
from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
from U_net import *
numbers_of_contours = 30
seed = 1234
np.random.seed(seed)

#SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = "C:\\Users\\r_beh\\data"

TRAIN_CONTOUR_PATH_original = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\what"
second_run = True
TRAIN_CONTOUR_PATH = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\what"
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
VAL_CONTOUR_PATH = "C:\\Users\\r_beh\\data\\val_GT"
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = "C:\\Users\\r_beh\\data\\Sunnybrook Cardiac MR Database ContoursPart1\\Sunnybrook Cardiac MR Database ContoursPart1\\OnlineDataContours"
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online')
TEST_CONTOUR_PATH = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\test"
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_test')
ONLY_TRAIN_CONTOUR_PATH = "C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\30\\only train"

                        


#def shrink_case(case):
#    toks = case.split('-')
#    def shrink_if_number(x):
#        try:
#            cvt = int(x)
#            return str(cvt)
#        except ValueError:
#            return x
#    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))
        self.slice_no =  math.floor(self.img_no/20) if self.img_no%20 !=0 else math.floor(self.img_no/20)-1
        self.ED_flag = True if ((self.img_no%20) < 10 and (self.img_no % 20) !=0) else False
        self.is_weak = 0
   
    
    def __str__(self):
        return 'Contour for case %s, image %d' % (self.case, self.img_no)
    
    __repr__ = __str__
def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8') # shape is 256, 256


    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    # print(coords.shape)   (num_points , 2)
    # print("this is coords shape")
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask
#def read_contour(contour, data_path,GTpath):
#    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
#    GT_filename='IM-0001-%04d-icontour-manual.txt' % ( contour.img_no)
#    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
#    GTfull_path= os.path.join(GTpath, contour.case,'contours-manual','IRCCI-expert', GT_filename)
#    f = pydicom.dcmread(full_path)
#    img = f.pixel_array.astype('int')
#    mask = np.zeros_like(img, dtype='uint8')

#    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
#    cv2.fillPoly(mask, [coords], 1)
#    GTcoords = np.loadtxt(GTfull_path, delimiter=' ').astype('int')

#    if img.ndim < 3:
#        img = img[..., np.newaxis]
#        mask = mask[..., np.newaxis]
#        GTcoords=GTcoords[...,np.newaxis]
#    return img, mask,GTcoords

"""def read_contour(contour, data_path):
    filename = 'IM-0001-%04d.dcm' % ( contour.img_no)
    full_path = os.path.join(data_path, contour.case,'DICOM', filename)
    f = pydicom.dcmread(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask
"""
def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-'+contour_type+'contour-manual.txt')]
    #for dirpath, dirnames, files in os.walk(contour_path):
    #    print(files)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
        
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours
# choose n contours, n should be even
def choose_n_contours (number, contours_set, save_folder_n):
    tmp_contours= []
    ES_contours= []
    for contour in contours_set:
        if contour.ED_flag == True and contour.img_no < 150:
            tmp_contours.append(contour)
    tmp_contours = tmp_contours[0:number//2]
    print(tmp_contours)
    for contour2 in tmp_contours:
        tmp_slice_no = contour2.slice_no
        tmp_case = contour2.case
        for contour3 in contours_set:
        #ES_contour = [contour_2 for contour_2 in contours]
            if contour3.slice_no == tmp_slice_no and contour3.case == tmp_case and contour3.ED_flag == False :
                ES_contour = contour3
                ES_contours.append (ES_contour)
    all_contours = tmp_contours + ES_contours
    print(all_contours)
    
    return all_contours
    
    
def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(list(contours))))
    print(len(contours))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask
        

    return images, masks


if __name__== '__main__':
   # if len(sys.argv) < 3:
   #     sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    print(contour_type)
    #training_dataset= sys.argv[2]
    #os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 100

    print('Mapping ground truth '+contour_type+' contours to images in train...')
    # train_ctrs_original = map_all_contours(TRAIN_CONTOUR_PATH_original, contour_type, shuffle=True)

    #train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True)
    test_ctrs = list(map_all_contours(TEST_CONTOUR_PATH, contour_type, shuffle=True))
    test_ctrs = test_ctrs[0:len(test_ctrs)//10]
    a = list(map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=True))


    #train_ctrs_validation = map_all_contours(TRAIN_CONTOUR_PATH_validation, contour_type, shuffle=True)
    # picked_contours = choose_n_contours (numbers_of_contours, list(train_ctrs_original), "/") 
    print('Done mapping training set')
    #a=list(picked_contours)
    # for ctr in a:
        # save_folder ="C:\\Users\\r_beh\\OneDrive\\Desktop\\backup\\" + str(numbers_of_contours) + "\\only train\\" + ctr.case + "\\contours-manual\\IRCCI-expert\\"
        # if not (os.path.exists(save_folder)):
            # os.makedirs(save_folder)
            
        # file = open((save_folder + "\\" + str(ctr.img_no) + ".txt"), 'w')        
        # file.write('Welcome to Geeks for Geeks')
        # file.close()
        # file_name = "IM-0001-" + str(format(ctr.img_no, '04d')) + "-icontour-manual.txt"
        # target = TRAIN_CONTOUR_PATH_original + "\\" + ctr.case +  "\\contours-manual\\IRCCI-expert\\"
        # shutil.copy(target +file_name, save_folder + "\\" + file_name)

        
        

        
    #b=list(map_all_contours(TRAIN_CONTOUR_PATH_original, contour_type, shuffle=True))

    split = int(0*len(a))
    train_ctrs=a[split:]
    #dev_ctrs = b[0:split]
    print(len(a))
    print("before")
    # for element in dev_ctrs[:]:
        # for element2 in a:
            # if element.case == element2.case and element.img_no == element2.img_no:
                # a.remove(element2)
    # train_ctrs = a
    #print(len(a))
    #split = int(0.3*len(a))
    #dev_ctrs =b[0:25]
    #train_ctrs = a[:]
    print(len(train_ctrs))
    #print(train_ctrs[:])
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size)
    # print(np.array(mask_train).shape)
    # print("mask train shape")    (num_ctrs, 100, 100, 1)
    
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(test_ctrs,
                                            TEST_IMG_PATH,
                                            crop_size=crop_size)
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    #weights = 'C:\\Users\\r_beh\\cardiac-segmentation-master\\cardiac-segmentation-master\\model_logs_backupl\\sunnybrook_i_epoch_40.h5'
    model = fcn_model(input_shape, num_classes, weights=None)
    # print(model.summary())
    #model = unet(input_size = input_shape, pretrained_weights=None)    
    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 40
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    
    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(int(len(img_train)/mini_batch_size)):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask, sample_weight = 1)
            
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print(model.metrics_names, train_result)
        print('\nEvaluating dev set ...')
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print(model.metrics_names, result)
        save_file = '_'.join(['sunnybrook', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('realtime'):
            os.makedirs('realtime')
        save_path = os.path.join('realtime', save_file)
        #print(save_path)
        model.save_weights(save_path)
