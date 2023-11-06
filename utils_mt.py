# Import libraries
import os
import sys
import glob
import yaml
import math
import time
import random
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy import ndimage
import skimage.morphology 
#from osgeo import ogr, gdal
import matplotlib.pyplot as plt
from skimage.filters import rank
from sklearn.utils import shuffle
from skimage.morphology import disk
from skimage.transform import resize
#import tensorflow.keras.backend as K
#from keras.layers import *
#from tensorflow.keras.layers import * ### ok
# ****
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
#from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, load_model, Sequential, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# ***
from contextlib import redirect_stdout
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skimage.util.shape import view_as_windows
from sklearn.metrics import average_precision_score
#from tensorflow.keras.initializers import RandomNormal
#from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from tensorflow.keras.models import Model, load_model, Sequential, save_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

# Function to load yaml configuration file
def load_config(CONFIG_PATH, config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

# Functions
def load_optical_image(patch):
    # Read tiff Image
    print (patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    img = np.transpose(img.copy(), (1, 2, 0))
    print('Image shape :', img.shape)
    return img

def load_SAR_image(patch):
    #Function to read SAR images
    print (patch)
    gdal_header = gdal.Open(patch)
    db_img = gdal_header.ReadAsArray()
    #img_tif = TIFF.open(patch)
    #db_img = img_tif.read_image()
    #db_img = np.transpose(db_img, (1, 2, 0))
    temp_db_img = 10**(db_img/10)
    temp_db_img[temp_db_img>1] = 1
    return temp_db_img

def load_tif_image(patch):
    # Read tiff Image
    print (patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    #print(np.unique(img))
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    #img = np.transpose(img.copy(), (1, 2, 0))
    print('Image shape :', img.shape)
    return img

def convert_binary(image_matrix, thresh_val):
    up_th = 1
    dw_th = 0
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, up_th)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, dw_th)
    return final_conv

def filter_outliers(img, bins=1000000, bth=0.01, uth=0.99, mask=[0]):
#def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img

def create_mask(size_rows, size_cols, grid_size=(6,3)):
    num_tiles_rows = size_rows//grid_size[0]
    num_tiles_cols = size_cols//grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows*grid_size[0], num_tiles_cols*grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count+1
            mask[num_tiles_rows*i:(num_tiles_rows*i+num_tiles_rows), num_tiles_cols*j:(num_tiles_cols*j+num_tiles_cols)] = patch*count
    #plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask

def extract_patches(input_image, reference,  patch_size, stride, percent):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    
    patches_images = []
    patches_labels = []
    
    for i in range(0,len(patches_ref)):
        patch = patches_ref[i]
        class1 = patch[patch==1]
        
        if len(class1) >= int((patch_size**2)*(percent/100)):
            patch_img = patches_array[i]
            patch_label = patches_ref[i]
            #img_aug, label_aug = data_augmentation(patch_img, patch_label)
            patches_images.append(patch_img)
            patches_labels.append(patch_label)
        else:
            continue
    
    patches_ = np.asarray(patches_images).astype(np.float32)
    labels_ = np.asarray(patches_labels).astype(np.float32)

    return patches_, labels_

def extract_random_patches(input_image, reference,  patch_size, stride, percent, max_patches):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    
    patches_images = []
    patches_labels = []
    
    if len(patches_ref) >= max_patches:
        ind_random_patches = np.random.choice(len(patches_ref), size = max_patches, replace=False)
    else:
        ind_random_patches = np.random.choice(len(patches_ref), size = len(patches_ref), replace=False)
        
    for i in range(0, max_patches):
        patch_img = patches_array[ind_random_patches[i]]
        patch_label = patches_ref[ind_random_patches[i]]
        patches_images.append(patch_img)
        patches_labels.append(patch_label)
    
    patches_ = np.asarray(patches_images).astype(np.float32)
    labels_ = np.asarray(patches_labels).astype(np.float32)
    print('random patches per tile: ', len(patches_), len(labels_))

    return patches_, labels_

def patch_tiles(tiles, mask_amazon, image_array, image_ref, patch_size, stride, percent, path_image, path_mask,
                img_name, random_patches = False, max_patches = 1000):
    
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    
    '''Extraction of image patches and labels '''
    patches_out = []
    label_out = []
    count = 0
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        
        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        if random_patches == True:
            print('[*] Generating random patches...')
            patches_img, patch_ref = extract_random_patches(tile_img, tile_ref, patch_size, stride, percent, max_patches)
        else:
            print('[*] Random: False')
            patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride, percent)
        print('tile :', num_tile, patches_img.shape, patch_ref.shape)
        if len(patch_ref) == 0:
            continue
        for i in range(0, patch_ref.shape[0]):
            count = count+1
            np.save(path_image+img_name+'_'+str(count)+'.npy', patches_img[i])
            np.save(path_mask+img_name+'_'+str(count)+'.npy', patch_ref[i]) 
    print('Total patches: ', count)
    del patches_img, patch_ref

def patch_tiles_ok(tiles, mask_amazon, image_array, image_ref, patch_size, stride, percent, path_image, path_mask, img_name):
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    
    '''Extraction of image patches and labels '''
    patches_out = []
    label_out = []
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        
        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride, percent)
        print('tile :', num_tile, patches_img.shape, patch_ref.shape)
        if len(patch_ref) == 0:
            continue
        patches_out.append(patches_img)
        label_out.append(patch_ref)
        
    patches_out_ = np.concatenate(patches_out).astype(np.float32)
    label_out_ = np.concatenate(label_out).astype(np.float32)
    print('Total patches : ', patches_out_.shape, label_out_.shape)
    
    count = 0
    for i in range(0, len(patches_out_)): 
        count = count + 1
        np.save(path_image+img_name+'_'+str(count)+'.npy', patches_out_[i])
        np.save(path_mask+img_name+'_'+str(count)+'.npy', label_out_[i])
    #return patches_out_, label_out_

def patch_tiles_old(tiles, mask_amazon, image_array, image_ref, patch_size, stride, percent, path_image, path_mask, img_name):
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    
    '''Extraction of image patches and labels '''
    patches_out = []
    label_out = []
    count = 0
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        
        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride, percent)
        print('tile :', num_tile, patches_img.shape, patch_ref.shape)
        if len(patch_ref) == 0:
            continue
        for i in range(0, patch_ref.shape[0]):
            count = count+1
            np.save(path_image+img_name+'_'+str(count)+'.npy', patches_img[i])
            np.save(path_mask+img_name+'_'+str(count)+'.npy', patch_ref[i])
    print('Total patches: ', count)
    del patches_img, patch_ref

def retrieve_num_samples_base(current_folder, list_dir, folder, patches_sr):
    content_list = {}
    for index, val in enumerate(list_dir):
        print('path patches: ', current_folder,val, patches_sr,folder, 'path_image')
        path = os.path.join(current_folder, val, patches_sr, folder, 'path_image')
        #print(path)
        content_list[ list_dir[index] ] = os.listdir(path)

    file_list = list(content_list.values())

    complete_list = []
    for name in range(0, len(list_dir)):
        complete_list = complete_list + file_list[name]
    
    return complete_list, len(complete_list)

def retrieve_num_samples(current_folder, list_dir, folder, patches_tg, multi_sr=False):
    content_list = {}
    
    for index, val in enumerate(list_dir):
        if multi_sr == False:
            if index == 0:
                #print('source')
                path = os.path.join(current_folder, val, 'patches_prodes', folder, 'path_image')
            else:
                #print('target')
                path = os.path.join(current_folder, val, patches_tg, folder, 'path_image')
        if multi_sr == True:
            if index < 2:
                #print('source')
                path = os.path.join(current_folder, val, 'patches_prodes', folder, 'path_image')
            else:
                #print('target')
                path = os.path.join(current_folder, val, patches_tg, folder, 'path_image')
        content_list[ list_dir[index] ] = os.listdir(path)

    file_list = list(content_list.values())
    
    return file_list, len(file_list)
        
        
def data_gen_base(current_folder, list_dir, folder, patches_set, batch_size, patch_size, channels, num_classes):
    
    complete_list, _ = retrieve_num_samples_base(current_folder, list_dir, folder, patches_set)
    c = 0
    #n = os.listdir(img_folder) #List of training images
    n = complete_list
    random.shuffle(n)

    while (True):
        img = np.zeros((batch_size, patch_size, patch_size, channels)).astype('float32')
        
        img = np.zeros((batch_size, patch_size, patch_size, channels)).astype('float32')
        mask = np.zeros((batch_size, patch_size, patch_size, num_classes)).astype('float32')

        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            
            split_name = n[i].split('_')
            name_data = split_name[0]

            img_folder  = current_folder + name_data + '_1C/' + patches_set + '/' + folder + '/path_image/'
            mask_folder = current_folder + name_data + '_1C/' + patches_set + '/' + folder + '/path_mask/'
            
            train_img = np.load(img_folder+'/'+n[i])
            train_mask = np.load(mask_folder+'/'+n[i])
            train_mask = tf.keras.utils.to_categorical(train_mask , num_classes)
            
            if np.random.rand() < 0.3:
                train_img = np.rot90(train_img, 1)
                train_mask = np.rot90(train_mask, 1)
                
            if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
                train_img = np.flip(train_img, 0)
                train_mask = np.flip(train_mask, 0)
            
            if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
                train_img = np.flip(train_img, 1)
                train_mask = np.flip(train_mask, 1)
                
            if np.random.rand() > 0.7:
                train_img = train_img
                train_mask = train_mask
            
            img[i-c] = train_img
            mask[i-c] = train_mask

        c+=batch_size
        if(c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n) # "randomizing again"
        yield img, mask

##
def data_gen(dir_data, list_dir, folder, batch_size, patch_size, channels, num_classes, patches_tg, multi_sr, patch_dis, patch_dis_dim):
    c = 0
    c_list, n_domains = retrieve_num_samples(dir_data, list_dir, folder, patches_tg, multi_sr)

    for i in range(0, n_domains):
        random.shuffle(c_list[i])

    while (True):
        batch_img = np.zeros((batch_size, patch_size, patch_size, channels)).astype('float32')
        batch_mask = np.zeros((batch_size, patch_size, patch_size, num_classes)).astype('uint8')
        batch_img_ = np.zeros((batch_size, patch_size, patch_size, channels, n_domains)).astype('float32')
        batch_mask_ = np.zeros((batch_size, patch_size, patch_size, num_classes, n_domains)).astype('uint8')

        if patch_dis:
            batch_dom = np.zeros((batch_size, patch_dis_dim, patch_dis_dim, n_domains)).astype('uint8')
            batch_dom_ = np.zeros((batch_size, patch_dis_dim, patch_dis_dim, n_domains, n_domains)).astype('uint8')
        else:
            batch_dom = np.zeros((batch_size, n_domains)).astype('uint8')
            batch_dom_ = np.zeros((batch_size, n_domains, n_domains)).astype('uint8')

        len_dir = np.zeros((n_domains))

        for dom in range(0, n_domains):
            split_name = list_dir[dom].split('_')
            name_data = split_name[0]
            # Multi-sour e
            # if dom < 2:
            if dom == 0:
                # print('source')
                folder_img = dir_data + name_data + '_1C/' + 'patches_prodes/' + folder + '/path_image/'
                folder_mask = dir_data + name_data + '_1C/' + 'patches_prodes/' + folder + '/path_mask/'
            else:
                # print('target')
                folder_img = dir_data + name_data + '_1C/' + patches_tg + '/' + folder + '/path_image/'
                folder_mask = dir_data + name_data + '_1C/' + patches_tg + '/' + folder + '/path_mask/'
            len_dir[dom] = len(os.listdir(folder_img))
            # print('***** hereee ****: ', np.min(len_dir))

            for i in range(c, c + batch_size):
                patch_img = np.load(folder_img + c_list[dom][i])
                patch_mask = np.load(folder_mask + c_list[dom][i])
                patch_mask = tf.keras.utils.to_categorical(patch_mask, num_classes)
                if patch_dis:
                    dom_label = tf.keras.utils.to_categorical(np.ones((patch_dis_dim, patch_dis_dim)) * dom, n_domains)
                else:
                    dom_label = tf.keras.utils.to_categorical(dom, n_domains)

                if np.random.rand() < 0.3:
                    patch_img = np.rot90(patch_img, 1)
                    patch_mask = np.rot90(patch_mask, 1)

                if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
                    patch_img = np.flip(patch_img, 0)
                    patch_mask = np.flip(patch_mask, 0)

                if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
                    patch_img = np.flip(patch_img, 1)
                    patch_mask = np.flip(patch_mask, 1)

                if np.random.rand() > 0.7:
                    patch_img = patch_img
                    patch_mask = patch_mask

                batch_img[i - c, :, :, :] = patch_img
                batch_mask[i - c, :, :, :] = patch_mask
                if patch_dis:
                    batch_dom[i - c, :, :, :] = dom_label
                else:
                    batch_dom[i - c, :] = dom_label

            batch_img_[:, :, :, :, dom] = batch_img
            batch_mask_[:, :, :, :, dom] = batch_mask
            if patch_dis:
                batch_dom_[:, :, :, :, dom] = batch_dom
            else:
                batch_dom_[:, :, dom] = batch_dom

            # print(batch_img_.shape,batch_mask_.shape, batch_dom_.shape)

        c += batch_size
        if (c + batch_size >= np.min(len_dir)):
            c = 0
            # random.shuffle(c_list) # "randomizing again"
            for i in range(0, n_domains):
                random.shuffle(c_list[i])
        yield batch_img_, batch_mask_, batch_dom_

def data_gen_patch(dir_data, list_dir, folder, batch_size, patch_size, channels, num_classes, patches_tg, patch_dis):
    c = 0
    c_list, n_domains =  retrieve_num_samples(dir_data, list_dir, folder, patches_tg)
    
    for i in range(0, n_domains):
        random.shuffle(c_list[i])
    
    while (True):
        batch_img  = np.zeros((batch_size, patch_size, patch_size, channels)).astype('float32')
        batch_mask = np.zeros((batch_size, patch_size, patch_size, num_classes)).astype('uint8')
        batch_dom  = np.zeros((batch_size, patch_dis, patch_dis, n_domains)).astype('uint8')

        batch_img_  = np.zeros((batch_size, patch_size, patch_size, channels, n_domains)).astype('float32')
        batch_mask_ = np.zeros((batch_size, patch_size, patch_size, num_classes, n_domains)).astype('uint8')
        batch_dom_  = np.zeros((batch_size, patch_dis, patch_dis, n_domains, n_domains)).astype('uint8')
        
        len_dir = np.zeros((n_domains))

        for dom in range(0, n_domains):
            split_name = list_dir[dom].split('_')
            name_data = split_name[0]
            #print(name_data)
            #folder_img  = dir_data + name_data + '_1C/' + 'patches/' + folder + '/path_image/'
            #folder_mask = dir_data + name_data + '_1C/' + 'patches/' + folder + '/path_mask/'
            if dom == 0:
                #print('source')
                folder_img  = dir_data + name_data + '_1C/' + 'patches_prodes/' + folder + '/path_image/'
                folder_mask = dir_data + name_data + '_1C/' + 'patches_prodes/' + folder + '/path_mask/'
            else:
                #print('target')
                folder_img  = dir_data + name_data + '_1C/' + patches_tg + '/' + folder + '/path_image/'
                folder_mask = dir_data + name_data + '_1C/' + patches_tg + '/' + folder + '/path_mask/'
            len_dir[dom] = len(os.listdir(folder_img))

            for i in range(c, c+batch_size): 
                patch_img = np.load(folder_img +  c_list[dom][i])
                patch_mask = np.load(folder_mask + c_list[dom][i])
                #print(folder_img +  c_list[dom][i])
                #print(folder_mask + c_list[dom][i])
                patch_mask = tf.keras.utils.to_categorical(patch_mask , num_classes)
                dom_label = tf.keras.utils.to_categorical(np.ones((patch_dis,patch_dis))*dom , n_domains)
                
                if np.random.rand() < 0.3:
                    patch_img = np.rot90(patch_img, 1)
                    patch_mask = np.rot90(patch_mask, 1)
                
                if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
                    patch_img = np.flip(patch_img, 0)
                    patch_mask = np.flip(patch_mask, 0)

                if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
                    patch_img = np.flip(patch_img, 1)
                    patch_mask = np.flip(patch_mask, 1)

                if np.random.rand() > 0.7:
                    patch_img = patch_img
                    patch_mask = patch_mask

                batch_img[i-c,:,:,:]  = patch_img
                batch_mask[i-c,:,:,:] = patch_mask
                batch_dom[i-c,:,:,:] = dom_label        

            batch_img_[:,:,:,:,dom]  = batch_img
            batch_mask_[:,:,:,:,dom] = batch_mask
            batch_dom_[:,:,:,:,dom] = batch_dom

            #print(batch_img_.shape,batch_mask_.shape, batch_dom_.shape) 
            
        c+=batch_size
        if(c+batch_size>=np.min(len_dir)):
            c=0
            #random.shuffle(c_list) # "randomizing again"
            for i in range(0, n_domains):
                random.shuffle(c_list[i])
        yield batch_img_, batch_mask_, batch_dom_

        
# new model
def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1

def get_patches_batch(image, rows, cols, radio, batch):
    temp = []
    for i in range(0, batch):
        batch_patches = image[rows[i]-radio:rows[i]+radio+1, cols[i]-radio:cols[i]+radio+1, :]
        temp.append(batch_patches)
    patches = np.asarray(temp)
    return patches

def pred_recostruction(patch_size, pred_labels, image_ref):
    # Reconstruction 
    stride = patch_size
    h, w = image_ref.shape
    num_patches_h = int(h/stride)
    num_patches_w = int(w/stride)
    count = 0
    img_reconstructed = np.zeros((num_patches_h*stride,num_patches_w*stride))
    for i in range(0,num_patches_w):
        for j in range(0,num_patches_h):
            img_reconstructed[stride*j:stride*(j+1),stride*i:stride*(i+1)]=pred_labels[count]
            #img_reconstructed[32*i:32*(i+1),32*j:32*(j+1)]=p_labels[count]
            count+=1
    return img_reconstructed

def entropy_criterion(logits):
    ll = tf.nn.softmax(logits)
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=ll, logits=logits))

def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            #loss = y_true * K.log(y_pred) * weights
            #loss = -K.sum(loss, -1)
            # loss ok *****
            loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
            #loss = loss * weights
            # Focal loss
            #gamma = 2
            #loss = y_true * K.log(y_pred) * (1-y_pred)**gamma  + (1-y_true) * K.log(1-y_pred) * (y_pred)**gamma 
            #loss = y_true * K.log(y_pred) * K.pow((1-y_pred), gamma)  + (1-y_true) * K.log(1-y_pred) * K.pow((y_pred), gamma)
            loss = loss * weights 
            #loss = - K.sum(loss, -1)
            loss = - K.mean(loss, -1)
            return loss
        return loss
    
      

def mask_no_considered(image_ref, past_ref, buffer):
    # Creation of buffer for pixel no considered
    image_ref_ = image_ref.copy()
    im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
    im_erosion = skimage.morphology.erosion(image_ref_, disk(buffer))
    inner_buffer = image_ref_ - im_erosion
    inner_buffer[inner_buffer == 1] = 2
    outer_buffer = im_dilate-image_ref_
    outer_buffer[outer_buffer == 1] = 2
    
    # 1 deforestation, 2 unknown
    image_ref_[outer_buffer + inner_buffer == 2 ] = 2
    #image_ref_[outer_buffer == 2 ] = 2
    image_ref_[past_ref == 1] = 2
    return image_ref_

   


def color_map(prob_map, ref_reconstructed, mask_no_considered, clipping_mask_, th):
    reconstructed = prob_map.copy()
    reconstructed[reconstructed >= th] = 1
    reconstructed[reconstructed < th] = 0
    
    true_positives = (reconstructed*ref_reconstructed)
    diff_image = reconstructed-ref_reconstructed
    
    output_map = np.zeros((ref_reconstructed.shape)).astype(np.float32)
    output_map[true_positives == 1] = 1
    output_map[diff_image == 1] = 2
    output_map[diff_image==-1] = 3
    output_map[mask_no_considered == 2] = 4
    output_map[clipping_mask_ == 0] = 0
    return output_map

def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics

        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'

    """

    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=1)
    tp_fp = np.sum(confusion_matrix, axis=0)

    has_no_rp = tp_fn == 0
    has_no_pp = tp_fp == 0

    tp_fn[has_no_rp] = 1
    tp_fp[has_no_pp] = 1

    percentages = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recalls = tp / tp_fn

    p_zero = precisions == 0
    precisions[p_zero] = 1

    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    ious = tp / (tp_fn + tp_fp - tp)

    precisions[has_no_pp] *= 0.0
    precisions[p_zero] *= 0.0
    recalls[has_no_rp] *= 0.0

    f1s[p_zero] *= 0.0
    f1s[percentages == 0.0] = np.nan
    ious[percentages == 0.0] = np.nan

    mf1 = np.nanmean(f1s)
    miou = np.nanmean(ious)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    metrics = {'percentages': percentages,
               'precisions': precisions,
               'recalls': recalls,
               'f1s': f1s,
               'mf1': mf1,
               'ious': ious,
               'miou': miou,
               'oa': oa}

    return metrics
def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    #print('unique values: ', np.unique(ref_reconstructed))
    thresholds = thresholds_    
    metrics_all = []
    
    for thr in thresholds:
        print(thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        #mask_no_consider = mask_areas_pred * mask_borders 
        #ref_consider = mask_no_consider * ref_reconstructed
        #pred_consider = mask_no_consider*img_reconstructed
        
        #ref_final = ref_consider[mask_amazon_ts_==1]
        #pre_final = pred_consider[mask_amazon_ts_==1]
        mask_no_consider = mask_areas_pred * mask_borders * mask_amazon_ts_
        
        ref_final = ref_reconstructed[mask_no_consider == 1]
        pre_final = img_reconstructed[mask_no_consider == 1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #tn, fp, fn, tp = confusion_matrix(ref_final, pre_final).ravel()
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        #aa = (TP+FP)/len(ref_final)
        f1 = 2 * (precision_ * recall_) / (precision_ + recall_)
        mm = np.hstack((recall_, precision_, f1))
        metrics_all.append(mm)
        if thr == 0.5:
            print('confusion matrix', cm)
            print(get_confusion_metrics(cm))
    metrics_ = np.asarray(metrics_all)
    return metrics_, cm

def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed


    
# Function to compute mAP
def Area_under_the_curve(X, Y):
    #X -> Recall
    #Y -> Precision
    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])
    
    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b                
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))
                    
    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))
    
    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)
    
    return area

def complete_nan_values(metrics):
    vec_prec = metrics[:,1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2*vec_prec[j+1]-vec_prec[j+2]
            if vec_prec[j] >= 1:
                vec_prec[j] = 1
    metrics[:,1] = vec_prec
    return metrics 

def plot_images(img_t0, reference, vmin=-1, vmax=1, figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    i = np.random.choice(np.arange(len(img_t0)))
    
    x0 = img_t0[i,:,:,:3]
    x0 = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
    
    x1 = img_t0[i,:,:,3:]
    x1 = (x1-np.min(x1))/(np.max(x1)-np.min(x1)) 
    
    x2 = reference[i,:,:]
    x2 = (x2-np.min(x2))/(np.max(x2)-np.min(x2))

    ax1 = fig.add_subplot(131)
    plt.title('t0')
    ax1.imshow(x0, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.axis('off')

    ax2 = fig.add_subplot(132)
    plt.title('t1')
    ax2.imshow(x1, cmap='jet', vmin=vmin, vmax=vmax)
    ax2.axis('off')

    ax3 = fig.add_subplot(133)
    plt.title('def ref')
    ax3.imshow(x2, cmap='jet', vmin=vmin, vmax=vmax)
    ax3.axis('off')

    plt.show()
    # fig.savefig(save_img_path+name+'_img_pt_br_2_elastic_'+str(epoch))
    plt.close()

def plot_prediction(batch_sr, rec_sr, true_sr, pred_sr, batch_tr1, rec_tr1, true_tr1, pred_tr1, batch_tr2, rec_tr2, true_tr2, pred_tr2, vmin = -1, vmax = 1, index=0):
    vb_t0 = [2,1,0]
    vb_t1 = [12,11,10]
    #i = np.random.randint(32)
    i = index
    x_sr_t0 = normalization(batch_sr[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_sr_t0 = normalization(rec_sr[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_sr_t1 = normalization(batch_sr[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_sr_t1 = normalization(rec_sr[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    #x_sr_t0 = (batch_sr[i,:,:,vb_t0] - batch_sr[i,:,:,vb_t0].min())/(batch_sr[i,:,:,vb_t0].max()-batch_sr[i,:,:,vb_t0].min())
    #r_sr_t0 = (rec_sr[i,:,:,vb_t0] - rec_sr[i,:,:,vb_t0].min())/(rec_sr[i,:,:,vb_t0].max()-rec_sr[i,:,:,vb_t0].min())
    #x_sr_t1 = (batch_sr[i,:,:,vb_t1] - batch_sr[i,:,:,vb_t1].min())/(batch_sr[i,:,:,vb_t1].max()-batch_sr[i,:,:,vb_t1].min())
    #r_sr_t1 = (rec_sr[i,:,:,vb_t1] - rec_sr[i,:,:,vb_t1].min())/(rec_sr[i,:,:,vb_t1].max()-rec_sr[i,:,:,vb_t1].min())
    lb_true_sr = true_sr[i].argmax(axis=-1)
    lb_pred_sr = pred_sr[i].argmax(axis=-1)
    
    x_tr1_t0 = normalization(batch_tr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_tr1_t0 = normalization(rec_tr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_tr1_t1 = normalization(batch_tr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_tr1_t1 = normalization(rec_tr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    #x_tr1_t0 = (batch_tr1[i,:,:,vb_t0] - batch_tr1[i,:,:,vb_t0].min())/(batch_tr1[i,:,:,vb_t0].max()-batch_tr1[i,:,:,vb_t0].min())
    #r_tr1_t0 = (rec_tr1[i,:,:,vb_t0] - rec_tr1[i,:,:,vb_t0].min())/(rec_tr1[i,:,:,vb_t0].max()-rec_tr1[i,:,:,vb_t0].min())
    #x_tr1_t1 = (batch_tr1[i,:,:,vb_t1] - batch_tr1[i,:,:,vb_t1].min())/(batch_tr1[i,:,:,vb_t1].max()-batch_tr1[i,:,:,vb_t1].min())
    #r_tr1_t1 = (rec_tr1[i,:,:,vb_t1] - rec_tr1[i,:,:,vb_t1].min())/(rec_tr1[i,:,:,vb_t1].max()-rec_tr1[i,:,:,vb_t1].min())
    lb_true_tr1 = true_tr1[i].argmax(axis=-1)
    lb_pred_tr1 = pred_tr1[i].argmax(axis=-1)
    
    x_tr2_t0 = normalization(batch_tr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_tr2_t0 = normalization(rec_tr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_tr2_t1 = normalization(batch_tr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_tr2_t1 = normalization(rec_tr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    #x_tr2_t0 = (batch_tr2[i,:,:,vb_t0] - batch_tr2[i,:,:,vb_t0].min())/(batch_tr2[i,:,:,vb_t0].max()-batch_tr2[i,:,:,vb_t0].min())
    #r_tr2_t0 = (rec_tr2[i,:,:,vb_t0] - rec_tr2[i,:,:,vb_t0].min())/(rec_tr2[i,:,:,vb_t0].max()-rec_tr2[i,:,:,vb_t0].min())
    #x_tr2_t1 = (batch_tr2[i,:,:,vb_t1] - batch_tr2[i,:,:,vb_t1].min())/(batch_tr2[i,:,:,vb_t1].max()-batch_tr2[i,:,:,vb_t1].min())
    #r_tr2_t1 = (rec_tr2[i,:,:,vb_t1] - rec_tr2[i,:,:,vb_t1].min())/(rec_tr2[i,:,:,vb_t1].max()-rec_tr2[i,:,:,vb_t1].min())
    lb_true_tr2 = true_tr2[i].argmax(axis=-1)
    lb_pred_tr2 = pred_tr2[i].argmax(axis=-1)

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(3,6,1)
    plt.title('x_sr_t0')
    ax1.imshow(x_sr_t0, vmin=vmin, vmax=vmax)
    ax1.axis('off')

    ax2 = fig.add_subplot(3,6,2)
    plt.title('rec_sr_t0')
    ax2.imshow(r_sr_t0, vmin=vmin, vmax=vmax)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(3,6,3)
    plt.title('x_sr_t1')
    ax3.imshow(x_sr_t1, vmin=vmin, vmax=vmax)
    ax3.axis('off')

    ax4 = fig.add_subplot(3,6,4)
    plt.title('rec_sr_t1')
    ax4.imshow(r_sr_t1, vmin=vmin, vmax=vmax)
    ax4.axis('off')
        
    ax5 = fig.add_subplot(3,6,5)
    plt.title('true sr')
    ax5.imshow(lb_true_sr, cmap ='jet',vmax=2)
    ax5.axis('off')

    ax6 = fig.add_subplot(3,6,6)
    plt.title('pred sr')
    ax6.imshow(lb_pred_sr, cmap ='jet', vmax=2)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(3,6,7)
    plt.title('x_tr1_t0')
    ax7.imshow(x_tr1_t0, vmin=vmin, vmax=vmax)
    ax7.axis('off')

    ax8 = fig.add_subplot(3,6,8)
    plt.title('rec_tr1_t0')
    ax8.imshow(r_tr1_t0, vmin=vmin, vmax=vmax)
    ax8.axis('off')
    
    ax9 = fig.add_subplot(3,6,9)
    plt.title('x_tr1_t1')
    ax9.imshow(x_tr1_t1, vmin=vmin, vmax=vmax)
    ax9.axis('off')

    ax10 = fig.add_subplot(3,6,10)
    plt.title('rec_tr1_t1')
    ax10.imshow(r_tr1_t1, vmin=vmin, vmax=vmax)
    ax10.axis('off')
    
    ax11 = fig.add_subplot(3,6,11)
    plt.title('true t1r')
    ax11.imshow(lb_true_tr1, cmap ='jet', vmax=2)
    ax11.axis('off')
    
    ax12 = fig.add_subplot(3,6,12)
    plt.title('pred tr1')
    ax12.imshow(lb_pred_tr1, cmap ='jet', vmax=2)
    ax12.axis('off')    

    # target 2
    ax13 = fig.add_subplot(3,6,13)
    plt.title('x_tr2_t0')
    ax13.imshow(x_tr2_t0, vmin=vmin, vmax=vmax)
    ax13.axis('off')

    ax14 = fig.add_subplot(3,6,14)
    plt.title('rec_tr2_t0')
    ax14.imshow(r_tr2_t0, vmin=vmin, vmax=vmax)
    ax14.axis('off')
    
    ax15 = fig.add_subplot(3,6,15)
    plt.title('x_tr2_t1')
    ax15.imshow(x_tr2_t1, vmin=vmin, vmax=vmax)
    ax15.axis('off')

    ax16 = fig.add_subplot(3,6,16)
    plt.title('rec_tr2_t1')
    ax16.imshow(r_tr2_t1, vmin=vmin, vmax=vmax)
    ax16.axis('off')
    
    ax17 = fig.add_subplot(3,6,17)
    plt.title('true t12')
    ax17.imshow(lb_true_tr2, cmap ='jet', vmax=2)
    ax17.axis('off')
    
    ax18 = fig.add_subplot(3,6,18)
    plt.title('pred tr1')
    ax18.imshow(lb_pred_tr2, cmap ='jet', vmax=2)
    ax18.axis('off')  
    plt.show()
    plt.close()
    
def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed

def new_shape_tiles (tr_img, n_pool, n_rows, n_cols):
    rows, cols = tr_img.shape[:-1]
    pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
    pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
    print(pad_rows, pad_cols)

    npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
    tr_img = np.pad(tr_img.copy(), pad_width=npad, mode='reflect')

    h, w, c = tr_img.shape
    patch_size_rows = h//n_rows
    patch_size_cols = w//n_cols
    num_patches_x = int(h/patch_size_rows)
    num_patches_y = int(w/patch_size_cols)
    
    return patch_size_rows, patch_size_cols


def inference_classifier_tiles(tr_img, n_pool, n_rows, n_cols, output_c_dim,
                               model_enc_sh, new_model_enc_sh, model_classifier, new_model_classifier):
    rows, cols = tr_img.shape[:-1]
    pad_rows = rows - np.ceil(rows / (n_rows * 2 ** n_pool)) * n_rows * 2 ** n_pool
    pad_cols = cols - np.ceil(cols / (n_cols * 2 ** n_pool)) * n_cols * 2 ** n_pool
    print(pad_rows, pad_cols)

    npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
    tr_img_pad = np.pad(tr_img.copy(), pad_width=npad, mode='reflect')

    h, w, c = tr_img_pad.shape
    patch_size_rows = h // n_rows
    patch_size_cols = w // n_cols
    num_patches_x = int(h / patch_size_rows)
    num_patches_y = int(w / patch_size_cols)

    for l1 in range(1, len(model_enc_sh.layers)):
        new_model_enc_sh.layers[l1].set_weights(model_enc_sh.layers[l1].get_weights())

    for l2 in range(1, len(model_classifier.layers)):
        new_model_classifier.layers[l2].set_weights(model_classifier.layers[l2].get_weights())

    patch_t = []
    for i in range(0, num_patches_y):
        for j in range(0, num_patches_x):
            patch = tr_img_pad[patch_size_rows * j:patch_size_rows * (j + 1),
                    patch_size_cols * i:patch_size_cols * (i + 1), :]
            feats_sh = new_model_enc_sh.predict(np.expand_dims(patch, axis=0))
            _, predictions_ = new_model_classifier.predict(feats_sh)
            del patch
            patch_t.append(predictions_[:, :, :, 1])
            del predictions_
    patches_pred = np.asarray(patch_t).astype(np.float32)
    prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols,
                                         patches_pred)
    prob_recontructed = prob_recontructed[:rows, :cols]
    print('[***] prob size DA', prob_recontructed.shape)
    return prob_recontructed

def img_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w, 6)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1),:]=patches_pred[count]
            count+=1
    return img_reconstructed

def inference_decoder_tiles(tr_img, n_pool, n_rows, n_cols, output_c_dim,
                            model_sh, new_model_sh, model_ex, new_model_ex, model_dec, new_model_dec):
    
    rows, cols = tr_img.shape[:-1]
    pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
    pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
    print(pad_rows, pad_cols)

    npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
    tr_img_pad = np.pad(tr_img.copy(), pad_width=npad, mode='reflect')

    h, w, c = tr_img_pad.shape
    patch_size_rows = h//n_rows
    patch_size_cols = w//n_cols
    num_patches_x = int(h/patch_size_rows)
    num_patches_y = int(w/patch_size_cols)
    
    for l1 in range(1, len(model_sh.layers)):
        new_model_sh.layers[l1].set_weights(model_sh.layers[l1].get_weights())

    for l2 in range(1, len(model_ex.layers)):
        new_model_ex.layers[l2].set_weights(model_ex.layers[l2].get_weights())
     
    for l3 in range(1, len(model_dec.layers)):
        new_model_dec.layers[l3].set_weights(model_dec.layers[l3].get_weights())
       
    patch_t = []
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            patch = tr_img_pad[patch_size_rows*j:patch_size_rows*(j+1), patch_size_cols*i:patch_size_cols*(i+1), :]
            feats_sh = new_model_sh.predict(np.expand_dims(patch, axis=0))
            feats_ex = new_model_ex.predict(np.expand_dims(patch, axis=0))
            predictions_ = new_model_dec.predict([feats_sh, feats_ex])
            del patch 
            patch_t.append(predictions_)
            del predictions_
    patches_pred = np.asarray(patch_t).astype(np.float32)
    img_recontructed = img_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
    
    return img_recontructed

def plot_prediction_base(batch_img, batch_mask, pred_cl, vmin = -1, vmax = 1, index=0):
    #i = np.random.randint(32)
    i = index
    x_sr_t0 = (batch_img[i,:,:,:3] - batch_img[i,:,:,:3].min())/(batch_img[i,:,:,:3].max()-batch_img[i,:,:,:3].min())
    x_sr_t1 = (batch_img[i,:,:,3:] - batch_img[i,:,:,3:].min())/(batch_img[i,:,:,3:].max()-batch_img[i,:,:,3:].min())
    
    lb_true_sr = batch_mask[i].argmax(axis=-1)
    lb_pred_sr = pred_cl[i].argmax(axis=-1)

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,4,1)
    plt.title('x_t0')
    ax1.imshow(x_sr_t0, vmin=vmin, vmax=vmax)
    ax1.axis('off')

    ax2 = fig.add_subplot(1,4,2)
    plt.title('x_t1')
    ax2.imshow(x_sr_t1, vmin=vmin, vmax=vmax)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1,4,3)
    plt.title('ref')
    ax3.imshow(lb_true_sr, cmap ='jet',vmax=2)
    ax3.axis('off')

    ax4 = fig.add_subplot(1,4,4)
    plt.title('pred')
    ax4.imshow(lb_pred_sr, cmap ='jet',vmax=2)
    ax4.axis('off') 

    plt.show()
    plt.close()
    
def inference_classifier_patch(patch_size, overlap_percent, tr_img, output_c_dim, model_enc_sh, model_classifier):
    overlap = round(patch_size * overlap_percent)
    # the overlap will be multiple of 2
    overlap -= overlap % 2
    stride = patch_size - overlap
    print('overlap: ', overlap)
    print('stride: ', stride)
    
    num_rows, num_cols = tr_img.shape[:-1]
    print('image shape: ', num_rows, num_cols)

    # Add Padding to the image to match with the patch size and the overlap
    step_row = (stride - num_rows % stride) % stride
    step_col = (stride - num_cols % stride) % stride

    pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
    tr_img = np.pad(tr_img.copy(), pad_tuple, mode = 'symmetric')
    
    # Number of patches: k1xk2
    k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
    print('Number of patches: %d x %d' %(k1, k2))
    
    # Inference
    probs = np.zeros((k1*stride, k2*stride, output_c_dim), dtype='float32')
    print(probs.shape)

    for i in range(k1):
        for j in range(k2):
            patch = tr_img[i*stride:(i*stride + patch_size), j*stride:(j*stride + patch_size), :]
            patch = patch[np.newaxis,...]
            feats_sh = model_enc_sh.predict(patch)
            _, infer = model_classifier.predict(feats_sh)

            probs[i*stride : i*stride+stride, j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride, overlap//2 : overlap//2 + stride, :]
        #print('row %d' %(i+1))

    # Taken off the padding
    probs = probs[:k1*stride-step_row, :k2*stride-step_col]
    return probs

def plot_prediction_multi_sr_tr(batch_sr1, rec_sr1, true_sr1, pred_sr1, batch_sr2, rec_sr2, true_sr2, pred_sr2, batch_tr1, rec_tr1, true_tr1, pred_tr1, batch_tr2, rec_tr2, true_tr2, pred_tr2, vmin = -1, vmax = 1, index=0):
    vb_t0 = [2,1,0]
    vb_t1 = [12,11,10]
    #i = np.random.randint(32)
    i = index
    x_sr1_t0 = normalization(batch_sr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_sr1_t0 = normalization(rec_sr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_sr1_t1 = normalization(batch_sr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_sr1_t1 = normalization(rec_sr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    lb_true_sr1 = true_sr1[i].argmax(axis=-1)
    lb_pred_sr1 = pred_sr1[i].argmax(axis=-1)
    
    x_sr2_t0 = normalization(batch_sr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_sr2_t0 = normalization(rec_sr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_sr2_t1 = normalization(batch_sr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_sr2_t1 = normalization(rec_sr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    lb_true_sr2 = true_sr2[i].argmax(axis=-1)
    lb_pred_sr2 = pred_sr2[i].argmax(axis=-1)
    
    x_tr1_t0 = normalization(batch_tr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_tr1_t0 = normalization(rec_tr1[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_tr1_t1 = normalization(batch_tr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_tr1_t1 = normalization(rec_tr1[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    #x_tr1_t0 = (batch_tr1[i,:,:,vb_t0] - batch_tr1[i,:,:,vb_t0].min())/(batch_tr1[i,:,:,vb_t0].max()-batch_tr1[i,:,:,vb_t0].min())
    #r_tr1_t0 = (rec_tr1[i,:,:,vb_t0] - rec_tr1[i,:,:,vb_t0].min())/(rec_tr1[i,:,:,vb_t0].max()-rec_tr1[i,:,:,vb_t0].min())
    #x_tr1_t1 = (batch_tr1[i,:,:,vb_t1] - batch_tr1[i,:,:,vb_t1].min())/(batch_tr1[i,:,:,vb_t1].max()-batch_tr1[i,:,:,vb_t1].min())
    #r_tr1_t1 = (rec_tr1[i,:,:,vb_t1] - rec_tr1[i,:,:,vb_t1].min())/(rec_tr1[i,:,:,vb_t1].max()-rec_tr1[i,:,:,vb_t1].min())
    lb_true_tr1 = true_tr1[i].argmax(axis=-1)
    lb_pred_tr1 = pred_tr1[i].argmax(axis=-1)
    
    x_tr2_t0 = normalization(batch_tr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    r_tr2_t0 = normalization(rec_tr2[i,:,:,vb_t0].transpose(1,2,0), norm_type = 2)
    x_tr2_t1 = normalization(batch_tr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    r_tr2_t1 = normalization(rec_tr2[i,:,:,vb_t1].transpose(1,2,0), norm_type = 2)
    #x_tr2_t0 = (batch_tr2[i,:,:,vb_t0] - batch_tr2[i,:,:,vb_t0].min())/(batch_tr2[i,:,:,vb_t0].max()-batch_tr2[i,:,:,vb_t0].min())
    #r_tr2_t0 = (rec_tr2[i,:,:,vb_t0] - rec_tr2[i,:,:,vb_t0].min())/(rec_tr2[i,:,:,vb_t0].max()-rec_tr2[i,:,:,vb_t0].min())
    #x_tr2_t1 = (batch_tr2[i,:,:,vb_t1] - batch_tr2[i,:,:,vb_t1].min())/(batch_tr2[i,:,:,vb_t1].max()-batch_tr2[i,:,:,vb_t1].min())
    #r_tr2_t1 = (rec_tr2[i,:,:,vb_t1] - rec_tr2[i,:,:,vb_t1].min())/(rec_tr2[i,:,:,vb_t1].max()-rec_tr2[i,:,:,vb_t1].min())
    lb_true_tr2 = true_tr2[i].argmax(axis=-1)
    lb_pred_tr2 = pred_tr2[i].argmax(axis=-1)

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(4,6,1)
    plt.title('x_sr1_t0')
    ax1.imshow(x_sr1_t0, vmin=vmin, vmax=vmax)
    ax1.axis('off')

    ax2 = fig.add_subplot(4,6,2)
    plt.title('rec_sr1_t0')
    ax2.imshow(r_sr1_t0, vmin=vmin, vmax=vmax)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(4,6,3)
    plt.title('x_sr1_t1')
    ax3.imshow(x_sr1_t1, vmin=vmin, vmax=vmax)
    ax3.axis('off')

    ax4 = fig.add_subplot(4,6,4)
    plt.title('rec_sr1_t1')
    ax4.imshow(r_sr1_t1, vmin=vmin, vmax=vmax)
    ax4.axis('off')
        
    ax5 = fig.add_subplot(4,6,5)
    plt.title('true sr1')
    ax5.imshow(lb_true_sr1, cmap ='jet',vmax=2)
    ax5.axis('off')

    ax6 = fig.add_subplot(4,6,6)
    plt.title('pred sr1')
    ax6.imshow(lb_pred_sr1, cmap ='jet', vmax=2)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(4,6,7)
    plt.title('x_sr2_t0')
    ax7.imshow(x_sr2_t0, vmin=vmin, vmax=vmax)
    ax7.axis('off')

    ax8 = fig.add_subplot(4,6,8)
    plt.title('rec_sr2_t0')
    ax8.imshow(r_sr2_t0, vmin=vmin, vmax=vmax)
    ax8.axis('off')
    
    ax9 = fig.add_subplot(4,6,9)
    plt.title('x_sr2_t1')
    ax9.imshow(x_sr2_t1, vmin=vmin, vmax=vmax)
    ax9.axis('off')

    ax10 = fig.add_subplot(4,6,10)
    plt.title('rec_sr2_t1')
    ax10.imshow(r_sr2_t1, vmin=vmin, vmax=vmax)
    ax10.axis('off')
        
    ax11 = fig.add_subplot(4,6,11)
    plt.title('true sr2')
    ax11.imshow(lb_true_sr2, cmap ='jet',vmax=2)
    ax11.axis('off')

    ax12 = fig.add_subplot(4,6,12)
    plt.title('pred sr2')
    ax12.imshow(lb_pred_sr2, cmap ='jet', vmax=2)
    ax12.axis('off')
    
    ax13 = fig.add_subplot(4,6,13)
    plt.title('x_tr1_t0')
    ax13.imshow(x_tr1_t0, vmin=vmin, vmax=vmax)
    ax13.axis('off')

    ax14 = fig.add_subplot(4,6,14)
    plt.title('rec_tr1_t0')
    ax14.imshow(r_tr1_t0, vmin=vmin, vmax=vmax)
    ax14.axis('off')
    
    ax15 = fig.add_subplot(4,6,15)
    plt.title('x_tr1_t1')
    ax15.imshow(x_tr1_t1, vmin=vmin, vmax=vmax)
    ax15.axis('off')

    ax16 = fig.add_subplot(4,6,16)
    plt.title('rec_tr1_t1')
    ax16.imshow(r_tr1_t1, vmin=vmin, vmax=vmax)
    ax16.axis('off')
    
    ax17 = fig.add_subplot(4,6,17)
    plt.title('true t1r')
    ax17.imshow(lb_true_tr1, cmap ='jet', vmax=2)
    ax17.axis('off')
    
    ax18 = fig.add_subplot(4,6,18)
    plt.title('pred tr1')
    ax18.imshow(lb_pred_tr1, cmap ='jet', vmax=2)
    ax18.axis('off')    

    # target 2
    ax19 = fig.add_subplot(4,6,19)
    plt.title('x_tr2_t0')
    ax19.imshow(x_tr2_t0, vmin=vmin, vmax=vmax)
    ax19.axis('off')

    ax20 = fig.add_subplot(4,6,20)
    plt.title('rec_tr2_t0')
    ax20.imshow(r_tr2_t0, vmin=vmin, vmax=vmax)
    ax20.axis('off')
    
    ax21 = fig.add_subplot(4,6,21)
    plt.title('x_tr2_t1')
    ax21.imshow(x_tr2_t1, vmin=vmin, vmax=vmax)
    ax21.axis('off')

    ax22 = fig.add_subplot(4,6,22)
    plt.title('rec_tr2_t1')
    ax22.imshow(r_tr2_t1, vmin=vmin, vmax=vmax)
    ax22.axis('off')
    
    ax23 = fig.add_subplot(4,6,23)
    plt.title('true tr2')
    ax23.imshow(lb_true_tr2, cmap ='jet', vmax=2)
    ax23.axis('off')
    
    ax24 = fig.add_subplot(4,6,24)
    plt.title('pred tr2')
    ax24.imshow(lb_pred_tr2, cmap ='jet', vmax=2)
    ax24.axis('off')  
    plt.show()
    plt.close()

def predictive_variance(pred_probs, axis=-1):
    pred_var = np.var(pred_probs, axis=axis)
    return pred_var

def predictive_entropy(pred_probs, axis = -1):
    epsilon = 1e-15
    pred_mean = np.mean(pred_probs, axis = axis)
    pred_entropy = - (pred_mean * np.log(pred_mean + epsilon))
    return pred_entropy


def inference_classifier_tiles_base(tr_img, n_pool, n_rows, n_cols, output_c_dim, model_classifier,
                                    new_model_classifier):
    rows, cols = tr_img.shape[:-1]
    pad_rows = rows - np.ceil(rows / (n_rows * 2 ** n_pool)) * n_rows * 2 ** n_pool
    pad_cols = cols - np.ceil(cols / (n_cols * 2 ** n_pool)) * n_cols * 2 ** n_pool
    print(pad_rows, pad_cols)

    npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
    tr_img_pad = np.pad(tr_img.copy(), pad_width=npad, mode='reflect')

    h, w, c = tr_img_pad.shape
    patch_size_rows = h // n_rows
    patch_size_cols = w // n_cols
    num_patches_x = int(h / patch_size_rows)
    num_patches_y = int(w / patch_size_cols)

    for l2 in range(1, len(model_classifier.layers)):
        new_model_classifier.layers[l2].set_weights(model_classifier.layers[l2].get_weights())

    patch_t = []
    for i in range(0, num_patches_y):
        for j in range(0, num_patches_x):
            patch = tr_img_pad[patch_size_rows * j:patch_size_rows * (j + 1),
                    patch_size_cols * i:patch_size_cols * (i + 1), :]
            # feats_sh = new_model_enc_sh.predict(np.expand_dims(patch, axis=0))
            _, predictions_ = new_model_classifier.predict(np.expand_dims(patch, axis=0))
            del patch
            patch_t.append(predictions_[:, :, :, 1])
            del predictions_
    patches_pred = np.asarray(patch_t).astype(np.float32)
    prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols,
                                         patches_pred)
    prob_recontructed = prob_recontructed[:rows, :cols]

    return prob_recontructed


def load_samples(current_folder, plabels, folder, patch_size, channels, num_classes):
    content_list_img = {}
    content_list_msk = {}
    for index, val in enumerate(plabels):
        print(index, val)
        path_img = os.path.join(current_folder, val, folder, 'path_image')
        path_msk = os.path.join(current_folder, val, folder, 'path_mask')
        content_list_img[plabels[index]] = os.listdir(path_img)
        content_list_msk[plabels[index]] = os.listdir(path_msk)

    file_list = list(content_list_img.values())
    mask_list = list(content_list_msk.values())

    content_list_img = []
    content_list_msk = []
    for name in range(0, len(plabels)):
        content_list_img = content_list_img + file_list[name]
        content_list_msk = content_list_msk + mask_list[name]

    matrix_samples = np.zeros((len(content_list_img), patch_size, patch_size, channels))
    for i in range(0, len(content_list_img)):
        matrix_samples[i, :, :, :] = np.load(path_img + '/' + content_list_img[i])

    matrix_labels = np.zeros((len(content_list_msk), patch_size, patch_size, num_classes))
    for j in range(0, len(content_list_msk)):
        matrix_labels[j, :, :, :] = tf.keras.utils.to_categorical(np.load(path_msk + '/' + content_list_msk[j]), num_classes)

    print(matrix_samples.shape, matrix_labels.shape)
    return matrix_samples, matrix_labels

def get_batch(n, all_samples, all_labels, num_classes=3, p_pos=None, return_inds=False):
    n_train_samples = all_labels.shape[0]
    train_inds = np.random.permutation(np.arange(n_train_samples))
    #print(train_inds)
    selected_inds = np.random.choice(train_inds, size=n, replace=False, p=p_pos)
    sorted_inds = np.sort(selected_inds)
    train_img = all_samples[sorted_inds,:,:,:]
    train_label = all_labels[sorted_inds,:,:,:]
    #train_label = tf.keras.utils.to_categorical(train_label , num_classes)
    return (train_img, train_label, sorted_inds) if return_inds else (train_img, train_label)

def get_latent_mu(images, encoder, encoder_sh, encoder_ex, latent_dim, batch_size=1024):
    N = images.shape[0]
    mu = np.zeros((N, latent_dim))
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind+batch_size, N+1)
        batch = images[start_ind:end_ind]
        batch_sh = encoder_sh.predict(batch)
        batch_ex = encoder_ex.predict(batch)
        encoder_output_ = encoder.predict([batch_sh, batch_ex])
        batch_mu = encoder_output_[:, :latent_dim]
        #print('batch_mu: ', batch_mu.shape)
        #_, batch_mu, _ = dbvae.encode(batch)
        mu[start_ind:end_ind] = batch_mu
    return mu


def get_training_sample_probabilities(images, encoder, encoder_sh, encoder_ex, latent_dim, bins=10, smoothing_fac=0.01):
    print("Recomputing the sampling probabilities")

    # run the input batch and get the latent variable means
    mu = get_latent_mu(images, encoder, encoder_sh, encoder_ex, latent_dim)
    print('shape mu: ', mu.shape)

    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])

    # consider the distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data sample falls in
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        # call the digitize function to find which bins in the latent distribution
        #    every data sample falls in to
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        bin_idx = np.digitize(latent_distribution, bin_edges)

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])

        # normalize all probabilities
        p = p / np.sum(p)

        # update sampling probabilities by considering whether the newly
        # computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(p, training_sample_p)

    # final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p


def GaussianSampling_np(inputs, ndims):
    means, logvar = inputs
    epsilon = np.random.normal(0., 1.,  size=(1, ndims))
    samples = means + np.exp(0.5*logvar)*epsilon
    return samples