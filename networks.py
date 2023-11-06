# Import libraries
import tensorflow as tf
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.initializers import RandomNormal
#from tensorflow.keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, load_model, Sequential, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def reconstruction_loss(img, pred):
    loss = tf.squared_difference(img, pred)
    return loss

def softmax_loss_d(onehot_labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels= onehot_labels, logits=logits)
    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels= onehot_labels, logits=logits)
    return loss

def softmax_loss_c1(onehot_labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels= onehot_labels, logits=logits)
    return loss

def softmax_loss_c2(onehot_labels, logits):
    #print(tf.shape(logits), tf.shape(onehot_labels))
    logits = tf.reshape(logits, shape = (tf.shape(logits)[0],tf.shape(logits)[1]* tf.shape(logits)[2], tf.shape(logits)[3]))
    labels = tf.reshape(onehot_labels, shape = (tf.shape(onehot_labels)[0],tf.shape(onehot_labels)[1]* tf.shape(onehot_labels)[2], tf.shape(onehot_labels)[3]))                   
    class_weights = tf.constant([0.2, 0.6, 0.2])
    weights = tf.reduce_sum(labels*class_weights, axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels= onehot_labels, logits=logits) 
    #print(tf.shape(loss))
    loss = loss * weights
    return loss

def softmax_loss_c(onehot_labels, logits):
    class_weights = tf.constant([1.0, 1.0, 0.0])
    logits = tf.reshape(logits, shape = (tf.shape(logits)[0],tf.shape(logits)[1]* tf.shape(logits)[2], tf.shape(logits)[3]))
    labels = tf.reshape(onehot_labels, shape = (tf.shape(onehot_labels)[0],tf.shape(onehot_labels)[1]* tf.shape(onehot_labels)[2], tf.shape(onehot_labels)[3]))                   
    weight_map = tf.multiply(labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=-1)
    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(labels= labels, logits=logits)
    weighted_loss = tf.multiply(loss_map, weight_map) 
    return weighted_loss

def entropy_criterion(logits):
    ll = tf.nn.softmax(logits)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ll, logits=logits))

def conv_block(input_data, n_filters, k_size=3, strides=2, activation='relu', padding='same',
               batchnorm=False, instnorm = False, name='None', ind = 0):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    x = Conv2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'conv_net_'+str(ind))(input_data)
    if batchnorm:
        x = BatchNormalization(momentum=0.8, name=name+'bn_net_'+str(ind))(x)
    if instnorm:
        x = InstanceNormalization(name=name+'isnt_norm_net_'+str(ind))(x)
    if activation is 'LReLU':
        x = LeakyReLU(alpha=0.2, name=name+'LReLU_net_'+str(ind))(x)        
    elif activation is 'relu':
        x = Activation('relu', name=name+'relu_net_'+str(ind))(x)
    else:
        x = Activation('linear', name=name+'linear_net_'+str(ind))(x)
    return x

def upconv_block(input_data, n_filters, k_size=3, strides=2, activation='relu', padding='same',
                 batchnorm=False, instnorm = False, name='None', ind = 0):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    #x = Conv2DTranspose(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'deconv_net_'+str(ind))(input_data)
    x = Conv2D(n_filters, k_size, padding=padding, kernel_initializer=init,
               name=name+'upconv_net_'+str(ind))(UpSampling2D(size = (strides,strides), name=name+'upsamplig_net_'+str(ind))(input_data))
    if batchnorm:
        x = BatchNormalization(momentum=0.8, name=name+'bn_net_'+str(ind))(x)
    if instnorm:
        x = InstanceNormalization(name=name+'ins_norm_net_'+str(ind))(x)
    if activation is 'LReLU':
        x = LeakyReLU(alpha=0.2, name=name+'LReLU_net_'+str(ind))(x)        
    elif activation is 'relu':
        x = Activation('relu', name=name+'relu_net_'+str(ind))(x)
    else:
        x = Activation('linear', name=name+'linear_net_'+str(ind))(x)
    return x


def resnet_block(x, nb_filters, k_size, strides, ind, name, batchnorm=False, instnorm = False):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    x_init = x
    
    x = conv_block(x, nb_filters, k_size=k_size, strides=strides, activation='relu', padding='same',
                   batchnorm=batchnorm, instnorm = instnorm, name=name+'res1_', ind = ind)
    
    x = conv_block(x, nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                   batchnorm=batchnorm, instnorm = instnorm, name=name+'res2_', ind = ind)
    ## Add
    x = Add(name=name+'add_resnet_net'+str(ind))([x, x_init])

    return x

def build_encoder_sh(input_shape, nb_filters=64, name='encoder_sh'):
    '''Base network to be shared (eq. to feature extraction)''' 
    input_layer = Input(shape=input_shape, name=name+"input_enc_net")
    
    conv0 = conv_block(input_layer, nb_filters, k_size=7, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 0)
    conv1 = conv_block(conv0, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 1)
    #pool1 = MaxPool2D((2 , 2), name = name+'maxp_'+str(1))(conv1)
    conv2 = conv_block(conv1, 4*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 2)
    #pool2 = MaxPool2D((2 , 2), name = name+'maxp_'+str(2))(conv2)
    feats = conv2
    
    for i in range(6):
        feats = resnet_block(feats, 4*nb_filters, 3, 1, i+1, name, batchnorm=False, instnorm = True)    
          
    return Model(inputs=input_layer, outputs=feats, name=name+'model')


def build_encoder_ex(input_shape, nb_filters=64, name='encoder_ex'):
    '''Base network to be shared (eq. to feature extraction)'''  
    input_layer = Input(shape=input_shape, name=name+"input_enc_net")
    
    conv0 = conv_block(input_layer, nb_filters, k_size=7, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 0)
    conv1 = conv_block(conv0, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 1)
    #pool1 = MaxPool2D((2 , 2), name = name+'maxp_'+str(1))(conv1)
    conv2 = conv_block(conv1, 4*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 2)
    #pool2 = MaxPool2D((2 , 2), name = name+'maxp_'+str(2))(conv2)
    feats = conv2
    
    for i in range(6):
        feats = resnet_block(feats, 4*nb_filters, 3, 1, i+1, name, batchnorm=False, instnorm = True)

    return Model(inputs=input_layer, outputs=feats, name=name+'model')


def build_decoder(feats_sh, feats_tx=None, out_dim=4, nb_filters=32, name='decoder'):
    '''Base network to be shared (eq. to feature extraction)'''
    init = RandomNormal(stddev=0.02)  
    feats_sh = Input(shape=feats_sh, name=name+"feats_sh_dec_net")
    if feats_tx:
        feats_tx = Input(shape=feats_tx, name=name+"feats_ex_dec_net")  
        print(tf.shape(feats_sh), tf.shape(feats_tx))
        #feats = Add(name=name+"feats_dec_net")([feats_sh, feats_tx])
        feats = Concatenate(name=name+"feats_dec_net")([feats_sh, feats_tx])
        inputs = [feats_sh, feats_tx]
    else:
        feats = feats_sh
        inputs = feats_sh
    
    conv0 = conv_block(feats, 4*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 0)
    for i in range(4):
        rb_feats = resnet_block(conv0, 4*nb_filters, 3, 1, i+1, name, batchnorm=False, instnorm = True)
        
    upconv0 = upconv_block(rb_feats, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = True, name=name, ind = 1)
    upconv1 = upconv_block(upconv0, 1*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = True, name=name, ind = 2)
    out = Conv2D(out_dim, (1, 1), activation='tanh', kernel_initializer=init, padding="same",
               name=name+'conv_up_net_out')(upconv1)
        
    return Model(inputs=inputs, outputs=out, name=name+'model')
    

# define the standalone discriminator model
def build_discriminator(input_shape, nb_filters=64, num_domains=3, name='disc_'):
    '''Base network to be shared (eq. to feature extraction)'''

    init = RandomNormal(stddev=0.02)     
    input_layer = Input(shape=input_shape, name=name+"input_disc_net")
    
    conv0 = conv_block(input_layer, nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 0) 
    conv1 = conv_block(conv0, 2*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 1)
    conv2 = conv_block(conv1, 4*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 2)
    conv3 = conv_block(conv2, 8*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 3) 
    #for i in range(2):
    #    rb_feats = resnet_block(conv3, 8*nb_filters, 3, 1, i+1, name, batchnorm=False, instnorm = True)
        
    conv4 = conv_block(conv3, 1, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 4) 
    
    feats = Flatten()(conv4)
    logits = Dense(num_domains, activation=None, kernel_initializer=init, name=name+'dense_net_0')(feats)
    out = Activation('softmax', name=name+'act_softmax')(logits)
      
    return Model(inputs=input_layer, outputs=[logits, out], name=name+'model')

# define the standalone discriminator model
def build_patch_discriminator(input_shape, nb_filters=64, num_domains=3, name='disc_'):
    '''Base network to be shared (eq. to feature extraction)'''

    init = RandomNormal(stddev=0.02)     
    input_layer = Input(shape=input_shape, name=name+"input_disc_net")
    
    conv0 = conv_block(input_layer, nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 0) 
    conv1 = conv_block(conv0, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 1)
    conv2 = conv_block(conv1, 4*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 2)
    conv3 = conv_block(conv2, 8*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = True, name=name, ind = 3) 
    #for i in range(2):
    #    rb_feats = resnet_block(conv3, 8*nb_filters, 3, 1, i+1, name, batchnorm=False, instnorm = True)
        
    #conv4 = conv_block(conv3, num_domains, k_size=3, strides=1, activation='relu', padding='same',
    #                   batchnorm=False, instnorm = True, name=name, ind = 4) 
    
    #feats = Flatten()(conv4)
    #logits = Dense(num_domains, activation=None, kernel_initializer=init, name=name+'dense_net_0')(feats)
    #out = Activation('softmax', name=name+'act_softmax')(logits)
    logits = Conv2D(num_domains,(1,1), activation = None, padding = 'same', kernel_initializer=init,
                    name=name+'conv_output')(conv3)
    
    out = Activation('softmax', name=name+'act_softmax')(logits)
      
    return Model(inputs=input_layer, outputs=[logits, out], name=name+'model')

# define the standalone classifier model
def build_classifier_rb(input_shape, nb_filters=64, num_classes=3, name='class_'):
    '''Base network to be shared (eq. to feature extraction)'''  
    init = RandomNormal(stddev=0.02) 
    input_layer = Input(shape=input_shape, name=name+"input_class_net")
    
    conv1 = conv_block(input_layer, 8*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = False, name=name, ind = 1)
    rb1 = resnet_block(conv1, 8*nb_filters, 3, 1, 1, name, batchnorm=False, instnorm = False)
    pool1 = MaxPool2D((2 , 2), name = name+'maxp_'+str(1))(rb1)
    
    conv2 = conv_block(pool1, 8*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = False, name=name, ind = 2)
    rb2 = resnet_block(conv2, 8*nb_filters, 3, 1, 2, name, batchnorm=False, instnorm = False)
    pool2 = MaxPool2D((2 , 2), name = name+'maxp_'+str(2))(rb2)

    # Upsamplig
    upconv3 = upconv_block(pool2, 8*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 4)
    merged2 = concatenate([rb2, upconv3], name=name+'concatenate_2')
    
    upconv2 = upconv_block(merged2, 4*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 5)
    merged1 = concatenate([rb1, upconv2], name=name+'concatenate_1')
    
    upconv1 = upconv_block(merged1, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 6)
    #merged0 = concatenate([conv0, upconv1], name=name+'concatenate_0')
    
    upconv0 = upconv_block(upconv1, nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 7)
    
    logits = Conv2D(num_classes,(1,1), activation = None, padding = 'same', kernel_initializer=init,
                    name=name+'conv_output')(upconv0)
    
    out = Activation('softmax', name=name+'act_softmax')(logits)
    #output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
          
    return Model(inputs=input_layer, outputs=[logits, out] , name=name+'model')


def build_classifier(input_shape, nb_filters=64, num_classes=3, name='class_'):
    '''Base network to be shared (eq. to feature extraction)'''  
    init = RandomNormal(stddev=0.02) 
    input_layer = Input(shape=input_shape, name=name+"input_class_net")
    
    conv0 = conv_block(input_layer, 4*nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm = False, name=name, ind = 0)
        
    conv1 = conv_block(conv0, 8*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = False, name=name, ind = 1)
    #rb1 = resnet_block(conv1, 8*nb_filters, 3, 1, 1, name, batchnorm=False, instnorm = False)
    #pool1 = MaxPool2D((2 , 2), name = name+'maxp_'+str(1))(conv1)
    
    conv2 = conv_block(conv1, 8*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm = False, name=name, ind = 2)
    #rb2 = resnet_block(conv2, 8*nb_filters, 3, 1, 2, name, batchnorm=False, instnorm = False)
    #pool2 = MaxPool2D((2 , 2), name = name+'maxp_'+str(2))(conv2)

    # Upsamplig
    upconv3 = upconv_block(conv2, 8*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 4)
    merged2 = concatenate([conv1, upconv3], name=name+'concatenate_2')
    
    upconv2 = upconv_block(merged2, 4*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 5)
    merged1 = concatenate([conv0, upconv2], name=name+'concatenate_1')
    
    upconv1 = upconv_block(merged1, 2*nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 6)
    
    upconv0 = upconv_block(upconv1, nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm = False, name=name, ind = 7)
    
    logits = Conv2D(num_classes,(1,1), activation = None, padding = 'same', kernel_initializer=init,
                    name=name+'conv_output')(upconv0)
    
    out = Activation('softmax', name=name+'act_softmax')(logits)
    #output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
          
    return Model(inputs=input_layer, outputs=[logits, out] , name=name+'model')


def build_classifier_base(input_shape, nb_filters=64, num_classes=3, skip_c = False, name='class_'):
    '''Base network to be shared (eq. to feature extraction)'''
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=input_shape, name=name + "input_class_base_net")

    conv0 = conv_block(input_layer, nb_filters, k_size=7, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=0)
    conv1 = conv_block(conv0, 2 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=1)
    conv2 = conv_block(conv1, 4 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=2)
    conv3 = conv_block(conv2, 8 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=3)
    conv4 = conv_block(conv3, 8 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=4)

    feats = conv4
    for i in range(6):
        feats = resnet_block(feats, 8 * nb_filters, 3, 1, i + 1, name,
                             batchnorm=False, instnorm=True)

    # Upsamplig
    if skip_c:
        upconv3 = upconv_block(feats, 8 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=5)
        merged2 = concatenate([conv3, upconv3], name=name + 'concatenate_2')

        upconv2 = upconv_block(merged2, 4 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=6)
        merged1 = concatenate([conv2, upconv2], name=name + 'concatenate_1')

        upconv1 = upconv_block(merged1, 2 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=7)
        merged0 = concatenate([conv1, upconv1], name=name + 'concatenate_0')

        upconv0 = upconv_block(merged0, nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=8)
    else:
        upconv3 = upconv_block(feats, 8 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=5)
        upconv2 = upconv_block(upconv3, 4 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=6)
        upconv1 = upconv_block(upconv2, 2 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=7)
        upconv0 = upconv_block(upconv1, nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                               batchnorm=False, instnorm=False, name=name, ind=8)

    logits = Conv2D(num_classes, (1, 1), activation=None, padding='same', kernel_initializer=init,
                    name=name + 'conv_output')(upconv0)

    out = Activation('softmax', name=name + 'act_softmax')(logits)
    # output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)

    return Model(inputs=input_layer, outputs=[logits, out], name=name + 'model')


def latent_vae(feats_sh, feats_tx=None, name='vae'):
    '''Base network to be shared (eq. to feature extraction)'''
    init = RandomNormal(stddev=0.02)
    feats_sh = Input(shape=feats_sh, name=name + "feats_sh_dec_net")
    if feats_tx:
        feats_tx = Input(shape=feats_tx, name=name + "feats_ex_dec_net")
        print(tf.shape(feats_sh), tf.shape(feats_tx))
        # feats = Add(name=name+"feats_dec_net")([feats_sh, feats_tx])
        feats = Concatenate(name=name + "feats_dec_net")([feats_sh, feats_tx])
        inputs = [feats_sh, feats_tx]
    else:
        feats = feats_sh
        inputs = feats_sh

    ### lattent feature VAE
    feats_vae = Flatten(name=name + 'flatten_net_vae')(feats)
    feats_vae = Dense(100, kernel_initializer=init, name=name + 'dense_net_vae1')(feats_vae)
    logits_vae = Dense(100 * 2, kernel_initializer=init, name=name + 'dense_vae_out', activation=None)(feats_vae)

    return Model(inputs=inputs, outputs= logits_vae, name=name + 'model')

def GaussianSampling(inputs, ndims):
    means, logvar = inputs
    epsilon = tf.random.normal(shape=(1, ndims), mean=0., stddev=1.)
    samples = means + tf.exp(0.5*logvar)*epsilon
    return samples

def build_VAE_decoder(input_shape, out_dim=4, nb_filters=32, name='decoder'):
    '''Base network to be shared (eq. to feature extraction)'''
    init = RandomNormal(stddev=0.02)
    input_layer = Input(shape=input_shape, name=name+"input_vdec_net")

    dense1 = Dense(32 * 32 * 4 * nb_filters, kernel_initializer=init, name=name + 'dense_net_d2')(input_layer)
    reshape_l = Reshape((32, 32, 4 * nb_filters), name=name + 'reshape')(dense1)

    conv0 = conv_block(reshape_l, 4 * nb_filters, k_size=3, strides=1, activation='relu', padding='same',
                       batchnorm=False, instnorm=True, name=name, ind=0)
    for i in range(4):
        rb_feats = resnet_block(conv0, 4 * nb_filters, 3, 1, i + 1, name, batchnorm=False, instnorm=True)

    upconv0 = upconv_block(rb_feats, 2 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm=True, name=name, ind=1)
    upconv1 = upconv_block(upconv0, 1 * nb_filters, k_size=3, strides=2, activation='relu', padding='same',
                           batchnorm=False, instnorm=True, name=name, ind=2)
    out = Conv2D(out_dim, (1, 1), activation='tanh', kernel_initializer=init, padding="same",
                 name=name + 'conv_up_net_out')(upconv1)

    return Model(inputs=input_layer, outputs=out, name=name + 'model')


def build_vae(encoder, decoder, x_sh, x_ex, latent_dim):
    # Variational autoencoder
    encoder_output = encoder([x_sh, x_ex])
    # latent variable distribution parameters
    print(encoder_output)
    z_mean = encoder_output[:, :latent_dim]
    z_logsigma = encoder_output[:, latent_dim:]

    z = GaussianSampling([z_mean, z_logsigma], latent_dim)
    z = Reshape((1, latent_dim))(z)

    # reconstruction
    recon = decoder(z)

    return z_mean, z_logsigma, z, recon