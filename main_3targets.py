from utils_mt import *
from networks import *
root_path = os.getcwd()
print(root_path)

# folder to load config file
CONFIG_PATH = root_path
config = load_config(CONFIG_PATH, 'main_config.yaml')
#print('config: ', config)

exp = config['num_exp']
path_exp = root_path+'/' + config['folder_exp'] + '/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'

if not os.path.exists(path_exp):
    os.makedirs(path_exp)
if not os.path.exists(path_models):
    os.makedirs(path_models)
if not os.path.exists(path_maps):
    os.makedirs(path_maps)

dir_data = config['data_directory']
tr_folder = 'train'
vl_folder = 'val'
save_prob = False
multi_sr = False
times = config['times']
channels = config['channels'] # each image
num_classes = config['num_classes']
num_domains = config['num_domains']

patch_dis = config['patch_dis']
if patch_dis == True:
    patch_size_dis = config['patch_size_dis']
else:
    patch_size_dis = None

# list of folders to be merged
# SOURCE - TARGET1 - TARGET2
#list_dir = ['MT_1C', 'PA_1C', 'MA_1C']
list_dir = [config['sr_1'], config['tr_1'], config['tr_2'], config['tr_3']]
test_name = config['tr_1'][0:2]
patches_tg = 'patches_random'

# parameters
lr = config['lr']
batch_size = config['batch_size']
patch_size = config['patch_size']
cdims = config['channels']
#cdims = latent_dim//2
num_clases = config['num_classes']

lambda_r = config['lambda_r_']
lambda_ds = config['lambda_ds_']
lambda_de = config['lambda_de_']
lambda_c = config['lambda_c_']

print('Parameters')
print('\n', 'lr :', lr, '\n', 'batch_size: ', batch_size, '\n', 'patch_size: ', patch_size, '\n', 'num_clases: ', num_clases)

print('\n', 'lambda r :', lambda_r, '\n', 'lambda_ds: ', lambda_ds, '\n', 'lambda_de: ', lambda_de, '\n', 'lambda_c: ', lambda_c)

## computation graph
K.clear_session()

# Inputs
img_sr = K.placeholder(dtype=tf.float32, name="img_source", shape=(batch_size, patch_size, patch_size, cdims))
img_tr1 = K.placeholder(dtype=tf.float32, name="img_target1", shape=(batch_size, patch_size, patch_size, cdims))
img_tr2 = K.placeholder(dtype=tf.float32, name="img_target2", shape=(batch_size, patch_size, patch_size, cdims))
img_tr3 = K.placeholder(dtype=tf.float32, name="img_target3", shape=(batch_size, patch_size, patch_size, cdims))
class_label_sr = K.placeholder(dtype=tf.float32, name='source_mask_label',
                               shape=(batch_size, patch_size, patch_size, num_clases))
if patch_dis == True:
    domain_label_sr = K.placeholder(dtype=tf.float32, name='source_domain_label',
                                    shape=(batch_size, patch_size_dis, patch_size_dis, num_domains))
    domain_label_tr1 = K.placeholder(dtype=tf.float32, name='target1_domain_label',
                                     shape=(batch_size, patch_size_dis, patch_size_dis, num_domains))
    domain_label_tr2 = K.placeholder(dtype=tf.float32, name='target2_domain_label',
                                     shape=(batch_size, patch_size_dis, patch_size_dis, num_domains))
    domain_label_tr3 = K.placeholder(dtype=tf.float32, name='target3_domain_label',
                                     shape=(batch_size, patch_size_dis, patch_size_dis, num_domains))
else:
    domain_label_sr = K.placeholder(dtype=tf.float32, name='source_domain_label', shape=(batch_size, num_domains))
    domain_label_tr1 = K.placeholder(dtype=tf.float32, name='target1_domain_label', shape=(batch_size, num_domains))
    domain_label_tr2 = K.placeholder(dtype=tf.float32, name='target2_domain_label', shape=(batch_size, num_domains))
    domain_label_tr3 = K.placeholder(dtype=tf.float32, name='target3_domain_label', shape=(batch_size, num_domains))

# encoders definition
nb_filters_sh = 16
encoder_sh = build_encoder_sh((patch_size, patch_size, cdims), nb_filters=nb_filters_sh, name='enc_sh_')
encoder_ex = build_encoder_ex((patch_size, patch_size, cdims), nb_filters=nb_filters_sh, name='enc_ex_')

encoder_sh.summary()
encoder_ex.summary()

# Getting output feats shapes
sh_out_shape = encoder_sh.layers[-1].output_shape[1:]
ex_out_shape = sh_out_shape

# decoder definition
decoder_sh = build_decoder(sh_out_shape, ex_out_shape, out_dim = cdims, nb_filters=16, name='dec_')
decoder_sh.summary()

# discriminator definition
if patch_dis:
    domain_discriminator = build_patch_discriminator((sh_out_shape), nb_filters=4, num_domains=num_domains, name='dis_')
else:
    domain_discriminator = build_discriminator((sh_out_shape), nb_filters=4, num_domains = num_domains, name='dis_')
domain_discriminator.summary()

# classifier definition
nb_filters_c = 16
classifier = build_classifier((sh_out_shape), nb_filters=nb_filters_c, num_classes = 3, name='cl_')
classifier.summary()

# features shared
feats_sh_sr = encoder_sh(img_sr)
feats_sh_tr1 = encoder_sh(img_tr1)
feats_sh_tr2 = encoder_sh(img_tr2)
feats_sh_tr3 = encoder_sh(img_tr3)

# features exclusive
feats_ex_sr = encoder_ex(img_sr)
feats_ex_tr1 = encoder_ex(img_tr1)
feats_ex_tr2 = encoder_ex(img_tr2)
feats_ex_tr3 = encoder_ex(img_tr3)

# decoder
rec_sr = decoder_sh([feats_sh_sr, feats_ex_sr])
rec_tr1 = decoder_sh([feats_sh_tr1, feats_ex_tr1])
rec_tr2 = decoder_sh([feats_sh_tr2, feats_ex_tr2])
rec_tr3 = decoder_sh([feats_sh_tr3, feats_ex_tr3])

# domain discriminator
dom_sh_sr_logits, _ = domain_discriminator(feats_sh_sr)
dom_ex_sr_logits, _ = domain_discriminator(feats_ex_sr)

# target 1
dom_sh_tr1_logits, _ = domain_discriminator(feats_sh_tr1)
dom_ex_tr1_logits, _ = domain_discriminator(feats_ex_tr1)

# target 2
dom_sh_tr2_logits, _ = domain_discriminator(feats_sh_tr2)
dom_ex_tr2_logits, _ = domain_discriminator(feats_ex_tr2)

# target 3
dom_sh_tr3_logits, _ = domain_discriminator(feats_sh_tr3)
dom_ex_tr3_logits, _ = domain_discriminator(feats_ex_tr3)

# classifier
cl_sr_logits, _ = classifier(feats_sh_sr)
cl_tr1_logits, _ = classifier(feats_sh_tr1)
cl_tr2_logits, _ = classifier(feats_sh_tr2)
cl_tr3_logits, _ = classifier(feats_sh_tr3)

# Computing loss

# loss decoder
loss_dec = lambda_r * (tf.reduce_mean(tf.abs(img_sr-rec_sr)) + tf.reduce_mean(tf.abs(img_tr1-rec_tr1)) + \
                      tf.reduce_mean(tf.abs(img_tr2-rec_tr2)) + tf.reduce_mean(tf.abs(img_tr3-rec_tr3)))

# loss domain classifier

loss_feat_ex_dom_dis = lambda_de * (tf.reduce_mean(softmax_loss_d(domain_label_sr, dom_ex_sr_logits)) + \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr1, dom_ex_tr1_logits)) + \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr2, dom_ex_tr2_logits)) + \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr3, dom_ex_tr3_logits)))

loss_feat_sh_dom_dis = lambda_ds * (tf.reduce_mean(softmax_loss_d(domain_label_sr, dom_sh_sr_logits)) + \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr1, dom_sh_tr1_logits))+ \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr2, dom_sh_tr2_logits))+ \
                                   tf.reduce_mean(softmax_loss_d(domain_label_tr3, dom_sh_tr3_logits)))

loss_dis = loss_feat_sh_dom_dis + loss_feat_ex_dom_dis

# Regularization term
#cl_reg = lambda_c * entropy_criterion(cl_tr1_logits)

# loss classifier label
loss_cl = tf.reduce_mean(softmax_loss_c(class_label_sr, cl_sr_logits)) # + 0.01 * cl_reg

# loss classifier shared
loss_sh_cl = lambda_c * tf.reduce_mean(softmax_loss_c(class_label_sr, cl_sr_logits)) # + 0.01 * cl_reg

# loss shared encoder
loss_sh = loss_dec + loss_sh_cl - loss_feat_sh_dom_dis

# loss exclusive encoder
loss_ex = loss_dec + loss_feat_ex_dom_dis

# Collecting variables for training
t_vars = tf.trainable_variables()

e_sh_vars = [var for var in t_vars if 'enc_sh_' in var.name] # Encoder shared variables
e_ex_vars = [var for var in t_vars if 'enc_ex_' in var.name] # Encoder exclusive variables
cl_vars =   [var for var in t_vars if 'cl_' in var.name] # Classifier variables
dec_vars =  [var for var in t_vars if 'dec_' in var.name] # Decoder variables
dis_vars =  [var for var in t_vars if 'dis_' in var.name] # Discriminator variables

# Optimizer parameters
lr_d = 0.0002
beta1 = 0.5

# Assings variables and corresponding lossses to be minimized
sh_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(loss_sh, var_list = e_sh_vars )
ex_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(loss_ex, var_list = e_ex_vars)
cl_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(loss_cl, var_list = cl_vars)
dec_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(loss_dec, var_list = dec_vars)
dis_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(loss_dis, var_list = dis_vars)

sess = K.get_session()

with open(path_models + '/' + 'enc_sh_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        encoder_sh.summary()

with open(path_models + '/' + 'enc_ex_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        encoder_ex.summary()

with open(path_models + '/' + 'dec_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        decoder_sh.summary()

with open(path_models + '/' + 'dis_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        domain_discriminator.summary()

with open(path_models + '/' + 'cl_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        classifier.summary()

train_gen = data_gen(dir_data, list_dir, tr_folder, batch_size, patch_size, channels, num_classes, patches_tg, multi_sr, patch_dis, patch_size_dis)

tr_samples, _ = retrieve_num_samples(dir_data, list_dir, tr_folder, patches_tg, multi_sr)
print('Total training samples sr: ', len(tr_samples[0]))
print('Total training samples tr1: ', len(tr_samples[1]))
print('Total training samples tr2: ', len(tr_samples[2]))
print('Total training samples tr3: ', len(tr_samples[3]))
tr_samples_min = min(len(tr_samples[0]), len(tr_samples[1]), len(tr_samples[2]), len(tr_samples[3]))
print(tr_samples_min)

valid_gen = data_gen(dir_data, list_dir, vl_folder, batch_size, patch_size, channels, num_clases, patches_tg, multi_sr, patch_dis, patch_size_dis)
vl_samples,_ = retrieve_num_samples(dir_data, list_dir, vl_folder, patches_tg, multi_sr)
print('Total validation samples sr: ', len(vl_samples[0]))
print('Total validation samples tr1: ', len(vl_samples[1]))
print('Total validation samples tr2: ', len(vl_samples[2]))
print('Total validation samples tr3: ', len(vl_samples[3]))
vl_samples_min = min(len(vl_samples[0]), len(vl_samples[1]), len(vl_samples[2]), len(vl_samples[3]))
print(vl_samples_min)

tr_time = []
for tm in range(0, times):
    print('time: ', tm)

    path_model_tm = path_models + '/run_' + str(tm)
    path_loss_tm = path_exp + '/loss/' + '/run_' + str(tm)
    if not os.path.exists(path_model_tm):
        os.makedirs(path_model_tm)
    if not os.path.exists(path_loss_tm):
        os.makedirs(path_loss_tm)

    sess.run(tf.compat.v1.global_variables_initializer())
    index_plt = np.random.randint(batch_size)

    num_of_trn_batches = tr_samples_min // batch_size
    num_of_val_batches = vl_samples_min // batch_size
    epochs = 100
    best_val_loss = np.inf
    patience = 10
    tr_dec, tr_dis, tr_esh, tr_eex, tr_cls, tr_dsh, tr_dex = [], [], [], [], [], [], []
    vl_dec, vl_dis, vl_esh, vl_eex, vl_cls, vl_dsh, vl_dex = [], [], [], [], [], [], []

    for epoch in range(epochs):
        print('epoch :', epoch)
        enc_sh_loss, enc_ex_loss, dec_loss, dis_loss, ex_dis_loss, sh_dis_loss, cl_loss = [], [], [], [], [], [], []

        start_time = time.time()
        for idx in range(0, num_of_trn_batches):

            # selecting a batch of images
            batch_img, batch_mask, batch_dom = next(train_gen)
            batch_img_sr = batch_img[:, :, :, :, 0]
            batch_img_tr1 = batch_img[:, :, :, :, 1]
            batch_img_tr2 = batch_img[:, :, :, :, 2]
            batch_img_tr3 = batch_img[:, :, :, :, 3]
            batch_label_sr = batch_mask[:, :, :, :, 0]
            batch_label_tr1 = batch_mask[:, :, :, :, 1]
            batch_label_tr2 = batch_mask[:, :, :, :, 2]
            batch_label_tr3 = batch_mask[:, :, :, :, 3]
            if patch_dis:
                batch_dom_sr = batch_dom[:, :, :, :, 0]
                batch_dom_tr1 = batch_dom[:, :, :, :, 1]
                batch_dom_tr2 = batch_dom[:, :, :, :, 2]
                batch_dom_tr3 = batch_dom[:, :, :, :, 3]
            else:
                batch_dom_sr = batch_dom[:, :, 0]
                batch_dom_tr1 = batch_dom[:, :, 1]
                batch_dom_tr2 = batch_dom[:, :, 2]
                batch_dom_tr3 = batch_dom[:, :, 3]

            if batch_img_sr.shape[0] != batch_size or batch_img_tr1.shape[0] != batch_size or batch_img_tr2.shape[0] != batch_size or batch_img_tr3.shape[0] != batch_size:
                continue

            feed_dict = {img_sr: batch_img_sr, img_tr1: batch_img_tr1, img_tr2: batch_img_tr2, img_tr3: batch_img_tr3,
                         domain_label_sr: batch_dom_sr, domain_label_tr1: batch_dom_tr1, domain_label_tr2: batch_dom_tr2,
                         domain_label_tr3: batch_dom_tr3, class_label_sr: batch_label_sr}

            sess.run([sh_optim], feed_dict=feed_dict)
            sess.run([ex_optim], feed_dict=feed_dict)
            sess.run([cl_optim], feed_dict=feed_dict)
            sess.run([dec_optim], feed_dict=feed_dict)
            sess.run([dis_optim], feed_dict=feed_dict)

            with sess.as_default():
                enc_sh_loss_ = loss_sh.eval(feed_dict)
                enc_sh_loss.append(enc_sh_loss_)
                enc_ex_loss_ = loss_ex.eval(feed_dict)
                enc_ex_loss.append(enc_ex_loss_)
                cl_loss_ = loss_cl.eval(feed_dict)
                cl_loss.append(cl_loss_)
                dec_loss_ = loss_dec.eval(feed_dict)
                dec_loss.append(dec_loss_)
                dis_loss_ = loss_dis.eval(feed_dict)
                dis_loss.append(dis_loss_)
                loss_ex_dis_ = loss_feat_ex_dom_dis.eval(feed_dict)
                ex_dis_loss.append(loss_ex_dis_)
                loss_sh_dis_ = loss_feat_sh_dom_dis.eval(feed_dict)
                sh_dis_loss.append(loss_sh_dis_)

            if idx % num_of_trn_batches == 0:
                feats_sh_sr_ = encoder_sh.predict(batch_img_sr)
                feats_sh_tr1_ = encoder_sh.predict(batch_img_tr1)
                feats_sh_tr2_ = encoder_sh.predict(batch_img_tr2)
                feats_sh_tr3_ = encoder_sh.predict(batch_img_tr3)

                feats_ex_sr_ = encoder_ex.predict(batch_img_sr)
                feats_ex_tr1_ = encoder_ex.predict(batch_img_tr1)
                feats_ex_tr2_ = encoder_ex.predict(batch_img_tr2)
                feats_ex_tr3_ = encoder_ex.predict(batch_img_tr3)

                rec_sr_ = decoder_sh.predict([feats_sh_sr_, feats_ex_sr_])
                rec_tr1_ = decoder_sh.predict([feats_sh_tr1_, feats_ex_tr1_])
                rec_tr2_ = decoder_sh.predict([feats_sh_tr2_, feats_ex_tr2_])
                rec_tr3_ = decoder_sh.predict([feats_sh_tr3_, feats_ex_tr3_])

                _, dom_sh_sr_ = domain_discriminator.predict(feats_sh_sr_)
                _, dom_ex_sr_ = domain_discriminator.predict(feats_ex_sr_)

                _, dom_sh_tr1_ = domain_discriminator.predict(feats_sh_tr1_)
                _, dom_ex_tr1_ = domain_discriminator.predict(feats_ex_tr1_)

                _, dom_sh_tr2_ = domain_discriminator.predict(feats_sh_tr2_)
                _, dom_ex_tr2_ = domain_discriminator.predict(feats_ex_tr2_)

                _, dom_sh_tr3_ = domain_discriminator.predict(feats_sh_tr3_)
                _, dom_ex_tr3_ = domain_discriminator.predict(feats_ex_tr3_)

                _, cl_sr_ = classifier.predict(feats_sh_sr_)
                _, cl_tr1_ = classifier.predict(feats_sh_tr1_)
                _, cl_tr2_ = classifier.predict(feats_sh_tr2_)
                _, cl_tr3_ = classifier.predict(feats_sh_tr3_)

                true_sr = batch_label_sr[index_plt].argmax(axis=-1)
                pred_sr = cl_sr_[index_plt].argmax(axis=-1)
                true_tr1 = batch_label_tr1[index_plt].argmax(axis=-1)
                pred_tr1 = cl_tr1_[index_plt].argmax(axis=-1)
                true_tr2 = batch_label_tr2[index_plt].argmax(axis=-1)
                pred_tr2 = cl_tr2_[index_plt].argmax(axis=-1)
                true_tr3 = batch_label_tr3[index_plt].argmax(axis=-1)
                pred_tr3 = cl_tr3_[index_plt].argmax(axis=-1)

                acc_sr = accuracy_score(true_sr.flatten(), pred_sr.flatten())
                acc_tr1 = accuracy_score(true_tr1.flatten(), pred_tr1.flatten())
                acc_tr2 = accuracy_score(true_tr2.flatten(), pred_tr2.flatten())
                acc_tr3 = accuracy_score(true_tr3.flatten(), pred_tr3.flatten())

                print('prediction source domain sh/ex: ', dom_sh_sr_[index_plt], dom_ex_sr_[index_plt])
                print('prediction target1 domain sh/ex: ', dom_sh_tr1_[index_plt], dom_ex_tr1_[index_plt])
                print('prediction target2 domain sh/ex: ', dom_sh_tr2_[index_plt], dom_ex_tr2_[index_plt])
                print('prediction target3 domain sh/ex: ', dom_sh_tr3_[index_plt], dom_ex_tr3_[index_plt])
                print('prediction acc sr {:.2f}'.format(acc_sr))
                print('prediction acc tr1 {:.2f}'.format(acc_tr1))
                print('prediction acc tr2 {:.2f}'.format(acc_tr2))
                print('prediction acc tr3 {:.2f}'.format(acc_tr3))

        # Evaluating model on validation,
        val_dec_loss, val_dis_loss, val_sh_loss, val_ex_loss, val_ex_dis_loss, val_sh_dis_loss, val_cl_loss = [], [], [], [], [], [], []
        for _ in range(0, num_of_val_batches):
            # batch_t0, batch_t1 = next(valid_gen)
            batch_img, batch_mask, batch_dom = next(valid_gen)
            batch_img_sr = batch_img[:, :, :, :, 0]
            batch_img_tr1 = batch_img[:, :, :, :, 1]
            batch_label_sr = batch_mask[:, :, :, :, 0]
            batch_label_tr1 = batch_mask[:, :, :, :, 1]
            batch_label_tr2 = batch_mask[:, :, :, :, 2]
            batch_label_tr3 = batch_mask[:, :, :, :, 3]
            if patch_dis:
                batch_dom_sr = batch_dom[:, :, :, :, 0]
                batch_dom_tr1 = batch_dom[:, :, :, :, 1]
                batch_dom_tr2 = batch_dom[:, :, :, :, 2]
                batch_dom_tr3 = batch_dom[:, :, :, :, 3]
            else:
                batch_dom_sr = batch_dom[:, :, 0]
                batch_dom_tr1 = batch_dom[:, :, 1]
                batch_dom_tr2 = batch_dom[:, :, 2]
                batch_dom_tr3 = batch_dom[:, :, 3]

            if batch_img_sr.shape[0] != batch_size or batch_img_tr1.shape[0] != batch_size or batch_img_tr2.shape[0] != batch_size or batch_img_tr3.shape[0] != batch_size:
                continue

            feed_dict = {img_sr: batch_img_sr, img_tr1: batch_img_tr1, img_tr2: batch_img_tr2, img_tr3: batch_img_tr3,
                         domain_label_sr: batch_dom_sr, domain_label_tr1: batch_dom_tr1, domain_label_tr2: batch_dom_tr2,
                         domain_label_tr3: batch_dom_tr3, class_label_sr: batch_label_sr}

            with sess.as_default():
                v_sh_loss = loss_sh.eval(feed_dict)
                val_sh_loss.append(v_sh_loss)
                v_ex_loss = loss_ex.eval(feed_dict)
                val_ex_loss.append(v_ex_loss)
                v_cl_loss = loss_cl.eval(feed_dict)
                val_cl_loss.append(v_cl_loss)
                v_dec_loss = loss_dec.eval(feed_dict)
                val_dec_loss.append(v_dec_loss)
                v_dis_loss = loss_dis.eval(feed_dict)
                val_dis_loss.append(v_dis_loss)
                v_ex_dis_loss = loss_feat_ex_dom_dis.eval(feed_dict)
                val_ex_dis_loss.append(v_ex_dis_loss)
                v_sh_dis_loss = loss_feat_sh_dom_dis.eval(feed_dict)
                val_sh_dis_loss.append(v_sh_dis_loss)

        if best_val_loss > np.mean(val_cl_loss):
            patience = 10
            best_val_loss = np.mean(val_cl_loss)
            print('Saving best model and checkpoints')
            save_model(encoder_sh, path_model_tm + '/' + 'enc_sh.h5')
            save_model(encoder_ex, path_model_tm + '/' + 'enc_ex.h5')
            save_model(decoder_sh, path_model_tm + '/' + 'dec_sh.h5')
            save_model(classifier, path_model_tm + '/' + 'classifer.h5')
            save_model(domain_discriminator, path_model_tm + '/' + 'dis.h5')
            print('Ok')
        else:
            patience -= 1
        if patience < 0:
            print('[***] end training ...')
            break
        #print('[***] end training ...')
        elapsed_time = time.time() - start_time
        print('loss dec tr {:.2f}'.format(np.mean(dec_loss)), 'loss dec val {:.2f}'.format(np.mean(val_dec_loss)))
        print('loss dis tr {:.2f}'.format(np.mean(dis_loss)), 'loss dis val {:.2f}'.format(np.mean(val_dis_loss)))
        print('loss sh tr {:.2f}'.format(np.mean(enc_sh_loss)), 'loss sh val {:.2f}'.format(np.mean(val_sh_loss)))
        print('loss ex tr {:.2f}'.format(np.mean(enc_ex_loss)), 'loss ex val {:.2f}'.format(np.mean(val_ex_loss)))
        print('loss cl tr {:.2f}'.format(np.mean(cl_loss)), 'loss cl val {:.2f}'.format(np.mean(val_cl_loss)))
        print('loss sh dis tr {:.2f}'.format(np.mean(sh_dis_loss)),
              'loss sh dis val {:.2f}'.format(np.mean(val_sh_dis_loss)))
        print('loss ex dis tr {:.2f}'.format(np.mean(ex_dis_loss)),
              'loss ex dis val {:.2f}'.format(np.mean(val_ex_dis_loss)))
        # save training loss
        tr_dec.append(np.mean(dec_loss))
        tr_dis.append(np.mean(dis_loss))
        tr_esh.append(np.mean(enc_sh_loss))
        tr_eex.append(np.mean(enc_ex_loss))
        tr_cls.append(np.mean(cl_loss))
        tr_dsh.append(np.mean(sh_dis_loss))
        tr_dex.append(np.mean(ex_dis_loss))
        # save validation loss
        vl_dec.append(np.mean(val_dec_loss))
        vl_dis.append(np.mean(val_dis_loss))
        vl_esh.append(np.mean(val_sh_loss))
        vl_eex.append(np.mean(val_ex_loss))
        vl_cls.append(np.mean(val_cl_loss))
        vl_dsh.append(np.mean(val_sh_dis_loss))
        vl_dex.append(np.mean(val_ex_dis_loss))
        np.savetxt(path_loss_tm + '/training_loss.txt',
                   ["Decoder: %s" % np.asarray(tr_dec), "Discriminador: %s" % np.asarray(tr_dis),
                    "EncSH: %s" % np.asarray(tr_esh), "EncEX: %s" % np.asarray(tr_eex),
                    "Classifier: %s" % np.asarray(tr_cls),
                    "DisSH: %s" % np.asarray(tr_dsh), "DisEX: %s" % np.asarray(tr_dex), ], fmt='%s', delimiter='\n')
        np.savetxt(path_loss_tm + '/validation_loss.txt',
                   ["Decoder: %s" % np.asarray(vl_dec), "Discriminador: %s" % np.asarray(vl_dis),
                    "EncSH: %s" % np.asarray(vl_esh), "EncEX: %s" % np.asarray(vl_eex),
                    "Classifier: %s" % np.asarray(vl_cls),
                    "DisSH: %s" % np.asarray(vl_dsh), "DisEX: %s" % np.asarray(vl_dex), ], fmt='%s', delimiter='\n')
    tr_time.append(elapsed_time)

tr_time_ = np.asarray(tr_time)
np.save(path_exp + '/tr_times.npy', tr_time_)