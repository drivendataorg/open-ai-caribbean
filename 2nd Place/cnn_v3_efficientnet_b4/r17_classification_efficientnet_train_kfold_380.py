# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


from a00_common_functions import *
from albumentations import *
from multiprocessing.pool import ThreadPool
from functools import partial
from efficientnet.keras import EfficientNetB4


KFOLD_NUMBER = 5
FOLD_LIST = [0, 1, 2, 3, 4]
SHAPE_SIZE = (380, 380, 3)
MODELS_PATH_KERAS = MODELS_PATH + 'classification_efficientnet_b4_380/'
if not os.path.isdir(MODELS_PATH_KERAS):
    os.mkdir(MODELS_PATH_KERAS)
IMG_CACHE = None


def strong_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        RandomRotate90(p=1.0),
        # RGBShift(p=0.5, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
        # IAAAffine(scale=1.0, translate_percent=(-0.1, 0.1), translate_px=None, rotate=0.0, shear=(-10, 10), order=1, cval=0, mode='reflect', always_apply=False, p=0.01),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            MedianBlur(p=1.0, blur_limit=7),
            Blur(p=1.0, blur_limit=7),
            GaussianBlur(p=1.0, blur_limit=7),
        ], p=0.2),
        OneOf([
            IAAAdditiveGaussianNoise(p=1.0),
            GaussNoise(p=1.0),
        ], p=0.2),
        OneOf([
            ElasticTransform(p=1.0, alpha=1.0, sigma=30, alpha_affine=20),
            GridDistortion(p=1.0, num_steps=5, distort_limit=0.3),
            OpticalDistortion(p=1.0, distort_limit=0.5, shift_limit=0.5)
        ], p=0.2)
    ],  p=p)


def simple_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=1.0),
        RGBShift(p=0.1, r_shift_limit=(-5, 5), g_shift_limit=(-5, 5), b_shift_limit=(-5, 5)),
    ],  p=p)


def preproc_input_classifcation(img):
    from keras.applications.inception_resnet_v2 import preprocess_input
    return preprocess_input(img)


def augment_image(orig_img):
    delta = 100
    img = orig_img[:, :, :3]
    mask = orig_img[:, :, 3]
    res = strong_aug()(image=img, mask=mask)
    img, mask = res['image'], res['mask']
    mask = np.expand_dims(mask, axis=2)
    img = np.concatenate((img, mask), axis=2)

    # Get random part of image
    start_0 = random.randint(0, delta)
    end_0 = random.randint(img.shape[0]-delta, img.shape[0])
    start_1 = random.randint(0, delta)
    end_1 = random.randint(img.shape[1] - delta, img.shape[1])

    img = img[start_0:end_0, start_1:end_1, :]
    return img


def process_single_item(img_name, type):
    global IMG_CACHE

    img = IMG_CACHE[img_name + '.png'].copy()
    if type == 'train':
        img = augment_image(img)
    else:
        # We take only needed part
        img = img[50:-50, 50:-50]
    if (img.shape[0] != SHAPE_SIZE[0]) or (img.shape[1] != SHAPE_SIZE[1]):
        img = cv2.resize(img, (SHAPE_SIZE[1], SHAPE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    # show_image(img[:, :, :3])
    # show_image(img[:, :, 1:4])
    return img


def batch_generator_train(data, answ, batch_size, type):
    from keras.utils import to_categorical
    global IMG_CACHE

    threads = 4
    p = ThreadPool(threads)
    process_item_func = partial(process_single_item, type=type)

    start_point = 0
    index = np.arange(0, len(data))
    np.random.shuffle(index)

    while True:
        batch_answ = []

        if start_point + batch_size > len(data):
            np.random.shuffle(index)
            start_point = 0

        ind = index[start_point:start_point+batch_size]
        batch_data = p.map(process_item_func, data[ind])

        for i in range(batch_size):
            current_index = ind[i]
            current_answ = to_categorical(CLASSES.index(answ[current_index]), num_classes=NUM_CLASSES)
            batch_answ.append(current_answ)
        start_point += batch_size

        batch_data = np.array(batch_data, dtype=np.float32)
        batch_data = preproc_input_classifcation(batch_data)
        batch_answ = np.array(batch_answ, dtype=np.float32)

        yield batch_data, batch_answ


def freeze_model(model, freeze):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if freeze is True:
            layer.trainable = False
        else:
            layer.trainable = True
        if layer.name in ['prediction', 'classification']:
            layer.trainable = True
    return model


def model_EfficientNetB4(size, upd_ch):
    from keras.models import Model, load_model
    from keras.layers import Dense, Dropout, Input

    ref_ch = 3
    required_layer_name = 'stem_conv'

    weights_cache_path = CACHE_PATH + 'model_EfficientNetB4_{}ch_imagenet_{}.h5'.format(upd_ch, size)
    if not os.path.isfile(weights_cache_path):
        model_ref = EfficientNetB4(include_top=False,
                          weights='imagenet',
                          input_shape=(size, size, ref_ch),
                          pooling='avg',)
        model_upd = EfficientNetB4(include_top=False,
                                   weights=None,
                                   input_shape=(size, size, upd_ch),
                                   pooling='avg', )

        for i, layer in enumerate(model_ref.layers):
            print('Update weights layer [{}]: {}'.format(i, layer.name))
            if layer.name == required_layer_name:
                print('Recalc weights!')
                config = layer.get_config()
                use_bias = config['use_bias']
                if use_bias:
                    w, b = layer.get_weights()
                else:
                    w = layer.get_weights()[0]
                print('Use bias?: {}'.format(use_bias))
                print('Shape ref: {}'.format(w.shape))
                shape_upd = (w.shape[0], w.shape[1], upd_ch, w.shape[3])
                print('Shape upd: {}'.format(shape_upd))

                w_new = np.zeros(shape_upd, dtype=np.float32)
                for j in range(upd_ch):
                    w_new[:, :, j, :] = ref_ch * w[:, :, j%ref_ch, :] / upd_ch
                if use_bias:
                    model_upd.layers[i].set_weights((w_new, b))
                else:
                    model_upd.layers[i].set_weights((w_new,))
                continue
            else:
                model_upd.layers[i].set_weights(layer.get_weights())
        model_upd.save(weights_cache_path)
    else:
        model_upd = load_model(weights_cache_path)

    x = model_upd.layers[-1].output
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='prediction')(x)
    model = Model(inputs=model_upd.inputs, outputs=x)
    # print(model.summary())
    return model


def set_lr(model, lr):
    import keras.backend as K
    K.set_value(model.optimizer.lr, float(lr))


def create_keras_model(fold_number):
    print('Start fold: {}'.format(fold_number))

    global IMG_CACHE
    from keras import __version__
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from keras.optimizers import Adam, SGD, RMSprop
    print('Keras version: {}'.format(__version__))
    start_time = time.time()

    train_ids_all, train_answ_all, valid_ids_all, valid_answ_all = get_kfold_split(5, seed=20)
    train_ids = train_ids_all[fold_number]
    train_answ = train_answ_all[fold_number]
    valid_ids = valid_ids_all[fold_number]
    valid_answ = valid_answ_all[fold_number]

    print(len(train_ids), len(train_answ), len(valid_ids), len(valid_answ))
    # Exclude unverified from validation
    if 1:
        train_data = pd.read_csv(INPUT_PATH + 'train_labels.csv')
        bad_ids = train_data[train_data['verified'] == False]['id'].values
        if 1:
            condition = ~np.isin(train_ids, bad_ids)
            train_ids = train_ids[condition]
            train_answ = train_answ[condition]

        condition = ~np.isin(valid_ids, bad_ids)
        valid_ids = valid_ids[condition]
        valid_answ = valid_answ[condition]
        print('Exclude bad IDs from validation...')
        print(len(train_ids), len(train_answ), len(valid_ids), len(valid_answ))

    IMG_CACHE = preread_train_images()
    model_name = 'effnetB4_fold_{}'.format(fold_number)
    optim_name = 'Adam'
    batch_size_train = 8
    batch_size_valid = 8
    learning_rate = 0.00002
    epochs = 400
    patience = 6
    print('Batch size: {}'.format(batch_size_train))
    print('Learning rate: {}'.format(learning_rate))

    steps_per_epoch = len(train_ids) // batch_size_train
    validation_steps = len(valid_ids) // batch_size_valid
    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    final_model_path = MODELS_PATH_KERAS + '{}.h5'.format(model_name)
    cache_model_path = MODELS_PATH_KERAS + '{}_temp.h5'.format(model_name)
    cache_model_path_score = MODELS_PATH_KERAS + '{}'.format(model_name) + '_loss_{val_loss:.4f}_acc_{val_acc:.4f}_ep_{epoch:02d}.h5'

    model = model_EfficientNetB4(380, 4)
    # model = freeze_model(model, freeze=True)
    if 0:
        weights_path = cache_model_path
        print('Load weights: {}'.format(weights_path))
        model.load_weights(weights_path)
    # print(model.summary())
    print(get_model_memory_usage(batch_size_train, model))

    if optim_name == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, save_best_only=False, save_weights_only=True),
        ModelCheckpoint(cache_model_path_score, save_best_only=False, save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-9, epsilon=0.00001, verbose=1),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_{}_optim_{}_fold_{}.csv'.format(model_name, optim_name, fold_number), append=True),
    ]

    # set_lr(model, 1e-4)
    history = model.fit_generator(generator=batch_generator_train(train_ids, train_answ, batch_size_train, type='train'),
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator_train(valid_ids, valid_answ, batch_size_valid, type='valid'),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=10,
                                  initial_epoch=0,
                                  # class_weight=class_weight,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    max_acc = max(history.history['val_acc'])
    print('Minimum loss for given fold: ', min_loss)
    print('Maximum acc for given fold: ', max_acc)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    del model
    K.clear_session()
    tf.reset_default_graph()


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'Keras'

    for kf in range(KFOLD_NUMBER):
        if kf not in FOLD_LIST:
            continue
        create_keras_model(kf)
        copy_best_model(MODELS_PATH_KERAS, kf, clean=True)


'''
Train val
Ep 16: 875s - loss: 0.4532 - acc: 0.8253 - val_loss: 0.4374 - val_acc: 0.8360
Ep 13: 891s - loss: 0.4834 - acc: 0.8122 - val_loss: 0.4261 - val_acc: 0.8347
Ep 13: 941s - loss: 0.4868 - acc: 0.8097 - val_loss: 0.4533 - val_acc: 0.8295
Ep 16: 886s - loss: 0.4444 - acc: 0.8256 - val_loss: 0.4584 - val_acc: 0.8275
Ep 14: 954s - loss: 0.4696 - acc: 0.8150 - val_loss: 0.4682 - val_acc: 0.8209
'''