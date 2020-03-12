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


from cnn_v1_densenet121.r16_classification_d121_train_kfold_224 import *
from sklearn.metrics import accuracy_score, log_loss
import keras.backend as K


def get_mirror_image_by_index(image, index):
    """    
    :param image: 
    :param index: Maximum 8  
    :return: 
    """
    if index < 4:
        image = np.rot90(image, k=index)
    else:
        if len(image.shape) == 3:
            image = image[::-1, :, :]
        else:
            image = image[::-1, :]
        image = np.rot90(image, k=index-4)
    return image


def get_TTA_image_classification(img_orig):
    img_batch = []
    delta_list = [100, 75, 50, 25]
    # delta_list = [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
    for d in delta_list:
        img = img_orig[d:-d, d:-d, :].copy()
        if img.shape[0] < 10 or img.shape[1] < 10:
            print('Image shape is small: {} {}'.format(d, img.shape))
            continue
        if (img.shape[0] != SHAPE_SIZE[0]) or (img.shape[1] != SHAPE_SIZE[1]):
            img = cv2.resize(img, (SHAPE_SIZE[1], SHAPE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        for i in range(8):
            im = get_mirror_image_by_index(img, i)
            img_batch.append(im)

    img_batch = np.array(img_batch, dtype=np.float32)
    img_batch = preproc_input_classifcation(img_batch)
    return img_batch


def get_validation_score(models_paths):
    global IMG_CACHE
    from keras import __version__
    print('Keras version: {}'.format(__version__))

    train = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    verified_ids = train[train['verified'] == True]['id'].values

    train_ids_all, train_answ_all, valid_ids_all, valid_answ_all = get_kfold_split(5)
    IMG_CACHE = preread_train_images()

    all_frames = []
    for fold_number, model_path in enumerate(models_paths):
        fold_time = time.time()
        train_ids = train_ids_all[fold_number]
        train_answ = train_answ_all[fold_number]
        valid_ids = valid_ids_all[fold_number]
        valid_answ = valid_answ_all[fold_number]
        verified_cond = np.isin(valid_ids, verified_ids)
        valid_ids_verified = valid_ids[verified_cond]
        print('Fold: {} Load weights: {}'.format(fold_number+1, model_path))
        model = model_DenseNet121_multich(224, 4)
        model.load_weights(model_path)
        print('Valid full: {} Valid vrfd: {}'.format(len(valid_ids), len(valid_ids_verified)))

        all_preds = []
        all_answ = []
        for i in range(len(valid_ids)):
            print('Go for {}: {}'.format(i, valid_ids[i]))
            img = IMG_CACHE[valid_ids[i] + '.png']
            img_batch = get_TTA_image_classification(img.copy())
            preds = model.predict(img_batch, batch_size=32)
            preds = np.mean(preds, axis=0)
            all_preds.append(preds)
            # print(valid_answ[i])
            all_answ.append(CLASSES.index(valid_answ[i]))

        all_preds = np.array(all_preds)
        all_answ = np.array(all_answ)

        print(all_preds.shape)
        print(all_answ.shape)
        print(all_preds)
        print(all_answ)

        acc = accuracy_score(all_answ, all_preds.argmax(axis=1))
        loss = log_loss(all_answ, all_preds)
        print('Fold full: {} Accuracy: {:.6f} Log loss: {:.6f}'.format(fold_number, acc, loss))

        acc = accuracy_score(all_answ[verified_cond], all_preds.argmax(axis=1)[verified_cond])
        loss = log_loss(all_answ[verified_cond], all_preds[verified_cond])
        print('Fold vrfd: {} Accuracy: {:.6f} Log loss: {:.6f}'.format(fold_number, acc, loss))

        d = pd.DataFrame(valid_ids, columns=['id'])
        d['real_answ'] = all_answ
        preds_columns = []
        for i, c in enumerate(CLASSES):
            d['{}'.format(c)] = all_preds[:, i]
            preds_columns.append('{}'.format(c))
        all_frames.append(d)
        del model
        K.clear_session()

        print('Time: {:.2f} sec'.format(time.time() - fold_time))

    all_frames = pd.concat(all_frames, axis=0)
    subm_path = FEATURES_PATH + 'd121_kfold_valid_TTA_32_{}.csv'.format(len(models_paths))
    all_frames.to_csv(subm_path, index=False)

    print('Full validation stat')
    s = pd.read_csv(subm_path)
    answ = s['real_answ'].values
    pred_float = s[preds_columns].values
    loss = log_loss(answ, pred_float)
    print('Loss: {:.6f}'.format(loss))
    pred = pred_float.argmax(axis=1)
    acc = accuracy_score(answ, pred)
    print('Acc: {:.6f}'.format(acc))

    print('Only verified validation stat')
    s = pd.read_csv(subm_path)
    s = s[s['id'].isin(verified_ids)]

    answ = s['real_answ'].values
    pred_float = s[preds_columns].values
    loss = log_loss(answ, pred_float)
    print('Loss: {:.6f}'.format(loss))

    pred = pred_float.argmax(axis=1)
    acc = accuracy_score(answ, pred)
    print('Acc: {:.6f}'.format(acc))

    shutil.copy(subm_path, subm_path[:-4] + '_loss_{:.6f}_acc_{:.6f}.csv'.format(loss, acc))

    return pred_float


def proc_tst(models_paths):
    global IMG_CACHE
    from keras import __version__
    from keras.models import load_model
    print('Keras version: {}'.format(__version__))

    test_files = pd.read_csv(INPUT_PATH + 'submission_format.csv')['id'].values
    print('Files to proc: {}'.format(len(test_files)))

    IMG_CACHE = preread_test_images()
    overall_preds = []
    for fold_number, model_path in enumerate(models_paths):
        fold_time = time.time()
        print('Load merged model: {}'.format(model_path))
        model = model_DenseNet121_multich(224, 4)
        model.load_weights(model_path)
        all_preds = []
        for i in range(len(test_files)):
            id = test_files[i]
            print('Go for {}: {}'.format(i, id))
            img = IMG_CACHE[id + '.png']
            img_batch = get_TTA_image_classification(img.copy())
            preds = model.predict(img_batch, batch_size=32)
            preds = np.array(preds).mean(axis=0)
            all_preds.append(preds)

        all_preds = np.array(all_preds)
        overall_preds.append(all_preds)
        del model
        K.clear_session()
        print('Time: {:.2f} sec'.format(time.time() - fold_time))

    overall_preds = np.array(overall_preds).mean(axis=0)
    print(overall_preds.shape)

    d = pd.DataFrame(test_files, columns=['id'])
    for i, c in enumerate(CLASSES):
        d[c] = overall_preds[:, i]
    subm_path = FEATURES_PATH + 'd121_v2_test_TTA_32_{}.csv'.format(len(models_paths))
    d.to_csv(subm_path, index=False)

    shutil.copy2(subm_path, SUBM_PATH + 'd121_v2_test_TTA_32_{}.csv'.format(len(models_paths)))
    return overall_preds


if __name__ == '__main__':
    start_time = time.time()
    models_paths = get_best_models_for_cnn(MODELS_PATH_KERAS, KFOLD_NUMBER)
    print('Models to use: {}'.format(models_paths))
    get_validation_score(models_paths)
    proc_tst(models_paths)


'''
v1: TTA 4
Fold full: 0 Accuracy: 0.873950 Log loss: 0.328462
Fold vrfd: 0 Accuracy: 0.825008 Log loss: 0.445403
Time: 150.14 sec
Fold full: 1 Accuracy: 0.883577 Log loss: 0.326103
Fold vrfd: 1 Accuracy: 0.841946 Log loss: 0.428459
Time: 156.26 sec
Fold full: 2 Accuracy: 0.874501 Log loss: 0.332266
Fold vrfd: 2 Accuracy: 0.825151 Log loss: 0.438617
Time: 164.32 sec
Fold full: 3 Accuracy: 0.881412 Log loss: 0.334262
Fold vrfd: 3 Accuracy: 0.838275 Log loss: 0.443894
Time: 175.43 sec
Fold full: 4 Accuracy: 0.882196 Log loss: 0.326269
Fold vrfd: 4 Accuracy: 0.838111 Log loss: 0.437351
Time: 190.30 sec
Full validation stat
Loss: 0.329471 Acc: 0.879124
Only verified validation stat
Loss: 0.438743 Acc: 0.833692 LB: 0.4036

v2: TTA 32
Fold full: 0 Accuracy: 0.880805 Log loss: 0.320212
Fold vrfd: 0 Accuracy: 0.833054 Log loss: 0.434414
Time: 674.39 sec
Fold full: 1 Accuracy: 0.882027 Log loss: 0.323541
Fold vrfd: 1 Accuracy: 0.838591 Log loss: 0.425031
Time: 682.89 sec
Fold full: 2 Accuracy: 0.876497 Log loss: 0.325555
Fold vrfd: 2 Accuracy: 0.829523 Log loss: 0.427763
Time: 691.76 sec
Fold full: 3 Accuracy: 0.884077 Log loss: 0.330097
Fold vrfd: 3 Accuracy: 0.841307 Log loss: 0.434449
Time: 704.54 sec
Fold full: 4 Accuracy: 0.887086 Log loss: 0.320917
Fold vrfd: 4 Accuracy: 0.843170 Log loss: 0.430028
Time: 712.46 sec
Full validation stat
Loss: 0.324089 Acc: 0.882095
Only verified validation stat
Loss: 0.430336 Acc: 0.837122 LB: 0.4082

TTA 168:
Fold vrfd: 0 Accuracy: 0.833724 Log loss: 0.430979
Fold vrfd: 1 Accuracy: 0.841275 Log loss: 0.419799
Fold vrfd: 2 Accuracy: 0.832885 Log loss: 0.424797
Fold vrfd: 3 Accuracy: 0.839623 Log loss: 0.432236
Fold vrfd: 4 Accuracy: 0.844857 Log loss: 0.426756
Only verified validation stat
Loss: 0.426911 Acc: 0.838467 
'''
