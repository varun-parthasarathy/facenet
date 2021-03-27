"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted.
"""
import os
import sys
import math
import argparse
import numpy as np
from scipy import misc
import tensorflow as tf
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.model_selection import KFold
from adaptive_triplet_loss import AdaptiveTripletLoss
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss, AssortedTripletLoss

def _get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    classes = set()
    for pair in pairs:
        if len(pair) == 3:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
            classes.add(pair[0])
        elif len(pair) == 4:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
            classes.add(pair[0])
            classes.add(pair[2])
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            #path_list += (path0,path1)
            path_list.append(path0)
            path_list.append(path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('[INFO] Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list, list(classes)
  
def _add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def _read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def _get_preprocessor(model_type):
    if model_type[0:6] == 'resnet':
        preprocessor = 'tensorflow.keras.applications.resnet'
    elif model_type[0:13] == 'efficientnet':
        preprocessor = 'tensorflow.keras.applications.efficientnet'
    elif model_type == 'xception':
        preprocessor = 'tensorflow.keras.applications.xception'
    elif model_type == 'inception_v3':
        preprocessor = 'tensorflow.keras.applications.inception_v3'
    elif model_type == 'inception_resnet_v2':
        preprocessor = 'tensorflow.keras.applications.inception_resnet_v2'
    elif model_type == 'mobilenet':
        preprocessor = 'tensorflow.keras.applications.mobilenet'
    elif model_type == 'mobilenet_v2':
        preprocessor = 'tensorflow.keras.applications.mobilenet_v2'
    else:
        preprocessor = None

    if preprocessor is not None:
        preprocessor = importlib.import_module(preprocessor)

    return preprocessor

def get_LFW_dataset(data_path, image_size, batch_size, crop_size, train_classes=0,
                    use_mixed_precision=False, use_tpu=False, model_type=None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    assert batch_size % 2 == 0, '[ERROR] Batch size must be a multiple of 2'
    pairs = _read_pairs('./data/pairs.txt')
    path_list, actual_issame, CLASS_NAMES = _get_paths(data_path, pairs)
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(path_list)
    assert image_count % batch_size == 0, '[ERROR] Batch size must perfectly divide number of images'
    preprocessor = _get_preprocessor(model_type)

    ds = tf.data.Dataset.from_tensor_slices(path_list)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16)
            else:
                img = tf.cast(img, tf.float16)
        else:
            img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [crop_size, crop_size])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            img = img / 255.
        #img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        #img = tf.image.random_flip_left_right(img)
        return img, label

    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES), actual_issame

def _distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sqrt(np.sum(np.square(diff), axis=1))
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def _calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, 
                   distance_metric=0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        dist = _distance(embeddings1, embeddings2, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = _calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = _calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = _calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy, np.std(accuracy)

def _calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def _calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        dist = _distance(embeddings1, embeddings2, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = _calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = _calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def _calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def main(weights_path, lfw_path, image_size, crop_size, model_type, loss_type,
         batch_size=30, use_mixed_precision=False, use_tpu=False, embedding_size=512):
    model = None
    if loss_type == 'ADAPTIVE':
        loss_obj = ['AdaptiveTripletLoss', AdaptiveTripletLoss]
    elif loss_type == 'FOCAL':
        loss_obj = ['TripletFocalLoss', TripletFocalLoss]
    elif loss_type == 'BATCH_HARD':
        loss_obj = ['TripletBatchHardLoss', TripletBatchHardLoss]
    elif loss_type == 'BATCH_HARD_V2':
        loss_obj = ['TripletBatchHardV2Loss', TripletBatchHardV2Loss]
    elif loss_type == 'ASSORTED':
        loss_obj = ['AssortedTripletLoss', AssortedTripletLoss]
    else:
        loss_obj = None
    if loss_obj is not None:
        model = tf.keras.models.load_model(weights_path, custom_objects={loss_obj[0]:loss_obj[1]})
    else:
        model = tf.keras.models.load_model(weights_path)

    lfw_ds, nrof_images, _, actual_issame = get_LFW_dataset(data_path=lfw_path, 
                                                            image_size=image_size, 
                                                            batch_size=batch_size,
                                                            crop_size=crop_size,
                                                            use_mixed_precision=use_mixed_precision,
                                                            use_tpu=use_tpu,
                                                            train_classes=0,
                                                            model_type=model_type)

    actual_issame = np.asarray(actual_issame)
    embeddings = np.zeros((nrof_images, embedding_size))
    labels = np.zeros((nrof_images,))
    start_idx = 0
    print('[INFO] There are %d images to process in the dataset' % (nrof_images))

    for i, (xs, ys) in enumerate(lfw_ds):
        print('Processing batch : %d of %d' % (i+1, int(nrof_images / batch_size)), flush=True)
        embs = model.predict(xs)
        end_idx = start_idx + np.squeeze(embs).shape[0]
        labels[start_idx:end_idx] = np.squeeze(ys)
        embeddings[start_idx:end_idx] = np.squeeze(embs)
        start_idx = end_idx

    result_string = 'Accuracy : {}%+-{}% :: Validation rate : {}%+-{}% @FAR : {} :: AUC : {} :: EER : {}'
    thresholds = np.arange(0, 2, 0.01)
    y_pred = embeddings
    embeddings1 = y_pred[0::2]
    embeddings2 = y_pred[1::2]
    y_true = labels
    issame = np.equal(y_true[0::2], y_true[1::2])

    if np.all(actual_issame == issame) is False:
        count = 0
        for i in range(len(actual_issame)):
            if actual_issame[i] != issame[i]:
                count += 1
                print('%d : ' % (i+1), end='')
                print(actual_issame[i] + ' ' + issame[i])
        print('[ERROR] %d mismatches in labels!' % (count))
        raise AssertionError('[ERROR] There was a problem in creating the dataset. Please check the labels')

    tpr, fpr, accuracy, acc_std = _calculate_roc(thresholds, embeddings1, embeddings2,
                                                 actual_issame, nrof_folds=10, distance_metric=0)
    thresholds = np.arange(0, 2, 0.001)
    val, val_std, far = _calculate_val(thresholds, embeddings1, embeddings2,
                                       actual_issame, 1e-3, nrof_folds=10, distance_metric=0)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    acc = np.mean(accuracy)

    result_string.format(acc, acc_std, val, val_std, far, auc, eer)
    print(result_string)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('-l', '--lfw_path', type=str, required=True,
                        help='Path to LFW dataset')
    parser.add_argument('-s', '--image_size', required=True, type=int, default=280,
                        help='Image size (before random crop and preprocessing)')
    parser.add_argument('-c', '--crop_size', required=True, type=int, default=260,
                        help='Image size after random crop is applied')
    parser.add_argument('--model', required=True, type=str, default='efficientnet_b2',
                        choices=['resnet50', 'resnet101', 'resnet152', 'inception_v3', 'efficientnet_b0',
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
                                 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'inception_resnet_v2',
                                 'xception', 'mobilenet', 'mobilenet_v2'],
                        help='NN architecture to use. Default is EfficientNet-B2')
    parser.add_argument('--loss_type', type=str, default='FOCAL',
                        choices=['VANILLA', 'BATCH_HARD', 'BATCH_HARD_V2', 'FOCAL', 'ADAPTIVE', 'ASSORTED'],
                        help='Choice of triplet loss formulation. Default is FOCAL')

    args = vars(parser.parse_args())
    main(weights_path=args['weights_path'],
         lfw_path=args['lfw_path'],
         image_size=args['image_size'],
         crop_size=args['crop_size'],
         model_type=args['model'],
         loss_type=args['loss_type'])
