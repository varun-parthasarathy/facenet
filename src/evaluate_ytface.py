"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted.
"""
import os
import sys
import math
import argparse
import importlib
import numpy as np
from scipy import misc
import tensorflow as tf
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.model_selection import KFold
from adaptive_triplet_loss import AdaptiveTripletLoss
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss, AssortedTripletLoss
from model_utils import create_neural_network_v2
import model_utils


def _read_pairs(pairs_filename, lfw_path):
    pairs = []
    actual_issame = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split(',')
            for i in range(len(pair)):
                pair[i] = pair[i].strip()

            if not os.path.exists(os.path.join(lfw_path, pair[2])):
                continue
            if not os.path.exists(os.path.join(lfw_path, pair[3])):
                continue
            if not os.path.isdir(os.path.join(lfw_path, pair[2])):
                continue
            if not os.path.isdir(os.path.join(lfw_path, pair[3])):
                continue
            if len(os.listdir(os.path.join(lfw_path, pair[2]))) == 0:
                continue
            if len(os.listdir(os.path.join(lfw_path, pair[3]))) == 0:
                continue

            if int(pair[-2]) == 1:
                actual_issame.append(True)
            else:
                actual_issame.append(False)
            pairs.append(pair)
    return np.array(pairs), np.array(actual_issame), len(pairs)

def _get_preprocessor(model_type):
    if 'inception_resnet_v2' in model_type:
        preprocessor = 'tensorflow.keras.applications.inception_resnet_v2'
        print('[INFO] Loaded Inception-Resnet-V2 data preprocessor', flush=True)
    elif 'efficientnet' in model_type:
        preprocessor = 'tensorflow.keras.applications.efficientnet'
        print('[INFO] Loaded EfficientNet data preprocessor', flush=True)
    elif 'xception' in model_type:
        preprocessor = 'tensorflow.keras.applications.xception'
        print('[INFO] Loaded Xception data preprocessor', flush=True)
    elif 'inception_v3' in model_type:
        preprocessor = 'tensorflow.keras.applications.inception_v3'
        print('[INFO] Loaded Inception-V3 data preprocessor', flush=True)
    elif 'resnet' in model_type:
        preprocessor = 'tensorflow.keras.applications.resnet'
        print('[INFO] Loaded Resnet data preprocessor', flush=True)
    elif 'mobilenet_v2' in model_type:
        preprocessor = 'tensorflow.keras.applications.mobilenet_v2'
        print('[INFO] Loaded MobileNet-V2 data preprocessor', flush=True)
    elif 'mobilenet' in model_type:
        preprocessor = 'tensorflow.keras.applications.mobilenet'
        print('[INFO] Loaded MobileNet data preprocessor', flush=True)
    else:
        preprocessor = None
        print('[WARNING] Could not find appropriate pre-processor for model', flush=True)

    if preprocessor is not None:
        preprocessor = importlib.import_module(preprocessor)

    return preprocessor

def get_dataset(data_path, image_size, batch_size, crop_size, train_classes=0,
                use_mixed_precision=False, use_tpu=False, model_type=None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_path = pathlib.Path(data_path)
    image_count = len(list(data_path.glob('*.png')))
    preprocessor = _get_preprocessor(model_type)

    ds = tf.data.Dataset.list_files(str(data_path/"*.png"), shuffle=True)

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img, channels=3)
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
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            img = img / 255.

        return img

    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count

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
         batch_size=50, use_mixed_precision=False, use_tpu=False, embedding_size=512,
         load_from_file=False):
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
        #Another solution is skip the model_utils import and use : custom_objects={loss_obj[0]:loss_obj[1], 'tf':tf}
    else:
        model = tf.keras.models.load_model(weights_path)

    pairs, actual_issame, nrof_pairs = _read_pairs('./data/ytface_pairs.txt', lfw_path)
    embeddings = np.zeros((nrof_pairs*2, embedding_size))

    if load_from_file is None or load_from_file is False:
        for pair_num, pair in enumerate(pairs):
            temp_emb = None
            x_ds, ic = get_dataset(data_path=os.path.join(lfw_path, pair[2]), 
                                   image_size=image_size,
                                   batch_size=batch_size,
                                   crop_size=crop_size,
                                   use_mixed_precision=use_mixed_precision,
                                   use_tpu=use_tpu,
                                   train_classes=0,
                                   model_type=model_type)
            for i, xs in enumerate(x_ds):
                embs = model.predict(xs)
                embs = np.squeeze(embs)
                if temp_emb is None:
                    temp_emb = embs
                else:
                    temp_emb = np.vstack((temp_emb, embs))

            assert temp_emb.shape == (ic, embedding_size)
            
            mean_emb = np.squeeze(np.mean(temp_emb, axis=1))
            #assert mean_emb.shape[1] == embedding_size
            embeddings[2*pair_num] = mean_emb

            temp_emb = None
            x_ds, ic = get_dataset(data_path=os.path.join(lfw_path, pair[3]), 
                                   image_size=image_size,
                                   batch_size=batch_size,
                                   crop_size=crop_size,
                                   use_mixed_precision=use_mixed_precision,
                                   use_tpu=use_tpu,
                                   train_classes=0,
                                   model_type=model_type)
            for i, xs in enumerate(x_ds):
                embs = model.predict(xs)
                embs = np.squeeze(embs)
                if temp_emb is None:
                    temp_emb = embs
                else:
                    temp_emb = np.vstack((temp_emb, embs))

            assert temp_emb.shape == (ic, embedding_size)
            
            mean_emb = np.squeeze(np.mean(temp_emb, axis=1))
            #assert mean_emb.shape[1] == embedding_size
            embeddings[(2*pair_num) + 1] = mean_emb

        np.save('./ytface_embeddings.npy', embeddings)
    else:
        embeddings = np.load('./embeddings.npy')

    result_string = 'Accuracy : {}%+-{}% :: Validation rate : {}%+-{}% @FAR : {} :: AUC : {} :: EER : {}'
    thresholds = np.arange(0, 2, 0.01)
    y_pred = embeddings
    embeddings1 = y_pred[0::2]
    embeddings2 = y_pred[1::2]

    tpr, fpr, accuracy, acc_std = _calculate_roc(thresholds, embeddings1, embeddings2,
                                                 actual_issame, nrof_folds=10, distance_metric=0)
    thresholds = np.arange(0, 2, 0.001)
    val, val_std, far = _calculate_val(thresholds, embeddings1, embeddings2,
                                       actual_issame, 1e-3, nrof_folds=10, distance_metric=0)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, fill_value='extrapolate')(x), 0., 1.)
    acc = np.mean(accuracy)

    result_string = result_string.format(acc*100, acc_std*100, val*100, val_std*100, far, auc, eer)
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
    parser.add_argument('--load_from_file', action='store_true',
                        help='Load embeddings and labels from file instead of running inference again')

    args = vars(parser.parse_args())
    main(weights_path=args['weights_path'],
         lfw_path=args['lfw_path'],
         image_size=args['image_size'],
         crop_size=args['crop_size'],
         model_type=args['model'],
         loss_type=args['loss_type'],
         load_from_file=args['load_from_file'])
