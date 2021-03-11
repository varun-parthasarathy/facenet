import sys
import math
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.model_selection import KFold


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
        accuracy = np.mean(accuracy)

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


class TripletLossMetrics(tf.keras.metrics.Metric):
    def __init__(self, nrof_images, embedding_size, name='TripletLossMetrics', **kwargs):
        super(TripletLossMetrics, self).__init__(name=name, **kwargs)
        self.labels = np.zeros((nrof_images,))
        self.embeddings = np.zeros((nrof_images, embedding_size))
        self.nrof_batches = 0
        self.start_idx = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        end_idx = self.start_idx + tf.shape(y_pred).numpy()[0]
        self.labels[self.start_idx:end_idx] = y_true.numpy()
        self.embeddings[self.start_idx:end_idx] = y_pred.numpy()
        self.start_idx = end_idx
        self.nrof_batches += 1

    def result(self):
        result_string = 'Accuracy : {}%+-{}% :: Validation rate : {}%+-{}% @FAR : {} :: AUC : {} :: EER : {}'
        thresholds = np.arange(0, 2, 0.01)
        y_pred = self.embeddings
        embeddings1 = y_pred[0::2]
        embeddings2 = y_pred[1::2]
        y_true = self.labels
        actual_issame = np.equal(y_true[0::2], y_true[1::2])
        tpr, fpr, accuracy, acc_std = _calculate_roc(thresholds, embeddings1, embeddings2,
                                                     actual_issame, nrof_folds=10, distance_metric=0)
        thresholds = np.arange(0, 2, 0.001)
        val, val_std, far = _calculate_val(thresholds, embeddings1, embeddings2,
                                           actual_issame, 1e-3, nrof_folds=10, distance_metric=0)
        auc = metrics.auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        result_string.format(acc, acc_std, val, val_std, far, auc, eer)
        #DEBUG
        print(result_string, flush=True)
        #DEBUG ENDS
        return tf.convert_to_tensor(result_string, dtype=str)

    def reset_states(self):
        self.labels = np.zeros((nrof_images,))
        self.embeddings = np.zeros((nrof_images, embedding_size))
        self.nrof_batches = 0
        self.start_idx = 0


class RangeTestCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_lr, end_lr, n_imgs, batch_size):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_imgs = n_imgs
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []
        tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)
        n_steps = self.params['steps'] if self.params['steps'] is not None else round(self.n_imgs / self.batch_size)
        n_steps *= self.params['epochs']
        self.by = (self.end_lr - self.start_lr) / n_steps

    def on_batch_end(self, batch, logs={}):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)
        self.losses.append(logs.get('loss'))
        lr += self.by
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class DecayMarginCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_fn, margin, decay_rate=0.9965, target_margin=0.2):
        super().__init__()
        self.margin = margin
        self.decay = decay_rate
        self.loss_fn = loss_fn
        self.target_margin = target_margin

    def on_epoch_end(epoch, logs={}):
        self.loss_fn.margin = max(self.margin * np.power(self.decay, epoch), self.target_margin)