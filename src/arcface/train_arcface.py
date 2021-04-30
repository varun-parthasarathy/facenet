import os
import cv2
import pathlib
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import *
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from dataset_utils import generate_training_dataset, get_test_dataset, get_LFW_dataset
from triplet_callbacks_and_metrics import RangeTestCallback
from arcface_utils import create_neural_network, SoftmaxLoss


def get_learning_rate_schedule(schedule_name, image_count, batch_size, learning_rate=1e-3, max_lr=0.5,
                               step_size=6000):
    lr = None
    if schedule_name == 'triangular2':
        lr = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=learning_rate,
                                                           maximal_learning_rate=max_lr,
                                                           step_size=step_size,
                                                           scale_mode='iterations')
    elif schedule_name == 'triangular':
        lr = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=learning_rate,
                                                           maximal_learning_rate=max_lr,
                                                           step_size=step_size,
                                                           scale_mode='iterations')
    elif schedule_name == 'exponential_decay':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=step_size,
                                                            decay_rate=0.96,
                                                            staircase=False)
    elif schedule_name == 'staircase':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=step_size,
                                                            decay_rate=0.96,
                                                            staircase=True)
    elif schedule_name == 'constant':
        lr = learning_rate
    else:
        pass

    assert lr is not None, '[ERROR] The learning rate schedule is not specified correctly'

    return lr

def get_optimizer(optimizer_name, lr_schedule, weight_decay=1e-6):
    assert lr_schedule is not None, '[ERROR] Learning rate schedule is required'
    opt = None
    if optimizer_name == 'RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule,
                                          momentum=0.9,
                                          centered=True)
    elif optimizer_name == 'SGDW':
        opt = tfa.optimizers.SGDW(learning_rate=lr_schedule,
                                  weight_decay=weight_decay,
                                  momentum=0.9,
                                  nesterov=True)
    elif optimizer_name == 'ADAM':
        opt = tfa.optimizers.AdamW(learning_rate=lr_schedule,
                                   weight_decay=weight_decay,
                                   amsgrad=True)                    # Needs to be tested further
    elif optimizer_name == 'ADAGRAD':
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    elif optimizer_name == 'ADADELTA':
        opt = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
    elif optimizer_name == 'LOOKAHEAD_ADAM':
        base_opt = tfa.optimizers.AdamW(learning_rate=lr_schedule,
                                        weight_decay=weight_decay,
                                        amsgrad=True)
        opt = tfa.optimizers.Lookahead(optimizer=base_opt,
                                       sync_period=8,
                                       slow_step_size=0.5)
    elif optimizer_name == 'LOOKAHEAD_SGD':
        base_opt = tfa.optimizers.SGDW(learning_rate=lr_schedule,
                                       weight_decay=weight_decay,
                                       momentum=0.9,
                                       nesterov=True)
        opt = tfa.optimizers.Lookahead(optimizer=base_opt,
                                       sync_period=8,
                                       slow_step_size=0.5)
    elif optimizer_name == 'RANGER':
        min_lr = None
        if isinstance(lr_schedule, float):
            min_lr = max(lr_schedule/100., 1e-4)
        else:
            min_lr = 1e-4
        base_opt = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule,
                                                weight_decay=weight_decay,
                                                total_steps=5000,
                                                warmup_proportion=0.1,
                                                min_lr=min_lr,
                                                amsgrad=False)
        opt = tfa.optimizers.Lookahead(optimizer=base_opt,
                                       sync_period=8,
                                       slow_step_size=0.5)
    else:
        pass

    assert opt is not None, '[ERROR] The optimizer is not specified correctly'

    return opt

def train_model(data_path, batch_size, image_size, crop_size, lr_schedule_name, init_lr, max_lr, weight_decay, 
                optimizer, model_type, embedding_size, num_epochs, checkpoint_path, margin=0.5, cache_path=None,
                range_test=False, use_tpu=False, tpu_name=None,
                use_mixed_precision=False, distributed=False,
                eager_execution=False, weights_path='', checkpoint_interval=5000,
                step_size=6000, recompile=False, steps_per_epoch=None, logist_scale=64):

    if use_tpu is True:
        assert tpu_name is not None, '[ERROR] TPU name must be specified'
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        print("[INFO] TPUs: ", tf.config.list_logical_devices('TPU'))

    if use_mixed_precision is True:
        if use_tpu is True:
            policy = mixed_precision.Policy('mixed_bfloat16')
        else:
            policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("[INFO] Using mixed precision for training. This will reduce memory consumption\n")

    if distributed is True and use_tpu is False:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("[INFO] Using distributed training strategy on GPU")

    train_dataset, n_imgs, n_classes = generate_training_dataset(data_path=data_path, 
                                                                 image_size=image_size, 
                                                                 batch_size=batch_size, 
                                                                 crop_size=crop_size, 
                                                                 cache=cache_path,
                                                                 use_mixed_precision=use_mixed_precision,
                                                                 use_tpu=use_tpu,
                                                                 model_type=model_type)

    test_dataset = None

    run_eagerly = eager_execution if eager_execution is not None else False
    
    log_dir = './logs/log_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100,
                                                          write_graph=False)
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()

    metrics = [tf.keras.metrics.Accuracy, tf.keras.metrics.AUC]

    loss_fn = SoftmaxLoss()

    if range_test is True:
        range_finder = RangeTestCallback(start_lr=init_lr,
                                         end_lr=max_lr,
                                         n_imgs=n_imgs,
                                         batch_size=batch_size)
        opt = get_optimizer(optimizer_name=optimizer,
                            lr_schedule=1e-5,
                            weight_decay=weight_decay)
        if use_tpu is True:
            with strategy.scope():
                model, compiled = create_neural_network(model_type=model_type,
                                                        embedding_size=embedding_size,
                                                        weights_path=weights_path,
                                                        n_classes=n_classes,
                                                        recompile=recompile,
                                                        input_shape=[crop_size, crop_size, 3],
                                                        training=True,
                                                        margin=margin,
                                                        logist_scale=logist_scale)
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=metrics,
                                  run_eagerly=run_eagerly)
        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model, compiled = create_neural_network(model_type=model_type,
                                                        embedding_size=embedding_size,
                                                        weights_path=weights_path,
                                                        n_classes=n_classes,
                                                        recompile=recompile,
                                                        input_shape=[crop_size, crop_size, 3],
                                                        training=True,
                                                        margin=margin,
                                                        logist_scale=logist_scale)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=1e-5,
                                    weight_decay=weight_decay) # Optimizer must be created within scope!
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=metrics,
                                  run_eagerly=run_eagerly)
        else:
            model, compiled = create_neural_network(model_type=model_type,
                                                    embedding_size=embedding_size,
                                                    weights_path=weights_path,
                                                    n_classes=n_classes,
                                                    recompile=recompile,
                                                    input_shape=[crop_size, crop_size, 3],
                                                    training=True,
                                                    margin=margin,
                                                    logist_scale=logist_scale)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
            if compiled is False:
                print('[INFO] Recompiling model using passed optimizer and loss arguments')
                model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=metrics,
                              run_eagerly=run_eagerly)

        callback_list = [range_finder, tensorboard_callback, stop_on_nan]

        train_history = model.fit(train_dataset, epochs=num_epochs, 
                                  callbacks=callback_list)

        print('\n[INFO] Training complete. Range test results can be found at "./range_test_result.png"')

        return
    else:
        lr_schedule = get_learning_rate_schedule(schedule_name=lr_schedule_name,
                                                 learning_rate=init_lr,
                                                 max_lr=max_lr,
                                                 image_count=n_imgs,
                                                 batch_size=batch_size,
                                                 step_size=step_size)
        opt = get_optimizer(optimizer_name=optimizer,
                            lr_schedule=lr_schedule,
                            weight_decay=weight_decay)

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        #checkpoint_name = checkpoint_path + '/' + 'cp-{epoch:03d}.ckpt'
        model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'full_model'),
                                                         save_weights_only=False,
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_best_only=False,
                                                         save_freq=checkpoint_interval)
        weights_saver = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'model_weights'),
                                                           save_weights_only=True,
                                                           monitor='val_loss',
                                                           mode='min',
                                                           save_best_only=False,
                                                           save_freq=checkpoint_interval)
        if use_tpu is True:
            with strategy.scope():
                model, compiled = create_neural_network(model_type=model_type,
                                                        embedding_size=embedding_size,
                                                        weights_path=weights_path,
                                                        n_classes=n_classes,
                                                        recompile=recompile,
                                                        input_shape=[crop_size, crop_size, 3],
                                                        training=True,
                                                        margin=margin,
                                                        logist_scale=logist_scale)
                assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=metrics,
                                  run_eagerly=run_eagerly)
        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model, compiled = create_neural_network(model_type=model_type,
                                                        embedding_size=embedding_size,
                                                        weights_path=weights_path,
                                                        n_classes=n_classes,
                                                        recompile=recompile,
                                                        input_shape=[crop_size, crop_size, 3],
                                                        training=True,
                                                        margin=margin,
                                                        logist_scale=logist_scale)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=lr_schedule,
                                    weight_decay=weight_decay) # Optimizer must be created within scope!
                assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=metrics,
                                  run_eagerly=run_eagerly)
        else:
            model, compiled = create_neural_network(model_type=model_type,
                                                    embedding_size=embedding_size,
                                                    weights_path=weights_path,
                                                    n_classes=n_classes,
                                                    recompile=recompile,
                                                    input_shape=[crop_size, crop_size, 3],
                                                    training=True,
                                                    margin=margin,
                                                    logist_scale=logist_scale)
            assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
            if compiled is False:
                print('[INFO] Recompiling model using passed optimizer and loss arguments')
                model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=metrics,
                              run_eagerly=run_eagerly)

        callback_list = [model_saver, weights_saver, tensorboard_callback, stop_on_nan]

        train_history = model.fit(train_dataset, 
                                  epochs=num_epochs, 
                                  callbacks=callback_list, 
                                  steps_per_epoch=None if steps_per_epoch == 0 else steps_per_epoch)
        
        if not os.path.exists('./results'):
            os.mkdir('./results')

        model_name = './results/model-' + datetime.now().strftime("%Y%m%d-%H%M%S")
        model.save(model_name)
        print('\n[INFO] Training complete. Saved model can be found in "./results"')

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', required=True,
                        help='Path to training dataset. It should be a folder of folders')
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=128,
                        help='Batch size to use for training')
    parser.add_argument('-s', '--image_size', required=False, type=int, default=246,
                        help='Image size (before random crop and preprocessing)')
    parser.add_argument('-c', '--crop_size', required=False, type=int, default=224,
                        help='Image size after random crop is applied')
    parser.add_argument('--lr_schedule', required=False, type=str, default='triangular2',
                        choices=['triangular2', 'triangular', 'staircase', 'exponential_decay', 'constant'],
                        help='Choice of learning rate schedule. Default is a cyclic policy')
    parser.add_argument('--init_lr', required=False, type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', required=False, type=float, default=0.5,
                        help='Maximum learning rate. Should be set when using cyclic LR policies only')
    parser.add_argument('--weight_decay', required=False, type=float, default=0.000001,
                        help='Weight decay coefficient for regularization. Default value is 1e-6')
    parser.add_argument('--optimizer', required=False, default='RMSPROP',
                        choices=['RMSPROP', 'SGDW', 'ADAM', 'ADAGRAD', 'ADADELTA', 'LOOKAHEAD_SGD', 
                                 'LOOKAHEAD_ADAM', 'RANGER'],
                        help='Optimizer to use for training. Default is RMSprop')
    parser.add_argument('--model', required=False, type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152', 'inception_v3', 'efficientnet_b0',
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
                                 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'inception_resnet_v2',
                                 'xception', 'mobilenet', 'mobilenet_v2'],
                        help='NN architecture to use. Default is ResNet-50')
    parser.add_argument('--embedding_size', required=False, type=int, default=512,
                        help='Embedding size for triplet loss')
    parser.add_argument('--cache_path', required=False, type=str, default='./face_cache.tfcache',
                        help='Path to cache file to use for the training dataset')
    parser.add_argument('--epochs', required=False, type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--margin', required=False, type=float, default=0.2,
                        help='ArcFace margin to use')
    parser.add_argument('--checkpoint_path', required=False, type=str, default='./checkpoints',
                        help='Path to folder in which checkpoints are to be saved')
    parser.add_argument('--range_test', action='store_true',
                        help='Whether to run a range test or not. Default is no')
    parser.add_argument('--use_tpu', action='store_true',
                        help='Whether to use a TPU for training. Default is no')
    parser.add_argument('--tpu_name', required=False, type=str,
                        help='If using a TPU, specify the TPU name')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision for training. Can greatly reduce memory consumption')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training strategy for multiple GPUs. Does not work with TPU')
    parser.add_argument('--eager_execution', action='store_true',
                        help='Enable eager execution explicitly. May be needed for validation datasets')
    parser.add_argument('--weights_path', type=str, default='', required=False,
                        help='Path to saved weights/checkpoints (if using saved weights for further training)')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, required=False,
                        help='Frequency of model checkpointing. Default is every 5000 steps')
    parser.add_argument('--step_size', type=int, default=6000, required=False,
                        help='Step size for cyclic learning rate policies')
    parser.add_argument('--recompile', action='store_true',
                        help='Recompile model. Recommended for constant learning rate')
    parser.add_argument('--steps_per_epoch', type=int, default=0, required=False,
                        help='Number of steps before an epoch is completed. Default is 0')
    parser.add_argument('--logist_scale', type=int, default=64, required=False,
                        help='Logit scale for ArcFace')

    args = vars(parser.parse_args())

    train_model(data_path=args['data_path'],
                batch_size=args['batch_size'],
                image_size=args['image_size'],
                crop_size=args['crop_size'],
                lr_schedule_name=args['lr_schedule'],
                init_lr=args['init_lr'],
                max_lr=args['max_lr'],
                weight_decay=args['weight_decay'],
                optimizer=args['optimizer'],
                model_type=args['model'],
                embedding_size=args['embedding_size'],
                cache_path=args['cache_path'],
                num_epochs=args['epochs'],
                margin=args['margin'],
                checkpoint_path=args['checkpoint_path'],
                range_test=args['range_test'],
                use_tpu=args['use_tpu'],
                tpu_name=args['tpu_name'],
                use_mixed_precision=args['use_mixed_precision'],
                distributed=args['distributed'],
                eager_execution=args['eager_execution'],
                weights_path=args['weights_path'],
                checkpoint_interval=args['checkpoint_interval'],
                step_size=args['step_size'],
                recompile=args['recompile'],
                steps_per_epoch=args['steps_per_epoch'],
                logist_scale=args['logist_scale'])
