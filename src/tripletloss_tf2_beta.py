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
from adaptive_triplet_loss import AdaptiveTripletLoss
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import mixed_precision
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss, AssortedTripletLoss, ConstellationLoss
from custom_triplet_loss import HAP2S_ELoss, HAP2S_PLoss
from dataset_utils import generate_training_dataset, get_test_dataset, get_LFW_dataset, generate_training_dataset_v2
from triplet_callbacks_and_metrics import RangeTestCallback, DecayMarginCallback, TripletLossMetrics, ToggleMetricEval
from model_utils import create_neural_network_v2
from tensorflow_similarity.losses import MultiSimilarityLoss
from cyclic_learning_rate import CyclicLR


def get_learning_rate_schedule(schedule_name, image_count, batch_size, learning_rate=1e-3, max_lr=0.25,
                               step_size=30000):
    lr = None
    if schedule_name == 'triangular2':
        # lr = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=learning_rate,
        #                                                     maximal_learning_rate=max_lr,
        #                                                     step_size=step_size)
        lr = CyclicLR(base_lr=learning_rate,
                      max_lr=max_lr,
                      step_size=step_size,
                      mode='triangular2')
        print('[WARNING] Due to bugs in tensorflow_addons, cyclic LR is currently a callback, not an LR policy. Use it accordingly')
    elif schedule_name == 'triangular':
        # lr = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=learning_rate,
        #                                                    maximal_learning_rate=max_lr,
        #                                                    step_size=step_size)
        lr = CyclicLR(base_lr=learning_rate,
                      max_lr=max_lr,
                      step_size=step_size,
                      mode='triangular')
        print('[WARNING] Due to bugs in tensorflow_addons, cyclic LR is currently a callback, not an LR policy. Use it accordingly')
    elif schedule_name == 'exponential_decay':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=step_size,
                                                            decay_rate=0.90,
                                                            staircase=False)
    elif schedule_name == 'staircase':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=step_size,
                                                            decay_rate=0.90,
                                                            staircase=True)
    elif schedule_name == 'constant':
        lr = learning_rate
    elif schedule_name == 'cosine_restart':
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=max_lr,
                                                               first_decay_steps=step_size,
                                                               t_mul=1.0, # Can be 2.0 as well, but 1.0 works just fine
                                                               m_mul=0.80, # Decay the starting lr for each restart
                                                               alpha=learning_rate)
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
                optimizer, model_type, embedding_size, num_epochs, checkpoint_path, margin=0.35, cache_path=None,
                range_test=False, use_tpu=False, tpu_name=None, test_path='',
                use_mixed_precision=False, triplet_strategy='', images_per_person=35, 
                people_per_sample=12, distance_metric="L2", soft=True, 
                sigma=0.3, decay_margin_rate=0.0, use_lfw=True, target_margin=0.2, distributed=False,
                eager_execution=False, weights_path='', checkpoint_interval=5000, use_metrics=False,
                step_size=6000, recompile=False, steps_per_epoch=None, equisample=False, loss_to_load='',
                use_imagenet=False, sam_type='null'):

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
        mixed_precision.set_global_policy(policy)
        print("[INFO] Using mixed precision for training. This will reduce memory consumption\n")

    if distributed is True and use_tpu is False:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("[INFO] Using distributed training strategy on GPU")

    if use_imagenet is None:
        use_imagenet = False

    train_dataset, n_imgs, n_classes = generate_training_dataset_v2(data_path=data_path, 
                                                                    image_size=image_size, 
                                                                    batch_size=batch_size, 
                                                                    crop_size=crop_size, 
                                                                    cache=cache_path,
                                                                    use_mixed_precision=use_mixed_precision,
                                                                    images_per_person=images_per_person,
                                                                    people_per_sample=people_per_sample,
                                                                    use_tpu=use_tpu,
                                                                    model_type=model_type,
                                                                    equisample=equisample)

    if test_path is not None and len(test_path) > 1:
        if use_lfw is True:
            test_dataset, test_images, _ = get_LFW_dataset(data_path=test_path, 
                                                           image_size=image_size, 
                                                           batch_size=batch_size,
                                                           crop_size=crop_size,
                                                           cache='./lfw_dataset_cache.tfcache',
                                                           use_mixed_precision=use_mixed_precision,
                                                           use_tpu=use_tpu,
                                                           train_classes=n_classes,
                                                           model_type=model_type)
        else:
            test_dataset, test_images, _ = get_test_dataset(data_path=test_path, 
                                                            image_size=image_size, 
                                                            batch_size=30,
                                                            crop_size=crop_size,
                                                            cache='./test_dataset_cache.tfcache',
                                                            use_mixed_precision=use_mixed_precision,
                                                            use_tpu=use_tpu,
                                                            train_classes=n_classes,
                                                            model_type=model_type)
    else:
        test_dataset = None

    run_eagerly = eager_execution if eager_execution is not None else False
    if triplet_strategy == 'VANILLA':
        loss_fn = tfa.losses.TripletSemiHardLoss(margin=margin)
        print('[INFO] Using vanilla triplet loss')
    elif triplet_strategy == 'BATCH_HARD':
        loss_fn = TripletBatchHardLoss(margin=margin,
                                       soft=soft,
                                       distance_metric=distance_metric)
        print('[INFO] Using batch-hard strategy.')
    elif triplet_strategy == 'BATCH_HARD_V2':
        loss_fn = TripletBatchHardV2Loss(margin1=(-1.0*margin),
                                         margin2=(margin1/100.0),
                                         beta=0.002,
                                         distance_metric=distance_metric)
        print('[INFO] Using batch-hard V2 strategy')
    elif triplet_strategy == 'ADAPTIVE':
        loss_fn = AdaptiveTripletLoss(margin=margin,
                                      soft=soft,
                                      lambda_=sigma)
        run_eagerly = True
        print('[INFO] Using Adaptive Triplet Loss')
    elif triplet_strategy == 'ASSORTED':
        loss_fn = AssortedTripletLoss(margin=margin,
                                      focal=soft,
                                      sigma=sigma,
                                      distance_metric=distance_metric)
        print('[INFO] Using assorted triplet loss')
    elif triplet_strategy == 'CONSTELLATION':
        loss_fn = ConstellationLoss(k=int(margin) if margin > 1 else 4,
                                    batch_size=batch_size)
    elif triplet_strategy == 'MULTISIMILARITY':
        if distance_metric == 'L2':
            dist = 'euclidean'
        elif distance_metric == 'angular':
            dist = 'cosine'
        else:
            dist = 'squared_euclidean'
        loss_fn = MultiSimilarityLoss(distance=dist,
                                      alpha=1.0,
                                      beta=20,
                                      epsilon=margin,
                                      lmda=sigma if sigma < 1 else 0.5)
    elif triplet_strategy == 'HAP2S_E':
        loss_fn = HAP2S_ELoss(margin=margin,
                              soft=soft,
                              sigma=sigma,
                              distance_metric=distance_metric)
    elif triplet_strategy == 'HAP2S_P':
        loss_fn = HAP2S_PLoss(margin=margin,
                              soft=soft,
                              alpha=sigma,
                              distance_metric=distance_metric)
    else:
        loss_fn = TripletFocalLoss(margin=margin,
                                   sigma=sigma,
                                   soft=soft,
                                   distance_metric=distance_metric)
        print('[INFO] Using triplet focal loss.')

    if decay_margin_rate > 0 and triplet_strategy != 'BATCH_HARD_V2':
        decay_margin_callback = DecayMarginCallback(loss_fn, margin, 
                                                    decay_margin_rate, target_margin)
        print('[INFO] Using decayed margin to reduce intra-class variability (experimental)')
    else:
        decay_margin_callback = None

    log_dir = './logs/log_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100,
                                                          write_graph=False)
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()

    if test_dataset is not None:
        triplet_loss_metrics = TripletLossMetrics(test_images, embedding_size)
    else:
        triplet_loss_metrics = None
    toggle_metrics = ToggleMetricEval()

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
                model, compiled = create_neural_network_v2(model_type=model_type,
                                                           embedding_size=embedding_size,
                                                           weights_path=weights_path,
                                                           loss_type=loss_to_load,
                                                           loss_fn=loss_fn,
                                                           recompile=recompile,
                                                           input_shape=[crop_size, crop_size, 3],
                                                           use_imagenet=use_imagenet,
                                                           sam_type=sam_type)
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=[triplet_loss_metrics] if use_metrics is True else None,
                                  run_eagerly=run_eagerly)
        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model, compiled = create_neural_network_v2(model_type=model_type,
                                                           embedding_size=embedding_size,
                                                           weights_path=weights_path,
                                                           loss_type=loss_to_load,
                                                           loss_fn=loss_fn,
                                                           recompile=recompile,
                                                           input_shape=[crop_size, crop_size, 3],
                                                           use_imagenet=use_imagenet,
                                                           sam_type=sam_type)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=1e-5,
                                    weight_decay=weight_decay) # Optimizer must be created within scope!
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=[triplet_loss_metrics] if use_metrics is True else None,
                                  run_eagerly=run_eagerly)
        else:
            model, compiled = create_neural_network_v2(model_type=model_type,
                                                       embedding_size=embedding_size,
                                                       weights_path=weights_path,
                                                       loss_type=loss_to_load,
                                                       loss_fn=loss_fn,
                                                       recompile=recompile,
                                                       input_shape=[crop_size, crop_size, 3],
                                                       use_imagenet=use_imagenet,
                                                       sam_type=sam_type)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
            if compiled is False:
                print('[INFO] Recompiling model using passed optimizer and loss arguments')
                model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=[triplet_loss_metrics] if use_metrics is True else None,
                              run_eagerly=run_eagerly)

        callback_list = [range_finder, tensorboard_callback, stop_on_nan]
        if use_metrics is True:
            callback_list.append(toggle_metrics)
        if decay_margin_callback is not None:
            callback_list.append(decay_margin_callback)

        train_history = model.fit(train_dataset, epochs=num_epochs, 
                                  callbacks=callback_list,
                                  validation_data=test_dataset)

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
                            lr_schedule=lr_schedule if lr_schedule_name not in ['triangular2', 'triangular'] else init_lr,
                            weight_decay=weight_decay)

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        #checkpoint_name = checkpoint_path + '/' + 'cp-{epoch:03d}.ckpt'
        if checkpoint_interval == 0:
            checkpoint_interval = 'epoch'
        checkpoint_saver = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                              save_weights_only=False,
                                                              monitor='loss',
                                                              mode='min',
                                                              save_best_only=False,
                                                              save_freq=checkpoint_interval)
        if use_tpu is True:
            with strategy.scope():
                model, compiled = create_neural_network_v2(model_type=model_type,
                                                           embedding_size=embedding_size,
                                                           weights_path=weights_path,
                                                           loss_type=loss_to_load,
                                                           loss_fn=loss_fn,
                                                           recompile=recompile,
                                                           input_shape=[crop_size, crop_size, 3],
                                                           use_imagenet=use_imagenet,
                                                           sam_type=sam_type)
                assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=[triplet_loss_metrics] if use_metrics is True else None,
                                  run_eagerly=run_eagerly)
        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model, compiled = create_neural_network_v2(model_type=model_type,
                                                           embedding_size=embedding_size,
                                                           weights_path=weights_path,
                                                           loss_type=loss_to_load,
                                                           loss_fn=loss_fn,
                                                           recompile=recompile,
                                                           input_shape=[crop_size, crop_size, 3],
                                                           use_imagenet=use_imagenet,
                                                           sam_type=sam_type)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=lr_schedule if lr_schedule_name not in ['triangular2', 'triangular'] else init_lr,
                                    weight_decay=weight_decay) # Optimizer must be created within scope!
                assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
                if compiled is False:
                    print('[INFO] Recompiling model using passed optimizer and loss arguments')
                    model.compile(optimizer=opt,
                                  loss=loss_fn,
                                  metrics=[triplet_loss_metrics] if use_metrics is True else None,
                                  run_eagerly=run_eagerly)
        else:
            model, compiled = create_neural_network_v2(model_type=model_type,
                                                       embedding_size=embedding_size,
                                                       weights_path=weights_path,
                                                       loss_type=loss_to_load,
                                                       loss_fn=loss_fn,
                                                       recompile=recompile,
                                                       input_shape=[crop_size, crop_size, 3],
                                                       use_imagenet=use_imagenet,
                                                       sam_type=sam_type)
            assert model is not None, '[ERROR] There was a problem in loading the pre-trained weights'
            if compiled is False:
                print('[INFO] Recompiling model using passed optimizer and loss arguments')
                model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=[triplet_loss_metrics] if use_metrics is True else None,
                              run_eagerly=run_eagerly)

        callback_list = [checkpoint_saver, tensorboard_callback, stop_on_nan]
        if use_metrics is True:
            callback_list.append(toggle_metrics)
        if decay_margin_callback is not None:
            callback_list.append(decay_margin_callback)
        if lr_schedule_name in ['triangular2', 'triangular']:
            callback_list.append(lr_schedule)
            print('[INFO] Successfully added cyclic LR callback to model training')

        train_history = model.fit(train_dataset, 
                                  epochs=num_epochs, 
                                  callbacks=callback_list, 
                                  validation_data=test_dataset,
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
                        choices=['triangular2', 'triangular', 'staircase', 'exponential_decay', 'constant', 'cosine_restart'],
                        help='Choice of learning rate schedule. Default is a cyclic policy')
    parser.add_argument('--init_lr', required=False, type=float, default=0.001,
                        help='Initial learning rate. For cosine restarts, specifies lowest value of learning rate')
    parser.add_argument('--max_lr', required=False, type=float, default=0.25,
                        help='Maximum learning rate. Should be set when using cyclic LR or cosine restart policies only')
    parser.add_argument('--weight_decay', required=False, type=float, default=0.000001,
                        help='Weight decay coefficient for regularization. Default value is 1e-6')
    parser.add_argument('--optimizer', required=False, default='RMSPROP',
                        choices=['RMSPROP', 'SGDW', 'ADAM', 'ADAGRAD', 'ADADELTA', 'LOOKAHEAD_SGD', 
                                 'LOOKAHEAD_ADAM', 'RANGER'],
                        help='Optimizer to use for training. Default is RMSprop')
    parser.add_argument('--model', required=False, type=str, default='inception_v3',
                        choices=['resnet50', 'resnet101', 'resnet152', 'inception_v3', 'efficientnet_b0',
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
                                 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'inception_resnet_v2',
                                 'xception', 'mobilenet', 'mobilenet_v2', 'efficientnetv2-s', 'efficientnetv2-m',
                                 'efficientnetv2-l', 'efficientnetv2-xl', 'efficientnetv2-b0', 'efficientnetv2-b1',
                                 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-t'],
                        help='NN architecture to use. Default is InceptionV3')
    parser.add_argument('--embedding_size', required=False, type=int, default=512,
                        help='Embedding size for triplet loss')
    parser.add_argument('--cache_path', required=False, type=str, default='./face_cache.tfcache',
                        help='Path to cache file to use for the training dataset')
    parser.add_argument('--epochs', required=False, type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--margin', required=False, type=float, default=0.2,
                        help='Margin to use for triplet loss. Specifies k for ConstellationLoss if margin > 1, but must be an int in this case')
    parser.add_argument('--checkpoint_path', required=False, type=str, default='./checkpoints',
                        help='Path to folder in which checkpoints are to be saved')
    parser.add_argument('--range_test', action='store_true',
                        help='Whether to run a range test or not. Default is no')
    parser.add_argument('--use_tpu', action='store_true',
                        help='Whether to use a TPU for training. Default is no')
    parser.add_argument('--tpu_name', required=False, type=str,
                        help='If using a TPU, specify the TPU name')
    parser.add_argument('--test_path', required=False, type=str,
                        help='Path to test dataset, if you want to check validation loss. Optional but recommended')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision for training. Can greatly reduce memory consumption')
    parser.add_argument('--triplet_strategy', type=str, default='FOCAL',
                        choices=['VANILLA', 'BATCH_HARD', 'BATCH_HARD_V2', 'FOCAL', 'ADAPTIVE', 'ASSORTED', 
                                 'CONSTELLATION', 'MULTISIMILARITY', 'HAP2S_E', 'HAP2S_P'],
                        help='Choice of triplet loss formulation. Default is FOCAL')
    parser.add_argument('--images_per_person', required=False, type=int, default=35,
                        help='Average number of images per class. Default is 35 (from MS1M cleaned + AsianCeleb)')
    parser.add_argument('--people_per_sample', required=False, type=int, default=50,
                        help='Number of people per sample. Helps fill buffer for shuffling the dataset properly')
    parser.add_argument('--distance_metric', required=False, type=str, default='L2',
                        choices=['L2', 'squared-L2', 'angular'],
                        help='Choice of distance metric. Default is Euclidean distance')
    parser.add_argument('--soft', action='store_true',
                        help='Use soft margin. For ASSORTED strategy, sets whether to use triplet focal loss or not')
    parser.add_argument('--sigma', type=float, required=False, default=0.3,
                        help='Value of sigma for FOCAL strategy. For ADAPTIVE and MULTISIMILARITY strategies, specifies lambda')
    parser.add_argument('--decay_margin_rate', type=float, required=False, default=0.0,
                        help='Decay rate for margin. Recommended value to set is 0.9965')
    parser.add_argument('--use_lfw', action='store_true',
                        help='Specifies whether test dataset is the LFW dataset or not')
    parser.add_argument('--target_margin', type=float, default=0.2, required=False,
                        help='Minimum margin when using decayed margin')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training strategy for multiple GPUs. Does not work with TPU')
    parser.add_argument('--eager_execution', action='store_true',
                        help='Enable eager execution explicitly. May be needed for validation datasets')
    parser.add_argument('--weights_path', type=str, default='', required=False,
                        help='Path to saved weights/checkpoints (if using saved weights for further training)')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, required=False,
                        help='Frequency of model checkpointing. Default is every 5000 steps')
    parser.add_argument('--use_metrics', action='store_true',
                        help='Include triplet metric evaluation during training. Not recommended when checkpointing is mandatory as custom metrics cannot be restored properly')
    parser.add_argument('--step_size', type=int, default=6000, required=False,
                        help='Step size for cyclic learning rate policies')
    parser.add_argument('--recompile', action='store_true',
                        help='Recompile model. Recommended for constant learning rate')
    parser.add_argument('--steps_per_epoch', type=int, default=0, required=False,
                        help='Number of steps before an epoch is completed. Default is 0')
    parser.add_argument('--equisample', action='store_true',
                        help='Determines whether to sample images from each class equally to form a batch. Will have performance drawbacks if enabled')
    parser.add_argument('--loss_to_load', type=str, default='FOCAL',
                        choices=['VANILLA', 'BATCH_HARD', 'BATCH_HARD_V2', 'FOCAL', 'ADAPTIVE', 'ASSORTED', 'CONSTELLATION', 'MULTISIMILARITY', 'HAP2S_E', 'HAP2S_P'],
                        help='Choice of triplet loss object for loading models. Default is FOCAL')
    parser.add_argument('--use_imagenet', action='store_true',
                        help='Use pre-trained ImageNet weights')
    parser.add_argument('--sam_type', type=str, default='null', choices=['null', 'SAM', 'ESAM'],
                        help='Use Sharpness-Aware Minimization enabled models for training. Training will be slower as a result')

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
                test_path=args['test_path'],
                use_mixed_precision=args['use_mixed_precision'],
                triplet_strategy=args['triplet_strategy'],
                images_per_person=args['images_per_person'],
                people_per_sample=args['people_per_sample'],
                distance_metric=args['distance_metric'],
                soft=args['soft'],
                sigma=args['sigma'],
                decay_margin_rate=args['decay_margin_rate'],
                use_lfw=args['use_lfw'],
                target_margin=args['target_margin'],
                distributed=args['distributed'],
                eager_execution=args['eager_execution'],
                weights_path=args['weights_path'],
                checkpoint_interval=args['checkpoint_interval'],
                use_metrics=args['use_metrics'],
                step_size=args['step_size'],
                recompile=args['recompile'],
                steps_per_epoch=args['steps_per_epoch'],
                equisample=args['equisample'],
                loss_to_load=args['loss_to_load'],
                use_imagenet=args['use_imagenet'],
                sam_type=args['sam_type'])
