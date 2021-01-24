import os
import re
import cv2
import pathlib
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cyclic_learning_rate as clr
from scipy.signal import savgol_filter
from sam_optimizer import SAMOptimizer
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import *
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss
from dataset_utils import generate_training_dataset, get_test_dataset, get_LFW_dataset
from triplet_callbacks_and_metrics import TripletLossMetrics


def create_neural_network(model_type='resnet50', embedding_size=512, input_shape=None, weights_path=''):
    base_model = None
    if model_type == 'resnet50':
        base_model = ResNet50(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'resnet101':
        base_model = ResNet101(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'resnet152':
        base_model = ResNet152(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'inception_v3':
        base_model = InceptionV3(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b0':
        base_model = EfficientNetB0(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b1':
        base_model = EfficientNetB1(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b2':
        base_model = EfficientNetB2(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b3':
        base_model = EfficientNetB3(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b4':
        base_model = EfficientNetB4(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b5':
        base_model = EfficientNetB5(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b6':
        base_model = EfficientNetB6(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'efficientnet_b7':
        base_model = EfficientNetB7(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'inception_resnet_v2':
        base_model = InceptionResNetV2(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'xception':
        base_model = Xception(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'mobilenet':
        base_model = MobileNet(weights=None, classes=embedding_size, classifier_activation=None)
    elif model_type == 'mobilenet_v2':
        base_model = MobileNetV2(weights=None, classes=embedding_size, classifier_activation=None)
    else:
        pass

    assert base_model is not None, '[ERROR] The model name was not correctly specified'

    logits = base_model.outputs
    embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), dtype='float32',
                                        name='embeddings')(logits)
    # embeddings = tf.keras.layers.Activation('linear', dtype='float32')(embeddings)
    model = Model(inputs=base_model.input, outputs=embeddings)

    if input_shape is not None:
        in_shape = np.array(model.input.shape)
        in_shape = tuple(in_shape[-3:])
        assert in_shape == input_shape, '[ERROR] The model input shape and the given input shape do not match'

    if len(weights_path) > 1 and os.path.exists(weights_path):
        print('[INFO] Attempting to load weights from most recently saved checkpoint')
        try:
            latest = tf.train.latest_checkpoint(weights_path)
            model.load_weights(latest)
        except:
            try:
                model = tf.keras.models.load_model(weights_path)
                print('[INFO] Loading model from SavedModel format')
            except:
                print('[ERROR] Weights did not match the model architecture specified, or path was incorrect')
                return None
    else:
        print('[WARNING] Could not load weights. Using random initialization instead')

    return model

def get_learning_rate_schedule(schedule_name, learning_rate=1e-3, max_lr=0.5, image_count, batch_size):
    lr = None
    if schedule_name == 'triangular2':
        lr = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=learning_rate,
                                                           maximal_learning_rate=max_lr,
                                                           step_size=6*(int(image_count/batch_size)+1),
                                                           scale_mode='iterations')
    elif schedule_name == 'triangular':
        lr = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=learning_rate,
                                                           maximal_learning_rate=max_lr,
                                                           step_size=6*(int(image_count/batch_size)+1),
                                                           scale_mode='iterations')
    elif schedule_name == 'exponential_decay':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=5*(int(image_count/batch_size)+1),
                                                            decay_rate=0.96,
                                                            staircase=False)
    elif schedule_name == 'staircase':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                            decay_steps=5*(int(image_count/batch_size)+1),
                                                            decay_rate=0.96,
                                                            staircase=True)
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
    else:
        pass

    assert opt is not None, '[ERROR] The optimizer is not specified correctly'

    opt = SAMOptimizer(rho=0.05,
                       base_optimizer=opt)

    return opt

def train_model(data_path, batch_size, image_size, crop_size, lr_schedule_name, init_lr, max_lr, weight_decay, 
                optimizer, model_type, embedding_size, cache_path=None, num_epochs, margin=0.35, 
                checkpoint_path, range_test=False, use_tpu=False, tpu_name=None, test_path='',
                use_mixed_precision=False, triplet_strategy='', images_per_person=35, 
                people_per_sample=12, pretrained_model='', distance_metric="L2", soft=True, 
                sigma=0.3, use_lfw=True, distributed=False):

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
                                                                 images_per_person=images_per_person,
                                                                 people_per_sample=people_per_sample,
                                                                 use_tpu=use_tpu)

    if len(test_path) > 1:
        if use_lfw is True:
            test_dataset, test_images, _ = get_LFW_dataset(data_path=test_path, 
                                                           image_size=image_size, 
                                                           batch_size=batch_size,
                                                           crop_size=crop_size,
                                                           cache='./lfw_dataset_cache.tfcache',
                                                           use_mixed_precision=use_mixed_precision,
                                                           use_tpu=use_tpu,
                                                           train_classes=n_classes)
        else:
            test_dataset, test_images, _ = get_test_dataset(data_path=test_path, 
                                                            image_size=image_size, 
                                                            batch_size=30,
                                                            crop_size=crop_size,
                                                            cache='./test_dataset_cache.tfcache',
                                                            use_mixed_precision=use_mixed_precision,
                                                            use_tpu=use_tpu,
                                                            train_classes=n_classes)
    else:
        test_dataset = None

    if triplet_strategy == 'VANILLA':
        loss_fn = tfa.losses.TripletSemiHardLoss(margin=margin,
                                                 reduction=tf.keras.losses.Reduction.NONE)
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
    else:
        loss_fn = TripletFocalLoss(margin=margin,
                                   sigma=sigma,
                                   soft=soft,
                                   distance_metric=distance_metric)
        print('[INFO] Using triplet focal loss.')

    triplet_loss_metrics = TripletLossMetrics(test_images, embedding_size)

#-------------------------------------------------------------------------------------------------------
    def train_step(x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        perturbations = opt.first_step(grads, model)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.second_step(grads, model, perturbations)

        return loss_value

    @tf.function
    def distributed_train_step(x_batch_train, y_batch_train):
        per_replica_loss = mirrored_strategy.run(train_step, args=(x_batch_train, y_batch_train, ))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                        axis=None)
#-------------------------------------------------------------------------------------------------------

    if range_test is True:
        range_finder = RangeTestCallback(start_lr=1e-5,
                                         end_lr=10)
        opt = get_optimizer(optimizer_name=optimizer,
                            lr_schedule=1e-5,
                            weight_decay=weight_decay)
        if use_tpu is True:
            with strategy.scope():
                model = create_neural_network(model_type=model_type,
                                              embedding_size=embedding_size)
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'

        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model = create_neural_network(model_type=model_type,
                                              embedding_size=embedding_size)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=1e-5,
                                    weight_decay=weight_decay)
            train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
            test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
        else:
            model = create_neural_network(model_type=model_type,
                                          embedding_size=embedding_size)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'

        callback_list = [range_finder]
        if decay_margin_callback is not None:
            callback_list.append(decay_margin_callback)

        lrs = []
        losses = []
        for epoch in range(5):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if distributed is True and use_tpu is False:
                    loss_value = distributed_train_step(x_batch_train, y_batch_train)
                else:
                    loss_value = train_step(x_batch_train, y_batch_train)
                losses.append(float(loss_value))
                lrs.append(opt.base_optimizer.lr.numpy())
                if step % 200 == 0:
                    print("Step : %d :: Current loss : %f" % (step, float(loss_value)))
            for x_batch_test, y_batch_test in test_dataset:
                val_logits = model(x_batch_test, training=False)
                triplet_loss_metrics.update_state(y_batch_test, val_logits)
            result = triplet_loss_metrics.result()
            print(str(result))
            triplet_loss_metrics.reset_states()

        plt.xscale('log')
        plt.plot(lrs, losses, color='blue')
        smooth_losses = savgol_filter(train_history.losses, 5)
        plt.plot(lrs, smooth_losses, color='red')
        plt.xlabel('Log learning rate')
        plt.ylabel('Loss')
        plt.savefig('./range_test_result.png')
        print('\n[INFO] Training complete. Range test results can be found at "./range_test_result.png"')

        return
    else:
        lr_schedule = get_learning_rate_schedule(schedule_name=lr_schedule_name,
                                                 learning_rate=init_lr,
                                                 max_lr=max_lr,
                                                 image_count=n_imgs,
                                                 batch_size=batch_size)
        opt = get_optimizer(optimizer_name=optimizer,
                            lr_schedule=lr_schedule,
                            weight_decay=weight_decay)

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint_name = checkpoint_path + '/' + 'cp-{epoch:03d}.ckpt'
        
        if use_tpu is True:
            with strategy.scope():
                model = create_neural_network(model_type=model_type,
                                              embedding_size=embedding_size)
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
        elif distributed is True and use_tpu is False:
            with mirrored_strategy.scope():
                model = create_neural_network(model_type=model_type,
                                              embedding_size=embedding_size)
                opt = get_optimizer(optimizer_name=optimizer,
                                    lr_schedule=lr_schedule,
                                    weight_decay=weight_decay)
            train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
            test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
        else:
            model = create_neural_network(model_type=model_type,
                                          embedding_size=embedding_size)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'

        for epoch in range(num_epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if distributed is True and use_tpu is False:
                    loss_value = distributed_train_step(x_batch_train, y_batch_train)
                else:
                    loss_value = train_step(x_batch_train, y_batch_train)
                if step % 200 == 0:
                    print("Step : %d :: Current loss : %f" % (step, float(loss_value)))
            for x_batch_test, y_batch_test in test_dataset:
                val_logits = model(x_batch_test, training=False)
                triplet_loss_metrics.update_state(y_batch_test, val_logits)
            result = triplet_loss_metrics.result()
            print(str(result))
            triplet_loss_metrics.reset_states()
            if epoch % 5 == 0:
                model.save(checkpoint_name.format(epoch))
        
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
    parser.add_argument('-b', '--batch_size', required=False, default=128,
                        help='Batch size to use for training')
    parser.add_argument('-s', '--image_size', required=False, default=246,
                        help='Image size (before random crop and preprocessing)')
    parser.add_argument('-c', '--crop_size', required=False, default=224,
                        help='Image size after random crop is applied')
    parser.add_argument('--lr_schedule', required=False, type=str, default='triangular2',
                        choices=['triangular2', 'triangular', 'staircase', 'exponential_decay'],
                        help='Choice of learning rate schedule. Default is a cyclic policy')
    parser.add_argument('--init_lr', required=False, type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', required=False, type=float, default=0.5,
                        help='Maximum learning rate. Should be set when using cyclic LR policies only')
    parser.add_argument('--weight_decay', required=False, type=float, default=0.000001,
                        help='Weight decay coefficient for regularization. Default value is 1e-6')
    parser.add_argument('--optimizer', required=False, default='RMSPROP',
                        choices=['RMSPROP', 'SGDW', 'ADAM', 'ADAGRAD', 'ADADELTA'],
                        help='Optimizer to use for training. Default is RMSprop')
    parser.add_argument('--model', required=False, type=str, default='inception_v3',
                        choices=['resnet50', 'resnet101', 'resnet152', 'inception_v3', 'efficientnet_b0',/ 
                                 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',/ 
                                 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'inception_resnet_v2',/ 
                                 'xception', 'mobilenet', 'mobilenet_v2'],/
                        help='NN architecture to use. Default is InceptionV3')
    parser.add_argument('--embedding_size', required=False, type=int, default=512,
                        help='Embedding size for triplet loss')
    parser.add_argument('--cache_path', required=False, type=str, default='./face_cache.tfcache',
                        help='Path to cache file to use for the training dataset')
    parser.add_argument('--epochs', required=False, type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--margin', required=False, type=float, default=0.2,
                        help='Margin to use for triplet semi-hard loss')
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
                        choices=['VANILLA', 'BATCH_HARD', 'BATCH_HARD_V2', 'FOCAL'],
                        help='Choice of triplet loss formulation. Default is FOCAL')
    parser.add_argument('--images_per_person', required=False, type=int, default=35,
                        help='Average number of images per class. Default is 35 (from MS1M cleaned + AsianCeleb)')
    parser.add_argument('--people_per_sample', required=False, type=int, default=50,
                        help='Number of people per sample. Helps fill buffer for shuffling the dataset properly')
    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Path to pretrained model OR folder containing previously trained checkpoints')
    parser.add_argument('--distance_metric', required=False, type=str, default='L2',
                        choices=['L2', 'squared-L2', 'angular'],
                        help='Choice of distance metric. Default is Euclidean distance')
    parser.add_argument('--soft', action='store_true',
                        help='Use soft margin for BATCH_HARD strategy')
    parser.add_argument('--sigma', type=float, required=False, default=0.3,
                        help='Value of sigma for FOCAL strategy')
    parser.add_argument('--use_lfw', action='store_true',
                        help='Specifies whether test dataset is the LFW dataset or not')


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
                pretrained_model=args['pretrained_model'],
                distance_metric=args['distance_metric'],
                soft=args['soft'],
                sigma=args['sigma'],
                use_lfw=args['use_lfw'])
