import os
import cv2
import keras
import pathlib
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cyclic_learning_rate as clr
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
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss


class RangeTestCallback(tf.keras.callbacks.Callback):
  def __init__(self, start_lr, end_lr):
    super().__init__()
    self.start_lr = start_lr
    self.end_lr = end_lr

  def on_train_begin(self, logs={}):
    self.lrs = []
    self.losses = []
    tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)
    n_steps = self.params['steps'] if self.params['steps'] is not None else round(self.params['samples'] / self.params['batch_size'])
    n_steps *= self.params['epochs']
    self.by = (self.end_lr - self.start_lr) / n_steps

  def on_batch_end(self, batch, logs={}):
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    self.lrs.append(lr)
    self.losses.append(logs.get('loss'))
    lr += self.by
    tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def generate_training_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                              use_mixed_precision=False, images_per_person=35, people_per_sample=50, use_tpu=False):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CLASS_NAMES = [item.name for item in data_path.glob('*') if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(list(data_path.glob('*/*.jpg')))

    classes_ds = tf.data.Dataset.list_files(str(data_path/'*/'))

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return np.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16) / 255.0
            else:
                img = tf.cast(img, tf.float16) / 255.0
        else:
            img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [image_size, image_size])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_jpeg_quality(img, 60, 100)
        return img, label

    def parse_class(class_path):
        return tf.data.Dataset.list_files(str(class_path/'*.jpg'), shuffle=True)

    if len(cache) > 1:
        classes_ds = classes_ds.cache(cache)
    ds = classes_ds.shuffle(len(CLASS_NAMES), reshuffle_each_iteration=True)
    ds = ds.interleave(lambda x: tf.data.Dataset(x).map(parse_class, num_parallel_calls=AUTOTUNE),
                       cycle_length=1, block_length=images_per_person)
    ds = ds.batch(batch_size).map(lambda x: tf.random.shuffle(x))
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.repeat()
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)

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
    embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), dtype='float32')(logits)
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

    return opt

def train_model(data_path, batch_size, image_size, crop_size, lr_schedule_name, init_lr, max_lr, weight_decay, 
                optimizer, model_type, embedding_size, cache_path=None, num_epochs, margin=0.35, 
                checkpoint_path, range_test=False, use_tpu=False, tpu_name=None, test_path='',
                use_mixed_precision=False, triplet_strategy='', images_per_person=35, 
                people_per_sample=12, pretrained_model='', squared=False, soft=True, sigma=0.3):

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

    train_dataset, n_imgs, n_classes = generate_training_dataset(data_path, image_size, batch_size, 
                                                                 crop_size, cache_path,
                                                                 train_classes=0,
                                                                 use_mixed_precision=use_mixed_precision,
                                                                 images_per_person=images_per_person,
                                                                 people_per_sample=people_per_sample,
                                                                 use_tpu=use_tpu)

    if len(test_path) > 1:
        test_dataset, _, _ = generate_training_dataset(test_path, image_size, batch_size, crop_size, 
                                                       cache='./test_dataset_cache.tfcache',
                                                       train_classes=n_classes,
                                                       use_mixed_precision=use_mixed_precision,
                                                       images_per_person=10,
                                                       people_per_sample=10,
                                                       use_tpu=use_tpu)
    else:
        test_dataset = None

    if triplet_strategy == 'VANILLA':
        loss_fn = tfa.losses.TripletSemiHardLoss(margin=margin)
        print('[INFO] Using vanilla triplet loss')
    elif triplet_strategy == 'BATCH_HARD':
        loss_fn = TripletBatchHardLoss(margin=margin,
                                       soft=soft,
                                       squared=squared)
        print('[INFO] Using batch-hard strategy.')
    elif triplet_strategy == 'BATCH_HARD_V2':
        loss_fn = TripletBatchHardV2Loss(margin1=(-1.0*margin),
                                         margin2=(margin1/100.0),
                                         beta=0.002,
                                         squared=squared)
        print('[INFO] Using batch-hard V2 strategy')
    else:
        loss_fn = TripletFocalLoss(margin=margin,
                                   sigma=sigma,
                                   squared=squared)
        print('[INFO] Using triplet focal loss.')

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
                model.compile(optimizer=opt,
                              loss=loss_fn)
        else:
            model = create_neural_network(model_type=model_type,
                                          embedding_size=embedding_size)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
            model.compile(optimizer=opt,
                          loss=loss_fn)
        train_history = model.fit(train_dataset, epochs=5, callbacks=[range_finder], validation_data=test_dataset)
        plt.xscale('log')
        plt.plot(train_history.lrs, train_history.losses, color='blue')
        smooth_losses = savgol_filter(train_history.losses, 5)
        plt.plot(train_history.lrs, smooth_losses, color='red')
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
        checkpoint_saver = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name,
                                                              save_weights_only=True,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False,
                                                              save_freq='epoch')
        if use_tpu is True:
            with strategy.scope():
                model = create_neural_network(model_type=model_type,
                                              embedding_size=embedding_size)
                assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
                model.compile(optimizer=opt,
                              loss=loss_fn)
        else:
            model = create_neural_network(model_type=model_type,
                                          embedding_size=embedding_size)
            assert model is not None, '[ERROR] There was a problem while loading the pre-trained weights'
            model.compile(optimizer=opt,
                          loss=loss_fn)

        train_history = model.fit(train_dataset, epochs=num_epochs, callbacks=[checkpoint_saver], 
                                  validation_data=test_dataset)
        
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
    parser.add_argument('--squared', action='store_true',
                        help='Whether to use squared Euclidean distance or not')
    parser.add_argument('--soft', action='store_true',
                        help='Use soft margin for BATCH_HARD strategy')
    parser.add_argument('--sigma', type=float, required=False, default=0.3,
                        help='Value of sigma for FOCAL strategy')

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
                squared=args['squared'],
                soft=args['soft'],
                sigma=args['sigma'])
