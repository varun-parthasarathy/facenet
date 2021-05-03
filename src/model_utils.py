import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.applications.resnet import *
from adaptive_triplet_loss import AdaptiveTripletLoss
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss, AssortedTripletLoss


def Backbone(model_type='resnet50', use_imagenet=True):
    weights = None
    if use_imagenet is True:
        weights = 'imagenet'
    def backbone(x_in):
        base_model = None
        if model_type == 'resnet50':
            base_model = ResNet50(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                  pooling='avg')(x_in)
        elif model_type == 'resnet101':
            base_model = ResNet101(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                   pooling='avg')(x_in)
        elif model_type == 'resnet152':
            base_model = ResNet152(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                   pooling='avg')(x_in)
        elif model_type == 'inception_v3':
            base_model = InceptionV3(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                     pooling='avg')(x_in)
        elif model_type == 'efficientnet_b0':
            base_model = EfficientNetB0(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b1':
            base_model = EfficientNetB1(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b2':
            base_model = EfficientNetB2(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b3':
            base_model = EfficientNetB3(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b4':
            base_model = EfficientNetB4(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b5':
            base_model = EfficientNetB5(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b6':
            base_model = EfficientNetB6(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b7':
            base_model = EfficientNetB7(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish,
                                        pooling='avg')(x_in)
        elif model_type == 'inception_resnet_v2':
            base_model = InceptionResNetV2(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                           pooling='avg')(x_in)
        elif model_type == 'xception':
            base_model = Xception(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                  pooling='avg')(x_in)
        elif model_type == 'mobilenet':
            base_model = MobileNet(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                   pooling='avg')(x_in)
        elif model_type == 'mobilenet_v2':
            base_model = MobileNetV2(weights=weights, include_top=False, input_shape=x_in.shape[1:],
                                     pooling='avg')(x_in)
        else:
            pass

        assert base_model is not None, '[ERROR] The model name was not correctly specified'
        return base_model
    return backbone

def OutputLayer(embedding_size=512, name='OutputLayer'):
    def output_layer(x_in):
        inputs = Input(x_in.shape[1:])
        x = Dropout(0.3, name='top_dropout')(inputs)
        logits = Dense(embedding_size, activation=None)(x)
        embeddings = tf.keras.layers.Lambda(lambda k: tf.math.l2_normalize(k, axis=1), dtype='float32',
                                            name='embeddings')(logits)
        return Model(inputs=inputs, outputs=embeddings, name=name)(x_in)
    return output_layer

def create_neural_network_v2(model_type='resnet50', embedding_size=512, input_shape=[260, 260, 3], 
                             weights_path='', loss_type='ADAPTIVE', loss_fn=None, recompile=False, 
                             use_imagenet=True):

    assert input_shape is not None, '[ERROR] Input shape not specified correctly!'
    inputs = Input(input_shape, name='image_input')
    backbone = Backbone(model_type=model_type, use_imagenet=use_imagenet)(inputs)
    embeddings = OutputLayer(embedding_size=embedding_size)(backbone)
    
    model = Model(inputs=inputs, outputs=embeddings)
    model.summary()
    compiled = False

    if len(weights_path) > 1 and os.path.exists(weights_path):
        print('[INFO] Attempting to load weights from most recently saved checkpoint')
        loss_obj = None
        try:
            if recompile is True:
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
                compiled = False
                print('[WARNING] Model will be compiled again. If you wish to start from a previously saved optimizer state, this is not recommended')
            else:
                if loss_type == 'ADAPTIVE':
                    loss_obj = ['AdaptiveTripletLoss', loss_fn]
                elif loss_type == 'FOCAL':
                    loss_obj = ['TripletFocalLoss', loss_fn]
                elif loss_type == 'BATCH_HARD':
                    loss_obj = ['TripletBatchHardLoss', loss_fn]
                elif loss_type == 'BATCH_HARD_V2':
                    loss_obj = ['TripletBatchHardV2Loss', loss_fn]
                elif loss_type == 'ASSORTED':
                    loss_obj = ['AssortedTripletLoss', loss_fn]
                else:
                    loss_obj = None
                if loss_obj is not None:
                    model = tf.keras.models.load_model(weights_path, custom_objects={loss_obj[0]:loss_obj[1]})
                else:
                    model = tf.keras.models.load_model(weights_path)
                    print('[INFO] Loading model without custom objects')
                compiled = True
                print('[WARNING] Model is already compiled; ignoring passed optimizer, loss and learning rate parameters')
            print('[INFO] Loading model from SavedModel format')
        except:
            try:
                latest = tf.train.latest_checkpoint(weights_path)
                model.load_weights(latest)
                print('[WARNING] Loading model weights from ckpt format. Model state is not preserved')
            except:
                print('[ERROR] Weights did not match the model architecture specified, or path was incorrect')
                print('[WARNING] Could not load weights. Using random initialization instead')
                return model, False
    else:
        print('[WARNING] Could not load weights. Using random initialization instead')
    
    return model, compiled