import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from models.efficientnetv2 import effnetv2_model
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
from custom_triplet_loss import TripletBatchHardLoss, TripletFocalLoss, TripletBatchHardV2Loss, AssortedTripletLoss, ConstellationLoss
from custom_triplet_loss import HAP2S_ELoss, HAP2S_PLoss
from tensorflow_similarity.losses import MultiSimilarityLoss
from efficientnetv2_models import get_efficientnetv2_model


class MetricEmbedding(tf.keras.layers.Layer):
    def __init__(self, unit):
        """L2 Normalized `Dense` layer usually used as output layer.
        Args:
            unit: Dimension of the output embbeding. Commonly between
            32 and 512. Larger embeddings usually result in higher accuracy up
            to a point at the expense of making search slower.
        """
        self.unit = unit
        self.dense = Dense(unit, dtype=tf.float32)
        super().__init__()
        # FIXME: enforce the shape
        # self.input_spec = rank2

    def call(self, inputs):
        x = self.dense(inputs)
        normed_x = tf.math.l2_normalize(x, axis=1)
        return normed_x

    def get_config(self):
        return {'unit': self.unit}


class SAMModel(tf.keras.Model):
    def __init__(self, base_model, rho=0.05):
        super(SAMModel, self).__init__()
        self.base_model = base_model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.base_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm


class ESAMModel(tf.keras.Model):
    '''
    From Efficient Sharpness-aware Minimization for Improved Training of Neural Networks
    See https://arxiv.org/pdf/2110.03141.pdf
    This provides performance and efficiency improvements over regular SAM. This class
    overrides the default train step, making it easier to use SAM for training other models
    since a custom training loop is no longer required.
    '''
    def __init__(self, base_model, rho=0.05, beta=0.5, gamma=0.5, adaptive=False):
        super(SAMModel, self).__init__()
        self.base_model = base_model
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.adaptive = adaptive

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)
        loss_before = loss.copy()
        trainable_params = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients, trainable_params)
        scale = self.rho / (grad_norm + 1e-7) / self.beta

        for (grad, param) in zip(gradients, trainable_params):
            if grad is None:
                continue
            if self.adaptive is True:
                e_w = tf.pow(param, 2) * grad * scale
            else:
                e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss_after = self.compiled_loss(labels, predictions)

        instance_sharpness = loss_after - loss_before
        prob = self.gamma
        if prob >= 0.99:
            indices = range(len(targets))
        else:
            position = int(len(targets) * prob)
            cutoff, _ = tf.math.top_k(instance_sharpness, position)
            cutoff = cutoff[-1]
            indices = [instance_sharpness > cutoff]

        with tf.GradientTape() as tape:
            predictions = self.base_model(images[indices])
            loss = self.compiled_loss(labels[indices], predictions)
        
        sam_gradients = tape.gradient(loss, trainable_params)
        is_requires_grad = []
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
            if random.random() > self.beta:
                is_requires_grad.append(False)
            else:
                is_requires_grad.append(True)
        new_grads, vars_to_update = sam_gradients, trainable_params
        for g, p, requires_grad in zip(sam_gradients, trainable_params, is_requires_grad):
            if requires_grad is False:
                new_grads.remove(g)
                vars_to_update.remove(p)
        
        self.optimizer.apply_gradients(
            zip(new_grads, vars_to_update))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.base_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients, params):
        if self.adaptive is True:
            norm = tf.norm(
                tf.stack([
                    tf.norm(tf.math.abs(param) * grad) for grad, param in zip(gradients, params) if grad is not None
                ])
            )
        else:
            norm = tf.norm(
                tf.stack([
                    tf.norm(grad) for grad in gradients if grad is not None
                ])
            )
        return norm


def Backbone(model_type='resnet50', use_imagenet=True):
    weights = None
    if use_imagenet is True:
        weights = 'imagenet'
    activation = tfa.activations.mish
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
            base_model = EfficientNetB0(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b1':
            base_model = EfficientNetB1(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b2':
            base_model = EfficientNetB2(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b3':
            base_model = EfficientNetB3(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b4':
            base_model = EfficientNetB4(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b5':
            base_model = EfficientNetB5(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b6':
            base_model = EfficientNetB6(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
                                        pooling='avg')(x_in)
        elif model_type == 'efficientnet_b7':
            base_model = EfficientNetB7(weights=weights, include_top=False, input_shape=x_in.shape[1:], activation=activation,
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

def OutputLayer(embedding_size=512, name='OutputLayer', model_type='resnet50'):
    def output_layer(x_in):
        inputs = Input(x_in.shape[1:])
        '''
        if 'efficientnet' in model_type:
            x = Dropout(0.3, name='top_dropout')(inputs)
            logits = Dense(embedding_size, activation=None)(x)
        else:
            logits = Dense(embedding_size, activation=None)(inputs)
        embeddings = tf.keras.layers.Lambda(lambda k: tf.math.l2_normalize(k, axis=1), dtype='float32',
                                            name='embeddings')(logits)
        '''
        if 'efficientnet' in model_type:
            x = Dropout(0.3, name='top_dropout')(inputs)
            embeddings = MetricEmbedding(embedding_size)(x)
        else:
            embeddings = MetricEmbedding(embedding_size)(inputs)
        return Model(inputs=inputs, outputs=embeddings, name=name)(x_in)
    return output_layer

def create_neural_network_v2(model_type='resnet50', embedding_size=512, input_shape=[260, 260, 3], 
                             weights_path='', loss_type='ADAPTIVE', loss_fn=None, recompile=False, 
                             use_imagenet=True, sam_type='null'):

    assert input_shape is not None, '[ERROR] Input shape not specified correctly!'
    assert len(input_shape) == 3, '[ERROR] Input shape must be of the form [height, width, channels]'
    '''
        The above assertion is simply so that another Monty Python reference can be snuck into the code.
        It makes things more fun to read.

        And then the Lord spake, saying:
        "First, shalt thou take out the model's top layer. Then shalt thou count to three, 
        no more, no less. Three shall be the number thou shalt count, and the number of the 
        counting shall be three. Four shalt thou not count, neither count thou two, excepting 
        that thou then proceed to three. Five is right out. Once the number three, being the 
        third number, be reached, then createst thou thy input layer, which having a 3-dimensional 
        shape, shall serve you well."
    '''
    model = None
    if 'efficientnetv2' in model_type:
        if use_imagenet is True:
            weights = 'imagenet21k-ft1k'
            #weights = 'imagenet21k'
            #weights = 'imagenet'
        else:
            weights = None

        # model = tf.keras.models.Sequential([
        #             tf.keras.layers.InputLayer(input_shape=input_shape),
        #             effnetv2_model.get_model(model_type, include_top=False, weights=weights, input_size=input_shape[0]),
        #             tf.keras.layers.Dropout(rate=0.3),
        #             MetricEmbedding(embedding_size)
        #         ])
        base_model = get_efficientnetv2_model(model_type=model_type,
                                              input_shape=input_shape,
                                              num_classes=0, # Must always be 0
                                              pretrained=weights)
        base_output = base_model.output # NOT "outputs"! Should be singular!
        droput_layer = tf.keras.layers.Dropout(rate=0.3)(base_output)
        embeddings = MetricEmbedding(embedding_size)(droput_layer)
        model = Model(inputs=base_model.input, outputs=embeddings)

    else:
        inputs = Input(input_shape, name='image_input')
        backbone = Backbone(model_type=model_type, use_imagenet=use_imagenet)(inputs)
        embeddings = OutputLayer(embedding_size=embedding_size, model_type=model_type)(backbone)
        model = Model(inputs=inputs, outputs=embeddings)
    
    assert model is not None, '[ERROR] Could not create model!'
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
                elif loss_type == 'CONSTELLATION':
                    loss_obj = ['ConstellationLoss', ConstellationLoss]
                elif loss_type == 'HAP2S_E':
                    loss_obj = ['HAP2S_ELoss', HAP2S_ELoss]
                elif loss_type == 'HAP2S_P':
                    loss_obj = ['HAP2S_PLoss', HAP2S_PLoss]
                else:
                    loss_obj = None
                if loss_obj is not None:
                    model = tf.keras.models.load_model(weights_path, custom_objects={loss_obj[0]:loss_obj[1]}, compile=False)
                else:
                    model = tf.keras.models.load_model(weights_path, compile=False)
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
                elif loss_type == 'CONSTELLATION':
                    loss_obj = ['ConstellationLoss', ConstellationLoss]
                elif loss_type == 'HAP2S_E':
                    loss_obj = ['HAP2S_ELoss', HAP2S_ELoss]
                elif loss_type == 'HAP2S_P':
                    loss_obj = ['HAP2S_PLoss', HAP2S_PLoss]
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

    if sam_type == 'SAM':
        model = SAMModel(model)
    # elif sam_type == 'ESAM':
    #     model = ESAMModel(model)
    else:
        pass
    
    return model, compiled