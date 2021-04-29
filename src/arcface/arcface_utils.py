import tensorflow as tf
import tensorflow_addons as tfa
import math


class ArcMarginPenaltyLogits(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, self.logist_scale, 'arcface_logits')

        return logits


def OutputLayer(embd_shape, name='OutputLayer'):
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape)(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer

def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogits(num_classes=num_classes,
                                   margin=margin,
                                   logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head

def Backbone(model_type='resnet50'):
    def backbone(x_in):
        if model_type == 'resnet50':
            base_model = ResNet50(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'resnet101':
            base_model = ResNet101(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'resnet152':
            base_model = ResNet152(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'inception_v3':
            base_model = InceptionV3(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'efficientnet_b0':
            base_model = EfficientNetB0(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b1':
            base_model = EfficientNetB1(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b2':
            base_model = EfficientNetB2(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b3':
            base_model = EfficientNetB3(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b4':
            base_model = EfficientNetB4(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b5':
            base_model = EfficientNetB5(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b6':
            base_model = EfficientNetB6(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'efficientnet_b7':
            base_model = EfficientNetB7(weights=None, include_top=False, input_shape=x_in.shape[1:], activation=tfa.activations.mish)(x_in)
        elif model_type == 'inception_resnet_v2':
            base_model = InceptionResNetV2(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'xception':
            base_model = Xception(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'mobilenet':
            base_model = MobileNet(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        elif model_type == 'mobilenet_v2':
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=x_in.shape[1:])(x_in)
        else:
            pass

        assert base_model is not None, '[ERROR] The model name was not correctly specified'

        return base_model

    return backbone


def create_neural_network(model_type='resnet50', n_classes=2, embedding_size=512, input_shape=None, 
                          weights_path='', recompile=False, training=True, margin=0.5,
                          logist_scale=64):
    model = None
    assert input_shape is not None, '[ERROR] Input shape must not be None'
    assert len(input_shape) == 3, '[ERROR] Input shape is not of the form [size, size, channels]'

    X = inputs = Input(input_shape, name='image_inputs')
    X = Backbone(model_type=model_type)(X)
    embeddings = OutputLayer(embd_shape=embedding_size)
    if training is True:
        labels = Input([], name='labels')
        logits = ArcHead(num_classes=n_classes,
                         margin=margin,
                         logist_scale=logist_scale)(embeddings, labels)
        model = Model((inputs, labels), logits)
    else:
        model = Model(inputs, embeddings)

    compiled = False

    if len(weights_path) > 1 and os.path.exists(weights_path):
        print('[INFO] Attempting to load weights from most recently saved checkpoint')
        loss_obj = None
        try:
            if recompile is True:
                model = tf.keras.models.load_model(weights_path)
                compiled = False
                print('[WARNING] Model will be compiled again. If you wish to start from a previously saved optimizer state, this is not recommended')
            else:
                model = tf.keras.models.load_model(weights_path)
                compiled = True
                print('[WARNING] Model is already compiled; ignoring passed optimizer, loss and learning rate parameters')
            print('[INFO] Loading model from SavedModel format')
        except:
            try:
                try:
                    model.load_weights(weights_path)
                except:
                    latest = tf.train.latest_checkpoint(weights_path)
                    model.load_weights(latest)
                print('[WARNING] Loading model weights from ckpt format. Model state is not preserved')
            except:
                print('[ERROR] Weights did not match the model architecture specified, or path was incorrect')
                print('[WARNING] Could not load weights. Using random initialization instead')
                return model, False
    else:
        print('[WARNING] Could not load weights. Using random initialization instead')

    model.summary()
    
    return model, compiled


def SoftmaxLoss():
    def softmax_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss