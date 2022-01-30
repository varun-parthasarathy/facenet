import random
import tensorflow as tf


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
            if grad is None or param is None:
                continue
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
    def __init__(self, base_model, rho=0.05, beta=0.5, gamma=0.5, adaptive=False, loss_fn=None):
        super(ESAMModel, self).__init__()
        self.base_model = base_model
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.adaptive = adaptive
        self.loss_fn = loss_fn
        assert self.loss_fn is not None, '[ERROR] No loss object received'

    def train_step(self, data):
        images, labels = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.base_model(images)
            loss = self.compiled_loss(labels, predictions)
        loss_before = self.loss_fn(labels, predictions)
        trainable_params = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients, trainable_params)
        scale = (self.rho / (grad_norm + 1e-7)) / self.beta

        for (grad, param) in zip(gradients, trainable_params):
            if grad is None or param is None:
                continue
            if self.adaptive is True:
                e_w = tf.pow(param, 2) * grad * scale
            else:
                e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        loss_after = self.loss_fn(labels, self.base_model(images))

        instance_sharpness = loss_after - loss_before
        range_val = tf.cast(tf.shape(labels)[0], tf.float32)
        if self.gamma >= 0.99:
            indices = tf.range(tf.cast(range_val, tf.int32))
        else:
            position = tf.cast(range_val * self.gamma, tf.int32)
            cutoff, _ = tf.math.top_k(instance_sharpness, position)
            cutoff = cutoff[-1]
            indices = tf.where(tf.math.greater(instance_sharpness, cutoff))
            indices = tf.reshape(indices, (tf.shape(indices)[0],))

        with tf.GradientTape() as tape:
            predictions = self.base_model(tf.gather(images, indices))
            loss = self.compiled_loss(tf.gather(labels, indices), predictions)
        
        sam_gradients = tape.gradient(loss, trainable_params)
        new_grads = []
        new_params = []
        for (param, e_w, g) in zip(trainable_params, e_ws, sam_gradients):
            param.assign_sub(e_w)
            if random.random() > self.beta:
                pass
            else:
                new_grads.append(g)
                new_params.append(param)
        
        self.optimizer.apply_gradients(
            zip(new_grads, new_params))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.base_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients, trainable_params):
        if self.adaptive is True:
            norm = tf.norm(
                tf.stack([
                    tf.norm(tf.math.abs(param) * grad) for grad, param in zip(gradients, params) if (grad is not None and param is not None)
                ])
            )
        else:
            norm = tf.norm(
                tf.stack([
                    tf.norm(grad) for grad in gradients if grad is not None
                ])
            )
        return norm