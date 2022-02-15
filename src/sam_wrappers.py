import random
import tensorflow as tf
import copy


class SAMModel(tf.keras.Model):
    def __init__(self, base_model, rho=0.05, **kwargs):
        super(SAMModel, self).__init__(**kwargs)
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.compiled_loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, self.trainable_variables):
            if grad is None or param is None:
                continue
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.base_model(images, training=True)
            loss = self.compiled_loss(labels, predictions)    
        
        sam_gradients = tape.gradient(loss, self.trainable_variables)
        for (param, e_w) in zip(self.trainable_variables, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, self.trainable_variables))
        
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
    def __init__(self, rho=0.05, beta=0.5, gamma=0.5, adaptive=False, **kwargs):
        super(ESAMModel, self).__init__(**kwargs)
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.adaptive = adaptive
        self.loss_fn = None

    def train_step(self, data):
        images, labels = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.compiled_loss(labels, predictions)
        loss_before = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        grad_norm = self._grad_norm(gradients)
        scale = (self.rho / (grad_norm + 1e-7)) / self.beta

        for (grad, param) in zip(gradients, self.trainable_variables):
            if grad is None or param is None:
                continue
            if self.adaptive is True:
                e_w = tf.pow(param, 2) * grad * scale
            else:
                e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        loss_after = self.loss_fn(labels, self(images))

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
            predictions = self(tf.gather(images, indices), training=True)
            loss = self.compiled_loss(tf.gather(labels, indices), predictions)
        
        sam_gradients = tape.gradient(loss, self.trainable_variables)
        new_grads = []
        new_vars = []
        for i, (param, e_w, g) in enumerate(zip(self.trainable_variables, e_ws, sam_gradients)):
            param.assign_sub(e_w)
            if random.random() > self.beta:
                pass
            else:
                new_grads.append(g)
                new_vars.append(param)

        self.optimizer.apply_gradients(zip(new_grads, new_vars))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        if self.adaptive is True:
            norm = tf.norm(
                tf.stack([
                    tf.norm(tf.math.abs(param) * grad) for grad, param in zip(gradients, self.trainable_variables) if (grad is not None and param is not None)
                ])
            )
        else:
            norm = tf.norm(
                tf.stack([
                    tf.norm(grad) for grad in gradients if grad is not None
                ])
            )
        return norm

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        self.loss_fn = copy.deepcopy(loss)
        self.loss_fn._fn_kwargs['reduce_loss'] = False
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        steps_per_execution=steps_per_execution,
                        **kwargs)