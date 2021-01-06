import tensorflow as tf
import re


class SAMOptimizer(tf.keras.optimizers.Optimizer):
    '''Implements a custom optimizer for Sharpness Aware Minimization.
       See https://arxiv.org/abs/2010.01412 for more details.
       This algorithm achieves state of the art performance on image classification
       tasks, and could potentially boost performance for face recognition as well'''
    def __init__(self, rho, base_optimizer, name=None, **kwargs):
        super(SAMOptimizer, self).__init__(name=name)
        assert rho >= 0, "[ERROR] rho not defined correctly"

        self._rho = rho
        self.base_optimizer = base_optimizer

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        raise NotImplementedError("SAM requires a custom training loop")

    @tf.function
    def first_step(self, grads, model, name=None):
        grad_norm = self._grad_norm(grads)
        scale = self._rho / tf.math.add(grad_norm, 1e-12)
        perturbations = dict()
        param_name = self._get_variable_name(param.name)
        for i, pair in enumerate(zip(grads, model.trainable_weights)):
            grad, param = pair
            if grad is None or param is None:
                perturbations[param_name] = None
                continue
            e_w = grad * scale
            model.trainable_weights[i] = tf.math.add(param, e_w)
            perturbations[param_name] = e_w

        return perturbations

    @tf.function
    def second_step(self, grads, model, perturbations, name=None):
        new_grads = list()
        for i, pair in ennumerate(zip(grads, model.trainable_weights)):
            grad, param = pair
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            e_w = perturbations[param_name]
            if e_w is None:
                continue
            model.trainable_weights[i] = tf.math.subtract(param, e_w)
            new_grads.append(grad)

        self.base_optimizer.apply_gradients(zip(new_grads, model.trainable_weights))

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    @tf.function
    def _grad_norm(self, grads_and_vars):
        grads = [g for g, v in grads_and_vars]
        return tf.linalg.global_norm(grads)
