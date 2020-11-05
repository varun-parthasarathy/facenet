# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements triplet loss."""

import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from typeguard import typechecked
from typing import Optional, Union, Callable


def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


@tf.function
def triplet_focal_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 0.2,
    sigma: FloatTensorLike = 0.3,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    """Computes the triplet focal loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      sigma: Float, sigma term in the loss definition.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_focal_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = metric_learning.angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    p_hard = tf.math.exp(tf.math.divide(hard_positives, sigma))
    n_hard = tf.math.exp(tf.math.divide(hard_negatives, sigma))

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(p_hard - n_hard))
    else:
        triplet_loss = tf.maximum(p_hard - n_hard + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss



@tf.function
def triplet_batch_hard_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin: FloatTensorLike = 1.0,
    soft: bool = False,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    """Computes the triplet loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
      soft: Boolean, if set, use the soft margin version.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_batch_hard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    Returns:
      triplet_loss: float scalar with dtype of y_pred.
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = metric_learning.angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
    else:
        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss

@tf.function
def triplet_batch_hard_v2_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    margin1: FloatTensorLike = -1.0,
    margin2: FloatTensorLike = 0.01,
    beta: FloatTensorLike = 0.002,
    distance_metric: Union[str, Callable] = "L2",
) -> tf.Tensor:
    """Computes the triplet loss with hard negative and hard positive mining.
    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of
        multiclass integer labels.
      y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
      margin1: Float, margin term in the loss definition.
      margin2: Float, margin term in the loss definition.
      beta: Float, multiplier for intra-class constraint.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_batch_hard_v2_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )

      See https://ieeexplore.ieee.org/document/7780518
    """
    labels, embeddings = y_true, y_pred

    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape label tensor to [batch_size, 1].
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    if distance_metric == "L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=False
        )

    elif distance_metric == "squared-L2":
        pdist_matrix = metric_learning.pairwise_distance(
            precise_embeddings, squared=True
        )

    elif distance_metric == "angular":
        pdist_matrix = metric_learning.angular_distance(precise_embeddings)

    else:
        pdist_matrix = distance_metric(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # hard negatives: smallest D_an.
    hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

    batch_size = tf.size(labels)

    adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # hard positives: largest D_ap.
    hard_positives = _masked_maximum(pdist_matrix, mask_positives)

    triplet_loss = tf.maximum(hard_positives - hard_negatives, margin1) + tf.math.multiply(
                                              tf.maximum(hard_positives, margin2), beta)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss


class TripletFocalLoss(LossFunctionWrapper):
    """Computes the triplet loss with hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8558553.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      sigma: Float, sigma term in the loss definition.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self, margin: FloatTensorLike = 1.0, 
        sigma: FloatTensorLike = 0.3,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None, **kwargs
    ):
        super().__init__(triplet_focal_loss,
                         name = name, 
                         reduction = tf.keras.losses.Reduction.NONE,
                         margin = margin,
                         sigma = sigma,
                         soft = soft,
                         distance_metric = distance_metric)


class TripletBatchHardLoss(tf.keras.losses.Loss):
    """Computes the triplet loss with hard negative and hard positive mining.
    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(triplet_batch_hard_loss,
                         name = name, 
                         reduction = tf.keras.losses.Reduction.NONE,
                         margin = margin,
                         soft = soft,
                         distance_metric = distance_metric)


class TripletBatchHardV2Loss(tf.keras.losses.Loss):
    """Computes the triplet loss with hard negative and hard positive mining.
    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch. Intra-class variability is enforced through
    a second margin that places a constraint on the spread of the cluster.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://ieeexplore.ieee.org/document/7780518.
    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.
    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      name: Optional name for the op.
      distance_metric: str or function, determines distance metric:
                       "L2" for l2-norm distance
                       "squared-L2" for squared l2-norm distance
                       "angular" for cosine similarity
                        A custom function returning a 2d adjacency
                          matrix of a chosen distance metric can
                          also be passed here. e.g.
                          def custom_distance(batch):
                              batch = 1 - batch @ batch.T
                              return batch
                          triplet_semihard_loss(batch, labels,
                                        distance_metric=custom_distance
                                    )
    """

    @typechecked
    def __init__(
        self,
        margin1: FloatTensorLike = -1.0,
        margin2: FloatTensorLike = 0.01,
        beta: FloatTensorLike = 0.002,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(triplet_batch_hard_v2_loss,
                         name = name, 
                         reduction = tf.keras.losses.Reduction.NONE,
                         margin1 = margin1,
                         margin2 = margin2,
                         beta = beta,
                         distance_metric = distance_metric)