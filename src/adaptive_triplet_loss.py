"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf
import numpy as np


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def adapted_triplet_loss(labels, embeddings, lambda_=2.0, margin=1, soft=False, squared=False):
    """Build the apdaptive triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones and add it to the distribution shift.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        lambda_:trade-off parameter
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        adaptive_triplet_loss: scalar tensor containing the triplet loss (L = L_triplet + λ ∗ L_match)
    """
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.cast(mask_anchor_positive, dtype=tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    hard_positive_indices = tf.math.argmax(anchor_positive_dist, axis=1)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative, dtype=tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    hard_negative_indices = tf.math.argmin(anchor_negative_dist, axis=1)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if soft:
        triplet_loss = tf.math.log1p(tf.math.exp(hardest_positive_dist - hardest_negative_dist))
    else:
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    
    # Get final mean triplet loss
    L_triplet = tf.reduce_mean(triplet_loss)
    
    # Embeding dict stores the mean of all embeddings of each instance
    embedding_dict = dict()
    for i in range(len(labels)):
        if labels[i].numpy() not in embedding_dict:
            embedding_dict[labels[i].numpy()] = [embeddings[i]]
        else:
            embedding_dict[labels[i].numpy()].append(embeddings[i])
            
    # Taking mean of the embeddings in embedding_dict
    for label in embedding_dict:
        embedding_dict[label] = tf.math.reduce_mean(embedding_dict[label], axis=0)
        
    # L_match_dict stores the mean of embeddings of the instances choosen in the triplet selections
    L_match_dict = dict()
    
    # Adding instances from hard positives
    for i in hard_positive_indices.numpy():
        if labels[i].numpy() not in L_match_dict:
            L_match_dict[labels[i].numpy()] = [embeddings[i]]
        else:
            L_match_dict[labels[i].numpy()].append(embeddings[i])
            
    # Adding instances from hard negatives
    for i in hard_negative_indices.numpy():
        if labels[i].numpy() not in L_match_dict:
            L_match_dict[labels[i].numpy()] = [embeddings[i]]
        else:
            L_match_dict[labels[i].numpy()].append(embeddings[i])
            
    # Taking mean of the embeddings in L_match_dict
    for label in L_match_dict:
        L_match_dict[label] = tf.math.reduce_mean(L_match_dict[label], axis=0)
        
    # Find L Match using sum of l2 norm of L_triplet - L_match_dict
    l2_norms = []
    for ind in L_match_dict:
        l2_norm = np.linalg.norm(embedding_dict[ind] - L_match_dict[ind], ord=2)
        l2_norms.append(l2_norm)
    l2_norms = np.sum(l2_norms)
    
    # Calculate triplet loss, triplet loss = L_triplet + λ ∗ L_match
    triplet_loss = L_triplet + (lambda_*l2_norms)
    
    return triplet_loss

class AdaptiveTripletLoss(LossFunctionWrapper):
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
        lambda_: FloatTensorLike = 0.3,
        soft: bool = False,
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None, **kwargs
    ):
        super().__init__(adapted_triplet_loss,
                         name = name, 
                         reduction = tf.keras.losses.Reduction.NONE,
                         margin = margin,
                         lambda_ = lambda_,
                         soft = soft,
                         squared = False if distance_metric=="L2" else True)