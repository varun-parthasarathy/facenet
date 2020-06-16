import os
import cv2
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def _get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    classes = set()
    for pair in pairs:
        if len(pair) == 3:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
            classes.add(pair[0])
        elif len(pair) == 4:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
            classes.add(pair[0])
            classes.add(pair[2])
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('[INFO] Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list, list(classes)
  
def _add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def _read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def generate_training_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                              use_mixed_precision=False, images_per_person=35, people_per_sample=50, 
                              use_tpu=False):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CLASS_NAMES = [item.name for item in data_path.glob('*') if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(list(data_path.glob('*/*')))

    classes_ds = tf.data.Dataset.list_files(str(data_path/'*/'))

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return np.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        img = tf.io.decode_image(img, channels=3)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16)
            else:
                img = tf.cast(img, tf.float16)
        else:
            img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.0, 0.2)
        img = tf.image.random_jpeg_quality(img, 70, 100)
        img = img / 255.0
        return img, label

    def parse_class(class_path):
        class_path = pathlib.Path(class_path)
        return tf.data.Dataset.list_files(str(class_path/'*.jpg'), shuffle=True)

    if len(cache) > 1:
        classes_ds = classes_ds.cache(cache)
    ds = classes_ds.shuffle(len(CLASS_NAMES), reshuffle_each_iteration=True)
    ds = ds.interleave(lambda x: tf.data.Dataset(x).map(parse_class, num_parallel_calls=AUTOTUNE),
                       cycle_length=len(CLASS_NAMES), block_length=images_per_person,
                       num_parallel_calls=AUTOTUNE,
                       deterministic=True)
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).map(lambda x: tf.random.shuffle(x), num_parallel_calls=AUTOTUNE)
    #ds = ds.repeat() # Is this needed?
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)


def get_test_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                     use_mixed_precision=False, use_tpu=False):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    assert batch_size % 2 == 0, '[ERROR] Batch size must be a multiple of 2'
    CLASS_NAMES = [item.name for item in data_path.glob('*') if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(list(data_path.glob('*/*')))

    classes_ds = tf.data.Dataset.list_files(str(data_path/'*/'))

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return np.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        img = tf.io.decode_image(img, channels=3)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16)
            else:
                img = tf.cast(img, tf.float16)
        else:
            img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = img / 255.0
        return img, label

    def parse_class(class_path):
        class_path = pathlib.Path(class_path)
        return tf.data.Dataset.list_files(str(class_path/'*.jpg'), shuffle=True).take(3)

    if len(cache) > 1:
        classes_ds = classes_ds.cache(cache)
    ds = classes_ds.shuffle(len(CLASS_NAMES), reshuffle_each_iteration=True)
    ds = ds.interleave(lambda x: tf.data.Dataset(x).map(parse_class, num_parallel_calls=AUTOTUNE),
                       cycle_length=len(CLASS_NAMES), block_length=2,
                       num_parallel_calls=AUTOTUNE,
                       deterministic=True)
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)


def get_LFW_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                    use_mixed_precision=False, use_tpu=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    assert batch_size % 2 == 0, '[ERROR] Batch size must be a multiple of 2'
    pairs = _read_pairs('./data/pairs.txt')
    path_list, _, CLASS_NAMES = _get_paths(data_path, pairs)

    ds = tf.data.Dataset.from_tensor_slices(path_list)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return np.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        img = tf.io.decode_image(img, channels=3)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16)
            else:
                img = tf.cast(img, tf.float16)
        else:
            img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size])
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = img / 255.0
        return img, label

    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    if len(cache) > 1:
        ds = ds.cache(cache)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)
