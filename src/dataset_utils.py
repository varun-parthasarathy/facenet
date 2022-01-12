import os
import cv2
import pathlib
import importlib
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
            #path_list += (path0,path1)
            path_list.append(path0)
            path_list.append(path1)
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

def _get_preprocessor(model_type):
    if 'inception_resnet_v2' in model_type:
        preprocessor = 'tensorflow.keras.applications.inception_resnet_v2'
        print('[INFO] Loaded Inception-Resnet-V2 data preprocessor', flush=True)
    elif 'efficientnet' in model_type:
        preprocessor = 'tensorflow.keras.applications.efficientnet'
        print('[INFO] Loaded EfficientNet data preprocessor', flush=True)
    elif 'xception' in model_type:
        preprocessor = 'tensorflow.keras.applications.xception'
        print('[INFO] Loaded Xception data preprocessor', flush=True)
    elif 'inception_v3' in model_type:
        preprocessor = 'tensorflow.keras.applications.inception_v3'
        print('[INFO] Loaded Inception-V3 data preprocessor', flush=True)
    elif 'resnet' in model_type:
        preprocessor = 'tensorflow.keras.applications.resnet'
        print('[INFO] Loaded Resnet data preprocessor', flush=True)
    elif 'mobilenet_v2' in model_type:
        preprocessor = 'tensorflow.keras.applications.mobilenet_v2'
        print('[INFO] Loaded MobileNet-V2 data preprocessor', flush=True)
    elif 'mobilenet' in model_type:
        preprocessor = 'tensorflow.keras.applications.mobilenet'
        print('[INFO] Loaded MobileNet data preprocessor', flush=True)
    else:
        preprocessor = None
        print('[WARNING] Could not find appropriate pre-processor for model', flush=True)

    if preprocessor is not None:
        preprocessor = importlib.import_module(preprocessor)

    return preprocessor

def generate_training_dataset(data_path, image_size, batch_size, crop_size, cache='',
                              use_mixed_precision=False, images_per_person=35, people_per_sample=50, 
                              use_tpu=False, model_type=None, equisample=False):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CLASS_NAMES = [item.name for item in data_path.iterdir() if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(list(data_path.glob('*/*.png')))
    #batches = int(image_count / images_per_person)
    print("[INFO] Image count in training dataset : %d" % (image_count))
    preprocessor = _get_preprocessor(model_type)


    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(parts[-2] == CLASS_NAMES)

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img, channels=3)
        if use_mixed_precision is True:
            if use_tpu is True:
                img = tf.cast(img, tf.bfloat16)
            else:
                img = tf.cast(img, tf.float16)
        else:
            img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [image_size, image_size])
        '''if tf.shape(img)[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif len(tf.shape(img)) == 2:
            img = tf.image.grayscale_to_rgb(tf.expand_dims(img, axis=-1))
        else:
            pass'''
        return img

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            pass
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_brightness(img, 0.2)
        #img = tf.image.random_contrast(img, 0.8, 1.0)
        #img = tf.image.random_jpeg_quality(img, 75, 100)
        return img, label

    def get_images(f):
        return tf.data.Dataset.list_files(tf.strings.join([f, '/*.png']))

    '''ds = ds.batch(images_per_person)
    if len(cache) > 1:
        ds = ds.cache(cache)
    ds = ds.shuffle(batches+1, reshuffle_each_iteration=True)
    ds = ds.unbatch()'''
    ds = None
    if equisample is True:
        classes_in_path = [str(i) for i in data_path.iterdir() if i.is_dir()]
        #classes_ds = tf.data.Dataset.list_files(str(data_path/'*/'), shuffle=True)
        classes_ds = tf.data.Dataset.from_tensor_slices(classes_in_path)
        classes_ds = classes_ds.shuffle(len(classes_in_path), reshuffle_each_iteration=True)
        ds = classes_ds.interleave(lambda f: get_images(f), block_length=images_per_person,
                                   cycle_length=people_per_sample, num_parallel_calls=AUTOTUNE)
        #if len(cache) > 1:
        #    ds = ds.cache(cache)
        ds = ds.map(process_path, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size)#.shuffle(4096, reshuffle_each_iteration=True)
        ds = ds.prefetch(AUTOTUNE)
    else:
        ds = tf.data.Dataset.list_files(str(data_path/"*/*.png"), shuffle=False)
        ds = ds.shuffle(images_per_person * people_per_sample)
        ds = ds.map(process_path, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size).shuffle(512, reshuffle_each_iteration=True)
        #ds = ds.map(lambda x: tf.random.shuffle(x), num_parallel_calls=AUTOTUNE) # Useful for ConstellationLoss
        # The above line of code should not be absolutely necessary, since we are performing a shuffle
        # before this which should in theory ensure that the images within a batch are shuffled. However,
        # I am leaving it in as a comment just in case.
        ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)


def generate_training_dataset_v2(data_path, image_size, batch_size, crop_size, cache='',
                                 use_mixed_precision=False, images_per_person=35, people_per_sample=50, 
                                 use_tpu=False, model_type=None, equisample=False):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CLASS_NAMES = [item.name for item in data_path.iterdir() if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(list(data_path.glob('*/*.png')))
    batches = image_count / batch_size
    if batches != int(batches):
        batches = int(batches) + 1
    else:
        batches = int(batches)
    print("[INFO] Image count in training dataset : %d" % (image_count))
    print("[INFO] Using V2 method for generating dataset")
    print("[INFO] Number of batches in dataset : %d" % (batches))
    preprocessor = _get_preprocessor(model_type)


    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(parts[-2] == CLASS_NAMES)

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img, channels=3)
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
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            img = img / 255.
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_brightness(img, 0.2)
        #img = tf.image.random_contrast(img, 0.8, 1.0)
        #img = tf.image.random_jpeg_quality(img, 75, 100)
        return img, label

    def get_images(f):
        return tf.data.Dataset.list_files(tf.strings.join([f, '/*.png']))

    def process_path_bulk(batch):
        #reqd_dtype = tf.float16 if use_mixed_precision is True else tf.float32
        #reqd_dtype = tf.bfloat16 if use_tpu is True else reqd_dtype
        #batch = tf.map_fn(process_path, batch, fn_output_signature=reqd_dtype)
        # The above code may not be needed as vectorized_map should handle this without issues
        # However, I have left it in for safety purposes.
        batch = tf.vectorized_map(process_path, batch)
        return batch

    ds = None
    if equisample is True:
        classes_in_path = [str(i) for i in data_path.iterdir() if i.is_dir()]
        # classes_ds = tf.data.Dataset.from_tensor_slices(classes_in_path)
        # classes_ds = classes_ds.shuffle(len(classes_in_path), reshuffle_each_iteration=True)
        # ds = classes_ds.interleave(lambda f: get_images(f), block_length=images_per_person,
        #                            cycle_length=people_per_sample, num_parallel_calls=AUTOTUNE)
        classes_ds = [get_images(f) for f in classes_in_path]
        weights = [1./len(classes_in_path) for i in classes_in_path]
        ds = tf.data.Dataset.sample_from_datasets(classes_ds, weights=weights, stop_on_empty_dataset=False)
        ds = ds.batch(batch_size).shuffle(batches, reshuffle_each_iteration=True)
        ds = ds.map(process_path_bulk, num_parallel_calls=AUTOTUNE, deterministic=False)
        ds = ds.prefetch(AUTOTUNE)
        ds = ds.apply(tf.data.experimental.assert_cardinality(batches)) # To ensure the dataset has a fixed cardinality
                                                                        # while training - fixes the number of training steps
    else:
        ds = tf.data.Dataset.list_files(str(data_path/"*/*.png"), shuffle=False)
        ds = ds.shuffle(images_per_person * people_per_sample).batch(batch_size)
        ds = ds.shuffle(batches, reshuffle_each_iteration=True)
        ds = ds.map(process_path_bulk, num_parallel_calls=AUTOTUNE, deterministic=False)
        ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)


def get_test_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                     use_mixed_precision=False, use_tpu=False, model_type=None):
    data_path = pathlib.Path(data_path)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    assert batch_size % 2 == 0, '[ERROR] Batch size must be a multiple of 2'
    CLASS_NAMES = [item.name for item in data_path.iterdir() if item.is_dir()]
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(CLASS_NAMES)*3
    preprocessor = _get_preprocessor(model_type)

    #classes_ds = tf.data.Dataset.list_files(str(data_path/'*/'))

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img, channels=3)
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
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            img = img / 255.
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        return img, label

    ds = tf.data.Dataset.list_files(str(data_path/"*/*.png"), shuffle=False)
    ds = ds.batch(3)
    if len(cache) > 1:
        ds = ds.cache(cache)
    ds = ds.shuffle(1024, reshuffle_each_iteration=False)
    ds = ds.unbatch()
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)


def get_LFW_dataset(data_path, image_size, batch_size, crop_size, cache='', train_classes=0,
                    use_mixed_precision=False, use_tpu=False, model_type=None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    assert batch_size % 2 == 0, '[ERROR] Batch size must be a multiple of 2'
    pairs = _read_pairs('./data/pairs.txt')
    path_list, _, CLASS_NAMES = _get_paths(data_path, pairs)
    CLASS_NAMES.sort()
    CLASS_NAMES = np.array(CLASS_NAMES)
    image_count = len(path_list)
    preprocessor = _get_preprocessor(model_type)

    ds = tf.data.Dataset.from_tensor_slices(path_list)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.argmax(parts[-2] == CLASS_NAMES) + train_classes

    def decode_img(img):
        #img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.io.decode_png(img, channels=3)
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
        if preprocessor is not None:
            img = preprocessor.preprocess_input(img)
        else:
            img = img / 255.
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        return img, label

    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    if len(cache) > 1:
        ds = ds.cache(cache)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, image_count, len(CLASS_NAMES)
