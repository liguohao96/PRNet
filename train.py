from predictor import PosPrediction, resfcn256
import os
import tensorflow as tf
import skimage.io
import numpy as np


def train(loss_val, var_list):
    global_step = tf.Variable(FLAGS['start_step'], trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=FLAGS['decay_step'], decay_rate=FLAGS['decay_rate'],
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads, global_step=global_step), learning_rate, global_step


def _parse_function(input_file, label_file):
    image_string = tf.read_file(input_file)
    image_decoded = tf.image.decode_image(image_string)

    label_string = tf.read_file(label_file)
    label_decoded = tf.image.decode_image(label_string)
    return image_decoded, label_decoded


FLAGS = {'learning_rate': 1e-4,
         'log_dir': 'Data/net-data/',
         'weight_file': './Data/net-data/256_256_resfcn256_weight',
         'decay_step': 2500,
         'decay_rate': 0.25,
         'start_step': 0,
         'end_step': 15000,
         'batch_size': 8
         }

import random


class BatchData(object):
    def __init__(self, data_dir, batch_size=32, valid_percent=0.2):
        input_dir = os.path.join(data_dir, 'origin')
        label_dir = os.path.join(data_dir, 'uv_map')
        input_list = [os.path.join(input_dir, f)
                      for f in os.listdir(input_dir)]
        label_list = []
        for f in input_list:
            filename = os.path.splitext(os.path.split(f)[-1])[0]
            label_file = os.path.join(
                label_dir, filename+'_uv_position_map.npy')
            if os.path.exists(label_file):
                label_list.append(label_file)
            else:
                raise Exception('missing label for %s' % f)
        self.input_list, self.label_list = input_list, label_list
        self.batch_size = batch_size
        self._read_image()
        np.random.shuffle(self.data)
        valid_num = int(max(1, valid_percent*len(self.data)))
        self.valid_num = valid_num

    @property
    def train(self):
        return IterableBatchData(data=self.data[-self.valid_num:], batch_size=self.batch_size)

    @property
    def valid(self):
        return IterableBatchData(data=self.data[-self.valid_num:], batch_size=self.batch_size)

    def _read_image(self):
        self.images = np.array([self._input_transform(f)
                                for f in self.input_list])
        self.annotations = np.array([self._label_transform(f)
                                     for f in self.label_list])
        self.data = [(self.images[i, :, :, :], self.annotations[i, :, :, :])
                     for i in range(self.images.shape[0])]
        for item in self.data:
            image, annot = item
        print(self.images.shape)
        print(self.annotations.shape)

    def _label_transform(self, filename):
        image = np.load(filename)/256.
        return image.astype(np.float32)

    def _input_transform(self, filename):
        image = skimage.io.imread(filename)/255.
        return image.astype(np.float32)


class IterableBatchData(object):
    class Iterator(object):
        def __init__(self, ibd):
            self.index = 0
            self.batch_size = ibd.batch_size
            self.data = ibd.data
            self.length = ibd._length

        def __iter__(self):
            return self

        def next(self):
            return self.__next__()

        def __next__(self):
            index = self.index
            if index >= self.length:
                raise StopIteration()
            else:
                self.index = min(len(self.data),
                                 self.index + self.batch_size)
                ret = self.data[index:self.index]
                return [np.array([item[i] for item in ret]) for i in range(len(ret[0]))]

    def __init__(self, data, batch_size=32):
        self.batch_offset, self.batch_size = 0, batch_size
        self.data = data
        self._length = len(data)
        self._data_len = len(self.data[0])

    def get_next(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > self._length:
            # Shuffle the data
            # perm = np.arange(self._length)
            np.random.shuffle(self.data)
            start = 0
            self.batch_offset = self.batch_size
        end = self.batch_offset
        ret = self.data[start:end]
        # [(item[i] for i in range(self._data_len)) for item in ret]
        return [np.array([item[i] for item in ret]) for i in range(self._data_len)]

    def __iter__(self):
        return self.__class__.Iterator(self)

    def __len__(self):
        return self._length


def main():
    resolution_inp = 256
    resolution_op = 256
    MaxPos = resolution_inp*1.1

    # network type
    network = resfcn256(resolution_inp, resolution_op)

    # net forward
    mask_weight = np.load('../results/mask.npy')
    if len(mask_weight.shape) == 2:
        mask_weight = mask_weight[:, :, np.newaxis]
        mask_weight = np.repeat(mask_weight, 3, axis=2)

    constant_mask = tf.constant(mask_weight)
    x = tf.placeholder(
        tf.float32, shape=[None, resolution_inp, resolution_inp, 3])
    y_ = tf.placeholder(
        tf.float32, shape=[None, resolution_op, resolution_op, 3]
    )
    x_op = network(x, is_training=True)
    loss = tf.square(y_ - x_op)
    loss = tf.multiply(loss, constant_mask)
    loss = tf.reduce_mean(loss)
    loss_summary = tf.summary.scalar("entropy", loss)
    trainable_var = tf.trainable_variables()
    train_op, lr_op, global_step_op = train(loss, trainable_var)
    dataset = BatchData('../results/', batch_size=FLAGS['batch_size'])
    print('train length is {}, valid length is {}'.format(
        len(dataset.train), len(dataset.valid)))
    iterator = dataset.train
    valid = dataset.valid

    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=False))
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)
    if 'weight_file' in FLAGS and FLAGS['weight_file'] and os.path.exists(FLAGS['weight_file']+'.data-00000-of-00001'):
        tf.train.Saver(network.vars).restore(
            sess, FLAGS['weight_file'])

    saver = tf.train.Saver(max_to_keep=20)
    train_writer = tf.summary.FileWriter('./Data/logs/', sess.graph)
    valid_writer = tf.summary.FileWriter('./Data/logs/')
    start = FLAGS['start_step']
    end = FLAGS['end_step']
    loss_sum = valid_loss = 0.0
    itr = start
    while True:
        for (train_images, train_annotations) in iterator:
            itr += 1
            if itr >= end:
                break

            feed_dict = {x: train_images,
                         y_: train_annotations}
            net_out, _, train_loss, summary_str = sess.run(
                [x_op, train_op, loss, loss_summary], feed_dict=feed_dict)
            loss_sum += train_loss

            if itr % 100 == 0:
                valid_loss_sum = 0.0
                for i, (valid_images, valid_annotations) in enumerate(valid):
                    feed_dict = {
                        x: valid_images, y_: valid_annotations
                    }
                    result, valid_loss, valid_summary = sess.run(
                        [x_op, loss, loss_summary], feed_dict=feed_dict)
                    valid_loss_sum += valid_loss
                train_writer.add_summary(summary_str, itr)
                valid_writer.add_summary(valid_summary, itr)
                lr, global_step = sess.run([lr_op, global_step_op])
                print("Step: {}, Learning Rate: {:.8f}, Train_loss:{:.8f}, Valid_loss:{:.8f}".format(
                    global_step, lr, loss_sum/100, valid_loss_sum/i))
                loss_sum = 0.0

            if itr % 500 == 0:
                saver.save(sess, FLAGS['log_dir'] + "model.ckpt", itr)
        else:
            continue
        break

    saver.save(sess, FLAGS['log_dir'] + "model.ckpt", itr)


if __name__ == "__main__":
    main()
