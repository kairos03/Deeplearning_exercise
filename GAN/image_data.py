import os

import tensorflow as tf


def read_data_sets(data_path):
    images = []

    # data_path
    data_path = data_path + '/' if data_path[-1] != '/' else data_path

    # image list queue
    image_list = [os.path.join(data_path, s) for s in os.listdir(data_path)]
    image_queue = tf.train.string_input_producer(image_list, shuffle=False)

    # read
    reader = tf.WholeFileReader()
    filename, content = reader.read(image_queue)

    # decode, resize, standardization
    image = tf.image.decode_jpeg(content, channels=1)
    image = tf.image.resize_image_with_crop_or_pad(image, 100, 100)
    # image = tf.image.per_image_standardization(image)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        raw = [sess.run([image]) for _ in range(len(image_list))]
        images = [f[0] for f in raw]

        coord.request_stop()
        coord.join(threads)

    return InputData(images, len(image_list))


class InputData:
    def __init__(self, images, num):
        self.images = images
        self.num_images = num
        self.offset = 0

    def next_batch(self, batch_size):

        start = batch_size * self.offset
        end = batch_size * (self.offset + 1)
        if end < self.num_images:
            start = 0
            end = batch_size
            self.offset = 0

        batch = self.images[start:end]

        self.offset += 1
        return batch

    def next(self):
        return self.images[:]
