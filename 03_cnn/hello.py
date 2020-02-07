import tensorflow as tf
from com.zhmh.tf import generate_random_rgb_pictures, generate_one_conv

IMAGE_WIDTH = 8
IMAGE_HEIGHT = 8
SAMPLE_SIZE = 1

DATA = generate_random_rgb_pictures(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLE_SIZE)
input_x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
cnn_filter = generate_one_conv(
    input_x, deep=(3, 10),
    filter_shape=(2, 2), filter_step=(2, 2),
    pool_shape=(2, 2), pool_step=(2, 2)
)
y = cnn_filter['y']

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    y = sess.run(y, feed_dict={input_x: DATA})
    print("y: \n", y, y.shape)
