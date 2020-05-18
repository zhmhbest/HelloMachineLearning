import tensorflow as tf
from com.zhmh.tf import generate_random_rgb_pictures, generate_one_conv, show_rgb_picture

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
SAMPLE_SIZE = 1

DATA = generate_random_rgb_pictures(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLE_SIZE)
show_rgb_picture(DATA[0])
input_x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
cnn_filter = generate_one_conv(
    input_x, deep=(3, 9),
    filter_shape=2, filter_step=2,
    pool_shape=2, pool_step=2
)
y = cnn_filter['y']

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    y = sess.run(y, feed_dict={input_x: DATA})
    print("y: \n", y)
    print(y.shape)
    show_rgb_picture(y[0])
    show_rgb_picture(y[0][:, :, 0:3])
    show_rgb_picture(y[0][:, :, 3:6])
    show_rgb_picture(y[0][:, :, 6:9])
