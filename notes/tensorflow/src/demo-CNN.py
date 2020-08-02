import tensorflow as tf
from zhmh.dataset.picture import generate_random_rgb_pictures, show_rgb_picture
from zhmh.tf import generate_one_conv

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
SAMPLE_SIZE = 1

# 生成模拟数据
DATA = generate_random_rgb_pictures(IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLE_SIZE)
show_rgb_picture(DATA[0])

# 生成卷积网络
place_x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
y = cnn_filter = generate_one_conv(
    place_x, deep=(3, 9),
    filter_shape=(2, 2), filter_step=(2, 2),
    pool_shape=(2, 2), pool_step=(2, 2),
    w_initialize=1,
    b_initialize=0.001
)

# 查看卷积效果
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    y_val = sess.run(y, feed_dict={place_x: DATA})
    print("y: \n", y_val)
    print(y_val.shape)
    show_rgb_picture(y_val[0])
    show_rgb_picture(y_val[0][:, :, 0:3])
    show_rgb_picture(y_val[0][:, :, 3:6])
    show_rgb_picture(y_val[0][:, :, 6:9])
