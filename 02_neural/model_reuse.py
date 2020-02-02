import os
import tensorflow as tf
MODEL_STORAGE_DIR = './dump'


def normal_save(model_location):
    """
    保存普通模型
    :return:
    """
    input_x = tf.placeholder(tf.float32, (None, 2), name='input_x')
    weight = tf.Variable(0.5, dtype=tf.float32, name="weight")
    result = input_x[:, 0] * input_x[:, 1] * weight
    # print(result)

    # Saver必须在声明变量之后才能实例化
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("保存计算两个变量和的模型", sess.run(result, feed_dict={input_x: [
            [1, 2],
            [2, 3],
            [3, 4]
        ]}))
        saver.save(sess, model_location)


def normal_load(model_location):
    """
    自动加载图上的所有变量，不用再自定义变量
    :return:
    """
    saver = tf.train.import_meta_graph(model_location + '.meta')
    g = tf.get_default_graph()
    input_x = g.get_tensor_by_name('input_x:0')
    # weight = g.get_tensor_by_name('weight:0')
    result = g.get_tensor_by_name('mul_1:0')
    with tf.Session() as sess:
        saver.restore(sess, model_location)
        print(sess.run(result, feed_dict={
            input_x: [
                [2, 3],
                [4, 5]
            ]
        }))


def ema_save(model_location):
    """
    保存滑动平均模型
    :return:
    """
    v1 = tf.Variable(1.1, dtype=tf.float32, name="v1")
    v2 = tf.Variable(2.2, dtype=tf.float32, name="v2")
    result = v1 + v2

    for variables in tf.global_variables():
        print(variables.name, end=',')
    print()

    ema = tf.train.ExponentialMovingAverage(0.99)
    maintain_op = ema.apply(tf.global_variables())
    for variables in tf.global_variables():
        print(variables.name, end=',')
    print()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print('Before', sess.run([v1, ema.average(v1)]), sess.run([v2, ema.average(v2)]))
        sess.run(tf.assign(v1, 11))  # 修改变量v1
        sess.run(tf.assign(v2, 22))  # 修改变量v2
        sess.run(maintain_op)        # 修改影子变量
        print('After', sess.run([v1, ema.average(v1)]), sess.run([v2, ema.average(v2)]))

        saver.save(sess, model_location)


def ema_load(model_location):
    saver = tf.train.import_meta_graph(model_location + '.meta')
    g = tf.get_default_graph()
    v1 = g.get_tensor_by_name('v1:0')
    v2 = g.get_tensor_by_name('v2:0')
    ema_v1 = g.get_tensor_by_name('v1/ExponentialMovingAverage:0')
    ema_v2 = g.get_tensor_by_name('v2/ExponentialMovingAverage:0')
    with tf.Session() as sess:
        saver.restore(sess, model_location)
        print(sess.run([v1, v2]))
        print(sess.run([ema_v1, ema_v2]))


if __name__ == '__main__':
    normal_model = MODEL_STORAGE_DIR + '/normal.ckpt'

    if os.path.exists(normal_model + '.meta'):
        normal_load(normal_model)
    else:
        normal_save(normal_model)

    tf.reset_default_graph()

    ema_model = MODEL_STORAGE_DIR + '/ema.ckpt'
    if os.path.exists(ema_model + '.meta'):
        ema_load(ema_model)
    else:
        ema_save(ema_model)
