import tensorflow as tf

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                reduction=tf.keras.losses.Reduction.NONE)

def generator_loss(disc_op):
    return binary_cross_entropy(tf.ones_like(disc_op), disc_op)

def disc_loss(real_op, fake_op):
    return 0.5 * (binary_cross_entropy(tf.ones_like(real_op), real_op) + 
                  binary_cross_entropy(tf.zeros_like(fake_op), fake_op))

def similar_loss(real_img, similar_img, LAMBDA):
    return tf.reduce_mean(tf.abs(real_img - similar_img)) * LAMBDA