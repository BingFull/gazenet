# Imports
import tensorflow as tf
import cv2
import numpy as np
import sonnet as snt

# Import for us of the transform layer and loss function
import dsnt

# For the Sonnet Module
# from dsnt_snt import DSNT


img_size = 150
image_count = 200
train_percent = 0.75
train_image_count = int(train_percent * image_count)
test_image_count = image_count - train_image_count

images = []
targets = []
for _ in range(200):
    img = np.zeros((img_size, img_size, 3))
    row, col = np.random.randint(0, img_size), np.random.randint(0, img_size)
    radius = np.random.randint(8, 15)
    b, g, r = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
    cv2.circle(img, (row, col), radius, (b, g, r), -1)
    images.append(img)
    norm_row = row / img_size
    norm_col = col / img_size
    targets.append([norm_row, norm_col])

images = np.array(images)
targets = np.array(targets)
train_images = images[:train_image_count]
test_images = images[train_image_count:]
train_targets = targets[:train_image_count]
test_targets = targets[train_image_count:]

print('''
{} images total
training: {}
testing : {}'''.format(image_count, train_image_count, test_image_count))


def inference(inputs):
    inputs = snt.Conv2D(output_channels=166,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv1')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=32,
                        kernel_shape=3,
                        rate=2,
                        padding='SAME',
                        name='conv2')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=64,
                        kernel_shape=3,
                        rate=4,
                        padding='SAME',
                        name='conv3')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=128,
                        kernel_shape=3,
                        rate=8,
                        padding='SAME',
                        name='conv4')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=256,
                        kernel_shape=3,
                        rate=16,
                        padding='SAME',
                        name='conv5')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=256,
                        kernel_shape=3,
                        padding='SAME',
                        name='conv6')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=1,
                        kernel_shape=1,
                        padding='SAME',
                        name='conv7')(inputs)
    coords, norm_heatmap = dsnt.dsnt(inputs)

    # The Sonnet option
    # coords, norm_heatmap = DSNT()(inputs)
    return coords, norm_heatmap

tf.reset_default_graph()

input_x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
input_y = tf.placeholder(tf.float32, shape=[None, 2])

heatmaps, predictions = inference(input_x)
# The predictions are in the range [-1, 1] but I prefer to work with [0, 1]
predictions = (predictions + 1) / 2

# Coordinate regression loss
loss_1 = tf.losses.mean_squared_error(input_y, predictions)
# Regularization loss
loss_2 = dsnt.js_reg_loss(heatmaps, input_y)
loss = loss_1 + loss_2

optimizer = tf.train.AdamOptimizer(learning_rate=6e-5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for i in range(train_image_count):
            curr_img = train_images[i]
            curr_target = train_targets[i]
            _, loss_val = sess.run(
                [optimizer, loss],
                {
                    input_x: [curr_img],
                    input_y: [curr_target]
                }
            )

    def evaluate_total_mse(images, targets):
        '''
        Evaluate the mean-squared-error across the whole given batch of images, targets
        '''
        total_loss = 0
        image_count = images.shape[0]
        for i in range(image_count):
            curr_img = images[i]
            curr_target = targets[i]
            loss_val = sess.run(loss_1, {
                input_x: [curr_img],
                input_y: [curr_target]
            })
            total_loss += loss_val
        return total_loss / image_count

    print("Training MSE: {:.5f}".format(evaluate_total_mse(train_images, train_targets)))
    print("Testing MSE : {:.5f}".format(evaluate_total_mse(test_images, test_targets)))