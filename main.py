import os
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 100
BATCH_SIZE = 5
FREEZE = False
STDDEV = 0.001
LEARNING_RATE = 0.0001
# Adam Optimizer already uses interally learning rate decay!
LEARNING_RATE_DECAY = True
DECAY_STEPS = 116
DECAY_RATE = 0.96
KEEP_PROB = 0.5
BETA = 1.e-3
BATCH_NORMALIZATION = False
BN_DECAY = 0.999


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load MetaGraphDef
    meta_graph_def = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # get desired tensors
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers_no_bn(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, 
                 freeze=False, is_training=None, keep_prob=None):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # Note: The fully connected layers of the original VGG-16 network have
    # already been replaced by 1x1 convolutions, i.e. the loaded version of 
    # the VGG-16 model is already a Fully Convolutional Network (FCN-8)
    # However, the number of output nodes has to be reduced
    
    if freeze:
        assert not is_training is None
        assert not keep_prob is None
        # apply dropout
        vgg_layer7_out = tf.contrib.layers.dropout(vgg_layer7_out, 
                                        keep_prob, is_training=is_training)
        vgg_layer4_out = tf.contrib.layers.dropout(vgg_layer4_out, 
                                        keep_prob, is_training=is_training)
        vgg_layer3_out = tf.contrib.layers.dropout(vgg_layer3_out, 
                                        keep_prob, is_training=is_training)
        
    # Decoders: reduce number of classes (there are only 2: road and not-road)
    layer7_red = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, 
                kernel_size=1, strides=(1, 1), padding="SAME", 
                kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA), 
                name='fcn1')

    layer4_red = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, 
                kernel_size=1, strides=(1, 1), padding="SAME", 
                kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA), 
                name='fcn2')
    layer3_red = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, 
                kernel_size=1, strides=(1, 1), padding="SAME", 
                kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA), 
                name='fcn3')
                
    # Create Decoder / Upsampling by 32 (2x2x8)
    x = tf.layers.conv2d_transpose(layer7_red, filters=num_classes, 
                kernel_size=4, strides=(2, 2), padding="SAME", 
                kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(BETA), 
                name='fcn4')
    
    # skip layers: add scaled output from layer 4
    x = tf.add(x, layer4_red, name="add_layer4")
    # next upsampling step
    x = tf.layers.conv2d_transpose(x, filters=num_classes, kernel_size=4, 
                                   strides=(2, 2), padding="SAME", 
                                   kernel_initializer=
                                   tf.random_normal_initializer(stddev=STDDEV),
                                   kernel_regularizer=
                                   tf.contrib.layers.l2_regularizer(BETA), 
                                   name='fcn5')
    
    # skip layers: add scaled output from layer 3
    x = tf.add(x, layer3_red, name="add_layer3")
    x = tf.layers.conv2d_transpose(x, filters=num_classes, kernel_size=16, 
                                   strides=(8, 8), padding="SAME", 
                                   kernel_initializer=
                                   tf.random_normal_initializer(stddev=STDDEV),
                                   kernel_regularizer=
                                   tf.contrib.layers.l2_regularizer(BETA), 
                                   name='fcn6')
    return x


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, 
           freeze=False, batch_normalization=False, is_training=None, 
           keep_prob=None):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # Note: The fully connected layers of the original VGG-16 network have
    # already been replaced by 1x1 convolutions, i.e. the loaded version of 
    # the VGG-16 model is already a Fully Convolutional Network (FCN-8)
    # However, the number of output nodes has to be reduced

    if batch_normalization:
        assert not is_training is None
        
        # Batch Normalization Parameters
        batch_norm = tf.contrib.layers.batch_norm
        bn_params = {'is_training': is_training,
                     'decay': BN_DECAY,
                     'updates_collections': None}
    else:
        batch_norm = None
        bn_params = None

    with tf.contrib.framework.arg_scope(
            [tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose],
            num_outputs=num_classes, padding="SAME", 
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(stddev=STDDEV),
            weights_regularizer=tf.contrib.layers.l2_regularizer(BETA),
            normalizer_fn=batch_norm, normalizer_params=bn_params):

        if freeze:
            assert not keep_prob is None
            # apply dropout
            vgg_layer7_out = tf.contrib.layers.dropout(vgg_layer7_out, 
                                            keep_prob, is_training=is_training)
            vgg_layer4_out = tf.contrib.layers.dropout(vgg_layer4_out, 
                                            keep_prob, is_training=is_training)
            vgg_layer3_out = tf.contrib.layers.dropout(vgg_layer3_out, 
                                            keep_prob, is_training=is_training)
            
        # Decoders: reduce number of classes (there are only 2: road and not-road)
        layer7_red = tf.contrib.layers.conv2d(vgg_layer7_out, kernel_size=1, 
                                              stride=1, scope="fcn1")
        layer4_red = tf.contrib.layers.conv2d(vgg_layer4_out, kernel_size=1, 
                                              stride=1, scope="fcn2")
        layer3_red = tf.contrib.layers.conv2d(vgg_layer3_out, kernel_size=1, 
                                              stride=1, scope="fcn3")
                    
        # Create Decoder / Upsampling by 32 (2x2x8)
        x = tf.contrib.layers.conv2d_transpose(layer7_red, kernel_size=4, 
                                               stride=2, scope="fcn4")
        
        # skip layers: add scaled output from layer 4
        x = tf.add(x, layer4_red, name="add_layer4")
        # next upsampling step
        x = tf.contrib.layers.conv2d_transpose(x, kernel_size=4, stride=2, 
                                               scope="fcn5")
        
        # skip layers: add scaled output from layer 3
        x = tf.add(x, layer3_red, name="add_layer3")
        x = tf.contrib.layers.conv2d_transpose(x, kernel_size=16, stride=8, 
                                               scope="fcn6")
    return x

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, 
             global_step=None, freeze=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # get logits
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    # calculate cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                                        labels=correct_label, logits=logits)
    # calculate loss
    loss_op = tf.reduce_mean(cross_entropy)
    # add regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op = tf.add_n([loss_op] + reg_losses)
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if freeze:
        # freeze original weight (only update new weights/biases)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='fcn[123456]')
        train_op = optimizer.minimize(loss_op, global_step=global_step, 
                                      var_list=train_vars)
    else:
        train_op = optimizer.minimize(loss_op, global_step=global_step)
        
    return logits, train_op, loss_op
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
             cross_entropy_loss, input_image, correct_label, keep_prob, 
             learning_rate, freeze=False, batch_normalization=False, 
             is_training=None, layer3_out=None, layer4_out=None, 
             layer7_out=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
                           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    
    if freeze:
        for arg in (is_training, layer3_out, layer4_out, layer7_out):
            assert not arg is None

        l3_results, l4_results, l7_results = [], [], []
        for X_batch, y_batch, batch_idcs in get_batches_fn(batch_size):
            l3, l4, l7 = sess.run([layer3_out, layer4_out, layer7_out], 
                                  feed_dict={input_image: X_batch, keep_prob: 1.0})
            l3_results.append(l3)
            l4_results.append(l4)
            l7_results.append(l7)
            
        l3_results = np.concatenate(l3_results, axis=0)
        l4_results = np.concatenate(l4_results, axis=0)
        l7_results = np.concatenate(l7_results, axis=0)
        
    if batch_normalization:
        assert not is_training is None
    
    last_mean = 1.0e10
    count = 0
    for epoch in range(epochs):
        print("Epoch:", epoch+1, end="  ")
        i = 0
        mean_loss = 0
        for batch in get_batches_fn(batch_size):
            i += 1
            X_batch, y_batch = batch[:2]
            feed_dict = {input_image: X_batch, correct_label: y_batch, 
                         keep_prob: KEEP_PROB}
            
            if freeze:
                batch_idcs = batch[2]
                feed_dict[layer3_out] = l3_results[batch_idcs]
                feed_dict[layer4_out] = l4_results[batch_idcs]
                feed_dict[layer7_out] = l7_results[batch_idcs]
                feed_dict[is_training] = True
                
            if batch_normalization:
                feed_dict[is_training] = True
                
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict=feed_dict)
            mean_loss += loss
        mean_loss /= i
        if mean_loss - last_mean < 0.0:
            count = 0
        else:
            count += 1
            if count > 5:
                print("Mean loss:", mean_loss, " Last loss:", loss)
                print("No improvements in the last 5 epochs. "
                      "Breaking up training ...")
                break
            
        last_mean = mean_loss
        print("Mean loss:", mean_loss, " Last loss:", loss)

tests.test_train_nn(train_nn)



def run():
    num_classes = 2
    image_shape = (160, 576)
    height, width = image_shape
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
                os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # create placeholders
        correct_label = tf.placeholder(tf.float32,
                                      [None, height, width, num_classes],
                                      name="correct_label")
        
        # Learning Rate Decay Parameters
        if LEARNING_RATE_DECAY:
            global_step = tf.Variable(0, trainable=False, name="global_step") # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(LEARNING_RATE, 
                                        global_step, DECAY_STEPS, DECAY_RATE, 
                                        staircase=True, name="learning_rate")
        else:
            global_step = None
            learning_rate = tf.constant(LEARNING_RATE, dtype=tf.float32, 
                                        name="learning_rate")
            
        if FREEZE or BATCH_NORMALIZATION:
            is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        else:
            is_training = None

        # load vgg-16 network
        image_input, keep_prob, layer3_out, layer4_out, layer7_out \
                                                    = load_vgg(sess, vgg_path)
                                                    
        # get layers
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes, 
                        FREEZE, BATCH_NORMALIZATION, is_training, keep_prob)

        logits, train_op, loss_op = optimize(nn_last_layer, correct_label, 
                            learning_rate, num_classes, global_step, FREEZE)
        
        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss_op, 
                 image_input, correct_label, keep_prob, learning_rate, FREEZE, 
                 BATCH_NORMALIZATION, is_training, layer3_out, layer4_out, 
                 layer7_out)
            
        # TODO: Save inference data using helper.save_inference_samples
        saver = tf.train.Saver()
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, 
                            logits, keep_prob, image_input, is_training, saver)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
