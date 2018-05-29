'''#**--clone_on_cpu=True
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset_name', type=str, default='quiz')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='inception_v4')
    parser.add_argument('--checkpoint_exclude_scopes', type=str, default='InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clone_on_cpu', type=bool, default=False)
 
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--batch_size', type=int, default=32)

    # eval
    parser.add_argument('--dataset_split_name', type=str, default='validation')
    parser.add_argument('--eval_dir', type=str, default='validation')
    parser.add_argument('--max_num_batches', type=int, default=128)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed



#train_cmd = 'python ./train_image_classifier.py  --dataset_name={dataset_name} --dataset_dir={dataset_dir} --model_name={model_name} --checkpoint_exclude_scopes={checkpoint_exclude_scopes} --train_dir={train_dir} --learning_rate={learning_rate} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
#eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'
train_cmd = 'python ./train_image_classifier.py  --dataset_name={dataset_name} --dataset_dir={dataset_dir} --model_name={model_name} --checkpoint_exclude_scopes={checkpoint_exclude_scopes} --train_dir={train_dir} --learning_rate={learning_rate} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    step_per_epoch = 50000 // FLAGS.batch_size

    if FLAGS.checkpoint_path:
        ckpt = ' --checkpoint_path=' + FLAGS.checkpoint_path
    else:
        ckpt = ''
    for i in range(30):
        steps = int(step_per_epoch * (i + 1))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                         'model_name': FLAGS. model_name,
                                         'checkpoint_exclude_scopes': FLAGS.checkpoint_exclude_scopes, 'train_dir': FLAGS. train_dir,
                                         'learning_rate': FLAGS.learning_rate, 'optimizer': FLAGS.optimizer,
                                         'batch_size': FLAGS.batch_size, 'max_number_of_steps': steps, 'clone_on_cpu': FLAGS.clone_on_cpu}) + ckpt)
        for l in p:
            print(p.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'dataset_name': FLAGS.dataset_name, 'dataset_dir': FLAGS.dataset_dir,
                                        'dataset_split_name': 'validation', 'model_name': FLAGS. model_name,
                                        'checkpoint_path': FLAGS.train_dir, 'batch_size': FLAGS.batch_size,
                                        'eval_dir': FLAGS. eval_dir, 'max_num_batches': FLAGS. max_num_batches}))
        for l in p:
            print(p.strip())
'''
"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

def transition(net, num_outputs, scope='transition'):
    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=scope + '_conv1x1')
    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=scope + '_avgpool')
    return net
def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    #growth = 24
    growth = 32
    compression_rate = 0.5

    def reduce_dim(input_feature):

        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # Put your code here.
            #这里我使用论文里提到的dense121网络结构
            print(images.shape)
            net = images
            net = slim.conv2d(net, 2 * growth, 7, stride=2, scope='conv1')
            net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')

            net = block(net, 2, growth, scope='block1')
            net = transition(net, reduce_dim(net), scope='transition1')

            net = block(net, 4, growth, scope='block2')
            net = transition(net, reduce_dim(net), scope='transition2')

            net = block(net, 6, growth, scope='block3')
            net = slim.batch_norm(net, scope='last_batch_norm_relu')
            net = tf.nn.relu(net)

            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool2', keep_dims=True)

            biases_initializer = tf.constant_initializer(0.1)
            net = slim.conv2d(net, num_classes, [1, 1], biases_initializer=biases_initializer, scope='logits')

            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            print(logits)
            end_points['Logits'] = logits
            end_points['predictions'] = slim.softmax(logits, scope='predictions')
            print("#############end#############")
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
