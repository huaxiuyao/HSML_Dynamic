import csv
import pickle
import random

import numpy as np
import tensorflow as tf

tf.set_random_seed(1234)
from data_generator import DataGenerator
from maml import MAML
import ipdb
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet or mixture or multidataset')
flags.DEFINE_integer('test_dataset', -1,
                     'which dataset to be test: 0: bird, 1: texture, 2: aircraft, 3: fungi, -1 is test all')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_test_task', 1000, 'number of test tasks.')
flags.DEFINE_integer('test_epoch', -1, 'test epoch, only work when test start')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('update_batch_size_eval', 10,
                     'number of examples used for inner gradient test (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('num_groups', 1, 'number of groups.')
flags.DEFINE_integer('fix_embedding_sample', -1,
                     'if the fix_embedding sample is -1, all samples are used for embedding. Otherwise, specific samples are used')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('hidden_dim', 40, 'output dimension of task embedding')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('emb_loss_weight', 0.0, 'the weight of autoencoder')
flags.DEFINE_string('emb_type', 'sigmoid', 'sigmoid embedding')
flags.DEFINE_bool('no_val', False, 'if true, there are no validation set of Omniglot dataset')
flags.DEFINE_integer('tree_type', 1, 'select the tree type: 1 or 2')
flags.DEFINE_integer('task_embedding_num_filters', 32, 'number of filters for task embedding')
flags.DEFINE_string('task_embedding_type', 'rnn', 'rnn or mean')

## clustering information
flags.DEFINE_integer('cluster_layer_0', 4, 'number of clusters in the first layer')
flags.DEFINE_integer('cluster_layer_1', 2, 'number of clusters in the second layer')

## Online version
flags.DEFINE_bool('online_training', False, 'whether to train online or not')
flags.DEFINE_float('online_threshold', 1.5, 'the threshold of each method')
flags.DEFINE_integer('online_change', 8000, 'keep stable training')


## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_string('datadir', '/home/huaxiuyao/Data/', 'directory for datasets.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot


class HSML_Online():
    def __init__(self):
        if FLAGS.datasource in ['multidataset', 'mixture']:
            self.NUM_TEST_POINTS = FLAGS.num_test_task
        else:
            self.NUM_TEST_POINTS = 600
        self.clusters = FLAGS.cluster_layer_0

    def train(self, model, saver, exp_string, data_generator, resume_itr=0):
        param_dict = {}
        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        if FLAGS.datasource in ['sinusoid', 'mixture']:
            PRINT_INTERVAL = 1000
            TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
        else:
            PRINT_INTERVAL = 100
            TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

        if FLAGS.log:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, starting training.')

        if FLAGS.online_training:
            change_itr=0
            if model.classification:
                last_res = 0
            else:
                last_res = 9999

        prelosses, postlosses, embedlosses = [], [], []

        num_classes = data_generator.num_classes  # for classification, 1 otherwise

        for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
            feed_dict = {}
            if 'generate' in dir(data_generator):
                if FLAGS.datasource == 'sinusoid':
                    batch_x, batch_y, amp, phase = data_generator.generate()
                elif FLAGS.datasource == 'mixture':
                    batch_x, batch_y, para_func, sel_set = data_generator.generate(itr=itr)

                inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]  # b used for testing
                labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

            if itr < FLAGS.pretrain_iterations:
                input_tensors = [model.pretrain_op]
            else:
                input_tensors = [model.metatrain_op]

            input_tensors.extend(
                [model.summ_op, model.total_embed_loss, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

            result = self.sess.run(input_tensors, feed_dict)

            prelosses.append(result[-2])
            postlosses.append(result[-1])
            embedlosses.append(result[2])

            # online training criterion
            # if FLAGS.online_training and itr % check_interval == 0 and itr != 0 and itr != 25000:
            if FLAGS.online_training and itr % PRINT_INTERVAL == 0 and (
                    (np.mean(postlosses) > FLAGS.online_threshold * last_res and not model.classification) or (
                    model.classification and np.mean(postlosses) < FLAGS.online_threshold * last_res)) and ((itr-change_itr)>FLAGS.online_change):
                print(
                    'the old postlosses is {}, and the new postlosses is {}'.format(
                        last_res, np.mean(postlosses)))
                # if itr == 25100 and model.classification:
                tvars = tf.trainable_variables()
                tvars_vals = self.sess.run(tvars)
                for var, val in zip(tvars, tvars_vals):
                    # print(var.name, val)
                    param_dict[var.name] = val
                tf.reset_default_graph()
                self.clusters += 1
                model, saver, data_generator = self.construct_model()
                tf.train.start_queue_runners(self.sess)
                tf.global_variables_initializer().run()
                tvars = tf.trainable_variables()
                tvars_vals = self.sess.run(tvars)
                for var, val in zip(tvars, tvars_vals):
                    if var.name in param_dict:
                        self.sess.run(var.assign(param_dict[var.name]))
                # tvars = tf.trainable_variables()
                # tvars_vals = sess.run(tvars)
                # for var, val in zip(tvars, tvars_vals):
                #     print(var.name, val)
                change_itr = itr
                print(
                    'add a new cluster in the online training mode in epoch {}, the total number of cluster is {}'.format(
                        change_itr, self.clusters))

            if FLAGS.online_training and itr % PRINT_INTERVAL == 0 and itr != 0:
                last_res = np.mean(postlosses)
                print(last_res)


            if itr % SUMMARY_INTERVAL == 0:
                if FLAGS.log:
                    train_writer.add_summary(result[1], itr)

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                if itr < FLAGS.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                std = np.std(postlosses)
                ci95 = 1.96 * std / np.sqrt(PRINT_INTERVAL)
                print_str += ': preloss: ' + str(np.mean(prelosses)) + ', postloss: ' + str(
                    np.mean(postlosses)) + ', embedding loss: ' + str(np.mean(embedlosses)) + ', confidence: ' + str(
                    ci95)
                print(print_str)
                prelosses, postlosses, embedlosses = [], [], []

            if (itr != 0) and itr % SAVE_INTERVAL == 0:
                saver.save(self.sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

            # sinusoid is infinite data, so no need to test on meta-validation set.
            if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0 and (
                    FLAGS.datasource not in ['sinusoid', 'mixture']):
                if 'generate' not in dir(data_generator):
                    feed_dict = {}
                    if model.classification:
                        input_tensors = [model.metaval_total_accuracy1,
                                         model.metaval_total_accuracies2[FLAGS.num_updates - 1], model.summ_op]
                    else:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates - 1],
                                         model.summ_op]
                else:
                    if FLAGS.datasource == 'sinusoid':
                        batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                    elif FLAGS.datasource == 'mixture':
                        batch_x, batch_y, para_func = data_generator.generate(train=False)
                    inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                    inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
                    labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                    labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

                    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                                 model.meta_lr: 0.0}
                    if model.classification:
                        input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]]
                    else:
                        input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]]

                result = self.sess.run(input_tensors, feed_dict)
                print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

        saver.save(self.sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

    def test(self, model, saver, exp_string, data_generator, test_num_updates=None):
        num_classes = data_generator.num_classes  # for classification, 1 otherwise

        np.random.seed(1)
        random.seed(1)

        metaval_accuracies = []
        print(self.NUM_TEST_POINTS)
        for test_itr in range(self.NUM_TEST_POINTS):
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                feed_dict = {model.meta_lr: 0.0}
            else:
                if FLAGS.datasource == 'sinusoid':
                    batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                elif FLAGS.datasource == 'mixture':
                    batch_x, batch_y, para_func, sel_set = data_generator.generate(test_itr, train=False)

                inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                             model.meta_lr: 0.0}

            if model.classification:
                result = self.sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)

            else:  # this is for sinusoid
                result = self.sess.run([model.total_loss1] + model.total_losses2, feed_dict)

            metaval_accuracies.append(result)

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(self.NUM_TEST_POINTS)

        print('Mean validation accuracy/loss, stddev, and confidence intervals')
        print((means, stds, ci95))

    def construct_model(self):
        self.sess = tf.InteractiveSession()
        if FLAGS.train == False:
            orig_meta_batch_size = FLAGS.meta_batch_size
            # always use meta batch size of 1 when testing.
            FLAGS.meta_batch_size = 1

        if FLAGS.datasource in ['sinusoid', 'mixture']:
            data_generator = DataGenerator(FLAGS.update_batch_size + FLAGS.update_batch_size_eval,
                                           FLAGS.meta_batch_size)
        else:
            if FLAGS.metatrain_iterations == 0 and FLAGS.datasource in ['miniimagenet', 'multidataset']:
                assert FLAGS.meta_batch_size == 1
                assert FLAGS.update_batch_size == 1
                data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
            else:
                if FLAGS.datasource in ['miniimagenet', 'multidataset']:  # TODO - use 15 val examples for imagenet?
                    if FLAGS.train:
                        data_generator = DataGenerator(FLAGS.update_batch_size + 15,
                                                       FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                    else:
                        data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                                       FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                                   FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory

        dim_output = data_generator.dim_output

        dim_input = data_generator.dim_input

        if FLAGS.datasource in ['miniimagenet', 'omniglot', 'multidataset']:
            tf_data_load = True
            num_classes = data_generator.num_classes

            if FLAGS.train:  # only construct training model if needed
                random.seed(5)
                if FLAGS.datasource in ['miniimagenet', 'omniglot']:
                    image_tensor, label_tensor = data_generator.make_data_tensor()
                elif FLAGS.datasource == 'multidataset':
                    image_tensor, label_tensor = data_generator.make_data_tensor_multidataset(sel_num=self.clusters,
                                                                                             train=True)
                inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

            random.seed(6)
            if FLAGS.datasource in ['miniimagenet', 'omniglot']:
                image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
            elif FLAGS.datasource == 'multidataset':
                image_tensor, label_tensor = data_generator.make_data_tensor_multidataset(sel_num=self.clusters,
                                                                                         train=False)
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
        else:
            tf_data_load = False
            input_tensors = None

        model = MAML(self.sess, dim_input, dim_output, test_num_updates=self.test_num_updates)

        model.cluster_layer_0 = self.clusters

        if FLAGS.train or not tf_data_load:
            model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        if tf_data_load:
            model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
        model.summ_op = tf.summary.merge_all()
        saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

        if FLAGS.train == False:
            # change to original meta batch size when loading model.
            FLAGS.meta_batch_size = orig_meta_batch_size

        if FLAGS.train_update_batch_size == -1:
            FLAGS.train_update_batch_size = FLAGS.update_batch_size
        if FLAGS.train_update_lr == -1:
            FLAGS.train_update_lr = FLAGS.update_lr

        return model, saver, data_generator

    def main(self):
        if FLAGS.datasource in ['sinusoid', 'mixture']:
            if FLAGS.train:
                self.test_num_updates = 1
            else:
                self.test_num_updates = 10
        else:
            if FLAGS.datasource in ['miniimagenet', 'multidataset']:
                if FLAGS.train == True:
                    self.test_num_updates = 1  # eval on at least one update during training
                else:
                    self.test_num_updates = 10
            else:
                self.test_num_updates = 10

        model, saver, data_generator = self.construct_model()

        exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
            FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
            FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.emb_loss_weight' + str(
            FLAGS.emb_loss_weight) + '.num_groups' + str(FLAGS.num_groups) + '.emb_type' + str(
            FLAGS.emb_type) + '.hidden_dim' + str(FLAGS.hidden_dim)

        if FLAGS.num_filters != 64:
            exp_string += 'hidden' + str(FLAGS.num_filters)
        if FLAGS.max_pool:
            exp_string += 'maxpool'
        if FLAGS.stop_grad:
            exp_string += 'stopgrad'
        if FLAGS.norm == 'batch_norm':
            exp_string += 'batchnorm'
        elif FLAGS.norm == 'layer_norm':
            exp_string += 'layernorm'
        elif FLAGS.norm == 'None':
            exp_string += 'nonorm'
        else:
            print('Norm setting not recognized.')

        resume_itr = 0
        model_file = None

        tf.train.start_queue_runners(self.sess)
        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables))
        if FLAGS.resume or not FLAGS.train:
            if FLAGS.train == True:
                # model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
                model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
            else:
                print(FLAGS.test_epoch)
                model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1 + 5:])
                print("Restoring model weights from " + model_file)
                saver.restore(self.sess, model_file)

        if FLAGS.train:
            self.train(model, saver, exp_string, data_generator, resume_itr)
        else:
            self.test(model, saver, exp_string, data_generator, self.test_num_updates)

        print('The number of clusters are {}'.format(self.clusters))


if __name__ == "__main__":
    online_model = HSML_Online()
    online_model.main()
