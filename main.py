import os
import numpy as np
import argparse

from train import train
from config import cfg
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument(
    '--middle_start',
    type=bool,
    default=False,
    help='Starting from the middle')
parser.add_argument(
    '--discriminative', type=bool, default=False, help='Discriminative or not')
parser.add_argument('--epoch', type=int, default=15, help='Epoch to train')
parser.add_argument(
    '--batch_size_train', type=int, default=cfg.CONST.BATCH_SIZE_TRAIN, help='Batch size')
parser.add_argument(
    '--batch_size_test', type=int, default=cfg.CONST.BATCH_SIZE_TEST, help='Batch size')
parser.add_argument('--ini_epoch', type=int, default=0, help='Initial epoch')
parser.add_argument(
    '--conf_epoch',
    type=int,
    default=10000,
    help='Confirmation epoch to evaluate interpolate, reconstruction')
parser.add_argument(
    '--mode', type=str, default='train', help='train or validate')
parser.add_argument(
    '--data_list',
    type=str,
    default='./train_3rscan.list',
    help='train or validate')
parser.add_argument(
    '--learning_rate_G',
    type=float,
    default=cfg.LEARNING_RATE_G,
    help='Learning rate for Generator of Adam')
parser.add_argument(
    '--learning_rate_D',
    type=float,
    default=cfg.LEARNING_RATE_D,
    help='Learning rate for Discriminator of Adam')

FLAGS = parser.parse_args()


def main():
    if not os.path.exists(cfg.DIR.CHECK_POINT_PATH):
        os.makedirs(cfg.DIR.CHECK_POINT_PATH)
    if not os.path.exists(cfg.DIR.TRAIN_OBJ_PATH):
        os.makedirs(cfg.DIR.TRAIN_OBJ_PATH)
    if not os.path.exists(cfg.DIR.EVAL_PATH):
        os.makedirs(cfg.DIR.EVAL_PATH)
    if FLAGS.middle_start:
        print('middle_start')

    if FLAGS.mode == 'train':
        train(FLAGS.epoch, FLAGS.learning_rate_G, FLAGS.learning_rate_D,
              FLAGS.batch_size_train, FLAGS.middle_start, FLAGS.ini_epoch,
              FLAGS.discriminative, FLAGS.data_list)
    elif FLAGS.mode == 'evaluate_recons' or 'evaluate_interpolate' or 'evaluate_noise':
        from evaluate import evaluate
        if FLAGS.mode == 'evaluate_recons':
            mode = 'recons'
        elif FLAGS.mode == 'evaluate_interpolate':
            mode = 'interpolate'
        else:
            mode = 'noise'
        evaluate(FLAGS.batch_size_test, FLAGS.conf_epoch, mode, FLAGS.discriminative,
                 FLAGS.data_list)


if __name__ == '__main__':
    main()
