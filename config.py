#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''

class TrainConfig(object):
    WORD_EMBEDDING_DIM = 50
    IDF_NUM = 3
    TRAIN_EPOCHS = 10
    W2V_MIN_COUNT = 5
    LOSS_FUNC = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    CONV_K_SIZE = 2

class EvaluationConfig(object):
    IDF_WEIGHT = 0.5

class ExececutionConfig(object):
    TRAIN_CASE = "training"
    TEST_CASE = "randomed_labeled_testing"   # "training" #"randomed_labeled_testing" # "develop"
    USE_TRAINED = True # Continue training based on previous model.
    PREPROCESS = False
    TRAIN = True
    EVALUATE = True
    INIT_MODEL = False # Only matters when TRAIN == False.
