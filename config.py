#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''

class TrainConfig(object):
    WORD_EMBEDDING_DIM = 50
    IDF_NUM = 3
    TRAIN_EPOCHS = 5
    W2V_MIN_COUNT = 5
    LOSS_FUNC = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    CONV_K_SIZE_L1 = 2

class EvaluationConfig(object):
    IDF_WEIGHT = 0.5

class ExececutionConfig(object):
    TRAIN_CASE = "training"
    TEST_CASE = "both"   # "both" # "training" #"randomed_labeled_testing" # "develop"
    USE_TRAINED = False # Continue training based on previous model. Only matters when TRAIN == True.
    PREPROCESS = False
    TRAIN = True
    EVALUATE = True
    INIT_MODEL = False # Use a initialized model for testing. Only matters when TRAIN == False.
