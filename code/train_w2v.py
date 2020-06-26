import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from gensim.models import Word2Vec
from sklearn.utils import shuffle

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
logging.getLogger().setLevel(logging.INFO)

train_path = 'data/train_preliminary/'
test_path = 'data/test/'
w2v_model_path = './w2v_model/'

def initiate_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info('===================================')
    logger.info('开始执行时间: {}'.format(time.ctime()))
    logger.info('===================================')
    return logger

def merge_csv(target, logger=None):
    logger.info('start merge csv')
    df_ad = pd.read_csv(train_path + 'ad.csv')
    df_click_log = pd.read_csv(train_path + 'click_log.csv')

    df_ad.loc[df_ad['product_id'] == '\\N', 'product_id'] = 0
    df_ad.loc[df_ad['industry'] == '\\N', 'industry'] = 0

    df_merge_train = df_click_log.merge(df_ad, how='left', on='creative_id')

    df_ad_test = pd.read_csv(test_path + "ad.csv")
    df_click_log_test = pd.read_csv(test_path + "click_log.csv")

    df_ad_test.loc[df_ad_test['product_id'] == '\\N', 'product_id'] = 0
    df_ad_test.loc[df_ad_test['industry'] == '\\N', 'industry'] = 0

    df_merge_test = df_click_log_test.merge(df_ad_test, how='left', on='creative_id')

    df_merge_train[target] = df_merge_train[target].apply(str)
    df_merge_test[target] = df_merge_test[target].apply(str)

    df_merge_all = pd.concat([df_merge_train, df_merge_test], ignore_index=True)
    logger.info('end merge csv')

    return  df_merge_all


def train_w2v_model(target, embed_size, window_size, save_path, logger=None):

    df_concat = merge_csv(target,logger)
    sentences = df_concat.groupby(['user_id']).apply(lambda x: x[target].tolist()).tolist()
    sentences = shuffle(sentences)

    logger.info('start train w2vmodel:'+ target + ' embed_size' + str(embed_size) + ' window_size' + str(window_size))
    model = Word2Vec(sentences=sentences, size=embed_size, min_count=1, sg=1, hs=1, window=window_size, workers=12, seed=2020,
                     iter=10)
    logger.info('end train w2vmodel')

    model.save(save_path)
    logger.info('save w2vmodel successful:'+save_path)


if __name__=='__main__':
    assert (len(sys.argv)==4)
    target, embed_size, window_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    if not os.path.exists(w2v_model_path):
        os.makedirs(w2v_model_path)

    logger = initiate_logger('train_w2v.log')
    save_path = w2v_model_path+'word2vec_'+ target +'_'+str(embed_size)+'_win'+ str(window_size)+'.model'

    train_w2v_model(target, embed_size,window_size,save_path,logger)
