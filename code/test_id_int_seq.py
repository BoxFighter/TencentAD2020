import pandas as pd
import logging
import sys
import os
import time
from gensim.models import Word2Vec
from tqdm import tqdm

train_path = 'data/train_preliminary/'
test_path = 'data/test/'
id_seq_path = './id_seq_path/'
w2v_model_path = './w2v_model/'
save_name = 'df_all_id_seq_test.csv'
feat = ['creative_id','ad_id','advertiser_id','product_id','product_category','industry']
# feat = ['creative_id','ad_id','advertiser_id','product_id']

def initiate_logger(log_path):
    """
    Initialize a logger with file handler and stream handler
    """
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

def load_w2v_index(feat_name):
    if feat_name == 'product_category' or feat_name == 'industry':
        path = w2v_model_path + 'word2vec_' + feat_name + '_64_win100.model'
    else:
        path = w2v_model_path + 'word2vec_' + feat_name + '_128_win100.model'
    logger.info('start load:' + path)

    w2v_model = Word2Vec.load(path)
    print(w2v_model)
    vocab_list = [word for word, Vocab in w2v_model.wv.vocab.items()]
    word_index = {" ": 0}
    # word_vector = {}
    # embedding_matrix = np.zeros((len(vocab_list) + 1, w2v_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：索引
        # word_vector[word] = w2v_model.wv[word] # 词语：词向量
        # embedding_matrix[i + 1] = w2v_model.wv[word]  # 词向量矩阵
    # print(embedding_matrix.shape)

    return word_index

def get_id_seq(df_merge_train,feat,logger=None):
    df_id_seq = pd.DataFrame(df_merge_train.groupby(['user_id']).apply(lambda x: x[feat].tolist())).reset_index()
    df_id_seq.columns = ['user_id', feat + '_seq']

    word_index = load_w2v_index(feat)
    logger.info(feat + ' word_index ready')

    result=[]
    hit=0
    miss=0
    for row in tqdm(df_id_seq[['user_id',feat+'_seq']].values,total=len(df_id_seq)):
        try:
            result.append([row[0],[word_index[str(i)]  for i in row[-1]]])
            hit+=1
        except Exception as e:
            miss+=1
    logger.info(feat + ' hit:'+str(hit)+' miss:'+str(miss))

    df_int_seq  = pd.DataFrame(result,columns=['user_id',feat+'_int_seq'])

    maxlen = 100
    for i, v in df_int_seq[feat+'_int_seq'].items():
        if len(v)>maxlen:
            df_int_seq[feat+'_int_seq'][i] = v[0:maxlen]

    logger.info(feat + ' sequence data ready')
    return df_int_seq


if __name__=='__main__':
    logger = initiate_logger('input_generate.log')
    if not os.path.exists(id_seq_path):
        os.makedirs(id_seq_path)

    df_ad_test = pd.read_csv(test_path + 'ad.csv')
    df_click_log_test = pd.read_csv(test_path + 'click_log.csv')

    df_ad_test.loc[df_ad_test['product_id'] == '\\N', 'product_id'] = 0
    df_ad_test.loc[df_ad_test['industry'] == '\\N', 'industry'] = 0

    df_merge_test = df_click_log_test.merge(df_ad_test, how='left', on='creative_id')

    df_id_seq = []

    for f in feat:
        logger.info(f + ' start handling')
        df_id_seq.append(get_id_seq(df_merge_train=df_merge_test, feat=f, logger=logger))

    df_res = None
    logger.info('start to merge')
    for i in range(len(df_id_seq)):
        if i == 0:
            df_res = df_id_seq[i]
        else:
            df_res = df_res.merge(df_id_seq[i], how='left', on='user_id')

    df_res.to_csv(id_seq_path + save_name, index=True)
    logger.info('end')