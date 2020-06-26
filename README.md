# TencentAD2020
腾讯广告算法大赛2020初赛

## 赛题描述
根据一组用户在长度为 91 天 (3 个月) 的时间窗口内的广 告点击历史记录作为训练数据集。每条记录中包含了日期 (从 1 到 91)、用户信息 (年龄，性别)，被点 击的广告的信息 (素材 id、广告 id、产品 id、产品类目 id、广告主 id、广告主行业 id 等)，以及该用 户当天点击该广告的次数，来预测用户的年龄和性别。

## 解决方案
由于总 creative_id 有 444w，若是直接做 onehot，然后 embedding，参数过大，很难进行训练。因此 我们使用 Word2vec 的思想，将每个用户历史点击记录看成句子，例如将访问过的所有 creative_id 当 作一个“句子”，每个广告为一个“单词”，对于用户 1234 来说:[’821396’, ’209778’, ’877468’, ’1683713’, ’122032’, ...]。这样我们总共可以形成 90w 个长度不等的序列，然后通过 Word2vec 模型进行训练(自 监督模型)，每个 creative_id 可以形成一个 K 维的向量。先把 id 看成一个词，然后把 id 拼接起来连 成一个句子，于是任务就变成了一个脱敏场景下的多输入文本分类问题。
### 网络结构设计
基于 LSTM 的多类文本输入分类模型网络结构:
整个模型由 4 部分构成:输入层、LSTM 层、池化层、全连接层。

1. 输入层(词嵌入层):

我选了’creative_id’,’ad_id’,’advertiser_id’,’product_id’ 作为输入文本。 使用预训练模型，即预先训练各个 id 的 word2vec 模型，将预训练的 word2vec 模型其嵌入 Embedding 层，固定住参数，之后训练时不调整这一层参数。每种 id 文本输入为一个定长 100 个词的文本序列，通过嵌入的预训练 word2vec 模型的 Embedding
层转化为 100 个词向量构成的句向量。

2.LSTM 层:

多个 id 句向量拼接后输入 LSTM 层使用双向 LSTM 提取文本特征，从输入的定长文本序列中， 利用局部词序信息，提取初级的特征，并组合初级的特征为高级特征。

3. 池化层:

使用了平均池化，即减少了模型的参数，又保证了在不定长的卷基层的输出上获得一个定长的全连接层的输入。

4. 全连接层:

全连接层的作用就是分类器，我使用了一层隐藏层的全连接网络进行分类，最后 log_softmax 输出分类结果。

![image](https://github.com/BoxFighter/TencentAD2020/blob/master/lstm2.png?raw=true)


![image](https://github.com/BoxFighter/TencentAD2020/blob/master/lstm.png?raw=true)

## 代码说明
- show-data.ipynb

下载数据，解压，简单分析数据

- train_id_int_seq.py

训练数据:执行脚本可以获得多 id 的用户点击序列，word2id 之后的 csv - test_id_int_seq.py

测试数据:执行脚本可以获得多 id 的用户点击序列，word2id 之后的 csv - train_w2v.py

训练 w2vmodel:python train_w2v.py target embed_size window_size 例如:python train_w2v.py ad_id 128 100

- MultTensorDataset.py

自定义的多 id 输入 tensordataset

- LSTM_MultID_Model.py

保存的训练模型

- demo_multlstm_age_train.ipynb

lstm age 模型训练

- demo_multlstm_gender_train.ipynb


lstm gender 模型训练

- demo_multlstm_pred.ipynb

lstm 测试结果生成
