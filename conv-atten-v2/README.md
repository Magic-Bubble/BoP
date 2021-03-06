STEP 1
训练词向量word2vec ../word2vec/
详见word2vec目录下的README.md
Done!

STEP 2
数据预处理 ./parse.py
1 -> 读取分词后的数据集 ../dataset/BoP2017-DBQA/BoP2017-DBQA.train.seg.txt 转化为qids questions, answers, labels
2 -> 将questions和answers转化为词典集, 并载入word2vec模型, 生成embedding matrix, 存储为vocab_embeddings.npy
3 -> 将questions中每个question转为其在词典中对应的下标列表, 存储为questions.npy
4 -> 将answers同上处理 存储为answers.npy
5 -> 将labels 存储为labels.npy, qids 存储为qids.npy
6 -> 计算overlap特征 去停词和不去停词 IDF加权与不加权 每个<q,a>对共4维 存储为overlap_feats.npy
7 -> 将questions和answers计算overlap indices 存储为q_overlap_indices.npy, a_overlap_indices.npy
python parse.py ../dataset/BoP2017-DBQA/BoP2017-DBQA.train.seg.txt ../word2vec/wiki.zh.text.model data/stoplist.txt data/train
python parse.py ../dataset/BoP2017-DBQA/BoP2017-DBQA.dev.seg.txt ../word2vec/wiki.zh.text.model data/stoplist.txt data/dev 
Done!

STEP 5
载入数据
1 -> 每次都载入一个batch的数据
2 -> 训练集shuffle, 并做weighted sample
3 -> 开发集不shuffle, 不做weighed sample

总体模型:
参考Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks(主)
参考Convolutional Deep Neural Networks for Document-based Question Answering(辅)

STEP 5
搭建句子矩阵映射模型
1 -> 载入 questions.npy, answers.npy, labels.npy
2 -> Lookup table, embedding 取出对应的向量
2 -> 卷积层 feature maps取100 kernel取[1, 3, 5] concat成L*300 
3 -> Attentive池化层 输出q和a对应的中间向量 均为feature map*3维

STEP 6
搭建句子匹配模型
1 -> 相似性匹配层 输出xsim
2 -> flatten层 结合Attentive层的中间向量 及 xsim 及 overlap_feats.npy
3 -> 全连接层 维度为1024 使用dropout p=0.5
4 -> softmax层 2维 输出为正负样本的概率

STEP 7
损失函数及优化器
1 -> LOSS = 分类损失函数 + L2正则项 对卷积层参数 参数取1e-5 对其他参数 取1e-4
2 -> 优化器 Adam 学习率 自动调整

STEP 8
训练 & 评估
1 -> batch_size 50
2 -> batch_size之后做一次loss反传
3 -> 一个epoch结束后, 用开发集计算MRR分数

STEP 9
测试
1 -> 计算MRR分数,用测试集
