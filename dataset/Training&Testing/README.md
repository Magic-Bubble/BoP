原始数据集
BoP2017-DBQA.train.txt
BoP2017-DBQA.dev.txt

python split.py BoP2017-DBQA.train.txt BoP2017-DBQA.train.ques.txt BoP2017-DBQA.train.answ.txt BoP2017-DBQA.train.label.txt
切分后的数据集
BoP2017-DBQA.train.ques.txt
BoP2017-DBQA.train.answ.txt
BoP2017-DBQA.train.label.txt
BoP2017-DBQA.dev.ques.txt
BoP2017-DBQA.dev.answ.txt
BoP2017-DBQA.dev.label.txt

分词
stanford segmenter 命令行调用.sh文件

分词后的questions
BoP2017-DBQA.train.ques.seg.txt
BoP2017-DBQA.dev.ques.seg.txt

分词后的answers
BoP2017-DBQA.train.answ.seg.txt
BoP2017-DBQA.dev.answ.seg.txt

python combine.py BoP2017-DBQA.train.ques.seg.txt BoP2017-DBQA.train.answ.seg.txt BoP2017-DBQA.train.label.txt BoP2017-DBQA.train.seg.txt
最终得到的分词后的数据集
BoP2017-DBQA.train.seg.txt
BoP2017-DBQA.dev.seg.txt

中间可能需要dos2unix filename命令去掉^M

Add
最终数据集
train.seg.txt BoP官方训练集 + nlpcc2016训练集+测试集
dev.seg.txt BoP官方开发集
