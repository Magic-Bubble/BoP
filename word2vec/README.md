STEP I    -> python process_wiki.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.text
处理wiki语料,存储成一行一片文章+"\n"的形式
STEP II   -> opencc -i wiki.zh.text -o wiki.zh.text.jian -c zht2zhs.ini
将wiki语料中的繁体字转为简体字
STEP III  -> ../../Software/stanford-segmenter-2015-12-09/segment.sh pku wiki.zh.text.jian UTF-8 0 > wiki.zh.text.jian.seg
将wiki语料进行切词处理,使用斯坦福的中文切词工具
STEP IV   -> python convert_unknown.py wiki.zh.text.jian.seg wiki.zh.text.jian.seg.unknown 5
将切词后的语料中词频低于5的词转为<UNKNOWN_TOKEN>
STEP V    -> python train_word2vec_model.py wiki.zh.text.jian.seg.unknown wiki.zh.text.model wiki.zh.text.vector
训练word2vec,size=400,min_count=5
