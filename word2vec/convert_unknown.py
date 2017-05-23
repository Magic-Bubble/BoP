#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os
import sys
import six
import nltk

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print("Using: python convert_unknown.py wiki.zh.text.seg wiki.zh.text.seg.unknown <min_count>")
        sys.exit(1)
    inp, outp, min_count = sys.argv[1:4]

    texts = open(inp).read()
    all_words=texts.split()
    fdist = nltk.probability.FreqDist(all_words)
    words = set(all_words)
    # Using set: more faster and faster than list when find
    lf_words = {w for w in words if fdist[w] < int(min_count)}

    texts = open(inp).readlines()
    output = open(outp, 'w')
    unknown_token = "<UNKNOWN_TOKEN>"
    space = " "
    i = 0
    for text in texts:
        text = text.split()
        text = [w if w not in lf_words else unknown_token for w in text]
        if six.PY3:
            output.write(space.join(map(lambda x: x.decode("utf-8"), text)) + '\n') 
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles") 
