#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os.path
import sys

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 5:
        print("Using: python combine.py BoP2017-DBQA.train.ques.seg.txt BoP2017-DBQA.train.answ.seg.txt BoP2017-DBQA.train.label.txt BoP2017-DBQA.train.seg.txt")
        sys.exit(1)
    inp1, inp2, inp3, outp = sys.argv[1:5]
    i = 0

    quess = open(inp1).readlines()
    answs = open(inp2).readlines()
    labels = open(inp3).readlines()
    output = open(outp, 'w')
    for i in range(len(quess)):
        output.write(labels[i].strip('\n') + '\t' + quess[i].strip('\n') + '\t' + answs[i])
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " q&a pairs")

    output.close()
    logger.info("Finished Saved " + str(i) + " q&a pairs")
