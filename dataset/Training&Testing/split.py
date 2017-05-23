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
        print("Using: python split.py BoP2017-DBQA.train.txt BoP2017-DBQA.train.ques.txt BoP2017-DBQA.train.answ.txt BoP2017-DBQA.train.label.txt")
        sys.exit(1)
    inp, outp1, outp2, outp3 = sys.argv[1:5]
    i = 0

    qas = open(inp).readlines()
    output1 = open(outp1, 'w')
    output2 = open(outp2, 'w')
    output3 = open(outp3, 'w')
    for qa in qas:
        parts = qa.split('\t')
        label = parts[0]
        ques = parts[1]
        answ = parts[2]
        output1.write(ques + '\n')
        output2.write(answ)
        output3.write(label + '\n')
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " q&a pairs")

    output1.close()
    output2.close()
    output3.close()
    logger.info("Finished Saved " + str(i) + " q&a pairs")
