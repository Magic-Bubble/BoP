#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import logging
import os.path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.cuda as cuda
from torch.utils import data
from tensorboard_logger import configure, log_value

from dataset import Dataset
from net import Net

def train():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    n_epochs = 100 
    batch_size = 50
    train_interval = 1000
    #test_interval = 500
    #test_steps = 100
    start_lr = 0.001
    end_lr = 1e-7

    cuda.set_device(1)
    configure("logs", flush_secs=5)

    train_data = np.load('./data/train/data.npz')
    dev_data = np.load('./data/dev/data.npz')
    train_dataset = Dataset(train_data)#, train=True)
    dev_dataset = Dataset(dev_data)#, train=False)
    class_sample_count = [15, 1]
    weight_per_class = 1 / torch.Tensor(class_sample_count).double()
    weights = [weight_per_class[label] for label in train_data['labels']]
    sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler)#, batch_size=batch_size, sampler=sampler)
    dev_dataloader = data.DataLoader(dev_dataset)#, shuffle=True)

    net = Net().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    #ignored_params = list(map(id, net.sentence.conv_q.parameters())) + list(map(id, net.sentence.conv_a.parameters()))
    ignored_params = list(map(id, net.sentence.conv1.parameters())) + list(map(id, net.sentence.conv2.parameters())) + list(map(id, net.sentence.conv3.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = optim.Adam([
        #{'params': net.sentence.conv_q.parameters(), 'weight_decay': 1e-5},
        #{'params': net.sentence.conv_a.parameters(), 'weight_decay': 1e-5},
        {'params': net.sentence.conv1.parameters()},
        {'params': net.sentence.conv2.parameters()},
        {'params': net.sentence.conv3.parameters()},
        {'params': base_params}
        ], lr=start_lr, weight_decay=1e-5)

    latest_epoch_num = 0
    model_path = './model/epoch_' + str(latest_epoch_num) + '.params'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        logger.info('Successfully loaded model: %s' % (model_path))
    else:
        logger.info('Could not find model: %s' % (model_path))
       
    old_mrr = 0
    lr = start_lr
    for epoch in range(n_epochs):
        if (lr < end_lr):
            break
        net.train()
        epoch += latest_epoch_num
        running_loss = 0.0
        correct = 0
        for i, train_data in enumerate(train_dataloader, 0):
            train_qids, train_questions, train_answers, train_overlap_feats, train_labels = train_data
            train_questions = Variable(train_questions.cuda())
            train_answers = Variable(train_answers.cuda())
            train_overlap_feats = Variable(train_overlap_feats.cuda())
            train_labels = Variable(train_labels.long().cuda())
            
            prob = net(train_questions, train_answers, train_overlap_feats)
            loss = criterion(prob, train_labels)
            loss.backward()

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            #optimizer.zero_grad()
            #optimizer.step()
            
            running_loss += loss.data[0]
            _, predicted = torch.max(prob.data, 1)
            correct += (predicted == train_labels.data).sum()

            if (i + 1) % train_interval == 0:
                logger.info('[%d, %5d] lr: %.7f, train_loss: %.3f, train_accuracy: %.3f' % (epoch+1, i+1, lr, running_loss / train_interval, correct / train_interval))
                log_value('train_loss', running_loss / train_interval)
                log_value('train_accuracy', correct / train_interval)
                running_loss = 0.0
                correct = 0

        logger.info("Finished %s epoch" % (epoch+1))
        torch.save(net.state_dict(), './model/epoch_%s.params' % (epoch+1))
        logger.info('Saved model: ./model/epoch_%s.params' % (epoch+1))

        # test on dev set
        net.eval()
        accurate = 0
        test_nums = 0
        unique_qid_nums = 0
        probs, labels = [], []
        qid_prev = 1
        rank_score = 0.0
        for j, test_data in enumerate(dev_dataloader, 0):
            test_qids, test_questions, test_answers, test_overlap_feats, test_labels = test_data
            test_questions = Variable(test_questions.cuda(), volatile=True)
            test_answers = Variable(test_answers.cuda(), volatile=True)
            test_overlap_feats = Variable(test_overlap_feats.cuda(), volatile=True)
            test_labels = Variable(test_labels.long().cuda(), volatile=True)
            
            if test_qids[0] != qid_prev:
                unique_qid_nums += 1
                probs = torch.Tensor(probs)
                labels = torch.from_numpy(np.array(labels))
                _, accurate_idx = torch.max(labels, 0)
                _, rank_idx = torch.sort(probs, 0, descending=True)
                _, rank = torch.max(rank_idx == accurate_idx[0], 0)
                rank_score += 1/(rank[0]+1)
                probs, labels = [], []
                qid_prev = test_qids[0]

            test_nums += test_questions.size()[0]

            prob = net(test_questions, test_answers, test_overlap_feats)

            _, predicted = torch.max(prob.data, 1)
            accurate += (predicted == test_labels.data).sum()

            probs.append(prob.data[0][1])
            labels.append(test_labels.data[0])

            #_, predicted = torch.max(prob.data, 1)
            #right += (predicted == test_labels.data).sum()

            #_, prediction = torch.max(prob.data[:, 1], 0)
            #_, accurate_idx = torch.max(test_labels.data, 0)
            #accurate += (prediction == accurate_idx)[0]
            #_, rank_idx = torch.sort(prob.data[:, 1], 0, descending=True)
            #_, rank = torch.max(rank_idx == accurate_idx[0], 0)
            #rank_score += 1/(rank[0]+1)
            #if (j + 1) == test_steps:
            #    break
        #logger.info('[%d, %5d] test_accuracy: %.3f, MAP: %.3f, MRR: %.3f' % (epoch+1, i+1, right / (test_nums), accurate / test_steps, rank_score / test_steps))
         
        unique_qid_nums += 1
        probs = torch.Tensor(probs)
        labels = torch.from_numpy(np.array(labels))
        _, accurate_idx = torch.max(labels, 0)
        _, rank_idx = torch.sort(probs, 0, descending=True)
        _, rank = torch.max(rank_idx == accurate_idx[0], 0)
        rank_score += 1/(rank[0]+1)
        mrr = rank_score / unique_qid_nums
        if (mrr < old_mrr):
            lr = lr * 0.5
            optimizer = optim.Adam([
                {'params': net.sentence.conv1.parameters(), 'weight_decay': 1e-5},
                {'params': net.sentence.conv2.parameters(), 'weight_decay': 1e-5},
                {'params': net.sentence.conv3.parameters(), 'weight_decay': 1e-5},
                {'params': base_params, 'weight_decay': 1e-4}
            ], lr=lr)
        old_mrr = mrr 

        logger.info('[%d] test_accuracy: %.3f, MRR: %.3f' % (epoch+1, accurate / test_nums, mrr))
        log_value('test_accuracy', accurate / test_nums)
        #log_value('MAP', accurate / test_steps)
        log_value('MRR', mrr)

    logger.info("Finished training")

if __name__ == '__main__':
    train()
