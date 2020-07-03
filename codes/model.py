import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from data import train_data
from data import test_data

class KGEModel(nn.Module,ABC):
    @abstractmethod
    def score(self, head, relation, tail, type):
        """
        Defined in a specific model
        """

    def foward(self,sample,type):
        if(type == 'head_batch'):
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif(type == 'tail_batch'):
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        return self.score(head, relation, tail, type)

    @staticmethod
    def train(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        
        model.train()
        optimizer.zero_grad()
        pos_triple, neg_triple, subsampling_weight, type = next(train_iterator)

        # CUDA is used by default.
        pos_triple = pos_triple.cuda()
        neg_triple = neg_triple.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # score
        if(args.model=='DistMult'):
            neg_score = model((pos_triple, neg_triple), type=type)
            pos_score = model(pos_triple)
            neg_score=subsampling_weight * neg_score
            pos_score=subsampling_weight * pos_score
        
        # loss
        if (args.model == 'DistMult'):
           loss = neg_score - pos_score + 1
           torch.clamp(loss, min=0.0)
           loss = loss.sum()
           loss = loss / subsampling_weight.sum()

        loss.backward()
        optimizer.step()

        log = {
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test(model, test_triple, all_triple, args):
        model.eval()

        test_dataloader_head = DataLoader(
            test_data(
                test_triple,
                all_triple,
                args.nen,
                args.nre,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=test_data.collate_fn
        )

        test_dataloader_tail = DataLoader(
            test_data(
                test_triple,
                all_triple,
                args.nen,
                args.nre,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=test_data.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        #CUDA is used by default.
        pos_triple = pos_triple.cuda()
        neg_triple = neg_triple.cuda()
        filter_bias = filter_bias.cuda()

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for pos_triple, neg_triple, filter_bias, type in test_dataset:
                    batch_size = pos_triple.size(0)
                    score = model((pos_triple, pos_triple), type)
                    score += filter_bias

                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = pos_triple[:, 0]
                    else:
                        positive_arg = pos_triple[:, 2]

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics

class DistMult(KGEModel):
    def __init__(self, nen, nre, hidden_dim,gamma):
        super(DistMult, self).__init__()
        self.nen=nen
        self.nre=nre
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(nen, self.hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nre, self.hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def score(self, head, relation, tail, type):
        if(type=='head_batch'):
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        score = score.sum(dim=2)

        return score



