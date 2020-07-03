import numpy as np
import torch
import os
from enum import Enum
from typing import Tuple, List, Dict

from torch.utils.data import Dataset

def word2id(data_path,data_name):
    with open(os.path.join(data_path, data_name)) as f:
        if(data_name=='entities.dict'):
            entity2id=dict()
            for line in f:
                id,entity=line.strip().split('\t')
                entity2id[entity]=int(id)
            return entity2id

        else:
            relation2id=dict()
            for line in f:
                id,entity=line.strip().split('\t')
                relation2id[entity]=int(id)
            return relation2id

def get_triple(data_path,data_name,entity2id,relation2id):
    triple=[]
    with open(os.path.join(data_path, data_name)) as f:
        for line in f:
            head,relation,tail=line.strip().split('\t')
            triple.append((entity2id[head], relation2id[relation], entity2id[tail]))
    return triple

def true_dict(triple):
    true_head={}
    true_tail={}
    for head,relation,tail in triple:
        if (head,relation) not in true_tail:
            true_tail[(head,relation)]=[]
            true_tail[(head, relation)].append(tail)
        elif tail not in true_tail[(head,relation)]:
            true_tail[(head, relation)].append(tail)
    for head, relation, tail in triple:
        if (relation,tail) not in true_head:
            true_head[(relation,tail)]=[]
            true_head[(relation, tail)].append(head)
        elif head not in true_head[(relation,tail)]:
            true_head[(relation, tail)].append(head)
    return true_head,true_tail

def frequency(triple):
    fre_hr={}
    fre_rt={}
    initial=3
    for head,relation,tail in triple:
        if (head,relation) not in fre_hr:
            fre_hr[(head,relation)]=initial
        else:
            fre_hr[(head,relation)]=fre_hr[(head,relation)]+1
        
        if (relation,tail) not in fre_rt:
            fre_rt[(relation,tail)]=initial
        else:
            fre_rt[(relation,tail)]=fre_rt[(relation,tail)]+1
        return fre_hr,fre_rt

class train_data(Dataset):
    def __init__(self,triple,entity2id,relation2id,n_size,type):
        self.triple=triple
        self.nen=len(entity2id)
        self.nre=len(relation2id)
        self.len=len(triple)
        self.type=type
        self.n_size=n_size
        self.true_head,self.true_tail=true_dict(triple)
        self.fre_hr,self.fre_rt=frequency(triple)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        pos_triple=self.triple[idx]
        head, relation, tail = pos_triple
        
        neg_triple=[]
        neg_size=0
        
        subsampling_weight = self.fre_hr[(head, relation)] + self.fre_rt[(relation,tail)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        while neg_size<self.n_size:
            neg_sample= np.random.randint(self.nen, size=self.n_size*2)
            if(self.type=='head_batch'):
                mask = np.in1d(
                    neg_sample,
                    self.ture_head[(relation,tail)],
                    assume_unique=True,
                    invert=True
                )
            else:
                mask = np.in1d(
                    neg_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            neg_sample=neg_sample[mask]
            neg_triple.append(neg_sample)
            neg_size+=neg_sample.size

        neg_triple = np.concatenate(neg_triple)[:self.n_size]
        neg_triple = torch.from_numpy(neg_triple)
        pos_triple = torch.LongTensor(pos_triple)
        return pos_triple,neg_triple,subsampling_weight,self.type

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        neg_triple = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        type = data[0][3]
        return pos_triple, neg_triple,subsampling_weight,type

class test_data(Dataset):
    def __init__(self, triple, all_triple, entity2id, relation2id, type):
        self.triple = triple
        self.nen = len(entity2id)
        self.nre = len(relation2id)
        self.len = len(triple)
        self.type = type
        self.triple_set = set(all_triple)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head,relation,tail=self.triple[idx]

        if(self.type=='head_batch'):
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nen)]
            tmp[head] = (0, head)
        else:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nen)]
            tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        neg_triple = tmp[:, 1]

        pos_triple = torch.LongTensor((head, relation, tail))

        return pos_triple, neg_triple, filter_bias, self.type

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        neg_triple = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        type = data[0][3]
        return pos_triple, neg_triple, filter_bias, type


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

