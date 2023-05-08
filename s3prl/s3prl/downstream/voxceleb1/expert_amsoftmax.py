# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import SpeakerClassifiDataset
from argparse import Namespace
from pathlib import Path
# from AdMSLoss import AdMSoftmaxLoss

class AMoSoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4, **kwargs):
        '''
        AM Softmax Loss
        '''
        super(AMoSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)


    def forward(self, x_BxH, labels_B):
        '''
        x shape: (B, H)
        labels shape: (B)
        '''
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        W = F.normalize(self.W, dim=0)

        x_BxH = F.normalize(x_BxH, dim=1)

        wf = torch.mm(x_BxH, W)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels_B]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels_B)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return wf, -torch.mean(L)

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = SpeakerClassifiDataset('train', root_dir, self.datarc['meta_data'], self.datarc['max_timestep'])
        self.dev_dataset = SpeakerClassifiDataset('dev', root_dir, self.datarc['meta_data'])
        self.test_dataset = SpeakerClassifiDataset('test', root_dir, self.datarc['meta_data'])
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = self.train_dataset.speaker_num,
            **model_conf,
        )

        #amsoftmax loss
        self.objective = AMoSoftmaxLoss(self.modelrc['projector_dim'], self.train_dataset.speaker_num, s=30.0, m=0.4)
        # self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        utterance_vector, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)
        predicted, loss = self.objective(utterance_vector,labels)

        # loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_speaker'] += SpeakerClassifiDataset.label2speaker(predicted_classid.cpu().tolist())
        records['truth_speaker'] += SpeakerClassifiDataset.label2speaker(labels.cpu().tolist())

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'voxceleb1/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_speaker"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth_speaker"])]
                file.writelines(lines)

        return save_names