# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import fairseq
from ..interfaces import UpstreamBase


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."
        
        model = torch.load(ckpt)
    

        model['cfg']['model']['label_rate']=int(model['cfg']['model']['label_rate'])
        model['cfg']['task']['label_rate']=int(model['cfg']['task']['label_rate'])

        try: 
            model['model']['final_proj.weight']=model['model']['final_proj_list.0.weight']
            del model['model']['final_proj_list.0.weight']
        except: pass   

        try: 
            model['model']['final_proj.bias']=model['model']['final_proj_list.0.bias']
            del model['model']['final_proj_list.0.bias']
        except: pass

        # a = [0,1,2,3]
        # for i in a:
        #     try: 
        #         key = f"final_proj_list.{i}.weight"
        #         del model['model'][key]
        #         key = f"final_proj_list.{i}.bias"
        #         del model['model'][key]
        #     except: pass


        torch.save(model,ckpt)

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt],
            strict=False
        )
        self.model = model[0]
        self.task = task
        self.cfg = cfg
        
        
        try: 
            self.test_pad = self.cfg['model']['test_pad']
            print("-------Using the utterance loss features-------")
        except: 
            self.test_pad = False
            print("-------NOT using the utterance loss features-------")
        #self.test_pad = False

# fairseq-hydra-train --config-dir config/pretrain/ --config-name hubert_base_librispeech.yaml  task.data=/home/hemant/hemant/librispeech/data/manifest/ task.label_dir=/home/hemant/hemant/librispeech/hubert_km_path_data/labels/ task.labels='["km3"]' model.label_rate=50 dataset.max_tokens=1420000 optimization.update_freq=[4] checkpoint.save_dir=/home/hemant/hubert_cluster_new/example/hubert/checkpoints_speaker_weights_10_only_500/ +model.encoder_layers=12 model.test_pad=true & fairseq-hydra-train --config-dir config/pretrain/ --config-name hubert_base_librispeech.yaml  task.data=/home/hemant/hemant/librispeech/data/manifest/ task.label_dir=/home/hemant/hemant/librispeech/hubert_km_path_data/labels/ task.labels='["km1"]' model.label_rate=50 dataset.max_tokens=1420000 optimization.update_freq=[4] checkpoint.save_dir=/home/hemant/hubert_cluster_new/example/hubert/checkpoints_speaker_weights_10_only_100/ +model.encoder_layers=12 model.test_pad=true

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            
            for module_id in range(len(eval(module_name))):
                
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
                
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))
            self.hook_postprocess = postprocess
          
    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
 
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
            test_pad=self.test_pad,
        )
        # print(features.shape)
        # exit()
 
        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
