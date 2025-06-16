from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import torch
import yaml


class Logger(object):
    def __init__(self, data, logscalars=True, save_embed = True):
        exp = 0
        if data["LAI"]:
            experience =  './logs/' + data['dataset'] +'/'+ data['model_arch'] + '_LAI/' + str(exp)
        else:
            experience =  './logs/' + data['dataset'] +'/'+ data['model_arch'] + '/' + str(exp)
        while os.path.isdir(experience) == True:
            exp+=1
            if data["LAI"]:
                experience = './logs/' + data['dataset'] +'/'+ data['model_arch'] + '_LAI/' + str(exp)
            else:
                experience = './logs/' + data['dataset'] +'/'+ data['model_arch'] + '/' + str(exp)
        self.logscalars = defaultdict(list)
        self.writer = SummaryWriter(experience)
        self.save_embed = save_embed
        self.savepath = experience
        self.log =logscalars
        with open(self.savepath + '/config.yaml', "w", encoding = "utf-8") as yaml_file:       
            dump = yaml.dump(data, default_flow_style = False, allow_unicode = True, encoding = None)
            yaml_file.write(dump) 

    def write_scalars(self, scalars, epoch, write_epoch=False):
        if write_epoch:
            self.logscalars['epoch'].append(epoch)
        for k, v in scalars.items():
            if self.log:
                self.logscalars[k].append(v)
            self.writer.add_scalar(k, v, epoch)   

    def save_log(self):
        acc_dict = {k: v for k, v in self.logscalars.items() if k[:3] == "Acc"}
        rest = {k: v for k, v in self.logscalars.items() if k[:3] != "Acc"}
        df = pd.DataFrame(acc_dict)
        df.to_csv(self.savepath + "/log_acc.txt")
        df = pd.DataFrame(rest)
        df.to_csv(self.savepath + "/log_train.txt")
        self.writer.flush()
        self.writer.close()

    def save_model(self, model):
        if len(self.logscalars['Test/mAP']) > 1:
            if self.logscalars['Test/mAP'][-1] >= np.max(self.logscalars['Test/mAP'][:-1]):
                torch.save(model.state_dict(), self.savepath  + '/' + 'best_mAP' +'.pt')
            if self.logscalars['Test/CMC1'][-1] >= np.max(self.logscalars['Test/CMC1'][:-1]):
                torch.save(model.state_dict(), self.savepath  + '/' + 'best_CMC' +'.pt')
        torch.save(model.state_dict(), self.savepath  + '/' + 'last' +'.pt')

    def write_embeddings(self, features, labels, images, epoch, tag='default'):
        #if self.logscalars['Accuraccy/mAP'][-1] > np.max(self.logscalars['Accuraccy/mAP']):
        self.writer.add_embedding(features, metadata=labels, label_img=images, global_step=epoch, tag=tag)
