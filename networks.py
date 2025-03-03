import torch
import torch.nn as nn
from collections import OrderedDict

from REPOS.RAFT.core.raft import RAFT as _RAFT

class RAFT(torch.nn.Module):
    def __init__(self, args, SINTEL=False, DENTAL=False, pre_model='RAFT_Step0'):
        super(RAFT, self).__init__()

        model = torch.nn.DataParallel(_RAFT(args))
        if SINTEL:
            path = 'REPOS/models/raft-sintel.pth'
            model.load_state_dict(torch.load(path))
        if DENTAL:
            if pre_model == 'RAFT_Step0':               # Step 0
                path = 'checkpoints/RAFT_Step0.pth'
            elif pre_model == 'RAFT_Step4_50_50_MIX':   # Step 4
                path = 'checkpoints/RAFT_Step4_50_50_MIX.pth'
            else:
                path = pre_model
            try:
                model.load_state_dict(torch.load(path))
            except:
                print(f'Error: model.load_state_dict(torch.load(path)): {path}')
        model = model.module
        self.model = model

    def forward(self, image1, image2, iters=20, test_mode=True):
        return self.model(image1, image2, iters=iters, test_mode=test_mode)

from REPOS.GMA.core.network import RAFTGMA as _RAFTGMA
class GMA(torch.nn.Module):
    def __init__(self, args, SINTEL=False, DENTAL=False, pre_model='GMA_CUSTOM'):
        super(GMA, self).__init__()
        args.position_only = False
        args.position_and_content=False
        args.mixed_precision = False
        model = torch.nn.DataParallel(_RAFTGMA(args))
        if SINTEL:
            path = 'REPOS/models/gma-sintel.pth'
            model.load_state_dict(torch.load(path))
        if DENTAL:
            if pre_model == 'GMA_Step0':
                path = 'checkpoints/GMA_Step0_Vident_synth'
            elif pre_model == 'GMA_Step4_50_50_MIX':
                path = 'checkpoints/GMA_Step4_50_50_MIX.pth'
            else:
                path = pre_model
            try:
                model.load_state_dict(torch.load(path))
            except:
                print(f'Error: model.load_state_dict(torch.load(path)): {path}')
        model = model.module
        self.model = model

    def forward(self, image1, image2, iters=12, test_mode=True):
        return self.model(image1, image2, iters=iters, test_mode=test_mode)

def loadModels(args, MODELS, DEVICE):
    models = []

    if MODELS != None:
        for MODEL in MODELS:
            if 'RAFT_' in MODEL:
                model = RAFT(args, SINTEL=False, DENTAL=True, pre_model=MODEL)
            elif MODEL =='RAFT':
                model = RAFT(args, SINTEL=True, DENTAL=False, pre_model=MODEL)
            elif 'GMA_' in MODEL:
                model = GMA(args, SINTEL=False, DENTAL=True, pre_model=MODEL)
            elif MODEL == 'GMA':
                model = GMA(args, SINTEL=True, DENTAL=False, pre_model=MODEL)
            model.to(DEVICE)
            model.eval()
            models.append(model)
            print(MODEL)
    return models

def getOpticalFlow(MODEL, model, image1, image2, image3):
    # Model name should include 'RAFT' or 'GMA' in it
    if 'RAFT' in MODEL or 'GMA' in MODEL:
        with torch.no_grad():
            _, flow_f    = model(image1, image2)
            _, flow_b    = model(image2, image1)
            _, flow_f_23 = model(image2, image3)
            _, flow_f_13 = model(image1, image3)
    else:
        print(f'Error: {MODEL} not found')
    return flow_f, flow_b, flow_f_23, flow_f_13