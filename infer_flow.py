import glob
import os
from PIL import Image
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from networks import loadModels, getOpticalFlow
from infer_seq import SOFA_data

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--DEVICE', help="device", default='cuda')
    parser.add_argument('--MODEL', help="restore checkpoint", default='checkpoints/RAFT_Sintel.pth')
    parser.add_argument('--DATA_PATH', help="data path", default='/home/emilia/DATASETS/Vident-real-100')
    parser.add_argument('--SPLIT', help="split", default='test')
    parser.add_argument('--IMAGE_DIR', help="image dir", default='GT')
    parser.add_argument('--SAVE_DATASET_DIR', help="save dataset", default='test_data_C1')


    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')

    args = parser.parse_args()

    DEVICE = args.DEVICE # DEVICE = 'cuda'
    DATA_PATH = args.DATA_PATH # DATA_PATH = '/home/emilia/DATASETS/Vident-real-100'
    SPLIT = args.SPLIT # SPLIT = 'test'
    IMAGE_DIR = args.IMAGE_DIR # IMAGE_DIR = 'GT'
    MODELS = [args.MODEL] # MODELS = ['GMA_Step4']
    models = loadModels(args, [args.MODEL], DEVICE)

    data_seq = glob.glob(os.path.join(DATA_PATH, SPLIT, '*'))

    for seq in data_seq:
        infer_data_seq = []
        images = glob.glob(os.path.join(seq, IMAGE_DIR, '*.png'))
        images.sort()
        images = images[:10] # for debugging
        for i, i2 , i3 in zip(images[:-2], images[1:-1], images[2:]):
            image1 = load_image(i)
            image2 = load_image(i2)
            image3 = load_image(i3)
            flow_f, flow_b, flow_f_23, flow_f_13 = getOpticalFlow(args.MODEL, models[0], image1, image2, image3)

            print(seq, i.split('/')[-1], i2.split('/')[-1], i3.split('/')[-1])
            infer_data_seq.append({
            'split': i.split('/')[-4],
            'seq': seq.split('/')[-1],
            'im1': i.split('/')[-1],
            'im2': i2.split('/')[-1],
            'im3': i3.split('/')[-1],
            'frame1': image1.permute(0, 2, 3, 1).cpu().numpy(),
            'frame2': image2.permute(0, 2, 3, 1).cpu().numpy(),
            'frame3': image3.permute(0, 2, 3, 1).cpu().numpy(),
            'OF12': flow_f.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32),
            'OF23': flow_f_23.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32),
            'OF13': flow_f_13.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32),
            'OF21': flow_b.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)})

        print('SOFA_data created ', seq)
        sofa_data = SOFA_data(data=infer_data_seq, model=args.MODEL, T=300, t=1.0)

        print('Selected frames C1...')
        C1 = sofa_data.c1

        print('Saving SOFA_data...')
        sofa_data.save_data(C1, args.SAVE_DATASET_DIR)

        print('Plotting metrics...')
        sofa_data.plot_metrics(sofa_data.metrics, f'{args.SAVE_DATASET_DIR}/plot_{seq.split("/")[-1]}.png')
