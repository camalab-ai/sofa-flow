import os
import random
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from infer_TMI_metrics import t_metrics

# Simulate the data
def simulateData(i=50):
    data = []
    for i in range(i):
        data.append({
            'split': 'train',
            'seq': 'seq_name',
            'im1': '00001.png',
            'im2': '00002.png',
            'im3': '00003.png',
            'frame1': np.random.rand(1, 800, 800, 3).astype(np.float32),
            'frame2': np.random.rand(1, 800, 800, 3).astype(np.float32),
            'frame3': np.random.rand(1, 800, 800, 3).astype(np.float32),
            'OF12': np.random.rand(1, 800, 800, 2).astype(np.float32),
            'OF23': np.random.rand(1, 800, 800, 2).astype(np.float32),
            'OF13': np.random.rand(1, 800, 800, 2).astype(np.float32),
            'OF21': np.random.rand(1, 800, 800, 2).astype(np.float32)})
    return data

class TMI_data():
    def __init__(self, data, model='model_name', T=300, t=1.0):
        self.model = model
        self.data = data
        self.T = T
        self.t = t

        self.metrics = []

        self.Data_t = self.data_t(data)
        self.Data_T = self.data_T(data)

        self.len_t = len(self.Data_t)

        self.c1 = self.C1()
        self.c2 = self.C2()
        self.c3 = self.C3()

    def data_t(self, data):
        accept = []
        for d in data:
            metrics = t_metrics(d['OF12'], d['OF23'], d['OF13'], d['OF21'])
            metrics['split'] = d['split']
            metrics['seq'] = d['seq']
            metrics['im1'] = d['im1']
            metrics['im2'] = d['im2']
            metrics['im3'] = d['im3']
            metrics['model'] = self.model
            self.metrics.append(metrics)
            if metrics['JEPE'] <= self.t and metrics['Ec'] <= self.t:
                accept.append(d)
        return accept

    def data_T(self, data):
        if len(data) > self.T:
            return random.sample(data, self.T)
        else:
            return data
    def C1(self):
        return self.data_T(self.Data_t)

    def C2(self):
        if len(self.Data_T) > self.len_t:
            return random.sample(self.Data_T, self.len_t)
        else:
            return self.Data_T

    def C3(self):
        data_t = self.Data_t
        data_T = self.Data_T
        num_samples = self.len_t if self.len_t < self.T else self.T
        accept = random.sample(data_t, int(num_samples/2)) + random.sample(data_T, num_samples-int(num_samples/2))
        return accept

    def save(self, save_data_path):
        # Save npz files
        for i in self.metrics:
            os.makedirs(save_data_path+'/'+i['split']+'/'+i['seq'], exist_ok=True)
            save_path = save_data_path+'/'+i['split']+'/'+i['seq']+'/'+i['im2'].replace('.png','.npz')
            np.savez(save_path, **i)

    def save_data(self, data, save_data_path ):
        # Save the data
        for i in data:
            name= i['im1'].split('/')[-1].replace('.png','')+'_'+i['im2'].split('/')[-1].replace('.png','')
            path = os.path.join(save_data_path, i['split'], i['seq'], name)
            os.makedirs(path, exist_ok=True)
            npz_data = {'optical_flow': i['OF12']}
            np.savez(os.path.join(path, i['im1'].split('/')[-1].replace('.png','.npz')), **npz_data)

            cv2.imwrite(os.path.join(path, i['im1'].split('/')[-1]), cv2.cvtColor(i['frame1'][0], cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(path, i['im2'].split('/')[-1]), cv2.cvtColor(i['frame2'][0], cv2.COLOR_BGR2RGB))
            print(path)
        return print("Saving complete")

    def plot_metrics(self, data_list, filename='plot.png'):
        """
        Creates subplot charts for the values 'Ec', 'JEPE', and 'smooth_2nd' from a list of dictionaries.

        Parameters:
        data_list (list): A list of dictionaries containing the keys 'Ec', 'JEPE', and 'smooth_2nd'.

        Returns:
        Matplotlib plots.
        """
        # Data
        indices = list(range(len(data_list)))
        ec_values = [item.get('Ec', 0) for item in data_list]
        epe_values = [item.get('JEPE', 0) for item in data_list]
        smooth_values = [item.get('smooth_2nd', 0) for item in data_list]

        # Subplot
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot Ec
        axs[0].plot(indices, ec_values, marker='o', linestyle='-', color='b', label="Ec")
        axs[0].set_ylabel("Ec")
        axs[0].legend()
        axs[0].grid(True)

        # Plot EPE (JEPE)
        axs[1].plot(indices, epe_values, marker='s', linestyle='-', color='r', label="JEPE")
        axs[1].set_ylabel("JEPE")
        axs[1].legend()
        axs[1].grid(True)

        # Plot smooth_2nd
        axs[2].plot(indices, smooth_values, marker='^', linestyle='-', color='g', label="smooth_2nd")
        axs[2].set_xlabel("Index")
        axs[2].set_ylabel("smooth_2nd")
        axs[2].legend()
        axs[2].grid(True)


        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

if __name__ == '__main__':
    # Simulate the data
    data = simulateData(i=2)

    # Initialize the TMI metrics
    tmi_data = TMI_data(data)

    # Print the metrics results
    print(tmi_data.metrics)

    # Generate data for C1, C2 and C3
    C1 = tmi_data.c1
    C2 = tmi_data.c2
    C3 = tmi_data.c3

    # Save the data
    tmi_data.save_data(C1, 'test_TMI_data_C1')
