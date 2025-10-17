"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in main.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class PathDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(PathDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, size))
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.num_nodes = size
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    # y = torch.cat((tour, tour[:, :1]), dim=1)
    y=tour

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()

import matplotlib.animation as animation
def plot_tour(data, best_path,save_path):
    fig, ax = plt.subplots()
    x = []
    y = []
    figure_list = []
    text_list = []
    for v in best_path:
        x.append(data[v][0])
        y.append(data[v][1])
        # print(data[v][0],data[v][1])
        text_list.append(str(v))

        ax.plot(x, y, 'c^', linewidth=2, markersize=15)
        ax.text(data[v][0], data[v][1], str(v), ha='center', va='center_baseline', size=8)

        figure = ax.plot(x, y, '--', linewidth=2, markersize=20)

        figure_list.append(figure)
    ani = animation.ArtistAnimation(fig, figure_list, interval=200, repeat_delay=0)
    ani.save(save_path)
    print('Save result at: {}'.format(save_path))
    # plt.show(block=False)
    # plt.pause(4.2)
    # ani.save('test2.gif')
    # plt.close()

def render(static, tour_indices, save_path):
    """Plots the found tours."""
    idx = tour_indices[0]
    idx = idx.expand(static.size(1), -1)
    data = torch.gather(static[0].data, 1, idx).cpu().numpy()
    plot_tour(data.T,[x for x in range(data.shape[1])],save_path)
