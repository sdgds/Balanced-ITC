# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
import scipy.io as scio
from umap import UMAP
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import pandas as pd
import warnings
warnings.filterwarnings('ignore')




### Model Results
def plot_bar(data_all, name):
    plt.figure(figsize=(6,6), dpi=300)
    plt.style.use('seaborn')
    font_1 = {"size": 15}
    sns.barplot(data=[data_all[:,2], data_all[:,1], data_all[:,0]],
                palette=[sns.color_palette()[0], sns.color_palette()[-1], sns.color_palette()[1]])
    plt.xlabel("Interconnectivity", font_1)
    plt.ylabel(name, font_1)
    plt.xticks(ticks = [0, 1, 2], fontsize = 11)
    plt.yticks(fontsize=12)



radii = np.load('radii_1000_random_neuron.npy')[:,3]
radii_RNN = np.load('radii_RNN.npy')
radii_AlexNet = np.load('radii_AlexNet.npy')
radii = np.hstack((radii_RNN.reshape(-1,1),
                   radii.reshape(-1,1),
                   radii_AlexNet.reshape(-1,1)))
plot_bar(radii, name='radii')

dimensions = np.load('dimensions_1000_random_neuron.npy')[:,3]
dimensions_RNN = np.load('dimensions_RNN.npy')
dimensions_AlexNet = np.load('dimensions_AlexNet.npy')
dimensions = np.hstack((dimensions_RNN.reshape(-1,1),
                        dimensions.reshape(-1,1),
                        dimensions_AlexNet.reshape(-1,1)))
plot_bar(dimensions, name='dimensions')



        
radii = np.load('radii_1000_random_neuron.npy')
dimensions = np.load('dimensions_1000_random_neuron.npy')
index = np.argsort(radii,axis=0)[:30]
# plot
fig, ax1 = plt.subplots(figsize=(3,4), dpi=300)  
d = np.array([dimensions[index[:,i],i] for i in range(7)])[[0,2,3,4,6]] 
ax1.plot(d.mean(1), 
         label='dimensions', color='red', marker='.')
ax1.fill_between([0,1,2,3,4], 
                 d.mean(1)-d.std(1), 
                 d.mean(1)+d.std(1), facecolor='red', alpha=0.3)
ax1.legend(bbox_to_anchor=(0.69,0.92))
ax2 = ax1.twinx()
ax2.grid()
r = np.array([radii[index[:,i],i] for i in range(7)])[[0,2,3,4,6]]
ax2.plot(r.mean(1), 
         label='radii', color='blue', marker='.')
ax2.fill_between([0,1,2,3,4], 
                 r.mean(1)-r.std(1), 
                 r.mean(1)+r.std(1), facecolor='blue', alpha=0.3)
ax2.legend(bbox_to_anchor=(0.5,0.99))






### Macaque Results
def plot_bar(data_1, data_2, name):
    plt.figure(figsize=(4,6), dpi=300)
    plt.style.use('seaborn')
    font_1 = {"size": 15}
    sns.barplot(data=[data_1, data_2],
                palette=[sns.color_palette()[-1], sns.color_palette()[1]])
    plt.ylabel(name, font_1)
    plt.xticks([0,0.6], ['TEp', 'TEa'], fontsize = 11)
    plt.yticks(fontsize=12)


radii = np.load('radii_100_random_neuron.npy')
dimensions = np.load('dimensions_100_random_neuron.npy')

plot_bar(radii[:,0], radii[:,2], name='radii')
plot_bar(dimensions[:,0], dimensions[:,2], name='dimensions')



def scatter_plot(data_1, data_2):
    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    plt.style.use('default')
    ax.scatter(data_1, data_2, color=sns.color_palette()[5])
    minv = min(min(data_1), min(data_2))
    maxv = max(max(data_1), max(data_2))
    l = (maxv-minv)/10
    ax.set_xlim([minv-l,maxv+l])
    ax.set_ylim([minv-l,maxv+l])
    ax.set_xlabel('TEO', fontsize = 11)
    ax.set_ylabel('TEa', fontsize = 11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    plt.style.use('default')
    ax.hist(data_1-data_2, bins=20, color=sns.color_palette()[7])
    maxv = max(data_1-data_2)
    minv = -maxv
    l = (maxv-minv)/10
    ax.set_xlim([minv-l,maxv+l])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

scatter_plot(dimensions[:,0], dimensions[:,2])
scatter_plot(radii[:,0], radii[:,2])

print(ttest_ind(dimensions[:,0], dimensions[:,2]))
print(ttest_ind(radii[:,0], radii[:,2]))


