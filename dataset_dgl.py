# -*- coding: utf-8 -*-

"""
Transforms YelpChi.mat to yelp.dgl 
Set the dataset_path in main [line 46] to where YelpChi.mat is present
For other names, change yelp_path [line 61] from YelpChi.mat to nyc,zip

To run : python Yelp_dgl_code.py --dataset yelp

Credit : https://github.com/shifengzhao/H2-FDetector/blob/master/src/data_preprocess.py
"""

import argparse
import os
import sys
import copy
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split


def generate_edges_labels(edges, labels, train_idx):
    row, col = edges
    edge_labels = []
    edge_train_mask = []
    train_idx = set(train_idx)
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(-1)
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    return edge_labels, edge_train_mask


if __name__ == '__main__':
    dataset_path = 'YelpChi/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    args = parser.parse_args()
    print('**********************************')
    print(f'Generate {args.dataset}')
    print('**********************************')
    if args.dataset == 'yelp':
        '''
        # generate yelp dataset
        '''
        if os.path.exists(dataset_path+'yelp.dgl'):
            print('Dataset yelp has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        yelp_path = dataset_path+'YelpChi.mat'
        yelp = scio.loadmat(yelp_path)
        feats = yelp['features'].todense()
        features = torch.from_numpy(feats)
        lbs = yelp['label'][0]
        labels = torch.from_numpy(lbs)
        homo = yelp['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        rur = yelp['net_rur']
        rur = rur+rur.transpose()
        rur = rur.tocoo()
        rtr = yelp['net_rtr']
        rtr = rtr+rtr.transpose()
        rtr = rtr.tocoo()
        rsr = yelp['net_rsr']
        rsr = rsr+rsr.transpose()
        rsr = rsr.tocoo()
        
        yelp_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','u','r'):(torch.tensor(rur.row), torch.tensor(rur.col)),
            ('r','t','r'):(torch.tensor(rtr.row), torch.tensor(rtr.col)),
            ('r','s','r'):(torch.tensor(rsr.row), torch.tensor(rsr.col))
        }
        yelp_graph = dgl.heterograph(yelp_graph_structure)
        yelp_graph.nodes['r'].data['feature'] = features
        yelp_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(len(lbs)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs, stratify=lbs, train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        
        yelp_graph.nodes['r'].data['train_mask'] = train_mask
        yelp_graph.nodes['r'].data['valid_mask'] = valid_mask
        yelp_graph.nodes['r'].data['test_mask'] = test_mask
        
        print('Generate edge labels.')
        homo_edges = yelp_graph.edges(etype='homo')
        homo_labels, homo_train_mask = generate_edges_labels(homo_edges, lbs, train_idx)
        yelp_graph.edges['homo'].data['label'] = homo_labels
        yelp_graph.edges['homo'].data['train_mask'] = homo_train_mask
        
        dgl.save_graphs(dataset_path+'yelp.dgl', yelp_graph)
        print(f'yelp dataset\'s num nodes:{yelp_graph.num_nodes("r")}, \
            rur edges:{yelp_graph.num_edges("u")}, \
            rtr edges:{yelp_graph.num_edges("t")}, \
            rsr edges:{yelp_graph.num_edges("s")}')
        print(f'Edge train num:{homo_train_mask.sum().item()}, pos num:{(homo_labels[homo_train_mask]==1).sum().item()}')
        

    print('***************endl****************')

