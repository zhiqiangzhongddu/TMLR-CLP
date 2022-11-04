import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from sklearn import preprocessing
import pickle as pkl

import torch

import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, WikiCS, WebKB, Actor, WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from NHB_dataset import load_nc_dataset

from generate_syn_dataset_utils import RandomPartitionGraph, make_feat


def read_label(dir):
    f_path = dir + '/' + 'labels.txt'
    fin_labels = open(f_path)
    labels = []
    node_id_mapping = dict()
    for new_id, line in enumerate(fin_labels.readlines()):
        old_id, label = line.strip().split()
        labels.append(int(label))
        node_id_mapping[old_id] = new_id
    fin_labels.close()
    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + '/' + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


def read_file(data_name, use_degree):
    directory = '../data/structure_label_data/' + data_name + '/'
    # read raw data
    raw_labels, node_id_mapping = read_label(directory)
    raw_edges = read_edges(directory, node_id_mapping)
    # generate raw nx-graph
    G = nx.Graph(raw_edges)
    # set up node attribute
    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    # attributes = np.arange(G.number_of_nodes(), dtype=np.float32)
    if use_degree:
        attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)
    G.graph['attributes'] = attributes

    return G, np.array(raw_labels)


def get_data(G, raw_labels):
    data = from_networkx(G=G)
    data.x = torch.FloatTensor(G.graph['attributes'])
    data.y = torch.LongTensor(raw_labels)

    return data


def split_data(data, train_ratio, data_name, split_id, fixed_split=True):
    if fixed_split:
        train_nodes = np.loadtxt(f'../input/fixed_split/{data_name}_TR_{train_ratio}_train_{split_id}.txt').astype(int)
        val_nodes = np.loadtxt(f'../input/fixed_split/{data_name}_TR_{train_ratio}_val_{split_id}.txt').astype(int)
        test_nodes = np.loadtxt(f'../input/fixed_split/{data_name}_TR_{train_ratio}_test_{split_id}.txt').astype(int)
    else:
        # set up train val and test
        shuffle = list(range(data.num_nodes))
        random.shuffle(shuffle)
        if train_ratio == 48:
            train_nodes = shuffle[: int(data.num_nodes * 48 / 100)]
            non_train_nodes = shuffle[int(data.num_nodes * 48 / 100):]
            val_nodes = non_train_nodes[: int(data.num_nodes * 32 / 100)]
            test_nodes = non_train_nodes[int(data.num_nodes * 32 / 100):]
        else:
            train_nodes = shuffle[: int(data.num_nodes * train_ratio / 100)]
            non_train_nodes = shuffle[int(data.num_nodes * train_ratio / 100):]
            val_nodes = non_train_nodes[: int(data.num_nodes * train_ratio / 100)]
            test_nodes = non_train_nodes[int(data.num_nodes * train_ratio / 100):]

    # set up train-val-test masks
    train_mask = np.array([False] * data.num_nodes)
    val_mask = np.array([False] * data.num_nodes)
    test_mask = np.array([False] * data.num_nodes)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    data.train_mask = torch.Tensor(train_mask).bool()
    data.val_mask = torch.Tensor(val_mask).bool()
    data.test_mask = torch.Tensor(test_mask).bool()

    return data


def load_wiki(use_feat):
    df = pd.read_csv('../data/others/Wiki/graph.txt', header=None, sep='\t', names=['source', 'target'])
    G = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    df_label = pd.read_csv('../data/others/Wiki/group.txt', header=None, sep='\t', names=['node_id', 'label'])
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    if use_feat:
        feature = np.zeros((df_label['node_id'].nunique(), 4973))
        with open('../data/others/wiki/tfidf.txt') as f:
            for line in f:
                id_1, id_2, value = line.split('\t')
                feature[int(id_1)][int(id_2)] = value
        feature = torch.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = Data(x=feature,
                y=torch.LongTensor(df_label['label']),
                edge_index=from_networkx(G=G).edge_index)

    return data


def load_emails(use_feat):
    df = pd.read_csv('../data/others/Emails/Email.txt', header=None, sep=' ', names=['source', 'target'])
    graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    df_label = pd.read_csv('../data/others/Emails/Email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
    df_label = df_label[
        df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts() > 20].index)]
    available_nodes = df_label['node_id'].unique()
    graph = graph.subgraph(available_nodes)
    keys = list(graph.nodes)
    vals = range(graph.number_of_nodes())
    mapping = dict(zip(keys, vals))

    G = nx.relabel_nodes(graph, mapping, copy=True)
    df_label['node_id'] = df_label['node_id'].replace(mapping)
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    identify_oh_feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = Data(x=identify_oh_feature,
                y=torch.LongTensor(df_label['label']),
                edge_index=tg.utils.from_networkx(G=G).edge_index)

    return data


def load_acm(use_feat):
    path = '../data/others/ACM/ACM_graph.txt'
    data = np.loadtxt('../data/others/ACM/ACM.txt')
    N = data.shape[0]
    idx = np.array([i for i in range(N)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    G = nx.from_scipy_sparse_matrix(adj)

    df_label = pd.read_csv('../data/others/ACM/ACM_label.txt', header=None).reset_index()
    df_label.columns = ['node_id', 'label']
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    if use_feat:
        feature = np.loadtxt('../data/others/ACM/ACM.txt')
        feature = torch.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = tg.data.Data(
        x=feature,
        y=torch.LongTensor(df_label['label']),
        edge_index=tg.utils.from_networkx(G=G).edge_index
    )

    return data


def load_dblp(use_feat):
    path = '../data/others/DBLP/DBLP_graph.txt'
    data = np.loadtxt('../data/others/DBLP/DBLP.txt')
    N = data.shape[0]
    idx = np.array([i for i in range(N)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32).toarray()
    adj[-1][-1] = 1
    G = nx.from_numpy_array(adj)

    df_label = pd.read_csv('../data/others/DBLP/DBLP_label.txt', header=None).reset_index()
    df_label.columns = ['node_id', 'label']
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    if use_feat:
        feature = np.loadtxt('../data/others/DBLP/DBLP.txt')
        feature = torch.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = Data(
        x=feature,
        y=torch.LongTensor(df_label['label']),
        edge_index=from_networkx(G=G).edge_index
    )

    return data


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def graphDict2Adj(graph):
    return nx.adjacency_matrix(nx.from_dict_of_lists(graph), nodelist=range(len(graph)))


def load_syn_products(h):
    if h == 11:
        path = '../data/Syn-OGB-Products/99/mixhop-n10000-h0.99-c10-r1-sample-ogbn_products.{}'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path.format(names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(path.format('test.index'))
        test_idx_range = np.sort(test_idx_reorder)
    else:
        path = '../data/Syn-OGB-Products/{}/mixhop-n10000-h{}0-c10-r1-sample-ogbn_products.{}'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path.format(h, float(h / 10), names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(path.format(h, float(h / 10), 'test.index'))
        test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = graphDict2Adj(graph).astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    data = Data(
        x=torch.FloatTensor(features.toarray()),
        y=torch.LongTensor(np.where(labels > 0)[1]),
        edge_index=tg.utils.from_scipy_sparse_matrix(adj)[0]
    )
    return data


def load_data(
        data_name: str, num_train: int,
        use_feat: bool = True, to_sparse: bool = False,
        fixed_split: bool = True, split_id: int = -1, feat_norm: bool = False,
        quiet: bool = False, add_feat: bool = False, device='cpu'
):
    '''
    num_train: train ratio; 0: follow the fixed split, otherwise follow random split
    '''
    if data_name in ['Cora', 'Pubmed', 'Citeseer', 'Cora-ML', 'Photo', 'Computers', 'CS', 'Physics', 'WikiCS',
                     "Cornell", "Texas", "Wisconsin", 'Actor', "Chameleon", "Squirrel"]:
        is_PyG_data = True
    elif 'MixHopSyn' in data_name:
        is_PyG_data = True
    else:
        is_PyG_data = False

    if is_PyG_data:
        path = '../data/PyG_data/' + data_name + '/'
        if data_name in ['Cora', 'Pubmed', 'Citeseer']:
            if feat_norm:
                data = Planetoid(path, data_name, transform=T.NormalizeFeatures())[0]
            else:
                data = Planetoid(path, data_name)[0]
        elif data_name in ['Cora-ML']:
            if feat_norm:
                data = CitationFull(path, 'Cora_ML', transform=T.NormalizeFeatures())[0]
            else:
                data = CitationFull(path, 'Cora_ML')[0]
        elif data_name in ['Photo', 'Computers']:
            if feat_norm:
                data = Amazon(path, data_name, transform=T.NormalizeFeatures())[0]
            else:
                data = Amazon(path, data_name)[0]
        elif data_name in ['CS', 'Physics']:
            if feat_norm:
                data = Coauthor(path, data_name, transform=T.NormalizeFeatures())[0]
            else:
                data = Coauthor(path, data_name)[0]
        elif data_name in ['WikiCS']:
            if feat_norm:
                # data = WikiCS(path, transform=T.NormalizeFeatures())[0]
                data = WikiCS(path)[0]
                data.x = torch.FloatTensor(dgl_normalize_features(data.x))
            else:
                data = WikiCS(path)[0]
        elif data_name in ["Cornell", "Texas", "Wisconsin"]:
            if feat_norm:
                data = WebKB(path, data_name, transform=T.NormalizeFeatures())[0]
            else:
                data = WebKB(path, data_name)[0]
            data.y = data.y.long()
        elif data_name in ["Chameleon", "Squirrel"]:
            if feat_norm:
                data = WikipediaNetwork(path, data_name, transform=T.NormalizeFeatures())[0]
            else:
                data = WikipediaNetwork(path, data_name)[0]
            data.y = data.y.long()
        elif data_name in ['Actor']:
            if feat_norm:
                data = Actor(path, transform=T.NormalizeFeatures())[0]
            else:
                data = Actor(path)[0]
        else:
            data = None
            print('Data information is wrong!')

        if num_train > 0:
            data = split_data(data, num_train, data_name, split_id, fixed_split)
        if not use_feat:
            feature = torch.FloatTensor(np.identity(data.num_nodes))
            data.x = feature

    elif is_PyG_data is not True:
        if data_name == 'Wiki':
            data = load_wiki(use_feat)
        elif data_name == 'Emails':
            data = load_emails(use_feat)
        elif data_name == 'ACM':
            data = load_acm(use_feat)
        elif data_name == 'DBLP':
            data = load_dblp(use_feat)
        # elif 'syn' in data_name:
        elif data_name[:3] == 'syn':
            dataset = RandomPartitionGraph('../data/syn/', name=data_name)
            data = dataset.data
        elif 'products-syn' in data_name:
            data_name = data_name.split('products-')[1]
            dataset = RandomPartitionGraph('../data/syn/', name=data_name)
            data = dataset.data
            data.x = make_feat(path=dataset.raw_dir, name=data_name, y=data.y)
        elif 'SynProducts' in data_name:
            h = int(data_name.split('-')[1])
            data = load_syn_products(h)
        elif 'Airports' in data_name:
            G, labels = read_file(data_name, use_degree=use_feat)
            data = get_data(G, labels)
            if not use_feat:
                feature = torch.FloatTensor(np.identity(data.num_nodes))
                data.x = feature
            feat_norm = False
        elif 'Twitch' in data_name:
            dataset = load_nc_dataset(data_name)
            data = Data(
                edge_index=dataset[0][0]['edge_index'],
                x=dataset[0][0]['node_feat'],
                y=dataset[0][1].long()
            )
        elif (data_name == 'deezer-europe') or (data_name == 'Yelp-Chi'):
            dataset = load_nc_dataset(data_name)
            data = Data(
                edge_index=dataset[0][0]['edge_index'],
                x=dataset[0][0]['node_feat'],
                y=dataset[0][1].long()
            )
        else:
            print('Wrong data name!')
            pass
        if num_train > 0:
            data = split_data(data, num_train, data_name, split_id, fixed_split)
        if feat_norm:
            norm = T.NormalizeFeatures()
            data = norm(data)

    if add_feat:
        embeddings = torch.cat([torch.load(f'../input/embeddings/diffusion{data_name}.pt'),
                                torch.load(f'../input/embeddings/spectral{data_name}.pt')], dim=-1)
        # embeddings = torch.load(f'../input/embeddings/diffusion{data_name}.pt')
        # embeddings = torch.load(f'../input/embeddings/spectral{data_name}.pt')
        data.x = torch.cat([data.x, embeddings], dim=-1)
    if to_sparse:
        to_sparse = T.ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)

    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    data.name = data_name
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '
              '{} training, {} validation and {} testing nodes'.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes,
                     sum(data.train_mask), sum(data.val_mask), sum(data.test_mask)))
    return data


def load_data_ogb(data_name: str, preprocess_mode: bool = False):
    if preprocess_mode:
        data = PygNodePropPredDataset(
            name='ogbn-' + data_name,
            root='../data/OGB'
        )[0]
        return data
    else:
        dataset = PygNodePropPredDataset(
            name='ogbn-' + data_name,
            root='../data/OGB', transform=T.ToSparseTensor()
        )
        data = dataset[0]

        evaluator = Evaluator(name=f'ogbn-{data_name}')
        split_idx = dataset.get_idx_split()

        return dataset, data, evaluator, split_idx


def dgl_normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
