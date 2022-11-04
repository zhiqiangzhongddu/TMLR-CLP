from torch import Tensor
from torch_sparse import SparseTensor
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition
import torch.nn as nn
from torch_sparse import matmul
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from load_data import *


def seed(value: int = 42):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    random.seed(value)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def statistic_info(data, pred, data_name, num_train, n_layers, hop, split_id, model_name, test_acc):
    # load homophily node set
    node_set_file = open(f'../input/dic_homophily_node/dic_homophily_node_set_{data_name}_hop_{hop}.pkl', 'rb')
    dic_homophily_node_set = pickle.load(node_set_file)
    node_set_file.close()
    # load homophily score file
    homophily_score_file = open(f'../input/dic_homophily_node/dic_homophily_score_{data_name}_hop_{hop}.pkl', 'rb')
    dic_homophily_score = pickle.load(homophily_score_file)
    homophily_score_file.close()

    node_labels = data.y.data.cpu().numpy()
    test_nodes = torch.where(data.test_mask == True)[0].tolist()
    acc_by_homophily = [test_acc]

    for score in range(11):
        score = score / 10
        nodes = dic_homophily_node_set[score]
        nodes = list(set(test_nodes).intersection(set(nodes)))
        acc = accuracy_score(y_pred=np.argmax(pred[nodes], axis=1),
                             y_true=node_labels[nodes])
        print('hop {}, h {}, #nodes {}, acc {}'.format(
            hop, score, len(nodes), round(acc, 4)
        ))
        acc_by_homophily.append(acc)

    acc_by_homophily = np.array([acc_by_homophily])
    acc_by_homophily = np.nan_to_num(acc_by_homophily)

    path = "../output/h_perform/homophily_perform_{}_{}_RT_{}_L_{}_hop_{}_split_{}.txt". \
        format(data_name, model_name, num_train, n_layers, hop, split_id)
    print(path)
    with open(path, "ab") as f:
        np.savetxt(f, acc_by_homophily, fmt='%1.4f', delimiter=' ', newline='\n')


def statistic_hop_info(data, pred, data_name, num_train, n_layers, hop, split_id, model_name, test_acc):
    # load homophily node set
    node_set_file = open(f'../input/dic_homophily_node/dic_hop_homophily_node_set_{data_name}_hop_{hop}.pkl', 'rb')
    dic_homophily_node_set = pickle.load(node_set_file)
    node_set_file.close()
    # load homophily score file
    homophily_score_file = open(f'../input/dic_homophily_node/dic_hop_homophily_score_{data_name}_hop_{hop}.pkl', 'rb')
    dic_homophily_score = pickle.load(homophily_score_file)
    homophily_score_file.close()

    node_labels = data.y.data.cpu().numpy()
    test_nodes = torch.where(data.test_mask == True)[0].tolist()
    acc_by_homophily = [test_acc]

    for score in range(11):
        score = score / 10
        nodes = dic_homophily_node_set[score]
        nodes = list(set(test_nodes).intersection(set(nodes)))
        acc = accuracy_score(y_pred=np.argmax(pred[nodes], axis=1),
                             y_true=node_labels[nodes])
        print('hop {}, h {}, #nodes {}, acc {}'.format(
            hop, score, len(nodes), round(acc, 4)
        ))
        acc_by_homophily.append(acc)

    acc_by_homophily = np.array([acc_by_homophily])
    acc_by_homophily = np.nan_to_num(acc_by_homophily)

    path = "../output/h_perform/hop_homophily_perform_{}_{}_RT_{}_L_{}_hop_{}_split_{}.txt". \
        format(data_name, model_name, num_train, n_layers, hop, split_id)
    print(path)
    with open(path, "ab") as f:
        np.savetxt(f, acc_by_homophily, fmt='%1.4f', delimiter=' ', newline='\n')


def corr_hf_hl(data_name, decomp=False, subgraph=True):
    # load dataset
    data = load_data(
        data_name, use_feat=True, num_train=10,
        feat_norm=False,
        to_sparse=False, device='cpu'
    )
    # remove self-loop
    data.edge_index = tg.utils.remove_self_loops(data.edge_index)[0]
    h_g = tg.utils.homophily_ratio(edge_index=data.edge_index, y=data.y)

    if decomp:
        # compress node features
        if data.num_features > 10:
            pca = decomposition.PCA(n_components=64)
            data.x = torch.FloatTensor(pca.fit_transform(data.x.data.numpy()))

    # load node homophily file (h_l)
    if subgraph:
        homophily_score_file = open(f'../input/dic_homophily_node/dic_homophily_score_{data_name}_hop_1.pkl', 'rb')
    else:
        homophily_score_file = open(f'../input/dic_homophily_node/dic_hop_homophily_score_{data_name}_hop_1.pkl', 'rb')
    dic_homophily_score = pickle.load(homophily_score_file)
    homophily_score_file.close()
    valid_nodes = [idx for idx in dic_homophily_score if dic_homophily_score[idx] > -1]
    print(f'There are {len(valid_nodes)} valid nodes')
    h_l = np.array([dic_homophily_score[idx] for idx in valid_nodes])

    # calculate h_f
    h_f = []
    for idx in valid_nodes:
        (fr, to) = data.edge_index[:, data.edge_index[1] == idx]
        tmp = cosine_similarity(data.x[fr], data.x[to], dense_output=False).diagonal()
        # normalisation from (-1, 1) to (0, 1)
        h_f.append(np.mean((tmp + 1) / 2))
    h_f = np.array(h_f)

    return h_g, h_l, h_f, np.corrcoef(h_l, h_f)[0, 1]


def corr_hmlp_hl(data_name, data, node_emb, subgraph=True):
    # load dataset
    data = data.to('cpu')
    # remove self-loop
    data.edge_index = tg.utils.remove_self_loops(data.edge_index)[0]
    # h_g = tg.utils.homophily_ratio(edge_index=data.edge_index, y=data.y)

    # load node homophily file (h_l)
    if subgraph:
        homophily_score_file = open(f'../input/dic_homophily_node/dic_homophily_score_{data_name}_hop_1.pkl', 'rb')
    else:
        homophily_score_file = open(f'../input/dic_homophily_node/dic_hop_homophily_score_{data_name}_hop_1.pkl', 'rb')
    dic_homophily_score = pickle.load(homophily_score_file)
    homophily_score_file.close()
    valid_nodes = [idx for idx in dic_homophily_score if dic_homophily_score[idx] > -1]
    print(f'There are {len(valid_nodes)} valid nodes have in-coming edges')
    h_l = np.array([dic_homophily_score[idx] for idx in valid_nodes])

    # calculate h_mlp
    h_mlp = []
    for idx in valid_nodes:
        (fr, to) = data.edge_index[:, data.edge_index[1] == idx]
        tmp = cosine_similarity(node_emb[fr], node_emb[to], dense_output=False).diagonal()
        # normalisation from (-1, 1) to (0, 1)
        h_mlp.append(np.mean((tmp + 1) / 2))
    h_mlp = np.array(h_mlp)

    return h_l, h_mlp, np.corrcoef(h_l, h_mlp)[0, 1]


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y].to(y.device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Embedding') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1.414)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1.414)


@torch.no_grad()
def masked_test(data, preds):
    accs = []
    if ('Twitch' in data.name) or (data.name == 'Yelp-Chi'):
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = preds[mask]
            accs.append(roc_auc_score(data.y[mask].detach().cpu().numpy(), pred.detach().cpu().numpy()[:, 1]))
    else:
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = preds[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     pred = preds[mask].max(1)[1]
    #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     accs.append(acc)
    return accs


@torch.no_grad()
def masked_test_OGB(pred, y_true, evaluator, train_idx, val_idx, test_idx):
    pred = pred.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y_true[train_idx],
        'y_pred': pred[train_idx]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[val_idx],
        'y_pred': pred[val_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[test_idx],
        'y_pred': pred[test_idx]
    })['acc']
    return train_acc, val_acc, test_acc


def makeDoubleStochastic(h, max_iterations=1000, delta_limit=1e-12):
    converge = False
    i = 0
    while not converge and i < max_iterations:
        prev_h = h.clone()
        h /= h.sum(0, keepdim=True)
        h /= h.sum(1, keepdim=True)

        delta = torch.linalg.norm(h - prev_h, ord=1)
        # print(i, delta)
        if delta < delta_limit:
            converge = True
        i += 1
    # if i == max_iterations:
    #     print("makeDoubleStochasticH: maximum number of iterations reached.")

    return h


def get_diag(src: SparseTensor) -> Tensor:
    """
    Copy from https://github.com/rusty1s/pytorch_sparse/blob/master/torch_sparse/diag.py
    :param src:
    :return:
    """
    row, col, value = src.coo()

    if value is None:
        value = torch.ones(row.size(0))

    sizes = list(value.size())
    sizes[0] = min(src.size(0), src.size(1))

    out = value.new_zeros(sizes)

    mask = row == col
    out[row[mask]] = value[mask]
    return out


def get_dist_h(A, y_true, n_hop, mask, nbp=False) -> Tensor:
    if not A.has_value():
        adj_t = A.set_value(torch.ones(A.storage.row().size(0)).float().to(y_true.device))
    else:
        adj_t = A

    if not nbp:
        # with Backtracking Paths
        for hop in range(1, n_hop):
            adj_t = matmul(adj_t, A)
    else:
        # without Backtracking Paths
        D = A.sum(-1).float()
        I = torch.ones_like(D)
        adj_ts = [adj_t]
        for i in range(1, n_hop):
            if i == 1:
                adj_t = matmul(A, adj_t)
                adj_t = adj_t.set_diag(get_diag(adj_t)-D)
                adj_ts.append(adj_t)
            else:
                adj_t = matmul(A, adj_t)
                adj_t = adj_t.set_diag(get_diag(adj_t) - (D-I)*get_diag(adj_ts[-2]))
                adj_ts.append(adj_t)

    Y = torch.zeros_like(F.one_hot(y_true.view(-1))).float()
    Y[mask] = F.one_hot(y_true.view(-1)).float()[mask]
    H = torch.matmul(matmul(adj_t, Y).transpose(0, 1), Y)
    H[H == 0] = 1e-7
    H = makeDoubleStochastic(H)

    return H