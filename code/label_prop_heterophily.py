from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor
from sklearn.metrics import roc_auc_score

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from utils import masked_test_OGB, makeDoubleStochastic


def get_myo_h(A, y_true, prior_estimation, mask) -> [Tensor, Tensor]:
    if A.has_value():
        A.set_value_(None)
    B = prior_estimation.clone()
    B[mask] = F.one_hot(y_true.view(-1)).float()[mask]

    # Y_true = F.one_hot(y_true.view(-1)).float()
    # H_true = torch.matmul(Y_true.transpose(0, 1), matmul(A, Y_true)) \
    #          / torch.matmul(Y_true.transpose(0, 1), matmul(A, Y_true)).sum(-1, keepdim=True)
    # H_true = makeDoubleStochastic(H_true)

    Y = torch.zeros_like(F.one_hot(y_true.view(-1))).float()
    Y[mask] = F.one_hot(y_true.view(-1)).float()[mask]
    # H = torch.matmul(Y.transpose(0, 1), matmul(A, B))
    H = torch.matmul(matmul(A, Y).transpose(0, 1), B)
    H[H == 0] = 1e-7
    H = makeDoubleStochastic(H)

    return H, B


def get_edge_weight(y_true, pred, adj, mask) -> Tensor:
    row, col, _ = adj.coo()
    H, B = get_myo_h(
        A=adj, y_true=y_true, prior_estimation=pred, mask=mask
    )
    # H = makeDoubleStochastic((dist_h + myo_h) / 2) if dist_h is not None else myo_h
    weight = torch.matmul(B[col], H) * B[row]
    return weight


def epoch_eval(y_true, out, eval_mask, test_mask, epoch, data_name, verbose):
    # epoch evaluation
    pred = out.max(1)[1]
    if ('Twitch' in data_name) or (data_name == 'Yelp-Chi'):
        acc_eval = roc_auc_score(y_true[eval_mask].detach().cpu().numpy(), out[eval_mask].detach().cpu().numpy()[:, 1])
        acc_test = roc_auc_score(y_true[test_mask].detach().cpu().numpy(), out[test_mask].detach().cpu().numpy()[:, 1])
    else:
        acc_eval = pred[eval_mask].eq(y_true[eval_mask]).sum().item() / eval_mask.sum().item()
        acc_test = pred[test_mask].eq(y_true[test_mask]).sum().item() / test_mask.sum().item()
    if verbose:
        print('Layer {}, Evaluation acc: {:.4f}, Test acc: {:.4f}'.format(
            epoch, acc_eval, acc_test
        ))
    return acc_eval, acc_test


class LabelPropagationHeterophily(MessagePassing):
    def __init__(self, max_layers: int, num_hops: int = 1):
        super(LabelPropagationHeterophily, self).__init__(aggr='add')
        self.max_layers = max_layers
        self.num_hops = num_hops

    def forward(
            self, y_true: Tensor, y_soft: Tensor, alpha: [Tensor, float],
            eval_mask: Tensor, test_mask: Tensor, data_name: str,
            adj: Adj, edge_weight: OptTensor = None, echo_weight: OptTensor = None,
            select_eval: bool = True,
            verbose: bool = True, post_step: Callable = lambda y: y.clamp_(0., 1.)
    ) -> (int, Tensor):
        # init
        best_eval = best_test = best_epoch = 0
        best_out = out = y_soft.clone()
        res = (1 - alpha) * out

        for epoch in range(1, self.max_layers+1):
            adj = adj.set_value(value=edge_weight)
            layer_out = out.clone()
            for _ in range(self.num_hops):
                layer_out = self.propagate(
                    edge_index=adj, x=layer_out, edge_weight=edge_weight, size=None
                )
            if epoch > 1 and echo_weight is not None:
                adj = adj.set_value(value=echo_weight)
                echo = self.propagate(
                    edge_index=adj, x=out, edge_weight=echo_weight, size=None
                )
                out = layer_out - echo
            else:
                out = layer_out
            out = F.normalize(out, p=1, dim=-1)
            out.mul_(alpha).add_(res)
            out = post_step(out)

            if select_eval:
                acc_eval, acc_test = epoch_eval(
                    y_true=y_true, out=out, eval_mask=eval_mask, test_mask=test_mask, epoch=epoch,
                    data_name=data_name, verbose=verbose
                )
                # save best states
                if acc_eval > best_eval:
                    best_epoch, best_eval, best_test, best_out = epoch, acc_eval, acc_test, out.clone()
                    # if verbose:
                    #     print('Update edge weight!')
                    # # TODO: speed up this calculation
                    # edge_weight = get_edge_weight(y_true, best_out, adj, spread_mask)

        if select_eval:
            return best_epoch, best_eval, best_test, best_out
        else:
            acc_eval, acc_test = epoch_eval(
                y_true=y_true, out=out, eval_mask=eval_mask, test_mask=test_mask, epoch=epoch,
                data_name=data_name, verbose=verbose
            )
            return -1, acc_eval, acc_test, out
        # return best_epoch, best_eval, best_test, best_out, -1, acc_eval, acc_test, out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        assert adj_t.has_value()
        edge_weight = adj_t.storage.value()
        if len(edge_weight.size()) == 1:
            # adj_t.set_value_(value=edge_weight)
            return matmul(adj_t, x, reduce=self.aggr)

        elif len(edge_weight.size()) == 2:
            res = []
            for idx in range(edge_weight.size(1)):
                adj_t = adj_t.set_value(edge_weight[:, idx])
                res.append(matmul(adj_t, x[:, idx].view(-1, 1), reduce=self.aggr))
            return torch.cat(res, dim=-1)

    def __repr__(self):
        return '{}(max_layers={})'.format(self.__class__.__name__, self.max_layers)


class LabelPropagationHeterophilyTog(MessagePassing):
    def __init__(self, max_layers: int, num_hops: int = 1):
        super(LabelPropagationHeterophilyTog, self).__init__(aggr='add')
        self.max_layers = max_layers
        self.num_hops = num_hops
        self.LPHete = LabelPropagationHeterophily(max_layers=max_layers, num_hops=num_hops)

    def forward(
            self, y_true: Tensor, y_soft: Tensor, alpha: [Tensor, float],
            spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor, data_name: str,
            adj: Adj, edge_weight: OptTensor = None, echo_weight: OptTensor = None,
            select_eval: bool = True,
            verbose: bool = True, post_step: Callable = lambda y: y.clamp_(0., 1.),
    ) -> (int, Tensor):
        """"""
        assert y_true.unique().size(0) == y_soft.size(1)

        # propagate prior belief
        out_1 = y_soft.clone()
        # propagate prior belief + True labels of spread nodes
        out_2 = y_soft.clone()
        out_2[spread_mask] = F.one_hot(y_true.view(-1)).float()[spread_mask]
        # propagate True labels of spread  (tested: same as the original LP)
        out_3 = torch.zeros_like(F.one_hot(y_true.view(-1)).float())
        out_3[spread_mask] = F.one_hot(y_true.view(-1)).float()[spread_mask]
        # propagate prior belief of spread nodes
        out_4 = torch.zeros_like(F.one_hot(y_true.view(-1)).float())
        out_4[spread_mask] = y_soft[spread_mask]

        if verbose:
            print('Method 1')
        epoch_1, eval_1, test_1, out_1 = self.LPHete(
            y_true=y_true, y_soft=out_1, alpha=alpha,
            eval_mask=eval_mask, test_mask=test_mask,
            adj=adj, edge_weight=edge_weight, echo_weight=echo_weight, data_name=data_name,
            verbose=verbose, post_step=post_step, select_eval=select_eval,
        )
        if verbose:
            print('Method 2')
        epoch_2, eval_2, test_2, out_2 = self.LPHete(
            y_true=y_true, y_soft=out_2, alpha=alpha,
            eval_mask=eval_mask, test_mask=test_mask,
            adj=adj, edge_weight=edge_weight, echo_weight=echo_weight, data_name=data_name,
            verbose=verbose, post_step=post_step, select_eval=select_eval,
        )
        if verbose:
            print('Method 3')
        epoch_3, eval_3, test_3, out_3 = self.LPHete(
            y_true=y_true, y_soft=out_3, alpha=alpha,
            eval_mask=eval_mask, test_mask=test_mask,
            adj=adj, edge_weight=edge_weight, echo_weight=echo_weight, data_name=data_name,
            verbose=verbose, post_step=post_step, select_eval=select_eval,
        )
        if verbose:
            print('Method 4')
        epoch_4, eval_4, test_4, out_4 = self.LPHete(
            y_true=y_true, y_soft=out_4, alpha=alpha,
            eval_mask=eval_mask, test_mask=test_mask,
            adj=adj, edge_weight=edge_weight, echo_weight=echo_weight, data_name=data_name,
            verbose=verbose, post_step=post_step, select_eval=select_eval,
        )
        res_eval = [eval_1, eval_2, eval_3, eval_4]
        idx = res_eval.index(max(res_eval))
        best_epoch = [epoch_1, epoch_2, epoch_3, epoch_4][idx]
        best_eval = [eval_1, eval_2, eval_3, eval_4][idx]
        best_test = [test_1, test_2, test_3, test_4][idx]
        best_out = [out_1, out_2, out_3, out_4][idx]

        return idx, best_epoch, best_eval, best_test, best_out

    def __repr__(self):
        return '{}(max_layers={})'.format(
            self.__class__.__name__,
            self.max_layers
        )
