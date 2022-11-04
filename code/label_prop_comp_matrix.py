from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor
from label_prop_heterophily import LabelPropagationHeterophily, LabelPropagationHeterophilyTog
from label_prop_heterophily import LabelPropagationHeterophilyOGB, LabelPropagationHeterophilyTogOGB
from utils import get_CM, get_edge_weight

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric as tg

from basic_model import Basic_MLP
from utils import makeDoubleStochastic


class CorrectAndSmoothHeterophily(torch.nn.Module):
    def __init__(self, num_correction_layers: int, correction_alpha: float,
                 num_smoothing_layers: int, smoothing_alpha: float,
                 autoscale: bool = True, scale: float = 1.0):
        super(CorrectAndSmoothHeterophily, self).__init__()
        self.autoscale = autoscale
        self.scale = scale
        self.correction_alpha = correction_alpha
        self.smoothing_alpha = smoothing_alpha

        self.prop1 = LabelPropagationHeterophily(max_layers=num_correction_layers)
        # self.prop2 = LabelPropagationHeterophily(max_layers=num_smoothing_layers)
        self.prop2 = LabelPropagationHeterophilyTog(max_layers=num_smoothing_layers)

    def correct(self, y_true: Tensor, y_soft: Tensor,
                spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
                edge_index: Adj, edge_weight: OptTensor = None,
                verbose: bool = True,) -> Tensor:
        # assert abs((float(y_soft.sum()) / y_soft.size(0)) - 1.0) < 1e-2

        numel = int(spread_mask.sum()) if spread_mask.dtype == torch.bool else spread_mask.size(0)
        # assert y_true.size(0) == numel

        error = torch.zeros_like(y_soft)
        error[spread_mask] = F.one_hot(y_true.view(-1)).float()[spread_mask] - y_soft[spread_mask]

        if self.autoscale:
            _, _, smoothed_error = self.prop1(
                y_true=None, y_soft=error,
                spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
                edge_index=edge_index, edge_weight=edge_weight,
                alpha=self.correction_alpha,
                select_eval=False, correct=True, verbose=verbose,
                post_step=lambda x: x.clamp_(-1., 1.),
            )
            sigma = error[spread_mask].abs().sum() / numel
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + scale * smoothed_error
        else:
            def fix_input(x):
                x[spread_mask] = error[spread_mask]
                return x
            _, _, smoothed_error = self.prop1(
                y_true=None, y_soft=error,
                spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
                edge_index=edge_index, edge_weight=edge_weight,
                alpha=self.correction_alpha,
                select_eval=False, correct=True, verbose=verbose,
                post_step=fix_input,
            )
            return y_soft + self.scale * smoothed_error

    def smooth(self, y_true: Tensor, y_soft: Tensor,
               spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
               edge_index: Adj, edge_weight: OptTensor = None,
               select_eval: bool = True, verbose: bool = True,
               post_step: Callable = lambda y: F.softmax(y, dim=-1)) -> Tensor:

        return self.prop2(
            y_true=y_true, y_soft=y_soft,
            spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
            edge_index=edge_index, edge_weight=edge_weight,
            alpha=self.smoothing_alpha,
            select_eval=select_eval, verbose=verbose,
            post_step=post_step,
        )

    def __repr__(self):
        L1, alpha1 = self.prop1.max_layers, self.correction_alpha
        L2, alpha2 = self.prop2.max_layers, self.smoothing_alpha
        return (f'{self.__class__.__name__}(\n'
                f'    correct: max_layers={L1}, alpha={alpha1}\n'
                f'    smooth:  max_layers={L2}, alpha={alpha2}\n'
                f'    autoscale={self.autoscale}, scale={self.scale}\n'
                ')')


class CorrectAndSmoothHeterophilyOGB(torch.nn.Module):
    def __init__(self, num_correction_layers: int, correction_alpha: float,
                 num_smoothing_layers: int, smoothing_alpha: float,
                 autoscale: bool = True, scale: float = 1.0):
        super(CorrectAndSmoothHeterophilyOGB, self).__init__()
        self.autoscale = autoscale
        self.scale = scale
        self.correction_alpha = correction_alpha
        self.smoothing_alpha = smoothing_alpha

        self.prop1 = LabelPropagationHeterophilyOGB(max_layers=num_correction_layers)
        # self.prop2 = LabelPropagationHeterophily(max_layers=num_smoothing_layers)
        self.prop2 = LabelPropagationHeterophilyTogOGB(max_layers=num_smoothing_layers)

    def correct(self, y_true: Tensor, y_soft: Tensor,
                spread_idx: Tensor, eval_idx: Tensor, test_idx: Tensor, evaluator,
                edge_index: Adj, edge_weight: OptTensor = None,
                verbose: bool = True,) -> Tensor:
        numel = int(spread_idx.sum()) if spread_idx.dtype == torch.bool else spread_idx.size(0)

        y_true_oh = F.one_hot(y_true.view(-1)).float()
        error = torch.zeros_like(y_soft)
        error[spread_idx] = y_true_oh[spread_idx] - y_soft[spread_idx]

        if self.autoscale:
            _, _, smoothed_error = self.prop1(
                y_true=None, y_soft=error,
                spread_idx=spread_idx, eval_idx=eval_idx, test_idx=test_idx, evaluator=evaluator,
                edge_index=edge_index, edge_weight=edge_weight,
                alpha=self.correction_alpha,
                select_eval=False, correct=True, verbose=verbose,
                post_step=lambda x: x.clamp_(-1., 1.),
            )
            sigma = error[spread_idx].abs().sum() / numel
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + scale * smoothed_error
        else:
            def fix_input(x):
                x[spread_idx] = error[spread_idx]
                return x
            _, _, smoothed_error = self.prop1(
                y_true=None, y_soft=error,
                spread_idx=spread_idx, eval_idx=eval_idx, test_idx=test_idx, evaluator=evaluator,
                edge_index=edge_index, edge_weight=edge_weight,
                alpha=self.correction_alpha,
                select_eval=False, correct=True, verbose=verbose,
                post_step=fix_input,
            )
            return y_soft + self.scale * smoothed_error

    def smooth(self, y_true: Tensor, y_soft: Tensor,
               spread_idx: Tensor, eval_idx: Tensor, test_idx: Tensor, evaluator,
               edge_index: Adj, edge_weight: OptTensor = None,
               select_eval: bool = True, verbose: bool = True,
               post_step: Callable = lambda y: F.softmax(y, dim=-1)) -> Tensor:

        return self.prop2(
            y_true=y_true, y_soft=y_soft,
            spread_idx=spread_idx, eval_idx=eval_idx, test_idx=test_idx, evaluator=evaluator,
            edge_index=edge_index, edge_weight=edge_weight,
            alpha=self.smoothing_alpha,
            select_eval=select_eval, verbose=verbose,
            post_step=post_step,
        )

    def __repr__(self):
        L1, alpha1 = self.prop1.max_layers, self.correction_alpha
        L2, alpha2 = self.prop2.max_layers, self.smoothing_alpha
        return (f'{self.__class__.__name__}(\n'
                f'    correct: max_layers={L1}, alpha={alpha1}\n'
                f'    smooth:  max_layers={L2}, alpha={alpha2}\n'
                f'    autoscale={self.autoscale}, scale={self.scale}\n'
                ')')


class LabelPropagationHeterophilyEnd2End1(MessagePassing):
    def __init__(self, max_layers: int, N: int):
        super(LabelPropagationHeterophilyEnd2End1, self).__init__(aggr='add')
        self.max_layers = max_layers
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        # self.alpha = nn.Parameter(torch.FloatTensor(N, 1))
        # self.alpha = nn.Parameter(torch.FloatTensor([0.9]*N).view(-1, 1))
        torch.nn.init.uniform_(self.alpha)
        # nn.init.xavier_normal_(self.alpha, gain=1.414)

        self.LPHete = LabelPropagationHeterophily(max_layers=max_layers)

    def forward(
            self, y_true: Tensor, y_soft: Tensor,
            spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
            edge_index: [Adj, Tensor], edge_weight: OptTensor = None,
            select_eval: bool = True, verbose: bool = True,
            post_step: Callable = lambda y: y.clamp_(0., 1.),
    ) -> (int, Tensor):

        _, _, out = self.LPHete(
            y_true=y_true, y_soft=y_soft,
            spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
            edge_index=edge_index, edge_weight=edge_weight,
            alpha=self.alpha, select_eval=select_eval, verbose=verbose,
            post_step=post_step,
        )

        return out

    def __repr__(self):
        return '{}(max_layers={})'.format(self.__class__.__name__,
                                                    self.max_layers)


class LabelPropagationHeterophilyEnd2End2(MessagePassing):
    def __init__(self, max_layers: int, N: int, in_dim: int, out_dim: int):
        super(LabelPropagationHeterophilyEnd2End2, self).__init__(aggr='add')
        self.max_layers = max_layers

        self.alpha = nn.Parameter(torch.FloatTensor(1))
        # self.alpha = nn.Parameter(torch.FloatTensor(N, 1))
        # self.alpha = nn.Parameter(torch.FloatTensor([0.9] * N).view(-1, 1))
        torch.nn.init.uniform_(self.alpha)
        # nn.init.xavier_normal_(self.alpha, gain=1.414)

        self.prior_estimator = Basic_MLP(
            in_dim=in_dim,
            out_dim=out_dim
        )
        self.LPHete = LabelPropagationHeterophily(max_layers=max_layers)

    def forward(
            self, data,
            spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
            select_eval: bool, verbose: bool = True,
            post_step: Callable = lambda y: F.softmax(y, dim=-1),
    ) -> (int, Tensor):
        """"""
        y_true = data.y.clone()
        edge_index = data.adj_t.clone()
        prior_pred = self.prior_estimator(data)
        y_soft = F.softmax(self.prior_estimator.get_embedding(), dim=-1)

        edge_weight = get_edge_weight(
            A=edge_index, y_true=y_true, y_soft=y_soft, mask=spread_mask,
        )

        _, _, out = self.LPHete(
            y_true=y_true, y_soft=y_soft,
            spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
            edge_index=edge_index, edge_weight=edge_weight,
            alpha=self.alpha, select_eval=select_eval, verbose=verbose,
            post_step=post_step,
        )

        return out, prior_pred

    def __repr__(self):
        return '{}(max_layers={})'.format(self.__class__.__name__,
                                                    self.max_layers)


def train_LPCM1(
        model, optim, y_true: Tensor, y_soft: Tensor,
        spread_mask: Tensor, train_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
        edge_index: [Adj, Tensor], edge_weight: OptTensor = None,
        select_eval: bool = True, verbose: bool = True,
):
    model.train()
    optim.zero_grad()

    output = model(
        y_true=y_true, y_soft=y_soft,
        spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
        edge_index=edge_index, edge_weight=edge_weight,
        select_eval=select_eval, verbose=verbose,
    )
    logits = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(logits[train_mask], y_true[train_mask])

    loss.backward()
    optim.step()
    return float(loss)


@torch.no_grad()
def test_LPCM1(
        model, y_true: Tensor, y_soft: Tensor,
        spread_mask: Tensor, train_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
        edge_index: [Adj, Tensor], edge_weight: OptTensor = None,
        select_eval: bool = True
):
    model.eval()

    logits, accs = model(
        y_true=y_true, y_soft=y_soft,
        spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
        edge_index=edge_index, edge_weight=edge_weight,
        verbose=False, select_eval=select_eval
    ), []

    for mask in [train_mask, eval_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y_true[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits


def train_LPCM2(
        model, optim, data,
        train_mask: Tensor,  spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
        select_eval: bool, post_step: Callable = lambda y: F.softmax(y, dim=-1),
):
    model.train()
    optim.zero_grad()

    y_true = data.y
    (output, prior_pred) = model(
        data=data, select_eval=select_eval,
        spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
        post_step=post_step,
    )
    logits = F.log_softmax(output, dim=-1)
    loss1 = F.nll_loss(logits[train_mask], y_true[train_mask])
    loss2 = F.nll_loss(prior_pred[train_mask], y_true[train_mask])
    loss = loss1 + loss2
    # loss = loss1 + 0.5 * loss2
    # loss = float(model.alpha) * loss1 + (1 - float(model.alpha)) * loss2
    # loss = loss2
    # loss = loss1

    loss.backward()
    optim.step()
    return float(loss)


@torch.no_grad()
def test_LPCM2(
        model, data,
        train_mask: Tensor, spread_mask: Tensor, eval_mask: Tensor, test_mask: Tensor,
        select_eval: bool, post_step: Callable = lambda y: F.softmax(y, dim=-1),
):
    model.eval()

    y_true = data.y
    (logits, _), accs = model(
        data=data, select_eval=select_eval,
        spread_mask=spread_mask, eval_mask=eval_mask, test_mask=test_mask,
        verbose=False, post_step=post_step,
    ), []

    for mask in [train_mask, eval_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y_true[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits


# def get_CM(data, prior_estimation, mask, method):
#     B = prior_estimation.clone()
#     B[mask] = F.one_hot(data.y.view(-1)).float()[mask]
#     A = data.adj_t.clone()
#
#     Y_true = F.one_hot(data.y.view(-1)).float()
#     E = torch.ones_like(F.one_hot(data.y.view(-1)).float())
#     # numerator = torch.matmul(Y_true.transpose(0, 1), matmul(A, Y_true))
#     # denominator = torch.matmul(Y_true.transpose(0, 1), matmul(A, E))
#     # denominator[denominator == 0] = 1e-7
#     # H_true = torch.divide(numerator, denominator)
#     H_true = torch.matmul(Y_true.transpose(0, 1), matmul(A, Y_true))\
#             / torch.matmul(Y_true.transpose(0, 1), matmul(A, Y_true)).sum(-1, keepdim=True)
#     H_true = makeDoubleStochastic(H_true)
#
#     if method == 1:
#         # Method - 1
#         Y = torch.zeros_like(F.one_hot(data.y.view(-1))).float()
#         Y[mask] = F.one_hot(data.y.view(-1)).float()[mask]
#         H = torch.matmul(Y.transpose(0, 1), matmul(A, B))
#         H[H == 0] = 1e-7
#         H = makeDoubleStochastic(H)
#     elif method == 2:
#         # Method - 2
#         Y = torch.zeros_like(F.one_hot(data.y.view(-1))).float()
#         Y[mask] = F.one_hot(data.y.view(-1)).float()[mask]
#         numerator = torch.matmul(Y.transpose(0, 1), matmul(A, B))
#         denominator = torch.matmul(Y.transpose(0, 1), matmul(A, B)).sum(-1, keepdim=True)
#         denominator[denominator == 0] = 1e-7
#         H = torch.divide(numerator, denominator)
#     elif method == 3:
#         # Method - 3
#         E = torch.ones_like(F.one_hot(data.y.view(-1)).float())
#         H = torch.divide(torch.matmul(B.transpose(0, 1), matmul(A, B)),\
#                            torch.matmul(B.transpose(0, 1), matmul(A, E)))
#     return H, H_true, A, B


# class LabelPropagationHeterophilyEnd2End(MessagePassing):
#     r"""
#         Args:
#         max_layers (int): The max number of propagations.
#     """
#
#     def __init__(self, max_layers: int, num_nodes: int, in_dim: int, out_dim: int):
#         super(LabelPropagationHeterophilyEnd2End, self).__init__(aggr='add')
#         self.max_layers = max_layers
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#
#         self.alpha = nn.Parameter(torch.FloatTensor(num_nodes, 2))
#         nn.init.xavier_normal_(self.alpha, gain=1.414)
#
#         self.basic_estimator = Basic_MLP(
#             in_dim, out_dim
#         )
#
#     def forward(
#             self, data,
#             spread_true_label: bool = False,
#             post_step: Callable = lambda y: F.softmax(y, dim=-1)
#     ) -> (int, Tensor):
#         """"""
#         y = data.y
#         eval_mask = data.train_mask + data.val_mask
#         test_mask = data.test_mask
#         device = data.x.device
#
#         # out = prior_estm
#         out = F.softmax(self.basic_estimator(data), dim=-1)
#         edge_index = data.edge_index
#
#         B = out.clone()
#         B[eval_mask] = F.one_hot(y.view(-1)).float().to(device)[eval_mask]
#         A = tg.utils.to_dense_adj(edge_index)[0]
#         E = torch.ones((data.num_nodes, self.out_dim)).to(device)
#         H = torch.divide(torch.matmul(torch.matmul(B.t(), A), B), torch.matmul(torch.matmul(B.t(), A), E))
#         cm_weight = H[out[edge_index[0]].max(1)[1], out[edge_index[1]].max(1)[1]]
#         gcn_weight = gcn_norm(edge_index, add_self_loops=False)[1]
#         edge_weight = cm_weight * gcn_weight
#
#         res = F.softmax(self.alpha, dim=-1)[:, 1].unsqueeze(-1) * out
#         best_epoch, best_eval = 0, 0
#         best_out = out
#         for epoch in range(self.max_layers):
#             if epoch > 0:
#                 # propagate_type: (y: Tensor, edge_weight: OptTensor)
#                 if spread_true_label:
#                     to_prop = out.clone()
#                     to_prop[eval_mask] = F.one_hot(y.view(-1)).float()[eval_mask]
#                     out = self.propagate(
#                         edge_index, x=to_prop,
#                         edge_weight=edge_weight, size=None
#                     )
#                 else:
#                     out = self.propagate(
#                         edge_index, x=out,
#                         edge_weight=edge_weight, size=None
#                     )
#                 # out.mul_(self.alpha).add_(res)
#                 out.mul_(F.softmax(self.alpha, dim=-1)[:, 0].unsqueeze(-1)).add_(res)
#                 out = post_step(out)
#
#             # epoch evaluation
#             pred = out.max(1)[1]
#             acc_eval = pred[eval_mask].eq(y[eval_mask]).sum().item() / eval_mask.sum().item()
#             acc_test = pred[test_mask].eq(y[test_mask]).sum().item() / test_mask.sum().item()
#             print('Epoch {}, Evaluation acc: {:.4f}, Test acc: {:.4f}'.format(
#                 epoch, acc_eval, acc_test
#             ))
#             # save best states
#             if acc_eval > best_eval:
#                 best_epoch = epoch
#                 best_eval = acc_eval
#                 best_out = out.clone()
#
#         return best_epoch, F.log_softmax(best_out, dim=-1)
#
#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
#
#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return matmul(adj_t, x, reduce=self.aggr)
#
#     def __repr__(self):
#         return '{}(max_layers={}, in_dim={}, out_dim={})'.format(
#             self.__class__.__name__,
#             self.max_layers, self.in_dim, self.out_dim
#         )