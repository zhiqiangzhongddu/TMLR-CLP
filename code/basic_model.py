from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import torch.nn as nn


class Basic_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, num_layers=3, dropout=0.5):
        super(Basic_MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = None

        self.conv_layers = nn.ModuleList()
        if num_layers == 1:
            self.lin = nn.Linear(self.in_dim, self.out_dim)
        else:
            for i in range(num_layers):
                if i == 0:
                    self.conv_layers.append(nn.Linear(self.in_dim, hid_dim))
                elif i == num_layers-1:
                    self.conv_layers.append(nn.Linear(hid_dim, self.out_dim))
                else:
                    self.conv_layers.append(nn.Linear(hid_dim, hid_dim))

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        if self.num_layers == 1:
            h = self.lin(h)
        else:
            for layer in self.conv_layers[:-1]:
                h = F.relu(layer(h))
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv_layers[-1](h)
        return h


def train_MLP(data, model, optim):
    model.train()
    optim.zero_grad()
    criterion = nn.CrossEntropyLoss()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optim.step()
    return float(loss)


@torch.no_grad()
def test_MLP(data, model):
    model.eval()

    output, accs = model(data), []
    logits = F.softmax(output, dim=-1)
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     pred = logits[mask].max(1)[1]
    #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     accs.append(acc)
    if ('Twitch' in data.name) or (data.name == 'Yelp-Chi'):
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask]
            # acc = eval_rocauc(
            #     y_true=data.y.view(-1, 1)[mask],
            #     y_pred=pred.detach()[:, 1]
            # )
            # accs.append(acc)
            accs.append(roc_auc_score(data.y[mask].detach().cpu().numpy(), pred.detach().cpu().numpy()[:, 1]))
    else:
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs, logits
