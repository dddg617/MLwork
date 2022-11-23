import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import GTNDataset
from model import SimpleHGNLayer

from dgl.nn.pytorch import HeteroLinear


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

parser = argparse.ArgumentParser(
    description="Training SimpleHGN"
)
parser.add_argument("--dataset", type=str, default="acm4GTN")
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--in_dim", type=int, default=256)
parser.add_argument("--edge_dim", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--out_dim", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--feat_drop", type=float, default=0.2)
parser.add_argument("--negative_slope", type=float, default=0.2)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--max_lr", type=float, default=1e-3)

args = parser.parse_args()

device = torch.device("cuda:1")

torch.manual_seed(0)

data = GTNDataset(args.dataset)
hg = data.__getitem__()
category = data.category
print(hg.ndata)
train_mask = hg.nodes[category].data.pop('train_mask')
val_mask = hg.nodes[category].data.pop('val_mask')
test_mask = hg.nodes[category].data.pop('test_mask')
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
hg = hg.to(device)
label = hg.nodes[category].data['label']
best_val_acc = torch.tensor(0)
best_test_acc = torch.tensor(0)
train_step = torch.tensor(0)
in_dim = [args.in_dim] + [args.hidden_dim] * args.num_layers
heads = [args.num_heads] * args.num_layers + [1]
model = SimpleHGNLayer(
    args.edge_dim,
    len(hg.etypes),
    in_dim,
    args.hidden_dim,
    args.out_dim,
    args.num_layers,
    heads,
    args.feat_drop,
    args.negative_slope,
    True,
    args.beta,
    hg.ntypes,
    ).to(device)
dim = {}
for ntype in hg.ntypes:
    dim[ntype] = hg.ndata['h'][ntype].shape[1]
linear = HeteroLinear(dim, args.in_dim).to(device)
out = nn.Linear(args.out_dim, label.max().item() + 1).to(device)
optimizer = torch.optim.AdamW([{'params': linear.parameters()},
                               {'params': model.parameters()}])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
print("Training SimpleHGN with #param: %d" % (get_n_params(model)))

for epoch in range(args.n_epoch):
    feature = linear(hg.ndata['h'])
    logits = model(hg, feature)[category]
    pred = out(logits)
    loss = F.cross_entropy(pred, label)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    if epoch % 5 == 0:
        model.eval()
        h = model(hg, feature)[category]
        logits = out(h)
        pred = logits.argmax(1)

        train_acc = (pred[train_idx] == label[train_idx]).float().mean()
        val_acc = (pred[val_idx] == label[val_idx]).float().mean()
        test_acc = (pred[test_idx] == label[test_idx]).float().mean()
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        print(
            "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
            % (
                epoch,
                optimizer.param_groups[0]["lr"],
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            )
        )    
    
