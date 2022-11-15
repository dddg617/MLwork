import argparse
import tqdm
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io
from model import SimpleHGNLayer

import dgl

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    g = dgl.to_homogeneous(G, ndata = "h")
    fanouts = [-1] * args.n_layers
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    train_loader = dgl.dataloading.DataLoader(g, train_idx.to(device), sampler,
                                            batch_size=args.batch_size, device=device,
                                            shuffle=True)
    # val_loader = dgl.dataloading.DataLoader(g, val_idx, sampler,
    #                                         batch_size=args.batch_size, device=device,
    #                                         shuffle=True)
    # test_loader = dgl.dataloading.DataLoader(g, test_idx, sampler,
    #                                         batch_size=args.batch_size, device=device,
    #                                         shuffle=True)
    out = nn.Linear(args.out_dim, labels.max().item() + 1).to(device)
    for epoch in np.arange(args.n_epoch) + 1:
        feature = g.ndata['h']
        # loader_tqdm = tqdm(train_loader, ncols=120)
        for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
            emb = feature[input_nodes]
            h = model(blocks, emb)["paper"]
            logits = out(h)
            # The loss is computed only for labeled nodes.
            loss = F.cross_entropy(logits, labels[seeds - G.num_nodes(ntype = "author")].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        if epoch % 5 == 0:
            model.eval()
            h = model(g, g.ndata['h'])["paper"]
            logits = out(h)
            pred = logits.argmax(1).cpu()

            train_acc = (pred[train_id] == labels[train_id]).float().mean()
            val_acc = (pred[val_id] == labels[val_id]).float().mean()
            test_acc = (pred[test_id] == labels[test_id]).float().mean()
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

torch.manual_seed(0)
data_url = "https://data.dgl.ai/dataset/ACM.mat"
data_file_path = "/tmp/ACM.mat"

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)

parser = argparse.ArgumentParser(
    description="Training GNN on ACM benchmark"
)
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--in_dim", type=int, default=256)
parser.add_argument("--edge_dim", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--out_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--feat_drop", type=float, default=0.2)
parser.add_argument("--negative_slope", type=float, default=0.2)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--max_lr", type=float, default=1e-3)

args = parser.parse_args()

device = torch.device("cuda:0")

G = dgl.heterograph(
    {
        ("paper", "written-by", "author"): data["PvsA"].nonzero(),
        ("author", "writing", "paper"): data["PvsA"].transpose().nonzero(),
        ("paper", "citing", "paper"): data["PvsP"].nonzero(),
        ("paper", "cited", "paper"): data["PvsP"].transpose().nonzero(),
        ("paper", "is-about", "subject"): data["PvsL"].nonzero(),
        ("subject", "has", "paper"): data["PvsL"].transpose().nonzero(),
    }
)
print(G)

pvc = data["PvsC"].tocsr()
p_selected = pvc.tocoo()
# generate labels
labels = pvc.indices
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_id = torch.tensor(shuffle[0:800]).long()
val_id = torch.tensor(shuffle[800:900]).long()
test_id = torch.tensor(shuffle[900:]).long()
train_idx = G.num_nodes(ntype = "author") + train_id
val_idx = G.num_nodes(ntype = "author") + val_id
test_idx = G.num_nodes(ntype = "author") + test_id
#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(
        torch.Tensor(G.number_of_nodes(ntype), args.in_dim), requires_grad=False
    )
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data["h"] = emb

G = G.to(device)

model = SimpleHGNLayer(
    args.edge_dim,
    len(G.etypes),
    args.in_dim,
    args.hidden_dim,
    args.out_dim,
    args.num_layers,
    args.num_heads,
    args.feat_drop,
    args.negative_slope,
    residual=True,
    beta = args.beta,
    ).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=args.n_epoch, max_lr=args.max_lr
)
print("Training SimpleHGN with #param: %d" % (get_n_params(model)))
train(model, G)

