import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from simplehgn_conv import SimpleHGNConv

class SimpleHGNLayer(nn.Module):
    r"""
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__

    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.

    Calculating the coefficient:
    
    .. math::
        \alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}} \quad (1)
    
    Residual connection including Node residual:
    
    .. math::
        h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)}) \quad (2)
    
    and Edge residual:
        
    .. math::
        \alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)} \quad (3)
        
    Multi-heads:
    
    .. math::
        h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (4)
    
    Residual:
    
        .. math::
            h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (5)
    
    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hidden_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    """    
    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop, negative_slope,
                residual, beta, ntypes):
        super(SimpleHGNLayer, self).__init__()
        self.ntypes = ntypes
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim[0],
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the SimpleHGN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        if hasattr(hg, 'ntypes'):
            # full graph training,
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata = 'h')
                h = g.ndata['h']
                for l in range(self.num_layers):  # noqa E741
                    h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)

            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        else:
            # for minibatch training, input h_dict is a tensor
            h = h_dict
            for layer, block in zip(self.hgn_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)

        return h_dict

def to_hetero_feat(h, type, name):
    """Feature convert API.
    
    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.
    
    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.
    
    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph
    
    Example
    -------
    
    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}
    
    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]

    return h_dict



class RGCN(nn.Module):
    """
    **Title:** `Modeling Relational Data with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`_

    **Authors:** Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling

    Parameters
    ----------
    in_dim : int
        Input feature size.
    hidden_dim : int
        Hidden dimension .
    out_dim : int
        Output feature size.
    etypes : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    num_hidden_layers: int
        Number of RelGraphConvLayer
    dropout : float, optional
        Dropout rate. Default: 0.0
    use_self_loop : bool, optional
        True to include self loop message. Default: False

    Attributes
    -----------
    RelGraphConvLayer: RelGraphConvLayer

    """
    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 etypes,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=True))
        # hidden 2 hidden
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # hidden 2 output
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, hg, h_dict):
        r"""
        Support full-batch and mini-batch training.

        Parameters
        ----------
        hg: dgl.HeteroGraph or dgl.blocks
            Input graph
        h_dict: dict[str, torch.Tensor]
            Input feature
        Returns
        -------
        h: dict[str, torch.Tensor]
            output feature
        """
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict

    def l2_penalty(self):
        loss = 0.0005 * torch.norm(self.layers[0].weight, p=2, dim=1)
        return loss


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    We use `HeteroGraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#heterographconv>`_ to implement the model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = False

        self.conv = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dgl.nn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}