# MLwork

This is the Group Work for Machine Learning Course.

## SimpleHGN[KDD 2021]

-   paper: [Are we really making much progress? Revisiting, benchmarking,and refining heterogeneous graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3447548.3467350)

## Basic Idea

- The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.
- At each layer, we calculate the coefficient:

$$
\alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}}
$$

- Residual connection including Node residual

$$
h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)})
$$

- where $h_i$ and $h_j$ is the features of the source and the target node. $r_{\psi(e)}$ is a $d$-dimension embedding for each edge type $\psi(e) \in T_e$.

- and Edge residual:

$$
\alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)}
$$

- Finally, a multi-head attention is used.

## Dataset information
|             | author | paper | Subject | Paper-Author | Paper-Subject | Features  | Train | Val | Test  |
| ----------- | ------ | ----- | ------- | ------------ | ------------- | --------- | ----- | --- | ----- |
| acm4GTN     | 5,912  | 3,025 | 57      | 9,936        | 3,025         | 1,902     | 600   | 300 | 2,125 |

|             | author | conference |  paper  | author-paper | conference-paper | Features  | Train | Val | Test  |
| ----------- | ------ | ---------- | ------- | ------------ | ---------------- | --------- | ----- | --- | ----- |
| dblp4GTN    |  4057  |     20     |  14328  |    19645     |      19645       |    334    |  800  | 400 | 2857  |

## Accuracy(%)
|        | acm4GTN | dblp4GTN |
| ------ | ------- | -------- |
| valid  |  98.67  |   94.75  |
|  test  |  98.21  |   94.71  |
 

## Requirements
- Python >= 3.6
- Pytorch >= 1.9.0
- DGL >= 0.8.0


## How to run

```bash
python simplehgn_trainer.py --dataset acm4GTN --n_epoch 200 --num_heads 4 --in_dim 256 --edge_dim 64 --hidden_dim 128 --out_dim 64 --num_layers 2 --feat_drop 0.2 --negative_slope 0.2 --beta 0.2 --clip 1.0 --max_lr 1e-3
```