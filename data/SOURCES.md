# 25 个图版本的数据来源与转换

这里列出训练实际使用的图版本。仓库提供参数、评价范围和校验信息。外部图的原始属性与边需从发布者或已有归档取得。下载源文件后还需按对应版本转换，几何参照依赖原节点顺序。各图的规模和归档 SHA-256 见[图清单](graphs.json)。

## 原生五图

原生训练程序使用 PyGOD 的 `load_data` 读取对应小写 ID，随后分开图输入与评价标签。原实验保留 PyGOD 发布的节点顺序、边方向及重复边，使用图中原有属性。原生图的两类张量可按[内容签名](native_signatures.json)核验。[PyGOD 数据仓库](https://github.com/pygod-team/data)

| ID | 节点数 | 特征数 | 边条目数 |
| --- | ---: | ---: | ---: |
| `books` | 1,418 | 21 | 3,695 |
| `disney` | 124 | 28 | 335 |
| `enron` | 13,533 | 18 | 176,987 |
| `reddit` | 10,984 | 64 | 168,016 |
| `weibo` | 8,405 | 400 | 407,963 |

## 外部二十图

同名图的输入可能不同。例如两种 Cora 节点数相同，但边数和异常标签来自不同发布物。表中“计分”是最终评价使用的节点数，训练使用整张无标签图。

| ID | 节点数 / 计分 | 图版本 | 原始发布物 |
| --- | ---: | --- | --- |
| `cora` | 2,708 / 2,708 | AutoGAD 的预注入 Cora，11,606 条边记录。 | [cora.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/cora.mat) |
| `acm` | 16,484 / 16,484 | AutoGAD 的预注入 ACM。 | [ACM.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/ACM.mat) |
| `facebook` | 1,081 / 1,081 | TAM 发布的 Facebook 属性图及其评价标签。 | [Facebook.mat](https://raw.githubusercontent.com/mala-lab/TAM-master/e82ead0584b9b807d4db56a9ffdf89ff94c74dfa/data/Facebook.mat) |
| `citeseer` | 3,327 / 3,327 | AutoGAD 的预注入 CiteSeer。 | [citeseer.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/citeseer.mat) |
| `pubmed` | 19,717 / 19,717 | AutoGAD 的预注入 PubMed。 | [pubmed.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/pubmed.mat) |
| `blogcatalog` | 5,196 / 5,196 | AutoGAD 的预注入 BlogCatalog。 | [BlogCatalog.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/BlogCatalog.mat) |
| `flickr` | 7,575 / 7,575 | AutoGAD 的 7,575 节点 Flickr 社交图。 | [Flickr.mat](https://raw.githubusercontent.com/ZhongLIFR/AutoGAD2024/78ad77c5366941d567eb3824901e63cf77e20cbf/ANEMONE/dataset/Flickr.mat) |
| `amazon` | 10,224 / 10,224 | TAM 的 10,224 节点 Amazon 版本，25 个属性。 | [Amazon.mat](https://raw.githubusercontent.com/mala-lab/TAM-master/e82ead0584b9b807d4db56a9ffdf89ff94c74dfa/data/Amazon.mat) |
| `yelpchi` | 23,831 / 23,831 | TAM 的 RUR 同用户关系图。使用发布文件中的 23,831 个节点。 | [YelpChi.mat](https://raw.githubusercontent.com/mala-lab/TAM-master/e82ead0584b9b807d4db56a9ffdf89ff94c74dfa/data/YelpChi.mat) |
| `inj_cora` | 2,708 / 2,708 | PyGOD/BOND 已注入的 Cora，11,060 条边记录。 | [inj_cora.pt.zip](https://raw.githubusercontent.com/pygod-team/data/77976df4129f5f753907beea4945bd3e21d0251f/inj_cora.pt.zip) |
| `inj_amazon` | 13,752 / 13,752 | PyGOD/BOND 已注入 Amazon，767 个属性。发布文件内有 511,042 条边记录。 | [inj_amazon.pt.zip](https://raw.githubusercontent.com/pygod-team/data/77976df4129f5f753907beea4945bd3e21d0251f/inj_amazon.pt.zip) |
| `inj_flickr` | 89,250 / 89,250 | PyGOD/BOND 已注入 Flickr，89,250 个节点。 | [inj_flickr.pt.zip](https://raw.githubusercontent.com/pygod-team/data/77976df4129f5f753907beea4945bd3e21d0251f/inj_flickr.pt.zip) |
| `arc_cs` | 18,333 / 18,333 | ARC 已注入的 Coauthor CS。原文件中的 Label 等于结构与属性异常标签的并集。 | [cs.mat](https://raw.githubusercontent.com/yixinliu233/ARC/f8be54f37710d8dc6449193b7fbc2260d433df34/dataset/cs.mat) |
| `arc_photo` | 7,650 / 7,650 | ARC 已注入的 Amazon Photo。保留发布者的混合异常标签。 | [photo.mat](https://raw.githubusercontent.com/yixinliu233/ARC/f8be54f37710d8dc6449193b7fbc2260d433df34/dataset/photo.mat) |
| `tolokers` | 11,758 / 11,758 | 异配图基准。原发布标签表示 Toloka 项目封禁状态。 | [tolokers.npz](https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/tolokers.npz) |
| `questions` | 48,921 / 48,921 | 异配图基准。沿用 Yandex Q 用户活跃度二元标签的原编码。 | [questions.npz](https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/questions.npz) |
| `amazon_all` | 11,944 / 8,639 | CARE-GNN 全关系 homo 图。节点 0 至 3304 属于无标签前缀。 | [Amazon.zip](https://raw.githubusercontent.com/YingtongDou/CARE-GNN/a64ff7523e187a24251f7ca88435d2c9d8f7dcd9/data/Amazon.zip) |
| `yelpchi_all` | 45,954 / 45,954 | CARE-GNN YelpChi 全关系 homo 图。保留 45,954 个节点。 | [YelpChi.zip](https://raw.githubusercontent.com/YingtongDou/CARE-GNN/a64ff7523e187a24251f7ca88435d2c9d8f7dcd9/data/YelpChi.zip) |
| `tfinance` | 39,357 / 39,357 | BWGNN 完整 DGL 图，保留 42,445,086 条有向边记录和全部 10 个属性。来源目录由作者维护。 | [作者数据目录](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing) |
| `elliptic` | 203,769 / 46,564 | Kaggle ellipticco 版本 1，静态全图与已知标签评价。 | [特征](https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset%2Felliptic_txs_features.csv?datasetVersionNumber=1) · [边](https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset%2Felliptic_txs_edgelist.csv?datasetVersionNumber=1) · [类别](https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset%2Felliptic_txs_classes.csv?datasetVersionNumber=1) |

## 从发布物得到对应图版本

- **AutoGAD、TAM 与 ARC 的 MAT 文件。** 取原始属性矩阵（通常为 `Attributes`），稀疏属性转稠密 float32，保留全部列。按原邻接矩阵（通常为 `Network`）非零位置建立源行、目标列的边索引，合并重复的二元边并按源、目标顺序排列。保留文件已有自环和方向，评价使用异常标签字段 `Label`。对已注入异常的发布物保持原标注。
- **BOND 三图。** 从各个固定版本的 PyG 图归档取得 `x` 和 `edge_index`，保留边的原列顺序、方向、自环及重数。评价将原 `y != 0` 映射为 1。这三个图保留发布文件的边顺序，无需按 MAT 的规则重新排序。
- **Tolokers 与 Questions。** 按发布者规定给原边列表补反边，再按节点对排序去重。保留原特征值和节点顺序，不另外添加自环。目标沿用原 0/1 编码。
- **CARE-GNN 两个全关系图。** 对 MAT 的 `homo` 邻接取非零二元支持，按源、目标排序去重，保留其已有方向和自环。Amazon（全关系）前 3,305 个节点的评价标签设为 −1，其余节点沿用 0/1。图训练保留所有节点。
- **T-Finance。** 从完整 `tfinance.dgl` 按 DGL 的 EID 顺序读边；原图已有双向边。保留原特征顺序并转 float32，评价标签取原两列 one-hot 标签的 argmax。原实验使用的 DGL 文件 SHA-256 为 `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`。外部共享目录的当前文件应自行对照该签名。
- **Elliptic。** 按特征文件的交易行顺序给节点编号，把有向边的两端交易 ID 映射为行号，保留边行顺序。去掉交易 ID 后保留含时间步在内的 166 个特征。原类别 1 映射为异常 1，类别 2 映射为正常 0，未知映射为 −1。157,205 个未知节点仍参与图训练。

## 已有冻结图时如何转为输入

若已有对应版本的纯张量归档，以下例子把它转成训练程序所需的 `inputs/cora.npz`。先按上面的数据版本规则准备张量。在已经冻结的图上保留给定边列顺序，避免重复执行上游补反边或排序操作。

```python
from pathlib import Path
import numpy as np
import torch

graph = torch.load("cora.pt", map_location="cpu", weights_only=True)
x = graph["x"].float().contiguous().numpy()
edges = graph["edge_index"].long().contiguous().numpy()
assert x.ndim == 2 and edges.ndim == 2 and edges.shape[0] == 2
Path("inputs").mkdir(exist_ok=True)
np.savez_compressed("inputs/cora.npz", x=x, edge_index=edges)
```

在仓库根目录先核验内容。核验通过后去掉 `--check-only` 即可运行该种子的外部训练。评价使用独立的整型 NPY 标签，约定见[输入与评价标签](README.md)。

```sh
python external/run.py cora 0 --data-dir inputs --check-only
```
