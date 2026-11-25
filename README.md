# FMGAD-stable

本仓库发布图节点异常检测模型 FMGAD 的稳定残差与图自适应极性版本，含主结果中 **25 个图版本的可执行参数**。每张图均使用种子 `0, 1, 2, 3, 42`，推断噪声种子固定为 `1729`。按图等权平均，归档主结果的 ROC-AUC 为 **75.882%**，平均精确率（AP）为 **26.656%**。[逐图配置表](configs/README.zh-CN.md) · [机器可读索引](configs/main_index.json) · [125 次运行的指标](results/main_per_seed.csv)

## 数据集与参数

- **5 张原生图**为 Books、Disney、Enron、Reddit、Weibo。入口直接读取对应 JSON 中的模型参数。[原生模型](fmgad) · [Books 配置示例](configs/main/books.json)
- **20 张外部图**为 Cora、ACM、Facebook、CiteSeer、PubMed、BlogCatalog、Flickr、Amazon (extension)、YelpChi-RUR、BOND Cora、BOND Amazon、BOND Flickr、ARC Coauthor CS、ARC Amazon Photo、Tolokers、Questions、Amazon (all relations)、YelpChi (all relations)、T-Finance、Elliptic。外部入口分别使用每图的训练参数和推断覆盖值，包含大图所用的稀疏计算实现。[外部运行说明](external/README.md) · [T-Finance 配置示例](configs/main/tfinance.json)

## 安装与运行

在带 CUDA 的环境中安装与显卡兼容的 PyTorch，再安装依赖。图输入及评价掩码见[数据格式说明](data/README.md)。原生图默认使用 PyGOD 加载；外部图使用各图版本的预处理输入。

```sh
python -m pip install -r requirements.txt
python run_native.py books 0 --output-dir runs
python external/run.py cora 0 --data-dir inputs --check-only
python external/run.py cora 0 --data-dir inputs --output-dir runs
python evaluate.py runs/cora/s0/stable.npz labels/cora.npy
```

两个入口都能从头训练。复现原生归档时，用 `--encoder` 传入对应种子的原自编码器；复现外部主表的种子 0、1、2 时，用 `--encoder-dir` 指定含归档编码器及校验信息的目录。外部种子 3、42 在主表中即从头拟合编码器。训练代码、图签名、外部图的几何参照和结果随仓库提供；原始图文件及归档编码器权重需另行准备。评分文件的 `raw` 保存极性校准之前的分数，`score` 用于论文主表。[图版本清单](data/graphs.json) · [编码器签名](data/native_signatures.json)

## 无监督选参与残差消融

外部 20 图从共同的 144 个候选中按图选择。目标由多比例 CSM、跨种子的排序一致性和高分集合重叠等权组成。用下列命令可从保存的候选统计量复算全部 20 个胜者。原生五图的纯 CSM 属于后续对照实验，主配置索引沿用原生主表的参数。[选参公式及工具](selection/README.md) · [五图研究配置](configs/studies/native_selectors.json)

```sh
python selection/select.py replay
python -m unittest discover -s tests -v
```

原生入口的 `--arm` 可取 `stable`（主模型）、`resid_center`（残差仅中心化并裁剪）、`old_agg`（回用单精度聚合）、`zero`（残差目标置零）或 `beta2`（残差损失权重为 2）。复算消融时，同图同种子共用原编码器并分别训练两支流。[原生残差配对结果](results/native_residual_ablation.csv) · [原生 CSM 配对结果](results/native_csm_summary.csv)

[MIT 许可证](LICENSE)
