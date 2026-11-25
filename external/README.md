# 外部 20 图的 FMGAD-stable 训练入口

`run.py` 根据图 ID 加载 [主表配置](../configs/main)，从输入图训练两个流，并保存校准前和校准后的逐节点异常分数。输入仅包含节点特征与边。评价标签在独立步骤中使用。

## 运行

从仓库根目录运行。下面的例子要求 `inputs/cora.npz` 包含 NumPy 数组 `x`（节点 × 特征）和 `edge_index`（2 × 边数）。输入对应 [25 图版本清单](../data/graphs.json)。程序核对两个数组的内容和顺序；相同数组可以重新压缩成 NPZ。模型训练使用 CUDA，输入校验可在 CPU 上完成。

```bash
python external/run.py --help
python external/run.py cora 0 --data-dir inputs --check-only
CUDA_VISIBLE_DEVICES=0 python external/run.py cora 0 --data-dir inputs --output-dir runs
```

可用种子为 0、1、2、3、42。每个图/种子在独立进程中运行。程序从 `configs/main/<id>.json` 读取训练参数、推断覆盖值、实际稀疏或稠密执行后端和随机种子。为复现原计算顺序，原始推断先计算重构分数，再根据所选配置做邻域平滑，最后应用固定的几何极性规则。图级诊断和节点参照保存在 [geometry](geometry) 中；参照与图版本和节点顺序绑定。该目录的 `code/` 保留该外部实验使用的骨干和稀疏算子。

## 编码器与输出

默认从头拟合自编码器。主表的搜索种子 0、1、2 复用了相同图、种子和训练配置的历史自编码器，种子 3、42 从头训练。若持有这些经过核对的权重，可传入 `--encoder-dir <目录>`，其中应包含 `<id>/s<seed>.pt` 和同名 JSON。JSON 记录 `seed`、`train`（配置中的训练索引）、`graph_sha256`（原归档图签名）、`epoch` 和 `ae_sha256`。程序验证签名、复制对应 epoch 的权重并从头训练无条件流和条件流。省略归档编码器得到的种子 0、1、2 结果属于重新训练。

[归档编码器签名](encoder_signatures.json) 列出主表使用的 60 个历史编码器对应的文件摘要和轮次。

完成后，`runs/<id>/s<seed>/stable.npz` 含 `raw` 和 `score`。主表评价使用 `score`，数值越大表示模型判定越异常。`stable.json` 记录配置、编码器来源、极性决策及输入与预测的哈希。`weights/` 保存本次训练权重。重新运行同一图与种子时可提供另一个输出目录。
