# 容量限制的软分配

samples/scene.json 是本地合成的原型候选和梯度交换。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 传输算子

`transport_op.py` 是可重用的熵正则容量传输算子：

- 前向软分配：行质量全部分出、正常列不超容量、弃置仅在允许范围内；报告行/列最大误差与可行目标同对偶下界的差，并区分非法输入、缺片、结构不可行、资源不足与未收敛。
- 反向代价梯度：接收对有效边质量与标量最优目标的导数，返回参与求解的原始代价的导数；临界列（正乘子与零松弛在容差内不可区分）不交付梯度。
- `ExchangeSession` 按稀疏块交接候选：声明/到达/钉版/更正/反传，旧版软标签不算当前版已确认结果，反传可沿绑定的旧版分配计算。

`run_scene.py` 按资料的时间序列执行交接并输出各版本分配、弃置量、容量状态与反向结果：

```sh
python3 run_scene.py samples/scene.json
python3 run_scene.py samples/capacity-edge.json
```
