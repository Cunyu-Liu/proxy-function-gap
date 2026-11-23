# Proxy–Function Gap：E0 VGSC-inspired fixed-pool

本仓库只包含 E0 固定完整序列候选池的合成验证。主方法是 **FP-REJ**（有界拒绝采样），**FP-TILT** 是解析 V 倾斜对照；另有 Random、BoN、按序列身份加权投票的 WSC 和 verifier reward argmax 的 greedy。没有真实标签、前缀生成、模型训练或 GPU 路径。

## 协议边界

见 `docs/E0_PROTOCOL.md` 和逐条复核响应 `docs/E0_REVIEW_RESPONSE.md`。联合 FP-REJ 的 product 权重是 `(V*Q)^alpha`，并提供 min/mean 敏感性；池级 Q 只用于覆盖拒选或 FP-REJ 接受成本，不进入 FP-TILT 精确选择。默认全零显式拒选，只有配置 `uniform` 才允许回退。

主预算是实际 `proxy_item_forwards`，同时记录 verifier/Q item-forward。Random=0；ensemble 按成员收费，BatchScorer 按 item 收费，cache 只有显式启用才命中，多轨迹共享预算。相同 C/N 不等于相同实际成本。

## 环境与运行

运行时仅使用 Python 标准库（Python >=3.9）：

```bash
/root/miniconda3/bin/python -m venv .venv
.venv/bin/python -m pip install 'pytest>=7,<9'
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
```

端到端合成 smoke：

```bash
.venv/bin/proxy-gap-e0 \
  --synthetic-seed 314159 --selection-seed 271828 \
  --proposal-cap 8 --trajectories 3 --top-k 2 --alpha 5 \
  --aggregation product --zero-weight-policy abstain \
  --output /teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/e0_v2_smoke.json
```

输出中的 synthetic regret 只用于正确性检查，拒选的 regret 为缺失，不是生物实验结论。

## 目录

- `src/proxy_function_gap/`：实现。
- `tests/`：接口、数学、预算、噪声和合成测试。
- `checks/independent_math_check.py`：C=1/2/3 全路径独立检查。
- `checks/code_audit.py`：源码边界审计。
- `docs/`：冻结协议、逐条复核响应和审查记录。

运行记录保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/`，不写入既有 `/home/rna_jev*` 项目。
