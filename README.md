# Proxy–Function Gap：E0 + Gate 0

本仓库包含 E0 固定完整序列候选池的合成验证，以及保留来源历史的 Sample 2019 Gate 0 可复现处理代码。E0 主方法是 **FP-REJ**（有界拒绝采样），**FP-TILT** 是解析 V 倾斜对照；另有 Random、BoN、按序列身份加权投票的 WSC 和 verifier reward argmax 的 greedy。Gate 0 只提供 proposed/unfrozen 的 split-ready groups；本整合不实现 split/vault，不做真实模型推理或训练，不进入 E1。

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

## Gate 0 集成

Gate 0 代码位于 `gate0/gate0_sample2019/`，来源为 `/home/proxy-function-gap-data` 的提交 `25d6fd4a8133aadde85f6cc4718503c0efe21249`。通过 Git subtree ancestry 和 `gate0/PROVENANCE.md` 保留 provenance；staging 未修改。打包、命令和数据边界见 `docs/GATE0_INTEGRATION.md`。原始与派生数据继续放在 `/teams`，不进入 Git。

## 目录

- `src/proxy_function_gap/`：E0 实现。
- `gate0/gate0_sample2019/`：Gate 0 处理实现。
- `tests/` 与 `gate0/tests/`：E0 和 Gate 0 测试。
- `checks/independent_math_check.py`：C=1/2/3 全路径独立检查。
- `checks/code_audit.py`：源码边界审计。
- `docs/`：冻结协议、逐条复核响应和审查记录。

E0 运行记录保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/`，整合记录保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/integration/`，不写入既有 `/home/rna_jev*` 项目。
