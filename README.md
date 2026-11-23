# Proxy–Function Gap：E0 固定池合成验证

本仓库当前只包含 E0 软件正确性骨架：固定完整序列候选池、约束掩码、random/BoN/WSC/greedy/V-only/joint 选择、逐模型调用预算和已知误差结构的合成测试。没有真实标签、前缀生成、模型训练或 GPU 路径。

## 公式边界

冻结定义见 `docs/E0_PROTOCOL.md`。特别注意：池级标量 Q 在精确归一化的 joint 倾斜中会约掉，因此精确 V-only 与 joint 相同；有限拒绝采样的接受率差异不能当作 Q 的科学增量。WSC 是事后 V 加权，greedy 是缓存提议上的联合权重最大化。

## 环境与运行

运行时仅使用 Python 标准库（Python >= 3.9）。审计环境的全局命令没有 `python3`，因此远程使用 `/root/miniconda3/bin/python` 创建项目私有环境：

```bash
/root/miniconda3/bin/python -m venv .venv
.venv/bin/python -m pip install 'pytest>=7,<9'
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
```

端到端合成 smoke：

```bash
.venv/bin/proxy-gap-e0 \
  --synthetic-seed 314159 --selection-seed 271828 --budget 8 --alpha 5 \
  --output /teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/e0_smoke.json
```

同种子输出包含稳定配置哈希。输出中的 regret 只是已知合成答案的正确性检查，不是生物实验结论。预算字段分 `proposals`、`proxy_calls`、`verifier_calls` 和 `qhat_calls` 报告；重复提议不缓存。

## 目录

- `src/proxy_function_gap/`：实现。
- `tests/`：pytest 接口、数学、约束、合成和预算测试。
- `checks/independent_math_check.py`：不复用生产倾斜公式的独立参考检查。
- `docs/INDEPENDENT_REVIEW.md`：第三轮代码审查清单与结论。
- `docs/plans/2026-11-23-e0-design.md`：设计取舍。

所有 E0 运行日志和 artifacts 保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/`，不写入既有 `/home/rna_jev*` 项目。
