# Proxy–Function Gap：E0 + Gate 0

本仓库包含 E0 固定完整序列候选池合成验证、Sample 2019 Gate 0/split-vault、三种子 fresh CUDA proxy、三成员推理、V/Q 校准与固定池选择。E0 主方法是 **FP-REJ**（有界拒绝采样），**FP-TILT** 是解析 V 倾斜对照；另有 Random、BoN、LCB、按序列身份加权投票的 WSC 和 verifier reward argmax 的 greedy。生产分析只使用冻结的 dev/calibration Learner 角色；outer 标签和 E1 评分仍关闭。

## 协议边界

见 `docs/E0_PROTOCOL.md` 和逐条复核响应 `docs/E0_REVIEW_RESPONSE.md`。联合 FP-REJ 的 product 权重是 `(V*Q)^alpha`，并提供 min/mean 敏感性；池级 Q 只用于覆盖拒选或 FP-REJ 接受成本，不进入 FP-TILT 精确选择。默认全零显式拒选，只有配置 `uniform` 才允许回退。

主预算是实际 `proxy_item_forwards`，同时记录 verifier/Q item-forward。Random=0；ensemble 按成员收费，BatchScorer 按 item 收费，cache 只有显式启用才命中，多轨迹共享预算。相同 C/N 不等于相同实际成本。

## 环境与运行

E0/Gate0 仅使用 Python 标准库；Sample runtime 和 fresh trainer 需要 Python >=3.10、numpy、torch 和 h5py。当前已核验 CUDA 环境为 `/root/miniconda3/envs/lucaone_cu128/bin/python`：

```bash
export PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:/home/proxy-function-gap/src:/home/proxy-function-gap/gate0:/home/proxy-function-gap/runtime/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/miniconda3/envs/lucaone_cu128/bin/python -m pytest -q
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

## Split-v2、Sample runtime 与 fresh trainer

split-v2 来源提交 `05eb96b5e730c9dd6cebc96486220bd6266c30d9` 通过 subtree merge `56d3e3b` 保留；独立 Sample runtime 来源提交 `3c8d047816a4a216a3954c66112c1e858c7dc2c7` 通过 subtree merge `981a3de` 保留。官方 GPL Python 代码和 HDF5 权重未复制进 Git，详见 `runtime/PROVENANCE.md`。

fresh trainer 协议见 `docs/TRAINER_PROTOCOL.md`。默认至少 3 seeds 在同一显式 CUDA 设备并行，生产 NPZ 与 fresh TSV 逐角色 row_id/source/order hash 强校验，train-only scaler、同一 eval 模型的 full-train/dev 停止、dev absolute best 重置、无 epoch/step 上限、原子 checkpoint 和完整 history 均有测试。完成态 checkpoint 拒绝续训，resume 校验 expected SHA256 及代码/模型/配置/scaler/data/order 身份。三种子正式训练已完成收敛，详见 `docs/FRESH_CNN_FORMAL_RUN1.md`；smoke 与训练收敛均不是科学功能结论。

生产推理与 V/Q/固定池协议见 `docs/ENSEMBLE_INFERENCE_RUN.md`、`docs/V_Q_SELECTION_PROTOCOL.md` 和 `docs/POOL_MANIFEST_SCHEMA.md`。唯一有效的 MRL 推理来源是 ensemble-v2；ensemble-v1 及其派生结果因缺少标签逆变换而作废。生产池严格绑定固定池来源、Learner 标签许可和推理 lineage，生产选择执行真实 CUDA forward 并记录实际成本。

VGSC 固定池 pilot 的已复核负结果与门禁决定见 `docs/PILOT_V3_RESULT.md`；outer 协议见 `docs/E1_OUTER_PROTOCOL.md`。FP-REJ/Q 未通过主线门禁，保留为基线与协议，不作为有效方法主张。三个嵌入点的执行与停止记录见 `docs/VGSC_THREE_PAPER_EXECUTION.md`。Paper 1 回顾性固定池证据包、预印本草稿和 PDF 已形成；投稿前仍须作者终审，且不声称同行评审或前瞻湿实验验证。

Paper 1 固定池外部压力结果见 `docs/PAPER1_EXTERNAL_RESULT.md` 和 `docs/PAPER1_EXTERNAL_PROTOCOL.md`。Castillo-Hair 的四个外部情境均未复现 Sample designed 的持续单调收益，支持压力效应的情境边界。分数匹配与共同 proxy 审计见 `docs/PAPER1_MECHANISM_RESULT.md`：差值区间均跨 0，当前更支持高分外推而非优化器特异失真。异构 4-mer MLP 对照见 `docs/PAPER1_CROSS_ARCH_RESULT.md`：四个上下文的误差相关性为 0.931–0.965，共同 proxy 风险跨越模型架构。替代池与输出噪声敏感性见 `docs/PAPER1_SENSITIVITY_NOISE_RESULT.md`：替代池仍非单调，BoN 不必然放大高斯输出噪声。

## Gate 0 集成

Schlusser 2024 多研究数据与 benchmark 见 `docs/SCHLUSSER2024_GATE0.md`。该包固定 Zenodo/ENA/GitLab 谱系、实测与预测字段、研究级角色及三轮验证；HepG2 endogenous TE 整研究冻结为后续确认性 OOD，Learner 不含 TE/预测列，当前未运行训练、推理或 outer 评分。其他研究的训练资源不冒充独立 outer。

Gate 0 代码位于 `gate0/gate0_sample2019/`，来源为 `/home/proxy-function-gap-data` 的提交 `25d6fd4a8133aadde85f6cc4718503c0efe21249`。通过 Git subtree ancestry 和 `gate0/PROVENANCE.md` 保留 provenance；staging 未修改。打包、命令和数据边界见 `docs/GATE0_INTEGRATION.md`。原始与派生数据继续放在 `/teams`，不进入 Git。

## 目录

- `src/proxy_function_gap/`：E0 实现。
- `gate0/gate0_sample2019/`：Gate 0 处理实现。
- `tests/` 与 `gate0/tests/`：E0 和 Gate 0 测试。
- `checks/independent_math_check.py`：C=1/2/3 全路径独立检查。
- `checks/code_audit.py`：源码边界审计。
- `docs/`：冻结协议、逐条复核响应和审查记录。

E0 运行记录保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/e0/`，整合记录保存在 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/integration/`，不写入既有 `/home/rna_jev*` 项目。
