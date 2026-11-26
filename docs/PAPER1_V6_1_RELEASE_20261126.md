# Paper 1 V6.1 当前发布代码归档（2026-11-26）

**Current：V6.1 是后续内部投稿准备的当前底稿；`paper/v6_20261126/` 中的 V6 保留为历史版本。** 本次为追加归档，不修改或删除 V6、Figures 16/17、PPT、科学数据或实验结果。

V6.1 来自独立终审后的文档级修正：一项 major 将 50k MAE 改善明确拆分为 r1 大幅、r2 小幅；两项 minor 分别披露 Random(N) 每条 pool–seed–cap 记录仅一次位置抽取、pool bootstrap 不积分 selector-randomization Monte Carlo 不确定性，以及将 ListNet 含糊的 “point improvements” 改为失败门禁下的有利点估计。核心实验数值、门禁和科学结论未改。

## 文件

- `paper/v6_1_20261126/PAPER1_TCELL_DECISION_V6_1_20261126.md`：当前主文，逐字节保留。
- `paper/v6_1_20261126/build_paper1_tcell_decision_v6_1.py`：V6.1 确定性 PDF 构建器。
- `paper/v6_1_20261126/verify_paper1_tcell_decision_v6_1.py`：V6.1 数值、叙事和 PDF 三轮验证器。
- `docs/PAPER1_TCELL_DECISION_V6_1_CLAIM_EVIDENCE_LEDGER_20261126.md`：当前 claim–evidence ledger；仅移除两处行尾空格。
- `docs/PAPER1_V6_INDEPENDENT_REVIEW_20261126.md`：独立终审报告；仅移除七处行尾空格，审查结论和证据不变。
- `docs/key_results/20261126/PAPER1_V6_1_SOURCE_HASHES_20261126.json`：原始来源与归档文件 SHA-256 对照。

本提交仍是 code-only 归档，不包含 PDF、PPTX、PNG、SVG、日志、验证输出 JSON、缓存、原始数据、序列标签、模型权重或大型数值数组。主文中“不声称新 V6.1 public release”的句子是定稿时历史表述，保留源文以避免篡改；本次仅追加代码归档，不创建 DOI。

## 可配置路径与运行

```bash
export PFG_EVIDENCE_ROOT=/path/to/frozen-evidence-root  # 含 deliverables/
export PFG_OUTPUT_ROOT=/path/to/build-output           # 可选，默认 Git 忽略的 build/publication/
export PFG_CANONICAL_PYTHON=/path/to/python             # 可选，默认当前解释器

python paper/v6_1_20261126/verify_paper1_tcell_decision_v6_1.py --round 1
python paper/v6_1_20261126/verify_paper1_tcell_decision_v6_1.py --round 2
python paper/v6_1_20261126/verify_paper1_tcell_decision_v6_1.py --round 3
```

Round 3 读取同一输出目录中的 Round 1/2 PASS JSON，并检查外部证据包中的交付 PDF。PDF、验证 JSON/日志和发布清单只写入 `PFG_OUTPUT_ROOT`。公开克隆不含科学输入，缺少文件时直接失败，不使用替代数据。

依赖与 V6 归档一致：Pandoc、Tectonic、PyMuPDF、NumPy、Poppler `pdfinfo`/`pdffonts`，以及 Times New Roman、Helvetica、Menlo、STIX Two Math 和图件字体。字体或工具版本变化可能改变布局与哈希；详细 PowerPoint 和 Figure/PPT 依赖见 `docs/PAPER1_V6_RELEASE_20261126.md`。

## 本次真实验证

- Round 1：302/302 PASS，覆盖数值、校正族、门禁、target=0、输入哈希和 ledger 映射。
- Round 2：94/94 PASS，新增 50k/ListNet/Random(N) 修正文本必要项及旧措辞禁用项均通过。
- Round 3：45/45 PASS；13 页 PDF 的文本、三幅图、嵌入 Unicode 字体、渲染、边界和空目录确定性重建均通过。
- 交付 PDF 与空目录重建 SHA-256 均为 `c7ca7e7f4fe56df6f1bd750ca10b673fdef0255f134648bbb71113b771476c2a`；PDF 本身不进入 Git。

独立终审属于内部科学一致性审查，不等于外部同行评审接收、前瞻盲测、独立生物复制或因果验证。V6.1 保留等预算 CNN BoN 的真实 MRL 选择价值，同时保留 50k、ListNet、HepG2 anchoring 的门禁失败及辅助程序 target evaluation=0 的边界。
