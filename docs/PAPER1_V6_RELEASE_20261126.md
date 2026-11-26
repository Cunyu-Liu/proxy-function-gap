# Paper 1 V6 发布代码归档（2026-11-26）

本目录归档 T-cell 决策导向 Paper 1 V6 的主文 Markdown、PDF 构建与独立验证脚本、主张—证据台账、Figures 16/17 构建与验证代码，以及 Paper 1 专用九页 PPT 的构建、配置与验证代码。主文和参考配置保持原文；台账只移除两处行尾空格以通过 `git diff --check`。其余归档适配只改变证据根目录、输出目录、解释器选择和参考 PDF/PPT 的位置，不改变科学数值、门禁或结论。

本提交是 code-only 归档，不包含 PDF、PPTX、PNG、日志、缓存、原始数据、序列表、模型权重或大型数值数组。主文中的“No new V6 public release”是主文定稿时的历史可用性表述，保留以避免篡改源文；本次提交仅归档代码，不创建 DOI，也不把历史回顾性证据升级为前瞻或独立生物验证。

## 文件

- `paper/v6_20261126/PAPER1_TCELL_DECISION_V6_20261126.md`：V6 主文，逐字节保留。
- `paper/v6_20261126/build_paper1_tcell_decision_v6.py`：确定性 PDF 构建器。
- `paper/v6_20261126/verify_paper1_tcell_decision_v6.py`：数值、叙事边界、PDF 字体/版式与空目录重建三轮验证。
- `docs/PAPER1_TCELL_DECISION_V6_CLAIM_EVIDENCE_LEDGER_20261126.md`：C01–C25 主张—证据台账；内容保留，仅移除两处行尾空格。
- `figures/v4/`：Figures 16/17 构建器、三轮验证器和冻结来源/输出哈希配置。
- `ppt/paper1_v4/`：Paper 1 专用九页 PPT 构建器、三轮验证器和冻结配置。
- `docs/key_results/20261126/PAPER1_V6_SOURCE_HASHES_20261126.json`：本地原始来源与归档文件的 SHA-256 对照。

## 可配置路径

所有代码路径均相对于仓库；科学输入和二进制交付不进入 Git。运行前设置：

```bash
# 包含 deliverables/ 的冻结证据包根目录；默认仓库根目录，但公开克隆刻意没有这些数据。
export PFG_EVIDENCE_ROOT=/path/to/frozen-evidence-root

# 可选；默认 build/publication/，已被 Git 忽略。
export PFG_OUTPUT_ROOT=/path/to/build-output

# 可选；原始渲染环境为 Python 3.13.5，默认使用当前解释器。
export PFG_CANONICAL_PYTHON=/path/to/python

# 可选；PPT 版式验证所用参考 PDF。
# 默认 $PFG_EVIDENCE_ROOT/deliverables/ppt/paper1_v4/paper1_tcell_proxy_decision_20261126.pdf
export PFG_PAPER1_DECK_PDF=/path/to/reference-deck.pdf
```

配置 JSON 中的 `deliverables/...` 均相对于 `PFG_EVIDENCE_ROOT`。缺少科学输入时直接失败，不使用示例数据替代。构建器生成的 PDF/PPTX/PNG/SVG、验证 JSON/日志和渲染文件只写入 `PFG_OUTPUT_ROOT`。

## 运行

```bash
python paper/v6_20261126/verify_paper1_tcell_decision_v6.py --round 1
python paper/v6_20261126/verify_paper1_tcell_decision_v6.py --round 2
python paper/v6_20261126/verify_paper1_tcell_decision_v6.py --round 3

python figures/v4/verify_paper1_figures_16_17.py

python ppt/paper1_v4/build_paper1_deck.py
python ppt/paper1_v4/verify_paper1_deck.py
```

Round 3 依赖同一输出目录中 Round 1/2 的 PASS JSON，因此应按顺序执行。PPT 验证器要求先构建暂存 PPTX；直接对空输出目录运行验证器会在读取 PPTX 时失败，这不是科学检查通过。

## 依赖、PowerPoint 与字体

- 原始确定性环境：Python 3.13.5、matplotlib 3.10.0、NumPy 2.1.3、pandas 2.2.3、python-pptx 1.0.2；另需 PyMuPDF 和 Pillow。
- 主文 PDF 构建需要 Pandoc、Tectonic，以及 Poppler 的 `pdfinfo`/`pdffonts`；Figure/PPT 的 PDF 字体检查也需要 `pdffonts`。
- 主文字体为 Times New Roman、Helvetica、Menlo、STIX Two Math；Figure 字体候选为 Arial、Helvetica、DejaVu Sans；PPT 指定 PingFang SC。缺少原字体时可能回退并改变换行、版式或哈希，不得把回退结果称为原发布的字节重建。
- Microsoft PowerPoint 是编辑和原生导出 PPT 的依赖，但本归档验证器不会自动调用 PowerPoint。当前冻结参考 PDF 实际由 LibreOffice 26.8.1.1 导出；此前两次 Microsoft PowerPoint 原生导出均因 AppleEvent 超时失败。第三轮如实检查该 LibreOffice PDF 的嵌入/Unicode 字体、页面边界、文本碰撞和逐页渲染，并明确记录 LibreOffice 字体回退；它不宣称完成原生 PowerPoint 导出验证。
- 可用 `PFG_PAPER1_DECK_PDF` 指向以后由 PowerPoint 原生导出的参考 PDF，但导出工具变化必须重新记录，不能沿用当前 LibreOffice 运行时描述冒充原生结果。

## 本次真实专项验证

- V6 主文：Round 1 数值与来源哈希 302/302 PASS；Round 2 叙事与证据边界 88/88 PASS；Round 3 PDF 文本、字体、渲染、科学输入保持和空目录字节确定性重建 45/45 PASS。
- Figures 16/17：70 项独立数值检查 PASS；13 个科学来源哈希通过；两次构建的 6 个 SVG/PDF/PNG 全部一致，并与原冻结交付 6/6 哈希一致；可编辑文本、300 dpi、边界和文本碰撞检查 PASS。
- Paper 1 PPT：31 项数值主张与 9 个来源哈希 PASS；51 个 `ppt/` 内部 payload 条目与原交付一致，两次重建确定；9 页参考 PDF 的字体、边界、碰撞和渲染检查 PASS。
- PPT 验证首次在空暂存输出目录直接运行，因 PPTX 尚未构建而失败；随后按上述先构建后验证入口完整重跑并 PASS。该失败没有被改写成测试成功，也没有跳过任何检查。

## 科学边界

主线保留同一 nested proposal prefix 下 CNN BoN 相对 Random(N) 的实测 MRL regret 改善，并以高分尾部排序损失解释 proxy–function gap。50k target adaptation、ListNet 和 HepG2 anchoring 均未通过各自成功门禁；residual、pairwise 和 multicontext 辅助程序的 T-cell 评价数保持为零。不同 test-v1/test-v2 池不混合比较，oracle 分解只表示 headroom，所有实验仍是已查看历史数据的回顾性/grouped-holdout 证据；无新训练、推理、盲测、outer 重开或因果主张。
