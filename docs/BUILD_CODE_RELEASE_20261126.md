# 论文、图与 PPT 构建代码归档（2026-11-26）

本目录归档统一 V5 预印本、V2/V3 图和 2026-11-26 PPT 的构建与验证代码。科学正文、数值、证据边界、统一故事和主张台账不作改写；代码-only 归档不包含 PDF、PPTX、PNG、原始数据、序列标签、模型权重、缓存或日志。

## 目录与证据边界

- `paper/v5_20261126/`：V5 主文、S2/S3 的原始 Markdown，确定性 PDF 构建器，以及独立科学复算器。
- `figures/v2/`：V2 Paper 1、Paper 2/3 和跨终点 OOD 图的构建与三轮验证脚本。
- `figures/v3/`：Figure 11–13 与 Figure 14/15 的构建、配置和三轮验证脚本。
- `ppt/v3/`：2026-11-26 十二页 PPT 构建器、参考配置和验证器。
- `docs/SCIENTIFIC_STORY_V5_20261126.md` 与 `docs/UNIFIED_PREPRINT_CLAIM_EVIDENCE_LEDGER_V5_20261126.md`：当前统一故事和主张—证据台账。
- `docs/key_results/20261126/`：此前已在原始 macOS 证据包上通过的历史构建/验证报告。这些报告不是对本代码-only 克隆已重建二进制的声明。

V5 的首要结果仍是特定历史 T-cell 情境中的等候选预算排序价值；组成调整衰减、N50/HepG2 边界、PERSIST 跨零负向迁移结果、Paper 2 冻结 calibration 失败和 Paper 3 非盲测诊断均完整保留。归档不重开 outer，不训练、不推理、不调阈值。

## 可配置路径

所有归档脚本使用仓库相对代码路径。科学输入与构建输出通过环境变量配置：

```bash
# 指向包含 deliverables/ 的原始冻结证据包根目录；不要把它复制进 Git。
export PFG_EVIDENCE_ROOT=/path/to/frozen-evidence-root

# 可选。默认是 Git 已忽略的仓库内 build/publication/；也可指向外部目录。
export PFG_OUTPUT_ROOT=/path/to/build-output

# 可选。需要恢复原始 Python 3.13.5 渲染环境时设置；默认使用当前解释器。
export PFG_CANONICAL_PYTHON=/path/to/python
```

`PFG_EVIDENCE_ROOT` 默认是仓库根目录，但本仓库刻意不含 `deliverables/` 原始证据包。因此真实科学复算必须显式提供证据包；缺文件时直接失败，不会用示例数据替代。配置 JSON 中的 `deliverables/...` 路径均相对于该证据根目录，SHA256 仍绑定原始冻结输入。

## 命令

```bash
# V5 独立科学复算；输出不会写入证据包。
python paper/v5_20261126/verify_unified_preprint_v5_20261126.py --round 1
python paper/v5_20261126/verify_unified_preprint_v5_20261126.py --round 2
python paper/v5_20261126/verify_unified_preprint_v5_20261126.py --round 3

# 图：验证器会先执行两次独立构建，再检查数值、确定性与格式。
python figures/v2/verify_v2_figures.py
python figures/v2/verify_v2_paper23_ood.py
python figures/v3/verify_v3_figures.py
python figures/v3/verify_v3_figures_14_15.py

# PPT：先构建 PPTX；完整验证还需要原生 PowerPoint 导出的同名 PDF。
python ppt/v3/build_modern_deck_20261126.py
python ppt/v3/verify_modern_deck_20261126.py

# V5 PDF 全套构建、版面/字体检查和空目录确定性重建。
python paper/v5_20261126/build_unified_preprint_v5.py
```

## Linux 与环境限制

- V3/PPT 原始确定性结果使用 Python 3.13.5、matplotlib 3.10.0、numpy 2.1.3、pandas 2.2.3、python-pptx 1.0.2。跨版本可以执行科学断言，但不能保证 PDF/SVG/PNG/PPTX 字节级一致；配置中的原始运行时和来源哈希不会被覆盖。
- 图形与论文版式依赖 Arial、Helvetica、Menlo、Arial Unicode MS、STIX Two Math 等字体。Linux 缺少这些 macOS 字体时可能回退，字形、换行、页面尺寸观感及渲染哈希可能不同；不得把字体回退后的产物称为原始发布物的字节重建。
- PPT 的第三轮验证要求 Microsoft PowerPoint 原生导出的 12 页 PDF，并检查 ArialUnicodeMS 嵌入、文本碰撞和逐页渲染。Linux 无 Microsoft PowerPoint，不能用 LibreOffice 导出冒充该验证；只能重建 PPTX，不能在本环境完成原生 PDF/视觉验证。
- V5 PDF 构建还要求 pandoc、Tectonic、PyMuPDF、NumPy 及 Poppler 的 `pdfinfo`/`pdffonts`。没有原字体或相同工具版本时，不承诺原 PDF 哈希一致。
- 独立科学验证器会读取外部证据包中的冻结聚合表、逐种子结果和部分序列级标签/NPY；这些输入一律不进入本仓库。Figure 14/15 验证还读取外部保存的历史远端核对日志，日志本身不归档。

## 历史关键结果入口

- `UNIFIED_PREPRINT_V5_RELEASE_MANIFEST.json`：V5 原科学正文哈希、三轮 PDF/字体/确定性检查及证据边界。
- `v2_validation_report.json`、`v2_paper23_ood_validation_report.json`：V2 图历史三轮 PASS。
- `v3_validation_report.json`、`v3_figures_14_15_validation_report.json`：V3 图历史数值、确定性和格式 PASS。
- `proxy_function_gap_modern_20261126_verification.json`：2026-11-26 PPT 历史 48 项数值主张、64 个 PPT payload 条目确定性及 12 页原生渲染 PASS。

## 本次归档验证

- 19 个归档 Python 文件在本地 macOS 与目标 Linux 环境均通过 `py_compile`。
- V5 独立复算器在原始冻结证据包上执行 `--round 1|2|3`，三轮均为 70/70 PASS；每轮绑定 74 个输入的 SHA256。
- V2/V3 四组验证器均 PASS。14 幅图的 SVG/PDF/PNG 共 42 个输出与此前已验证交付逐字节一致。
- V5 完整构建器三轮 PASS；主文、S2、S3 PDF 与此前已验证版本的 SHA256 分别保持 `98d967d1…`、`d5417b17…`、`fa73f9bd…`。
- PPT 48 项数值主张、两轮确定性重建和 12 页原生 PowerPoint PDF 渲染均 PASS；原交付与归档重建的 64 个 `ppt/` 内部 payload 条目哈希一致。ZIP 容器级哈希可随元数据变化，不作为科学内容差异。
- 目标 Linux/CUDA 工作树执行完整仓库测试：`325 passed, 106 subtests passed in 109.17s`。该测试不读取或重开 outer。
- 本地 macOS 全仓尝试另留记录：281 passed、18 skipped、26 failed；失败来自无 CUDA 和缺少远端 `/home` 固定测试资源，最终以目标环境完整测试为准。
