# ProxyFunctionGap

RBP 功能条件化低秩残差与泄漏受控生物序列长尾/OOD 基准。

当前状态（2026-11-26）：G4 基准/分析路线不变；M1a 已止损，M1b–M1f 禁止运行。E1 strict3train 主集 146 实验已最终验收，旧 `fold=-1` 主集结果继续撤回且仅作敏感性审计；E5 整 RBP 留出 72/72 pair 已完成正式汇总与独立三遍复核。PAR-CLIP patience20 三 seed 外部描述性结果及 E6 五划分分类与 pseudo-OOD v2 描述性结果保持各自证据边界。E1 仅支持 frozen RNA-FM BCE 相对簇去重余弦 kNN 的既定比较，E5 不支持统一可靠拒识；均不作 SOTA 或机制因果主张。

## 当前文档

- [预印本科学主线与证据链](论文证据/预印本科学主线与证据链_2026-11-25.md)
- [E1 strict3train 最终报告](E1_强基线/E1_主集严格三折冻结基线最终报告_2026-11-26.md)
- [E5 整 RBP 留出三 seed 双轨最终报告](E5_OOD/E5_整RBP留出三seed双轨最终报告_2026-11-26.md)
- [PAR-CLIP 三 seed 报告](E5_OOD/PAR-CLIP跨协议评测报告_2026-11-25.md)
- [E6 五划分与 OOD v2 报告](E6_蛋白迁移/E6_五划分冻结基线报告_2026-11-25.md)

## M1 历史入口（方法线已停止）

- `code_rareseq/m1_function_heads.py`：independent/shared/multitask/residual，显式 CUDA，validation tail AUPRC 收敛早停。
- `code_rareseq/m1_evaluate.py`：occurrence 对齐、validation-only 选参、实验内窗口配对 bootstrap。
- `code_rareseq/m1_submit.sh`：M1a 三 seed、双架构并行提交。
- `M1_功能残差/M1_预注册与执行合同_2026-11-24.md`：顺序门与验收合同。
- `M1_功能残差/交接状态_2026-11-24.md`：远程入口证据与恢复步骤。

## 目录纪律

服务器代码与提交放 `/home`；大型数据、嵌入、checkpoint 和 artifacts 放 `/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/`。禁止 CPU 静默降级、test 选参、把 smoke/pilot 写成最终结论。
