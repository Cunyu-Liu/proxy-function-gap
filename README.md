# ProxyFunctionGap

RBP 功能条件化低秩残差与泄漏受控生物序列长尾/OOD 基准。

当前状态：M1 训练和统计入口已完成静态、负向 CUDA 与合成计算检查；尚无 M1 GPU 科学结果。旧 G1/G3/G4 判决保持有效。

## 当前入口

- `code_rareseq/m1_function_heads.py`：independent/shared/multitask/residual，显式 CUDA，validation tail AUPRC 收敛早停。
- `code_rareseq/m1_evaluate.py`：occurrence 对齐、validation-only 选参、实验内窗口配对 bootstrap。
- `code_rareseq/m1_submit.sh`：M1a 三 seed、双架构并行提交。
- `M1_功能残差/M1_预注册与执行合同_2026-11-24.md`：顺序门与验收合同。
- `M1_功能残差/交接状态_2026-11-24.md`：远程入口证据与恢复步骤。

## 目录纪律

服务器代码与提交放 `/home`；大型数据、嵌入、checkpoint 和 artifacts 放 `/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/`。禁止 CPU 静默降级、test 选参、把 smoke/pilot 写成最终结论。
