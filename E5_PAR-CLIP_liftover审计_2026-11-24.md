# PAR-CLIP hg19→hg38 liftover 审计

日期：2026-11-24 ｜ 状态：坐标转换完成，**尚未做模型评价**

## 范围

冻结主集与 Mukherjee PAR-CLIP 的共有 RBP 共17个。`processed` 覆盖8个；其余9个仅存在于 `processed.full`，因此完整跨协议集合必须合并两个官方处理层级，不能把 processed 的8个误报为完整17个。

工具：CrossMap 0.7.3/0.7.4包（CLI报告0.7.3），UCSC `hg19ToHg38.over.chain.gz`。输入为单碱基 crosslink BED；输出校验仍为单碱基，按 chrom/start/end/strand 去重并保存每实验输入/输出 SHA256。

## 数量审计

| 来源层级 | 实验/RBP | 输入位点 | 成功转换 | 去重后 | 失败率 |
|---|---:|---:|---:|---:|---:|
| processed | 8 | 71,898 | 71,894 | 71,891 | 0.00556% |
| processed.full 独有 | 9 | 590,405 | 590,355 | 590,334 | 0.00847% |
| 合计 | 17 | 662,303 | 662,249 | 662,225 | 0.00815% |

失败54个位点，重复去除24个。失败和去重均逐实验记录，不静默删除。首次命名解析失败 run 保留；修正后两个有效 run 的审计表位于 `/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/parclip/hg38*`。

## 边界与下一步

坐标转换完成不等于跨协议模型评测完成。下一步需按冻结 BASIC 转录本宇宙筛位点、生成101nt窗口、计数非ACGU/区外位点，并用固定模型做 PAR-CLIP→eCLIP 评价；在此之前不得写跨协议性能结论。
