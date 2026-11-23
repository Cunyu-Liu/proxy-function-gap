# 独立复核意见逐条响应

## 1. 命名与算法边界

旧实现的 `v-only/joint` 同时指代精确倾斜和拒绝采样，容易混淆。现统一为 VGSC-inspired fixed-pool：FP-REJ 是主方法，FP-TILT 是解析对照；不声称官方生成式 VGSC 复现。

## 2. 权重与聚合

新增 `guidance_weight`，product 明确为 `(V*Q)^alpha`，min/mean 同样只在聚合后取一次指数。测试覆盖三种聚合、alpha=0、V 全零、Q=0 和 w=1。

## 3. WSC、greedy 与 top-k

旧 WSC 的最终加权抽样已删除，改为按序列身份求和投票。旧 greedy 的联合权重最大化已删除，固定为原始 verifier reward argmax。所有 top-k 按身份不放回；FP-REJ 通过多轨迹身份投票产生 top-k。

## 4. 池级 Q 的角色

FP-TILT 不再接受 Q。WSC/greedy 排序也不使用 Q。Q 只用于 `abstain_below_q` 覆盖门控或 FP-REJ 接受权重；后者只解释为成本/接受行为差异，不声称精确选择增量。

## 5. 正确性与预算覆盖

新增 C=1/2/3 全路径枚举、接受率/回退率、C=1 坍缩、常数 V、对抗/独立/oracle 真值、logit 零噪声和零概率、重复提议、ensemble、BatchScorer、显式 cache、多轨迹共享预算和 top-k 不放回测试。旧 50 tests 不作为充分性依据。

## 6. 噪声定义

协议明确为 V/Q 概率 logit 高斯扰动；sigma=0 保留端点。不再把该鲁棒性协议描述为无界 MRL 加噪。合成 proxy 的已知扰动与 V/Q 噪声协议分开命名。

## 7. 主预算

旧 `proxy_calls` 等字段改为实际 `proxy_item_forwards`、`verifier_item_forwards`、`qhat_item_forwards`。主预算为 proxy item-forward，Random=0。ensemble 按成员收费，batch 按 item 收费，cache 仅显式启用，多轨迹不重置计数。相同 C/N 不得宣称相同成本。

## 8. 全零主协议

默认 `zero_weight_policy=abstain`，空选择和原因进入结果；只有显式配置 `uniform` 才允许策略特定的均匀回退。删除了静默均匀选择。

## 范围结论

本次仍只做 E0。协议更新、完整测试、独立数学检查、源码审计和端到端复现全部通过后才会提交；任何失败都保留日志并阻止完成声明。
