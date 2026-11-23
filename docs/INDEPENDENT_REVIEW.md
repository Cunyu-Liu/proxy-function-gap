# E0 复核后三轮检查记录

日期：2026-11-23。本文对应 VGSC-inspired fixed-pool 重构；旧版 V-only/joint 记录只保留在历史 artifacts，不再描述当前实现。“独立”指不同角度和独立参考实现，不冒充外部人员评审。

## 第一轮：重构接口与数学门禁

- 初次新套件为 109 passed、2 failed；两处为共享 cache 测试预期和三候选掩码构造错误，修正后完整套件为 **111 passed**。
- 独立数学脚本初次发现“每提议接受率”参考公式误用期望值之比；改为逐接受路径累加 `路径概率/实际提议数` 后通过。失败日志均保留。
- C=1/2/3 分别穷举 2/4/8 条有序提议路径；每个 cap 使用 6000 次固定种子模拟，选择分布最大绝对误差分别为 0.011333、0.006000、0.001195，均小于 0.025。轨迹接受率、回退率和每提议接受率误差均小于 0.02。
- 源码审计覆盖 9 个运行时文件，仅允许标准库，确认真值隔离和无 GPU/训练/前缀路径。

## 第二轮：预算公平与语义空洞

新增并验证：

- WSC 按序列身份求和投票，greedy 只做原始 verifier reward argmax；FP-TILT 不接受 Q。
- product/min/mean 只取一次 alpha；alpha=0、全零、Q=0、w=1 和默认拒选/显式 uniform。
- 所有排序器 top-k 身份不放回，多轨迹共享总预算、cache 和分轨迹 delta 计数。
- ensemble 逐成员收费，BatchScorer 按 item 收费，显式 cache 才允许命中，Random 的 item-forward 为 0。
- adversarial/independent/oracle 合成真值、V/Q 概率 logit 零噪声和端点。
- 常数 V 的 FP-REJ 在 C=1/2/3 全路径上回到基础分布，最大误差约 1.11e-16；FP-TILT alpha=0 省略 V/Q 计分。

完整结果：**116 passed**，独立数学检查 PASS，源码审计 PASS。

## 第三轮：协议一致性与复现

- 完整 pytest 再次 **116 passed**，无 xfail。
- 独立数学检查再次 PASS；输出保存为 `round3_independent_math.json`。
- 源码审计增加默认 `ZeroWeightPolicy.ABSTAIN`、product 单次指数和 pool 不含 qhat 的静态断言，结果 PASS。
- 默认 CLI 双运行 JSON 字节一致。配置哈希为 `72c9f35629af2699ff455f1bdad11cb0cce6d11bdd1cd8a0441570c54ffdc4ed`，JSON SHA-256 为 `009632086edad208ca5d0e095dfd5b303ecd0cd65aa79fa261525cb75ee31354`。
- smoke 的实际成本：Random=0；BoN=24 proxy item-forward；WSC/greedy=24 verifier item-forward；FP-TILT=9 verifier item-forward；FP-REJ=20 verifier + 3 Q item-forward。相同 proposal cap 不代表相同实际成本。

## 审查发现与修复

1. 修正共享 cache 测试对首次轨迹计分次数的错误预期。
2. 修正三候选测试的可行掩码长度。
3. 修正独立参考脚本的每提议接受率期望定义。
4. 补充批量排序器 top-k、常数 V 和 alpha=0 成本测试。
5. 替换旧审查文档中 V-only/joint、联合 greedy 和静默均匀回退的过期描述。

## 未解决项与边界

- `ScoreCache` 只按 score kind 和 sequence ID 索引；调用方必须保证同一 cache 内 scorer/ensemble 已冻结，跨模型复用需另建 cache。
- 多轨迹采用固定 RNG 的顺序执行；未声称并行归约复现。
- ensemble 当前固定为成员均值；product/min/mean 是 V/Q 聚合，不是 ensemble 聚合。
- 全路径穷举是小候选池正确性证据，不是对所有分布的形式化证明，也不构成生物效果证据。
- FP-TILT 需要评分全部正提议质量身份，实际成本可能高于提前接受的 FP-REJ。
- 仍无真实标签、GPU 训练、前缀生成或 E1 结果。
