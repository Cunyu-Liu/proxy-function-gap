# E1 误差分层（描述性）

仅汇总已生成预测；频率切片的显著性只来自预指定配对 bootstrap，低同源切片不作显著性声明。

低同源定义：测试正例 d_hom < 0.60，配同实验全部测试负例；负例不按 d_hom 子采样。

| model | track | slice | n_experiments | auroc | auprc | auroc_sem | auprc_sem |
|---|---|---|---|---|---|---|---|
| knn | neg1c | head | 100 | 0.863909 | 0.843007 | 0.00625 | 0.007086 |
| knn | neg1c | mid | 28 | 0.850149 | 0.830712 | 0.01113 | 0.012696 |
| knn | neg1c | tail | 18 | 0.822089 | 0.798064 | 0.010996 | 0.012423 |
| knn | neg2 | head | 100 | 0.799754 | 0.768913 | 0.009065 | 0.009858 |
| knn | neg2 | mid | 28 | 0.759885 | 0.733774 | 0.012929 | 0.014319 |
| knn | neg2 | tail | 18 | 0.691325 | 0.660728 | 0.014472 | 0.01392 |
| modern_s0 | neg1c | head | 100 | 0.902912 | 0.894419 | 0.004855 | 0.005288 |
| modern_s0 | neg1c | mid | 28 | 0.885719 | 0.87693 | 0.008989 | 0.009928 |
| modern_s0 | neg1c | tail | 18 | 0.853571 | 0.841091 | 0.010293 | 0.012066 |
| modern_s0 | neg2 | head | 100 | 0.835857 | 0.820062 | 0.0082 | 0.00879 |
| modern_s0 | neg2 | mid | 28 | 0.792714 | 0.778776 | 0.011873 | 0.012828 |
| modern_s0 | neg2 | tail | 18 | 0.715311 | 0.696201 | 0.013218 | 0.014019 |
| modern_s1 | neg1c | head | 100 | 0.903032 | 0.894685 | 0.004842 | 0.005255 |
| modern_s1 | neg1c | mid | 28 | 0.88614 | 0.877234 | 0.008944 | 0.009854 |
| modern_s1 | neg1c | tail | 18 | 0.853383 | 0.841972 | 0.010327 | 0.011706 |
| modern_s1 | neg2 | head | 100 | 0.835842 | 0.820001 | 0.00818 | 0.008762 |
| modern_s1 | neg2 | mid | 28 | 0.792549 | 0.778658 | 0.011911 | 0.012927 |
| modern_s1 | neg2 | tail | 18 | 0.71554 | 0.695841 | 0.013549 | 0.014337 |
| modern_s2 | neg1c | head | 100 | 0.903088 | 0.894819 | 0.004847 | 0.00526 |
| modern_s2 | neg1c | mid | 28 | 0.885528 | 0.876475 | 0.009051 | 0.009991 |
| modern_s2 | neg1c | tail | 18 | 0.853407 | 0.842075 | 0.010486 | 0.01185 |
| modern_s2 | neg2 | head | 100 | 0.835933 | 0.820083 | 0.008181 | 0.008771 |
| modern_s2 | neg2 | mid | 28 | 0.793036 | 0.779123 | 0.011887 | 0.012846 |
| modern_s2 | neg2 | tail | 18 | 0.715466 | 0.694706 | 0.013617 | 0.014669 |
| knn | neg1c | low_hom<0.60 | 146 | 0.830752 | 0.697716 | 0.005249 | 0.008409 |
| knn | neg2 | low_hom<0.60 | 146 | 0.751782 | 0.585196 | 0.007274 | 0.009052 |
| modern_s0 | neg1c | low_hom<0.60 | 146 | 0.875228 | 0.781294 | 0.004336 | 0.007274 |
| modern_s0 | neg2 | low_hom<0.60 | 146 | 0.790413 | 0.653823 | 0.007027 | 0.009507 |
| modern_s1 | neg1c | low_hom<0.60 | 146 | 0.875358 | 0.781638 | 0.004336 | 0.007247 |
| modern_s1 | neg2 | low_hom<0.60 | 146 | 0.790605 | 0.65361 | 0.006989 | 0.009513 |
| modern_s2 | neg1c | low_hom<0.60 | 146 | 0.875409 | 0.781822 | 0.00433 | 0.007264 |
| modern_s2 | neg2 | low_hom<0.60 | 146 | 0.790682 | 0.653702 | 0.006994 | 0.009483 |

## Run ID
- knn: `E1-cv-prediction-combine-20261126-064253-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2/cv_combined/knn.tsv.gz`
- modern_s0: `E1-cv-prediction-combine-20261126-064253-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2/cv_combined/modern_bce_s0.tsv.gz`
- modern_s1: `E1-cv-prediction-combine-20261126-064253-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2/cv_combined/modern_bce_s1.tsv.gz`
- modern_s2: `E1-cv-prediction-combine-20261126-064253-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main_strict3train_p20_v2/cv_combined/modern_bce_s2.tsv.gz`
