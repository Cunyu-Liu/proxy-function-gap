# E1 首批误差分层（描述性）

仅汇总已生成预测；单折单种子不作显著性、G0 最终确认或论文结论。

低同源定义：测试正例 d_hom < 0.60，配同实验全部测试负例；负例不按 d_hom 子采样。

| model | track | slice | n_experiments | auroc | auprc | auroc_sem | auprc_sem |
|---|---|---|---|---|---|---|---|
| knn | neg1c | head | 100 | 0.863975 | 0.842995 | 0.00625 | 0.007084 |
| knn | neg1c | mid | 28 | 0.850345 | 0.830852 | 0.011113 | 0.012689 |
| knn | neg1c | tail | 18 | 0.822467 | 0.798722 | 0.010972 | 0.012385 |
| knn | neg2 | head | 100 | 0.799811 | 0.768893 | 0.009064 | 0.009854 |
| knn | neg2 | mid | 28 | 0.760151 | 0.733931 | 0.012908 | 0.014306 |
| knn | neg2 | tail | 18 | 0.691583 | 0.661345 | 0.014487 | 0.01399 |
| modern_s0 | neg1c | head | 100 | 0.901661 | 0.893252 | 0.004908 | 0.005308 |
| modern_s0 | neg1c | mid | 28 | 0.884522 | 0.875876 | 0.00912 | 0.010101 |
| modern_s0 | neg1c | tail | 18 | 0.852916 | 0.840943 | 0.009978 | 0.011439 |
| modern_s0 | neg2 | head | 100 | 0.834703 | 0.818602 | 0.008238 | 0.008801 |
| modern_s0 | neg2 | mid | 28 | 0.791608 | 0.777638 | 0.012005 | 0.013051 |
| modern_s0 | neg2 | tail | 18 | 0.718986 | 0.700678 | 0.013174 | 0.013959 |
| modern_s1 | neg1c | head | 100 | 0.901193 | 0.892449 | 0.004935 | 0.005372 |
| modern_s1 | neg1c | mid | 28 | 0.884077 | 0.874747 | 0.009226 | 0.010269 |
| modern_s1 | neg1c | tail | 18 | 0.853504 | 0.841673 | 0.010057 | 0.011425 |
| modern_s1 | neg2 | head | 100 | 0.834511 | 0.818408 | 0.008248 | 0.008832 |
| modern_s1 | neg2 | mid | 28 | 0.79166 | 0.777725 | 0.012064 | 0.013107 |
| modern_s1 | neg2 | tail | 18 | 0.717984 | 0.699192 | 0.013186 | 0.013976 |
| modern_s2 | neg1c | head | 100 | 0.901677 | 0.893074 | 0.004909 | 0.005328 |
| modern_s2 | neg1c | mid | 28 | 0.885175 | 0.875778 | 0.009128 | 0.010153 |
| modern_s2 | neg1c | tail | 18 | 0.854472 | 0.84328 | 0.01002 | 0.011444 |
| modern_s2 | neg2 | head | 100 | 0.834603 | 0.818613 | 0.008226 | 0.008794 |
| modern_s2 | neg2 | mid | 28 | 0.792475 | 0.778822 | 0.011877 | 0.01289 |
| modern_s2 | neg2 | tail | 18 | 0.718952 | 0.699919 | 0.013147 | 0.013868 |
| knn | neg1c | low_hom<0.60 | 146 | 0.830872 | 0.697819 | 0.005245 | 0.008399 |
| knn | neg2 | low_hom<0.60 | 146 | 0.751794 | 0.585163 | 0.007271 | 0.009037 |
| modern_s0 | neg1c | low_hom<0.60 | 146 | 0.874275 | 0.780478 | 0.004334 | 0.007176 |
| modern_s0 | neg2 | low_hom<0.60 | 146 | 0.7904 | 0.65339 | 0.006957 | 0.009388 |
| modern_s1 | neg1c | low_hom<0.60 | 146 | 0.873978 | 0.779358 | 0.004337 | 0.007207 |
| modern_s1 | neg2 | low_hom<0.60 | 146 | 0.790057 | 0.653012 | 0.006971 | 0.009437 |
| modern_s2 | neg1c | low_hom<0.60 | 146 | 0.874679 | 0.780821 | 0.004326 | 0.007202 |
| modern_s2 | neg2 | low_hom<0.60 | 146 | 0.790456 | 0.653612 | 0.006924 | 0.009393 |

## Run ID
- knn: `E1-cv-prediction-combine-20261125-052922-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main/cv_combined/knn.tsv.gz`
- modern_s0: `E1-cv-prediction-combine-20261125-052922-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main/cv_combined/modern_bce_s0.tsv.gz`
- modern_s1: `E1-cv-prediction-combine-20261125-052922-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main/cv_combined/modern_bce_s1.tsv.gz`
- modern_s2: `E1-cv-prediction-combine-20261125-052922-ZDLg6Y`；`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_main/cv_combined/modern_bce_s2.tsv.gz`
