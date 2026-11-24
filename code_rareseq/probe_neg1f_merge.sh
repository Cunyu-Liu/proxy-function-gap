#!/usr/bin/env bash
# Supplementary neg1f probe (pilot 5-fold + main fold0), merged into g0/ and g0_mainfold0/.
set -uo pipefail
W=/root/rareseq/work
E=/root/rareseq/data/e0
PYC="/root/miniconda3/envs/codonflow/bin/python"
export PYTHONPATH=/root/rareseq/pydeps
cd "$W"
$PYC e0_g0_models.py probe --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0_neg1f" --model rnafm --exps-file "$E/pilot_exps.txt" \
  --folds 5 --tracks neg1f
$PYC e0_g0_models.py probe --windows-dir "$E/windows" --neg-dir "$E/neg" \
  --emb-dir "$E/emb" --out "$E/g0_mainfold0_neg1f" --model rnafm --exps-file "$E/main_exps.txt" \
  --folds 1 --tracks neg1f
PYTHONPATH=/root/rareseq/pydeps /root/miniconda3/envs/codonflow/bin/python - <<'EOF'
import pandas as pd
E0="/root/rareseq/data/e0"
for main, extra in [(f"{E0}/g0", f"{E0}/g0_neg1f"),
                    (f"{E0}/g0_mainfold0", f"{E0}/g0_mainfold0_neg1f")]:
    a = pd.read_csv(f"{main}/probe_results.tsv", sep="\t")
    b = pd.read_csv(f"{extra}/probe_results.tsv", sep="\t")
    m = pd.concat([a, b], ignore_index=True).drop_duplicates(["experiment","fold","track","model"])
    m.to_csv(f"{main}/probe_results.tsv", sep="\t", index=False)
    s = m.groupby(["track","model"])[["auroc","auprc"]].mean().round(4).reset_index()
    s.to_csv(f"{main}/probe_summary.tsv", sep="\t", index=False)
    print(main, "rows=", len(m)); print(s.to_string(index=False))
EOF
echo NEG1F_PROBE_MERGED