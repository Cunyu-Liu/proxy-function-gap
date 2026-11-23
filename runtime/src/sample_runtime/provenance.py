from pathlib import Path

STAGING_ROOT = Path("/home/proxy-function-gap/runtime")
AUDIT_ROOT = Path("/home/proxy-function-gap-sample-audit")
OFFICIAL_ROOT = AUDIT_ROOT / "official-d53df410"
MODEL_ROOT = OFFICIAL_ROOT / "modeling" / "saved_models"
ORIGINAL_WEIGHTS = MODEL_ROOT / "evolution_model.hdf5"
RETRAINED_WEIGHTS = MODEL_ROOT / "retrained_evolution_model.hdf5"
DESIGNED_DATA = AUDIT_ROOT / "data" / "GSM3130443_designed_library.clean.csv.gz"
RANDOM_DATA = AUDIT_ROOT / "data" / "GSM3130435_egfp_unmod_1.csv.gz"
OFFICIAL_COMMIT = "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe"
OFFICIAL_REPOSITORY = "https://github.com/pjsample/human_5utr_modeling"
OFFICIAL_LICENSE = "GPL-3.0"
WEIGHT_SHA256 = {
    "original": "892052000f74bde49f5612fd10c83cfaa259b479647aa227807bff5d3adfea1c",
    "retrained": "fdf0f676b0c16a3ef594bb09c2382637266a42e7cee8b16730a8c0e2d2d42c9b",
}
DATA_SHA256 = {
    "designed": "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5",
    "random": "23bbd468ff6c6905f11e7dfdd7509601730e0f99c8ad2a78f37f3dfe99c31719",
}
SCALER = {"name": "egfp_unmod_1_first40000_ddof0", "mean": 5.993167213121654, "scale": 1.799413065189219, "ddof": 0}
TARGET_VALIDATION_AFFINE = {"mean": 6.007428661802911, "scale": 1.5849444899584102}
STEP_LIBRARIES = (
    "step_random_to_best_allow_uatg",
    "step_random_to_best_no_uatgs",
    "step_worst_to_best_allow_uatg",
    "step_worst_to_best_no_uatg",
)
EXPECTED_COUNTS = {"step": 5710, "target": 7106}
GATE0_SEQUENCE_INDEX = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/gate0_sample2019_v1/sequence_index.tsv.gz")
GATE0_SPLIT_V1_DIR = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/gate0_split_v1")
DEFAULT_MODEL_ROLE_MANIFEST = STAGING_ROOT / "configs" / "model_roles_v1.json"
