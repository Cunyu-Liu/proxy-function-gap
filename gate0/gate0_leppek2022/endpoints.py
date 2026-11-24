POOLED_ENDPOINTS = {
    "ribosome_load": {
        "column": "Ribosome load", "error_column": "Ribosome load error", "status": "measured",
        "provenance": "weighted_sum_of_polysome_sequencing_proportions", "units": "ribosomes_per_mRNA_weighted_load",
        "assay": "PERSIST-seq_polysome_profiling_16_fractions", "cell": "HEK293T", "time": "post_transfection_time_not_stated_in_S1",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "monosome_to_40s_60s_ratio": {
        "column": "Monosome to 40S/60S ratio", "status": "measured",
        "provenance": "ratio_of_polysome_sequencing_fractions", "units": "ratio",
        "assay": "PERSIST-seq_polysome_profiling_16_fractions", "cell": "HEK293T", "time": "post_transfection_time_not_stated_in_S1",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "polysome_to_monosome_ratio": {
        "column": "Polysome to monosome ratio", "status": "measured",
        "provenance": "ratio_of_polysome_sequencing_fractions", "units": "ratio",
        "assay": "PERSIST-seq_polysome_profiling_16_fractions", "cell": "HEK293T", "time": "post_transfection_time_not_stated_in_S1",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "polysome_to_40s_60s_ratio": {
        "column": "Polysome to 40S/60S ratio", "status": "measured",
        "provenance": "ratio_of_polysome_sequencing_fractions", "units": "ratio",
        "assay": "PERSIST-seq_polysome_profiling_16_fractions", "cell": "HEK293T", "time": "post_transfection_time_not_stated_in_S1",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "in_cell_degradation_coefficient": {
        "column": "In-cell degradation coefficient", "error_column": "In-cell degradation coefficient error", "status": "measured",
        "provenance": "fit_from_PERSIST-seq_timecourse_1_7_12_24h_3_replicates", "units": "per_hour",
        "assay": "PERSIST-seq_in-cell_RNA_stability", "cell": "HEK293T", "time": "fit_1_7_12_24h",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "in_cell_half_life": {
        "column": "In-cell half-life", "status": "measured",
        "provenance": "ln2_divided_by_measured_in-cell_degradation_coefficient", "units": "hour",
        "assay": "PERSIST-seq_in-cell_RNA_stability", "cell": "HEK293T", "time": "fit_1_7_12_24h",
        "construct": "pooled_full_length_mRNA_m7G_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "in_solution_degradation_coefficient": {
        "column": "In-solution degradation coefficient", "error_column": "In-solution degradation coefficient error", "status": "measured",
        "provenance": "fit_from_PERSIST-seq_solution_timecourse_0_0.5_1_2_3_4_5_6_16_24h", "units": "per_hour",
        "assay": "PERSIST-seq_in-solution_RNA_stability_CHES_pH10_10mM_MgCl2", "cell": "not_applicable", "time": "fit_0_0.5_1_2_3_4_5_6_16_24h",
        "construct": "pooled_full_length_mRNA_not_m7G_not_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "in_solution_half_life": {
        "column": "In-solution half-life", "status": "measured",
        "provenance": "ln2_divided_by_measured_in-solution_degradation_coefficient", "units": "hour",
        "assay": "PERSIST-seq_in-solution_RNA_stability_CHES_pH10_10mM_MgCl2", "cell": "not_applicable", "time": "fit_0_0.5_1_2_3_4_5_6_16_24h",
        "construct": "pooled_full_length_mRNA_not_m7G_not_polyA", "chemistry": "unmodified_unless_construct_name_states_otherwise",
    },
    "predicted_expression": {
        "column": "Predicted expression", "error_column": "Predicted expression error", "status": "predicted",
        "provenance": "kinetic_prediction_from_ribosome_load_and_in-cell_degradation_NOT_protein_measurement", "units": "arbitrary_predicted_units",
        "assay": "kinetic_model_not_assay", "cell": "HEK293T_model_context", "time": "model_dependent_not_measured",
        "construct": "pooled_full_length_mRNA_model", "chemistry": "not_applicable",
    },
}

CDS_ENDPOINTS = {
    "cds_in_solution_degradation_coefficient": {
        "column": "k_deg per hour", "error_column": "k_deg_err per hour", "status": "measured",
        "provenance": "fit_from_individual_timecourse_exact_n_in_Source_Data_Fig_4c", "units": "per_hour",
        "assay": "individual_CDS_in-solution_degradation", "cell": "not_applicable", "time": "timecourse_fit_source_timepoints_not_tabulated_in_S5",
        "construct": "individual_HBB_Nluc_CDS_mRNA", "chemistry": "pseudouridine_PSU",
    },
    "cds_in_solution_half_life": {
        "column": "half_life", "error_column": "half_life_err", "status": "measured",
        "provenance": "fit_from_individual_timecourse_exact_n_in_Source_Data_Fig_4c", "units": "hour",
        "assay": "individual_CDS_in-solution_degradation", "cell": "not_applicable", "time": "timecourse_fit_source_timepoints_not_tabulated_in_S5",
        "construct": "individual_HBB_Nluc_CDS_mRNA", "chemistry": "pseudouridine_PSU",
    },
    "nluc_fluc_6h": {
        "column": "Nluc_Fluc_6h_mean", "error_column": "Nluc_Fluc_6h_sd", "status": "measured",
        "provenance": "mean_of_at_least_3_biological_replicates_SD_in_error_field", "units": "Nluc_Fluc_ratio",
        "assay": "dual_luciferase_HEK293T", "cell": "HEK293T", "time": "6h",
        "construct": "individual_HBB_Nluc_CDS_mRNA_with_HBB_Fluc_control", "chemistry": "pseudouridine_PSU",
    },
    "nluc_fluc_24h": {
        "column": "Nluc_Fluc_24h_mean", "error_column": "Nluc_Fluc_24h_sd", "status": "measured",
        "provenance": "mean_of_at_least_3_biological_replicates_SD_in_error_field", "units": "Nluc_Fluc_ratio",
        "assay": "dual_luciferase_HEK293T", "cell": "HEK293T", "time": "24h",
        "construct": "individual_HBB_Nluc_CDS_mRNA_with_HBB_Fluc_control", "chemistry": "pseudouridine_PSU",
    },
    "cds_half_life_norm_to_start": {
        "column": "half_life_norm_to_START", "error_column": "half_life_err_norm_to_START", "status": "measured",
        "provenance": "normalized_from_measured_half_life_to_START_reference", "units": "ratio_to_START",
        "assay": "individual_CDS_in-solution_degradation", "cell": "not_applicable", "time": "timecourse_fit",
        "construct": "individual_HBB_Nluc_CDS_mRNA", "chemistry": "pseudouridine_PSU",
    },
    "nluc_fluc_6h_norm_to_start": {
        "column": "Nluc_Fluc_6h_mean_norm_to_START", "error_column": "Nluc_Fluc_6h_sd_norm_to_START", "status": "measured",
        "provenance": "normalized_from_measured_dual_luciferase_to_START_reference", "units": "ratio_to_START",
        "assay": "dual_luciferase_HEK293T", "cell": "HEK293T", "time": "6h",
        "construct": "individual_HBB_Nluc_CDS_mRNA_with_HBB_Fluc_control", "chemistry": "pseudouridine_PSU",
    },
    "nluc_fluc_24h_norm_to_start": {
        "column": "Nluc_Fluc_24h_mean_norm_to_START", "error_column": "Nluc_Fluc_24h_sd_norm_to_START", "status": "measured",
        "provenance": "normalized_from_measured_dual_luciferase_to_START_reference", "units": "ratio_to_START",
        "assay": "dual_luciferase_HEK293T", "cell": "HEK293T", "time": "24h",
        "construct": "individual_HBB_Nluc_CDS_mRNA_with_HBB_Fluc_control", "chemistry": "pseudouridine_PSU",
    },
}

POOLED_FEATURES = {
    "cai": "CAI", "cds_gc_content": "GC content (CDS)",
    "sup_degscore2_1": "SUP DegScore2.1", "sup_vienna": "SUP Vienna", "sup_eternafold": "SUP EternaFold",
    "aup_degscore2_1": "AUP DegScore 2.1", "aup_vienna": "AUP Vienna", "aup_eternafold": "AUP EternaFold",
    "mfe_vienna": "dG(MFE) Vienna", "mfe_eternafold": "dG(MFE) EternaFold",
}

CDS_FEATURES = {
    "cai": "CAI", "cds_gc_content": "GC content", "degscore_cds": "degscore CDS",
    "mfe_vienna_cds": "dG(MFE) Vienna CDS", "ensemble_vienna_cds": "dG(ensemble) Vienna CDS",
    "sup_vienna_cds": "SUP Vienna CDS", "aup_vienna_cds": "AUP Vienna CDS",
    "mfe_eternafold_cds": "dG(MFE) EternaFold CDS", "ensemble_eternafold_cds": "dG(ensemble) EternaFold CDS",
    "sup_eternafold_cds": "SUP EternaFold CDS", "aup_eternafold_cds": "AUP EternaFold CDS",
    "mld_cds": "MLD CDS",
}

ALL_ENDPOINTS = tuple(dict.fromkeys(tuple(POOLED_ENDPOINTS) + tuple(CDS_ENDPOINTS) + tuple(POOLED_FEATURES) + tuple(CDS_FEATURES)))
