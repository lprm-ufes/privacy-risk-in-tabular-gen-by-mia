import gc
import sys
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import numpy as np
import pandas as pd
import toml
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier

from flowmia import FlowMIA
from utils import utils
from utils.flowmia_config_builder import FlowMiaConfigBuilder
from utils.torch_mlp import TorchMLPClassifier
from utils.measure_domain_shift import evaluate_domain_shift

def get_threshold_by_fpr_rate(negative_class_scores, rate: float):
    if len(negative_class_scores) == 0:
        raise ValueError("negative_class_scores cannot be empty")

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")

    ordered_scores = np.sort(negative_class_scores)
    threshold_idx = int(np.ceil(len(ordered_scores) * (1 - rate)))

    if threshold_idx >= len(ordered_scores):
        return np.nextafter(ordered_scores[-1], np.inf)

    return ordered_scores[threshold_idx]

def get_tpr_rate_by_threshold(positive_class_scores, threshold, negative_class_scores=None, fpr_rate: float = None):
    if negative_class_scores is not None and fpr_rate is not None:
        y_true = np.concatenate([
            np.ones(len(positive_class_scores)),
            np.zeros(len(negative_class_scores)),
        ])
        y_score = np.concatenate([positive_class_scores, negative_class_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.interp(fpr_rate, fpr, tpr))
    return len(positive_class_scores[positive_class_scores >= threshold]) / len(positive_class_scores)

def build_classifiers(utility_params: dict, seed: int) -> list:
    rf_n_estimators = utility_params.get("rf_n_estimators", 100)
    knn_n_neighbors = utility_params.get("knn_n_neighbors", 5)
    mlp_hidden_layers = tuple(utility_params.get("mlp_hidden_layer_sizes", [100]))
    mlp_max_iter = utility_params.get("mlp_max_iter", 300)
    mlp_batch_size = utility_params.get("mlp_batch_size", utility_params.get("batch_size", 50))

    return [
        RandomForestClassifier(n_estimators=rf_n_estimators, random_state=seed),
        KNeighborsClassifier(n_neighbors=knn_n_neighbors),
        TorchMLPClassifier(
            hidden_layer_sizes=mlp_hidden_layers,
            max_iter=mlp_max_iter,
            batch_size=mlp_batch_size,
            random_state=seed,
        ),
    ]


def append_and_save_tracker(tracker_df: pd.DataFrame | None, row_details: dict, file_path: Path) -> pd.DataFrame:
    new_row_df = pd.DataFrame([row_details])
    if tracker_df is not None:
        tracker_df = pd.concat([tracker_df, new_row_df], ignore_index=True)
    else:
        tracker_df = new_row_df
    tracker_df.to_csv(file_path, index=False, encoding="utf-8")
    return tracker_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo de configuração.")
    args = parser.parse_args()

    config = utils.read_config(args.config)

    io_cfg = config["io"]
    member_path = utils.validate_file(io_cfg["member_dataset_input"])
    non_member_path = utils.validate_file(io_cfg["non_member_dataset_input"])
    test_path = utils.validate_file(io_cfg["test_dataset_input"])
    dataset_info_path = utils.validate_file(io_cfg["dataset_info"])

    synthetic_tracker_file = utils.validate_file(io_cfg["synthetic_tracker_file"])
    synthetic_datasets_folder = utils.validate_folder(io_cfg["synthetic_datasets_folder"], create=False)
    results_output_folder = utils.validate_folder(io_cfg["results_output_folder"], create=True)

    dataset_info = toml.load(dataset_info_path)
    total_train = dataset_info["sizes"].get("train", 0)
    total_test = dataset_info["sizes"].get("test", 0)
    total_reference = dataset_info["sizes"].get("reference", 0)

    num_columns = dataset_info["columns"].get("num_col", [])
    cat_columns = dataset_info["columns"].get("cat_col", [])
    ip_columns = dataset_info["columns"].get("ip_col", [])
    target_column = dataset_info["columns"]["label"]

    if len(ip_columns) > 0:
        num_columns = [col for col in num_columns if col not in ip_columns]
        cat_columns = [col for col in cat_columns if col not in ip_columns]

    exp_cfg = config.get("exp", {})
    remove_exps_folder = exp_cfg.get("remove_exps_folder", False)
    start_exp = exp_cfg.get("start_exp", None)
    end_exp = exp_cfg.get("end_exp", None)
    save_scores = exp_cfg.get("save_scores", True)
    run_attack = exp_cfg.get("run_attack", True)
    run_utility = exp_cfg.get("run_utility", True)
    run_fidelity = exp_cfg.get("run_fidelity", True)
    run_adversarial_shift = exp_cfg.get("run_adversarial_shift", False)
    plot = exp_cfg.get("plot", False)

    flowmia_params = config.get("flowmia_params", {})
    seed = flowmia_params.get("seed", 42)
    test_size = flowmia_params.get("test_size", min(total_train, total_reference))
    test_size = min(test_size, total_train, total_reference)
    use_wgan = flowmia_params.get("use_wgan", True)
    fpr_rate = flowmia_params.get("fpr_rate", 0.1)
    batch_size = flowmia_params.get("batch_size", 50)
    num_epochs = flowmia_params.get("num_epochs", 30)
    fcheckpoint = flowmia_params.get("fcheckpoint", 100)
    domias_epochs = flowmia_params.get("domias_epochs", 30)
    domias_batch_size = flowmia_params.get("domias_batch_size", 50)

    utility_params = config.get("utility_params", {})

    synthetic_tracker_df = pd.read_csv(synthetic_tracker_file, encoding="utf-8")
    idx_col = "exp_idx" if "exp_idx" in synthetic_tracker_df.columns else "test_idx"

    if start_exp is not None:
        synthetic_tracker_df = synthetic_tracker_df[synthetic_tracker_df[idx_col] >= start_exp]
    if end_exp is not None:
        synthetic_tracker_df = synthetic_tracker_df[synthetic_tracker_df[idx_col] <= end_exp]

    privacy_tracker_path = results_output_folder / "privacy_tracker.csv"
    utility_tracker_path = results_output_folder / "utility_tracker.csv"
    fidelity_tracker_path = results_output_folder / "fidelity_tracker.csv"

    privacy_df = pd.read_csv(privacy_tracker_path, encoding="utf-8") if privacy_tracker_path.is_file() else None
    utility_df = pd.read_csv(utility_tracker_path, encoding="utf-8") if utility_tracker_path.is_file() else None
    fidelity_df = pd.read_csv(fidelity_tracker_path, encoding="utf-8") if fidelity_tracker_path.is_file() else None

    base_columns = list(synthetic_tracker_df.columns)

    clf_short_names = {
        "RandomForestClassifier": "rf",
        "KNeighborsClassifier": "knn",
        "MLPClassifier": "mlp",
        "TorchMLPClassifier": "mlp",
    }

    if run_adversarial_shift:
        auc, pad = evaluate_domain_shift(pd.read_csv(member_path), pd.read_csv(non_member_path), num_columns, cat_columns)
        shift_df = pd.DataFrame({'aux': [auc], 'pad': [pad]})
        shift_df.to_csv(results_output_folder / "adversarial_shift.csv", index=False, encoding='utf=8')


    for _, row in synthetic_tracker_df.iterrows():
        exp_idx = row[idx_col]
        dataset_file = row["dataset_file"]
        synth_path = utils.validate_file(synthetic_datasets_folder / dataset_file)

        exp_output_folder_base = results_output_folder / "exps"
        exp_output_folder = utils.validate_folder(exp_output_folder_base / f"exp_{exp_idx}", create=True)
        domias_output_folder = utils.validate_folder(exp_output_folder / "domias", create=True)

        print(f"\n[Exp {exp_idx}] Avaliando dataset sintético: {synth_path}")

        builder = (
            FlowMiaConfigBuilder(
                member_path=str(member_path),
                non_member_path=str(non_member_path),
                synth_path=str(synth_path),
                categorical_cols=cat_columns,
                numerical_cols=num_columns,
                label_col=target_column,
            )
            .with_ip_columns(ip_columns)
            .with_test(str(test_path), test_size)
            .with_batch_size(batch_size)
            .with_num_epochs(num_epochs)
            .with_checkpoint_frequency(fcheckpoint)
            .with_save_path(str(exp_output_folder))
            .with_use_wgan(use_wgan)
            .with_seed(seed)
        )
        flowmia_config = builder.build()

        flowmia = FlowMIA(config=flowmia_config)

        effective_test_size = min(
            test_size,
            len(flowmia.X_member),
            len(flowmia.X_non_member),
        )

        base_details = {col: row[col] for col in base_columns}
        
        # Privacidade
        if run_attack:
            print(f"[Exp {exp_idx}] Executando avaliação de Privacidade...")
            flowmiagan_res = flowmia.flowmiagan(plot=plot, test_size=effective_test_size)

            print("FlowMiaGan scores:")
            print('score_members:', len(flowmiagan_res['score_members']))
            print('score_non_members:', len(flowmiagan_res['score_non_members']))

            threshold = get_threshold_by_fpr_rate(flowmiagan_res['score_non_members'], fpr_rate)
            flowmiagan_tpr = get_tpr_rate_by_threshold(
                flowmiagan_res['score_members'],
                threshold,
                flowmiagan_res['score_non_members'],
                fpr_rate,
            )
            print(f"TPR@FPR={fpr_rate}:", flowmiagan_tpr, "Threshold:", threshold)

            domias_scores, domias_auc = flowmia.domias(
                test_size=effective_test_size,
                epochs=domias_epochs,
                batch_size=domias_batch_size,
                save_path=str(domias_output_folder),
            )
            domias_scores_members = domias_scores[:effective_test_size]
            domias_scores_non_members = domias_scores[effective_test_size:]

            print("Domias scores:")
            print('score_members:', len(domias_scores_members))
            print('score_non_members:', len(domias_scores_non_members))

            threshold = get_threshold_by_fpr_rate(domias_scores_non_members, fpr_rate)
            domias_tpr = get_tpr_rate_by_threshold(
                domias_scores_members,
                threshold,
                domias_scores_non_members,
                fpr_rate,
            )
            print(f"TPR@FPR={fpr_rate}:", domias_tpr, "Threshold:", threshold)

            dcr_scores, dcr_auc = flowmia.compute_dcr(test_size=effective_test_size)

            dcr_scores_members = dcr_scores[:effective_test_size]
            dcr_scores_non_members = dcr_scores[effective_test_size:]

            print("DCR scores:")
            print('score_members:', len(dcr_scores_members))
            print('score_non_members:', len(dcr_scores_non_members))

            threshold = get_threshold_by_fpr_rate(dcr_scores_non_members, fpr_rate)
            dcr_tpr = get_tpr_rate_by_threshold(
                dcr_scores_members,
                threshold,
                dcr_scores_non_members,
                fpr_rate,
            )
            print(f"TPR@FPR={fpr_rate}:", dcr_tpr, "Threshold:", threshold)

            if save_scores:
                scores_folder = utils.validate_folder(exp_output_folder / "scores", create=True)
                np.save(
                    scores_folder / f"flowmiagan_scores_exp_{exp_idx}.npy",
                    {
                        "score_members": flowmiagan_res["score_members"],
                        "score_non_members": flowmiagan_res["score_non_members"],
                        "score_synthetic": flowmiagan_res["score_synthetic"],
                        "score_random": flowmiagan_res["score_random"],
                    },
                )
                np.save(scores_folder / f"domias_scores_exp_{exp_idx}.npy", domias_scores)
                np.save(scores_folder / f"dcr_scores_exp_{exp_idx}.npy", dcr_scores)

            privacy_row = base_details.copy()
            privacy_row.update({
                "fpr_fixed_rate": fpr_rate,
                "flowmiagan_tpr@fpr": flowmiagan_tpr,
                "flowmiagan_auc": flowmiagan_res["auc"],
                "domias_tpr@fpr": domias_tpr,
                "domias_auc": domias_auc,
                "dcr_tpr@fpr": dcr_tpr,
                "dcr_auc": dcr_auc,
                "flowmiagan_accuracy": flowmiagan_res["accuracy"],
                "flowmiagan_precision": flowmiagan_res["precision"],
                "flowmiagan_recall": flowmiagan_res["recall"],
                "flowmiagan_f1_score": flowmiagan_res["f1_score"],
                "flowmiagan_threshold": flowmiagan_res["threshold"],
                "flowmiagan_threshold_method": flowmiagan_res["threshold_method"],
                "flowmiagan_tp": flowmiagan_res["tp"],
                "flowmiagan_fp": flowmiagan_res["fp"],
                "flowmiagan_tn": flowmiagan_res["tn"],
                "flowmiagan_fn": flowmiagan_res["fn"],
                "mean_score_members": flowmiagan_res["mean_score_members"],
                "mean_score_non_members": flowmiagan_res["mean_score_non_members"],
                "mean_score_synthetic": flowmiagan_res["mean_score_synthetic"],
                "mean_score_random": flowmiagan_res["mean_score_random"],
                "std_score_members": flowmiagan_res["std_score_members"],
                "std_score_non_members": flowmiagan_res["std_score_non_members"],
                "std_score_synthetic": flowmiagan_res["std_score_synthetic"],
                "std_score_random": flowmiagan_res["std_score_random"],
                "score_gap_member_non_member": flowmiagan_res["score_gap_member_non_member"],
                "score_gap_member_synthetic": flowmiagan_res["score_gap_member_synthetic"],
                "score_gap_member_random": flowmiagan_res["score_gap_member_random"],
                "privacy_risk_score": flowmiagan_res["privacy_risk_score"],
                "wasserstein_member_synthetic": flowmiagan_res["wasserstein_member_synthetic"],
                "wasserstein_nonmember_synthetic": flowmiagan_res["wasserstein_nonmember_synthetic"],
            })
            privacy_df = append_and_save_tracker(privacy_df, privacy_row, privacy_tracker_path)
            print(f"Privacidade salva em: {privacy_tracker_path}")

        # Utilidade
        if run_utility:
            print(f"[Exp {exp_idx}] Executando avaliação de Utilidade...")
            classifiers = build_classifiers(utility_params, seed=seed)
            utility_res = flowmia.evaluate_utility(classifiers=classifiers, plot=plot)

            utility_row = base_details.copy()
            for clf_name, protocols in utility_res.items():
                short_name = clf_short_names.get(clf_name, clf_name.lower())
                for protocol_name in ["RTR", "TSTR"]:
                    metrics = protocols[protocol_name]
                    proto_prefix = f"{short_name}_{protocol_name.lower()}"
                    utility_row[f"{proto_prefix}_accuracy"] = metrics["Accuracy"]
                    utility_row[f"{proto_prefix}_precision"] = metrics["Precision"]
                    utility_row[f"{proto_prefix}_recall"] = metrics["Recall"]
                    utility_row[f"{proto_prefix}_f1_score"] = metrics["F1-Score"]

            utility_df = append_and_save_tracker(utility_df, utility_row, utility_tracker_path)
            print(f"Utilidade salva em: {utility_tracker_path}")

        # Fidelidade
        if run_fidelity:
            print(f"[Exp {exp_idx}] Executando avaliação de Fidelidade...")
            fidelity_res = flowmia.evaluate_fidelity(plot=plot)

            fidelity_row = base_details.copy()
            kl_values, js_values, wd_values = [], [], []
            per_col_metrics = {}

            for col_name, div_array in fidelity_res.items():
                kl, js, wd = div_array.flatten()
                kl_values.append(float(kl))
                js_values.append(float(js))
                wd_values.append(float(wd))
                per_col_metrics[f"{col_name}_kl"] = float(kl)
                per_col_metrics[f"{col_name}_js"] = float(js)
                per_col_metrics[f"{col_name}_wasserstein"] = float(wd)

            fidelity_row.update({
                "mean_kl": float(np.mean(kl_values)),
                "std_kl": float(np.std(kl_values)),
                "mean_js": float(np.mean(js_values)),
                "std_js": float(np.std(js_values)),
                "mean_wasserstein": float(np.mean(wd_values)),
                "std_wasserstein": float(np.std(wd_values)),
            })
            fidelity_row.update(per_col_metrics)

            fidelity_df = append_and_save_tracker(fidelity_df, fidelity_row, fidelity_tracker_path)
            print(f"Fidelidade salva em: {fidelity_tracker_path}")

        if remove_exps_folder and Path(exp_output_folder_base).exists():
            print("removing...", f"{exp_output_folder_base}")
            shutil.rmtree(exp_output_folder_base, ignore_errors=True)

        del flowmia
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAvaliação finalizada.")
