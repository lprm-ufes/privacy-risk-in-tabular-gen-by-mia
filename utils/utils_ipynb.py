import os
from typing import Dict, List, Optional, Tuple
from utils.utils import read_config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

DATASET_BASE_INFO_PATH = "./datasets/{0}/info.toml"
GENERATOR_ORDER = ['CTGAN', 'TVAE', 'TabDDPM', 'Tabula']
GENERATOR_COLORS = {'CTGAN': '#1f77b4', 'TVAE': '#ff7f0e', 'TabDDPM': '#2ca02c', 'Tabula': "#ff0000"}

def map_results_files(
    results_root: str,
    dataset_name: str,
    file_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = 'results'
) -> Dict[str, str]:
    """
    Mapeia a pasta 'results' e retorna os caminhos de um arquivo específico
    para cada gerador.

    Estrutura esperada:
        {results_root}/{gerador}/{dataset_name}/reference/{dataset_results}/{file_name}

    Args:
        results_root: Caminho da raiz da pasta 'results'.
        dataset_name: Nome do dataset a ser considerado.
        file_name: Nome do arquivo .csv (ex.: 'privacy_tracker.csv').
        exclude_models: Lista de geradores a serem ignorados.
        dataset_results: Nome da pasta que tem os resultados para o dataset

    Returns:
        Dicionário {nome_do_gerador: caminho_do_arquivo} apenas para os
        geradores em que o arquivo existe.
    """
    exclude = set(exclude_models or [])
    paths = {}

    for generator in sorted(os.listdir(results_root)):
        generator_dir = os.path.join(results_root, generator)
        if not os.path.isdir(generator_dir) or generator in exclude:
            continue

        file_path = os.path.join(
            generator_dir, dataset_name, "reference", dataset_results, file_name
        )
        if os.path.isfile(file_path):
            paths[generator] = file_path

    return paths


def get_all_results(
    results_root: str,
    dataset_name: str,
    file_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = 'results'
) -> pd.DataFrame:
    """
    Concatena um mesmo arquivo de resultados (.csv) de todos os geradores,
    adicionando a coluna 'generator' indicando a origem de cada linha.
    """
    paths = map_results_files(results_root, dataset_name, file_name, exclude_models, dataset_results)
    if not paths:
        raise FileNotFoundError(
            f"Nenhum arquivo '{file_name}' encontrado para o dataset "
            f"'{dataset_name}' em '{results_root}'."
        )

    dfs = []
    for generator, path in paths.items():
        df = pd.read_csv(path)
        df["generator"] = generator
        cols = ["generator"] + [c for c in df.columns if c != "generator"]
        dfs.append(df[cols])

    return pd.concat(dfs, ignore_index=True)


def get_all_privacy_results(
    results_root: str,
    dataset_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = 'results'
) -> pd.DataFrame:
    """Concatena o privacy_tracker.csv de todos os geradores."""
    return get_all_results(
        results_root, dataset_name, "privacy_tracker.csv", exclude_models, dataset_results
    )


def get_all_utility_results(
    results_root: str,
    dataset_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = 'results'
) -> pd.DataFrame:
    """Concatena o utility_tracker.csv de todos os geradores."""
    return get_all_results(
        results_root, dataset_name, "utility_tracker.csv", exclude_models, dataset_results
    )


def get_all_fidelity_results(
    results_root: str,
    dataset_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = 'results'
) -> pd.DataFrame:
    """Concatena o fidelity_tracker.csv de todos os geradores."""
    return get_all_results(
        results_root, dataset_name, "fidelity_tracker.csv", exclude_models, dataset_results
    )

def get_dataset_columns_name(dataset_name):
    dataset_info = read_config(DATASET_BASE_INFO_PATH.format(dataset_name))
    dataset_num = dataset_info['columns']['num_col']
    dataset_cat = dataset_info['columns']['cat_col']
    dataset_label = dataset_info['columns']['label']

    return dataset_num, dataset_cat, dataset_label

def _get_wasserstein_columns(dataset_columns):
    all_columns = ['mean', 'std'] + dataset_columns
    columns = [f'{col_name}_wasserstein' for col_name in all_columns]
    return columns

def _get_js_columns(dataset_columns):
    all_columns = ['mean', 'std'] + dataset_columns
    columns = [f'{col_name}_js' for col_name in all_columns]
    return columns

def _get_kl_columns(dataset_columns):
    all_columns = ['mean', 'std'] + dataset_columns
    columns = [f'{col_name}_kl' for col_name in all_columns]
    return columns

def get_expanded_fidelity_df(fidelity_df, dataset_columns):
    ws_cols = _get_wasserstein_columns(dataset_columns)
    js_cols = _get_js_columns(dataset_columns)
    kl_cols = _get_kl_columns(dataset_columns)

    common_columns = ['generator', 'exp_idx']
    generic_columns = common_columns + ['mean', 'std'] + dataset_columns

    ws_df = fidelity_df[common_columns + ws_cols].copy()
    ws_df.columns = generic_columns
    ws_df['fidelity_metric'] = 'EMD'

    js_df = fidelity_df[common_columns + js_cols].copy()
    js_df.columns = generic_columns
    js_df['fidelity_metric'] = 'JSD'

    kl_df = fidelity_df[common_columns + kl_cols].copy()
    kl_df.columns = generic_columns
    kl_df['fidelity_metric'] = 'KLD'

    columns_order = ['fidelity_metric'] + generic_columns

    full_df = pd.concat([ws_df, js_df, kl_df])[columns_order]

    return full_df

def get_expanded_utility_df(utility_df):
    utility_columns = ['rtr_accuracy','rtr_precision','rtr_recall','rtr_f1_score','tstr_accuracy','tstr_precision','tstr_recall','tstr_f1_score']

    rf_cols = [f'rf_{col}' for col in utility_columns]
    knn_cols = [f'knn_{col}' for col in utility_columns]
    mlp_cols = [f'mlp_{col}' for col in utility_columns]

    common_columns = ['generator', 'exp_idx']
    generic_columns = common_columns + utility_columns

    rf_df = utility_df[common_columns + rf_cols].copy()
    rf_df.columns = generic_columns
    rf_df['classifier'] = 'RF'

    knn_df = utility_df[common_columns + knn_cols].copy()
    knn_df.columns = generic_columns
    knn_df['classifier'] = 'KNN'

    mlp_df = utility_df[common_columns + mlp_cols].copy()
    mlp_df.columns = generic_columns
    mlp_df['classifier'] = 'MLP'

    columns_order = ['classifier'] + generic_columns
    full_df = pd.concat([rf_df, knn_df, mlp_df])[columns_order]
    
    return full_df

def get_expanded_privacy_df(privacy_df):
    privacy_columns = ['tpr@fpr','auc']

    flowmiagan_cols = [f'flowmiagan_{col}' for col in privacy_columns]
    domias_cols = [f'domias_{col}' for col in privacy_columns]
    dcr_cols = [f'dcr_{col}' for col in privacy_columns]

    common_columns = ['generator', 'exp_idx', 'fpr_fixed_rate']
    generic_columns = common_columns + privacy_columns

    flowmiagan_df = privacy_df[common_columns + flowmiagan_cols].copy()
    flowmiagan_df.columns = generic_columns
    flowmiagan_df['mia_attack'] = 'FlowMiaGan'

    domias_df = privacy_df[common_columns + domias_cols].copy()
    domias_df.columns = generic_columns
    domias_df['mia_attack'] = 'Domias'

    dcr_df = privacy_df[common_columns + dcr_cols].copy()
    dcr_df.columns = generic_columns
    dcr_df['mia_attack'] = 'DCR'

    columns_order = ['mia_attack'] + generic_columns
    full_df = pd.concat([flowmiagan_df, domias_df, dcr_df])[columns_order]
    
    return full_df

def plot_fidelity_per_column(fidelity_df, metric, col_names, dataset_name, n_parts=2, figsize=(18, 10), with_caption=True, log_scale=True):
    """Gráfico 1: barras agrupadas por coluna do dataset (uma barra por generator)."""
    sub_df = fidelity_df[fidelity_df['fidelity_metric'] == metric]
    col_list = list(col_names)

    mean_df = sub_df.groupby('generator')[col_list].mean()
    std_df = sub_df.groupby('generator')[col_list].std()

    run_counts = sub_df.groupby('generator').size()
    has_multiple_runs = (run_counts > 1).any()

    generators = [g for g in GENERATOR_ORDER if g in mean_df.index]
    generators += [g for g in mean_df.index if g not in generators]

    col_chunks = np.array_split(np.array(col_list), n_parts)
    bar_width = 0.8 / len(generators)

    fig, axes = plt.subplots(n_parts, 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    for part, (ax, cols) in enumerate(zip(axes, col_chunks), start=1):
        x = np.arange(len(cols))
        for i, gen in enumerate(generators):
            offset = (i - (len(generators) - 1) / 2) * bar_width
            values = [mean_df.loc[gen, c] if (gen in mean_df.index and c in mean_df.columns) else np.nan for c in cols]
            if has_multiple_runs:
                raw_errors = [std_df.loc[gen, c] if (gen in std_df.index and c in std_df.columns) else 0.0 for c in cols]
                raw_errors = [0.0 if (e is None or np.isnan(e)) else float(e) for e in raw_errors]

                if log_scale:
                    # Em escala logarítmica, a barra de erro inferior não pode atingir ou ultrapassar zero
                    lower_errors = []
                    upper_errors = []
                    for v, e in zip(values, raw_errors):
                        if np.isnan(v) or v <= 0:
                            lower_errors.append(0.0)
                            upper_errors.append(0.0)
                        else:
                            # Trunca o erro inferior para 90% do valor caso v - e <= 0
                            lower_errors.append(min(e, v * 0.9) if (v - e <= 0) else e)
                            upper_errors.append(e)
                    errors = [lower_errors, upper_errors]
                else:
                    errors = raw_errors
            else:
                errors = None

            ax.bar(x + offset, values, yerr=errors, capsize=3, width=bar_width,
                   label=gen, color=GENERATOR_COLORS.get(gen))
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ylabel_suffix = " (log)" if log_scale else ""
        ax.set_ylabel(f'{metric}{ylabel_suffix}')
        if log_scale:
            ax.set_yscale('log')
        if with_caption:
            sub_title_extra = f" (média ± desvio padrão entre {run_counts.max()} rodadas)" if has_multiple_runs else ""
            ax.set_title(f'{dataset_name} - {metric} por coluna{sub_title_extra} (parte {part}/{n_parts})')
        ax.grid(axis='y', linestyle='--', alpha=0.5, which='both' if log_scale else 'major')
        ax.set_axisbelow(True)
        ax.legend(title='Generator')

    fig.tight_layout()
    plt.show()


def plot_fidelity_mean(fidelity_df, metric, dataset_name, figsize=(7, 5), with_caption=True):
    """Gráfico 2: média da métrica (com desvio padrão) para cada generator."""
    sub_df = fidelity_df[fidelity_df['fidelity_metric'] == metric]

    generators = [g for g in GENERATOR_ORDER if g in sub_df['generator'].unique()]
    generators += [g for g in sub_df['generator'].unique() if g not in generators]

    run_counts = sub_df.groupby('generator').size()
    has_multiple_runs = (run_counts > 1).any()

    means = []
    stds = []
    for gen in generators:
        gen_df = sub_df[sub_df['generator'] == gen]
        if len(gen_df) > 1:
            # Múltiplas rodadas: média das médias e desvio padrão amostral entre rodadas
            means.append(float(gen_df['mean'].astype(float).mean()))
            gen_std = float(gen_df['mean'].astype(float).std(ddof=1))
            stds.append(0.0 if np.isnan(gen_std) else gen_std)
        elif len(gen_df) == 1:
            # Uma única rodada: usa a média e o std da própria rodada (entre colunas)
            means.append(float(gen_df['mean'].iloc[0]))
            stds.append(float(gen_df['std'].iloc[0]) if 'std' in gen_df.columns else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(generators, means, yerr=stds, capsize=6,
                  color=[GENERATOR_COLORS.get(g) for g in generators])
    ax.bar_label(bars, labels=[f'{m:.3f}' if not np.isnan(m) else '' for m in means], padding=3, label_type='center', color='white', fontweight='bold')
    ax.set_ylabel(f'{metric} médio')
    if with_caption:
        sub_title_extra = f"(média ± desvio padrão entre {run_counts.max()} rodadas)" if has_multiple_runs else "(± desvio padrão entre colunas)"
        ax.set_title(f'{dataset_name} - {metric} médio por generator {sub_title_extra}')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()



CLASSIFIER_ORDER = ['RF', 'KNN', 'MLP']
UTILITY_METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1-score',
}


def plot_utility(utility_df, metric, dataset_name, figsize=(10, 6), with_caption=False):
    """
    Gráfico de utilidade para uma métrica ('accuracy', 'precision', 'recall' ou 'f1_score').

    Para cada classifier há um grupo de barras (uma por generator) com o valor em TSTR.
    Sobre cada grupo é desenhada uma linha tracejada com o valor da métrica em RTR
    (baseline treinado com dados reais).
    Se houver mais de uma linha por (classifier, generator), usa a média (com barras de erro se houver repetições).
    """
    tstr_col, rtr_col = f'tstr_{metric}', f'rtr_{metric}'
    mean_df = utility_df.groupby(['classifier', 'generator'])[[tstr_col, rtr_col]].mean()
    std_df = utility_df.groupby(['classifier', 'generator'])[[tstr_col, rtr_col]].std()

    run_counts = utility_df.groupby(['classifier', 'generator']).size()
    has_multiple_runs = (run_counts > 1).any()

    present_clf = mean_df.index.get_level_values('classifier').unique()
    classifiers = [c for c in CLASSIFIER_ORDER if c in present_clf]
    classifiers += [c for c in present_clf if c not in classifiers]

    present_gen = mean_df.index.get_level_values('generator').unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gen]
    generators += [g for g in present_gen if g not in generators]

    group_width = 0.8
    bar_width = group_width / len(generators)
    x = np.arange(len(classifiers))
    metric_label = UTILITY_METRIC_LABELS.get(metric, metric)

    fig, ax = plt.subplots(figsize=figsize)

    for i, gen in enumerate(generators):
        offset = (i - (len(generators) - 1) / 2) * bar_width
        values = [mean_df[tstr_col].get((clf, gen), np.nan) for clf in classifiers]
        if has_multiple_runs:
            errors = [std_df[tstr_col].get((clf, gen), 0.0) for clf in classifiers]
            errors = [0.0 if (e is None or np.isnan(e)) else float(e) for e in errors]
        else:
            errors = None

        bars = ax.bar(x + offset, values, yerr=errors, capsize=4, width=bar_width,
                      label=f'{gen} (TSTR)', color=GENERATOR_COLORS.get(gen))
        ax.bar_label(bars, labels=[f'{v:.3f}' if not np.isnan(v) else '' for v in values], padding=2, fontsize=8)

    for j, clf in enumerate(classifiers):
        rtr_value = mean_df.loc[clf, rtr_col].mean()
        ax.hlines(rtr_value, x[j] - group_width / 2, x[j] + group_width / 2,
                  colors='black', linestyles='--', linewidth=2,
                  label='RTR' if j == 0 else None, zorder=3)
        if has_multiple_runs:
            rtr_std = std_df.loc[clf, rtr_col].mean()
            rtr_annot = f'RTR {rtr_value:.3f}' if (rtr_std is None or np.isnan(rtr_std) or rtr_std == 0) else f'RTR {rtr_value:.3f}±{rtr_std:.3f}'
        else:
            rtr_annot = f'RTR {rtr_value:.3f}'

        ax.annotate(rtr_annot, (x[j] + group_width / 2, rtr_value),
                    xytext=(3, 0), textcoords='offset points', va='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.set_xlabel('Classifier')
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.15)
    if with_caption:
        title_extra = f" (média ± desvio padrão entre {run_counts.max()} rodadas)" if has_multiple_runs else ""
        ax.set_title(f'{dataset_name} - {metric_label}: TSTR por generator vs RTR{title_extra}')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')

    fig.tight_layout()
    plt.show()


MIA_ATTACK_ORDER = ['FlowMiaGan', 'Domias', 'DCR']


def plot_privacy(privacy_df, metric, dataset_name, figsize=(10, 6)):
    """
    Gráfico de privacidade para uma métrica ('tpr@fpr' ou 'auc').

    Para cada ataque de MIA há um grupo de barras (uma por generator).
    Uma linha tracejada marca o desempenho de um atacante aleatório:
      - 'tpr@fpr': TPR = fpr_fixed_rate (um chute aleatório tem TPR = FPR);
      - 'auc': AUC = 0.5.
    Valores próximos da linha indicam ataque pouco melhor que o acaso (mais privacidade).
    Se houver mais de uma linha por (mia_attack, generator), usa a média (com barras de erro se houver repetições).
    """
    mean_df = privacy_df.groupby(['mia_attack', 'generator'])[metric].mean()
    std_df = privacy_df.groupby(['mia_attack', 'generator'])[metric].std()

    run_counts = privacy_df.groupby(['mia_attack', 'generator']).size()
    has_multiple_runs = (run_counts > 1).any()

    present_att = mean_df.index.get_level_values('mia_attack').unique()
    attacks = [a for a in MIA_ATTACK_ORDER if a in present_att]
    attacks += [a for a in present_att if a not in attacks]

    present_gen = mean_df.index.get_level_values('generator').unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gen]
    generators += [g for g in present_gen if g not in generators]

    if metric == 'tpr@fpr':
        fpr_values = privacy_df['fpr_fixed_rate'].unique()
        if len(fpr_values) > 1:
            print(f'Aviso: múltiplos fpr_fixed_rate {fpr_values}; usando a média na linha de referência.')
        baseline = float(np.mean(fpr_values))
        metric_label = f'TPR @ FPR={baseline:g}'
        baseline_label = f'Aleatório (TPR = FPR = {baseline:g})'
    elif metric == 'auc':
        baseline = 0.5
        metric_label = 'AUC'
        baseline_label = 'Aleatório (AUC = 0.5)'
    else:
        baseline, metric_label, baseline_label = None, metric, None

    group_width = 0.8
    bar_width = group_width / len(generators)
    x = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=figsize)

    all_max_candidates = []
    if baseline is not None:
        all_max_candidates.append(baseline)

    for i, gen in enumerate(generators):
        offset = (i - (len(generators) - 1) / 2) * bar_width
        values = [mean_df.get((att, gen), np.nan) for att in attacks]
        if has_multiple_runs:
            errors = [std_df.get((att, gen), 0.0) for att in attacks]
            errors = [0.0 if (e is None or np.isnan(e)) else float(e) for e in errors]
        else:
            errors = None

        bars = ax.bar(x + offset, values, yerr=errors, capsize=4, width=bar_width, label=gen,
                      color=GENERATOR_COLORS.get(gen))
        ax.bar_label(bars, labels=[f'{v:.3f}' if not np.isnan(v) else '' for v in values], padding=2, fontsize=8)

        for idx_v, v in enumerate(values):
            if not np.isnan(v):
                err = errors[idx_v] if errors else 0.0
                all_max_candidates.append(v + err)

    if baseline is not None:
        ax.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
                   label=baseline_label, zorder=3)

    max_value = np.nanmax(all_max_candidates) if all_max_candidates else 1.0
    ax.set_ylim(0, max_value * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_xlabel('Ataque de MIA')
    ax.set_ylabel(metric_label)
    title_extra = f" (média ± desvio padrão entre {run_counts.max()} rodadas)" if has_multiple_runs else ""
    ax.set_title(f'{dataset_name} - {metric_label} por ataque e generator{title_extra}')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')

    fig.tight_layout()
    plt.show()

def get_domain_shift_value(
    results_root: str,
    dataset_name: str,
    dataset_results: str,
    metric: str = "aux",
    exclude_models: Optional[List[str]] = None,
) -> float:
    """Obtém o valor escalar de domain shift (AUC ou PAD) para um determinado
    dataset de referência lendo o adversarial_shift.csv (o valor é idêntico entre
    os geradores).
    """
    df = get_all_results(
        results_root,
        dataset_name,
        "adversarial_shift.csv",
        exclude_models=exclude_models,
        dataset_results=dataset_results,
    )
    return float(df[metric].iloc[0])


def _load_mia_scores_array(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega um arquivo .npy de scores de MIA e retorna (members, non_members).
    Suporta tanto arrays 1D (primeira metade membros, segunda metade não-membros)
    quanto dicionários salvos em .npy (com chaves 'score_members' e 'score_non_members').
    """
    data = np.load(file_path, allow_pickle=True)
    if data.ndim == 0 and isinstance(data.item(), dict):
        scores_dict = data.item()
        members = np.asarray(scores_dict["score_members"], dtype=float).ravel()
        non_members = np.asarray(scores_dict["score_non_members"], dtype=float).ravel()
        return members, non_members

    arr = np.asarray(data, dtype=float).ravel()
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]


def map_mia_scores_files(
    results_root: str,
    dataset_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = "results",
) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Mapeia a pasta 'results' e retorna os caminhos dos arquivos .npy de scores
    dos ataques de MIA para cada gerador e experimento.

    Estrutura esperada:
        {results_root}/{gerador}/{dataset_name}/reference/{dataset_results}/exps/exp_{exp_idx}/scores/

    Returns:
        Dicionário {gerador: {exp_idx: {'dcr': path, 'domias': path, 'flowmiagan': path}}}
    """
    exclude = set(exclude_models or [])
    mapped: Dict[str, Dict[int, Dict[str, str]]] = {}

    for generator in sorted(os.listdir(results_root)):
        generator_dir = os.path.join(results_root, generator)
        if not os.path.isdir(generator_dir) or generator in exclude:
            continue

        exps_dir = os.path.join(
            generator_dir, dataset_name, "reference", dataset_results, "exps"
        )
        if not os.path.isdir(exps_dir):
            continue

        exp_entries = []
        for entry in os.listdir(exps_dir):
            if entry.startswith("exp_"):
                try:
                    exp_idx = int(entry.split("_", 1)[1])
                    exp_entries.append((exp_idx, os.path.join(exps_dir, entry)))
                except ValueError:
                    continue

        gen_exps: Dict[int, Dict[str, str]] = {}
        for exp_idx, exp_path in sorted(exp_entries, key=lambda x: x[0]):
            scores_dir = os.path.join(exp_path, "scores")
            if not os.path.isdir(scores_dir):
                continue

            dcr_path = os.path.join(scores_dir, f"dcr_scores_exp_{exp_idx}.npy")
            domias_path = os.path.join(scores_dir, f"domias_scores_exp_{exp_idx}.npy")
            if not os.path.isfile(domias_path):
                domias_path = os.path.join(scores_dir, f"domias_exp_{exp_idx}.npy")
            flowmiagan_path = os.path.join(scores_dir, f"flowmiagan_scores_exp_{exp_idx}.npy")

            attack_files = {}
            if os.path.isfile(dcr_path):
                attack_files["dcr"] = dcr_path
            if os.path.isfile(domias_path):
                attack_files["domias"] = domias_path
            if os.path.isfile(flowmiagan_path):
                attack_files["flowmiagan"] = flowmiagan_path

            if attack_files:
                gen_exps[exp_idx] = attack_files

        if gen_exps:
            mapped[generator] = gen_exps

    return mapped


def get_all_mia_scores_results(
    results_root: str,
    dataset_name: str,
    exclude_models: Optional[List[str]] = None,
    dataset_results: str = "results",
) -> pd.DataFrame:
    """
    Carrega e concatena os scores dos ataques de MIA (DCR, Domias e FlowMiaGan)
    de todos os geradores e experimentos em um DataFrame.

    A primeira metade dos scores corresponde aos membros (member = 1) e a
    segunda metade aos não-membros (member = 0).
    """
    mapped = map_mia_scores_files(
        results_root, dataset_name, exclude_models, dataset_results
    )
    if not mapped:
        raise FileNotFoundError(
            f"Nenhum arquivo de scores encontrado para o dataset "
            f"'{dataset_name}' em '{results_root}'."
        )

    dfs = []
    for generator, exps in mapped.items():
        for exp_idx, attack_files in exps.items():
            attack_scores = {}
            labels = None

            for attack_key, file_path in attack_files.items():
                members, non_members = _load_mia_scores_array(file_path)
                scores = np.concatenate([members, non_members])
                attack_scores[f"{attack_key}_score"] = scores

                if labels is None:
                    labels = np.concatenate([
                        np.full(len(members), 1, dtype=int),
                        np.full(len(non_members), 0, dtype=int),
                    ])

            if labels is not None:
                exp_df = pd.DataFrame({
                    "generator": generator,
                    "exp_idx": exp_idx,
                    "member": labels,
                    **attack_scores,
                })
                dfs.append(exp_df)

    return pd.concat(dfs, ignore_index=True)


def get_expanded_mia_scores_df(mia_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande o DataFrame retornado por `get_all_mia_scores_results` para o formato longo,
    com as colunas ['mia_attack', 'generator', 'exp_idx', 'member', 'score'].
    """
    common_columns = ["generator", "exp_idx", "member"]
    generic_columns = common_columns + ["score"]

    attack_mapping = [
        ("flowmiagan_score", "FlowMiaGan"),
        ("domias_score", "Domias"),
        ("dcr_score", "DCR"),
    ]

    dfs = []
    for col_name, attack_label in attack_mapping:
        if col_name in mia_scores_df.columns:
            sub_df = mia_scores_df[common_columns + [col_name]].copy()
            sub_df.columns = generic_columns
            sub_df["mia_attack"] = attack_label
            dfs.append(sub_df[["mia_attack"] + generic_columns])

    return pd.concat(dfs, ignore_index=True)


def plot_mia_roc_curve(
    mia_scores_df: pd.DataFrame,
    dataset_name: str,
    attacks: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    n_points: int = 200,
    fpr_max: Optional[float] = None,
    log_scale: bool = False,
    fpr_min: float = 1e-3,
    fpr_focus: Optional[List[float]] = None,
    separate_plots: bool = False,
    save_dir: Optional[str] = None,
    save_format: str = "png",
    dpi: int = 300,
    show: bool = True,
    with_caption = False
):
    """
    Plota a curva ROC (TPR vs FPR) para ataques de MIA (DCR, Domias, FlowMiaGan).
    Em cada gráfico há uma curva por gerador correspondente à média entre as
    rodadas (experimentos), acompanhada da sombra de ±1 desvio padrão.

    Configurado para monografia e artigos:
      - `separate_plots=True`: gera cada ataque em uma figura independente, com proporções
        ideais para inclusão direta em páginas A4 / documentos LaTeX.
      - Ticks e grade detalhados no eixo X (passos de 0.1 e minor ticks) para facilitar
        a visualização precisa dos valores de FPR.
      - `fpr_max`: permite zoom na região de baixo FPR (ex.: 0.1 ou 0.2), essencial para MIA.
      - `log_scale=True`: coloca o eixo X em escala logarítmica (padrão em benchmarks de MIA).
      - `fpr_focus`: desenha linhas verticais de referência em taxas fixas (ex.: [0.01, 0.05, 0.1]).
      - `save_dir`: salva automaticamente as figuras com resolução configurável (`dpi=300`).

    Args:
        mia_scores_df: DataFrame retornado por `get_all_mia_scores_results` ou `get_expanded_mia_scores_df`.
        dataset_name: Nome do dataset para títulos e nomes de arquivo.
        attacks: Lista opcional de ataques a plotar (ex.: ['FlowMiaGan', 'Domias']). Padrão: todos.
        figsize: Dimensões da figura. Se None, usa (6, 5) para plots separados ou (18, 5) para subplots.
        n_points: Número de pontos para interpolação da curva média.
        fpr_max: Limite máximo do eixo X para zoom em baixo FPR (ex.: 0.1 ou 0.2).
        log_scale: Se True, plota o eixo X em escala logarítmica.
        fpr_min: Limite inferior do eixo X quando `log_scale=True` (padrão: 1e-3).
        fpr_focus: Lista opcional de FPRs para destacar com linhas verticais (ex.: [0.01, 0.05, 0.1]).
        separate_plots: Se True, cria uma figura separada para cada ataque (ideal para monografia).
        save_dir: Pasta onde salvar os gráficos. Se None, não salva em disco.
        save_format: Formato de salvamento ('png', 'pdf', 'svg', etc.).
        dpi: Resolução da imagem exportada (padrão: 300).
        show: Se True, chama `plt.show()`.

    Returns:
        Lista de tuplas [(fig, ax), ...] se `separate_plots=True`, ou (fig, axes) se `separate_plots=False`.
    """
    if "mia_attack" not in mia_scores_df.columns:
        df = get_expanded_mia_scores_df(mia_scores_df)
    else:
        df = mia_scores_df

    present_att = df["mia_attack"].unique()
    if attacks is None:
        target_attacks = [a for a in MIA_ATTACK_ORDER if a in present_att]
        target_attacks += [a for a in present_att if a not in target_attacks]
    else:
        target_attacks = [a for a in attacks if a in present_att]
        if not target_attacks:
            target_attacks = list(attacks)

    present_gen = df["generator"].unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gen]
    generators += [g for g in present_gen if g not in generators]

    max_runs = df.groupby(["mia_attack", "generator"])["exp_idx"].nunique().max()
    has_multiple_runs = max_runs > 1
    title_extra = (
        f" (média ± desvio padrão entre {max_runs} rodadas)"
        if has_multiple_runs
        else ""
    )

    if log_scale:
        x_min = max(fpr_min, 1e-6)
        x_max = fpr_max if fpr_max is not None else 1.0
        mean_fpr = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    else:
        x_min = 0.0
        x_max = fpr_max if fpr_max is not None else 1.0
        mean_fpr = np.linspace(0.0, 1.0, n_points)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def _plot_curves_on_axis(ax, att_name):
        att_df = df[df["mia_attack"] == att_name]

        for gen in generators:
            gen_df = att_df[att_df["generator"] == gen]
            if gen_df.empty:
                continue

            tprs = []
            aucs = []

            for _, exp_df in gen_df.groupby("exp_idx"):
                y_true = (exp_df["member"] == 1).astype(int).to_numpy()
                y_score = exp_df["score"].astype(float).to_numpy()

                valid_mask = ~np.isnan(y_score)
                y_true = y_true[valid_mask]
                y_score = y_score[valid_mask]

                if len(np.unique(y_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = roc_auc_score(y_true, y_score)

                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                if not log_scale:
                    interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc_val)

            if not tprs:
                continue

            tprs_arr = np.array(tprs)
            mean_tpr = np.mean(tprs_arr, axis=0)
            if not log_scale and x_max >= 1.0:
                mean_tpr[-1] = 1.0
            mean_auc = float(np.mean(aucs))
            color = GENERATOR_COLORS.get(gen)

            if len(tprs) > 1:
                std_tpr = np.std(tprs_arr, axis=0, ddof=1)
                std_auc = float(np.std(aucs, ddof=1))
                tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
                tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
                ax.fill_between(
                    mean_fpr,
                    tpr_lower,
                    tpr_upper,
                    color=color,
                    alpha=0.2,
                )
                label = f"{gen} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
            else:
                label = f"{gen} (AUC = {mean_auc:.3f})"

            ax.plot(
                mean_fpr,
                mean_tpr,
                color=color,
                linewidth=2,
                label=label,
            )

    def _configure_axis(ax, is_first_col: bool = True):
        # Linha de referência de atacante aleatório
        if log_scale:
            ax.plot(
                [x_min, x_max],
                [x_min, x_max],
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Aleatório (AUC = 0.5)",
                zorder=2,
            )
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
        else:
            ax.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Aleatório (AUC = 0.5)",
                zorder=2,
            )
            ax.set_xlim(0.0, x_max)

            # Ticks detalhados e configuráveis para facilitar a leitura dos valores de FPR
            if x_max <= 0.05:
                ax.set_xticks(np.arange(0.0, x_max + 0.005, 0.01))
            elif x_max <= 0.1:
                ax.set_xticks(np.arange(0.0, x_max + 0.01, 0.02))
            elif x_max <= 0.25:
                ax.set_xticks(np.arange(0.0, x_max + 0.02, 0.05))
            elif x_max <= 0.5:
                ax.set_xticks(np.arange(0.0, x_max + 0.05, 0.1))
            else:
                ax.set_xticks(np.arange(0.0, 1.05, 0.1))

            ax.minorticks_on()
            ax.grid(True, which="major", linestyle="--", alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", alpha=0.25)

        # Linhas de referência verticais nos FPRs fixos de interesse
        if fpr_focus:
            for fp in fpr_focus:
                if x_min <= fp <= x_max:
                    ax.axvline(
                        fp,
                        color="gray",
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.7,
                        zorder=2,
                    )

        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("FPR (False Positive Rate)", fontsize=10)
        if is_first_col:
            ax.set_ylabel("TPR (True Positive Rate)", fontsize=10)

        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    if separate_plots:
        results = []
        fig_size = figsize or (6, 5)

        for att in target_attacks:
            fig, ax = plt.subplots(figsize=fig_size)
            _plot_curves_on_axis(ax, att)
            _configure_axis(ax, is_first_col=True)
            if with_caption:
                ax.set_title(f"{dataset_name} - Curva ROC: {att}{title_extra}", fontsize=11, pad=10)
            fig.tight_layout()

            if save_dir is not None:
                safe_att = att.lower().replace(" ", "_")
                fname = f"{dataset_name}_roc_{safe_att}.{save_format}"
                out_path = os.path.join(save_dir, fname)
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

            if show:
                plt.show()

            results.append((fig, ax))

        return results
    else:
        fig_size = figsize or (18, 5)
        fig, axes = plt.subplots(1, len(target_attacks), figsize=fig_size, sharey=True)
        axes = np.atleast_1d(axes)

        for idx, (ax, att) in enumerate(zip(axes, target_attacks)):
            _plot_curves_on_axis(ax, att)
            _configure_axis(ax, is_first_col=(idx == 0))
            ax.set_title(f"Ataque: {att}", fontsize=11, pad=10)

        fig.suptitle(
            f"{dataset_name} - Curva ROC (AUC) por ataque e generator{title_extra}",
            fontsize=13,
            y=1.02,
        )
        fig.tight_layout()

        if save_dir is not None:
            fname = f"{dataset_name}_roc_combined.{save_format}"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes


# ==============================================================================
# Funções para Geração de Tabelas Resumo (Fidelidade, Utilidade e Privacidade)
# ==============================================================================

def get_fidelity_table(
    fidelity_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    dataset_col_names: Optional[List[str]] = None,
    decimals: int = 3,
    formatted: bool = True,
) -> pd.DataFrame:
    """
    Constrói um DataFrame pandas com o resumo das métricas de fidelidade
    (EMD, JSD, etc.) por gerador (média ± desvio padrão entre as rodadas).

    Args:
        fidelity_df: DataFrame retornado por get_all_fidelity_results ou get_expanded_fidelity_df.
        metrics: Lista de métricas a incluir (padrão: ['EMD', 'JSD']).
        dataset_col_names: Lista das colunas do dataset (usado se fidelity_df não estiver expandido).
        decimals: Número de casas decimais para arredondamento (padrão: 3).
        formatted: Se True, retorna coluna única por métrica com string "média ± desvio".
                   Se False, retorna colunas numéricas separadas para média e desvio padrão.

    Returns:
        pd.DataFrame indexado por gerador com as métricas de fidelidade.
    """
    if metrics is None:
        metrics = ['EMD', 'JSD']

    if 'fidelity_metric' not in fidelity_df.columns:
        if dataset_col_names is not None:
            df = get_expanded_fidelity_df(fidelity_df, dataset_col_names)
        else:
            df = None
    else:
        df = fidelity_df

    present_gens = fidelity_df['generator'].unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gens]
    generators += [g for g in present_gens if g not in generators]

    rows = []
    for gen in generators:
        row = {'Gerador': gen}
        for metric in metrics:
            if df is not None:
                sub = df[(df['generator'] == gen) & (df['fidelity_metric'] == metric)]
                if len(sub) == 0:
                    mean_val, std_val = np.nan, np.nan
                elif len(sub) > 1:
                    mean_val = float(sub['mean'].astype(float).mean())
                    std_val = float(sub['mean'].astype(float).std(ddof=1))
                    if np.isnan(std_val):
                        std_val = 0.0
                else:
                    mean_val = float(sub['mean'].iloc[0])
                    std_val = float(sub['std'].iloc[0]) if 'std' in sub.columns else 0.0
            else:
                sub = fidelity_df[fidelity_df['generator'] == gen]
                metric_col = 'mean_wasserstein' if metric == 'EMD' else ('mean_js' if metric == 'JSD' else f'mean_{metric.lower()}')
                if metric_col in sub.columns and len(sub) > 0:
                    mean_val = float(sub[metric_col].astype(float).mean())
                    std_val = float(sub[metric_col].astype(float).std(ddof=1)) if len(sub) > 1 else 0.0
                    if np.isnan(std_val):
                        std_val = 0.0
                else:
                    mean_val, std_val = np.nan, np.nan

            if formatted:
                if np.isnan(mean_val):
                    row[metric] = '-'
                else:
                    row[metric] = f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
            else:
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val

        rows.append(row)

    table_df = pd.DataFrame(rows).set_index('Gerador')
    return table_df


def get_utility_table(
    utility_df: pd.DataFrame,
    metric: str = 'f1_score',
    classifiers: Optional[List[str]] = None,
    include_rtr_baseline: bool = True,
    decimals: int = 3,
    formatted: bool = True,
) -> pd.DataFrame:
    """
    Constrói um DataFrame pandas com o resumo das métricas de utilidade preditiva
    (TSTR vs RTR) por gerador e classificador (média ± desvio padrão entre as rodadas).

    Args:
        utility_df: DataFrame retornado por get_all_utility_results ou get_expanded_utility_df.
        metric: Métrica de utilidade ('f1_score', 'accuracy', 'precision', 'recall'),
                ou lista de métricas (ex.: ['f1_score', 'accuracy']). Padrão: 'f1_score'.
        classifiers: Lista de classificadores a incluir (padrão: ['RF', 'KNN', 'MLP']).
        include_rtr_baseline: Se True, inclui a linha 'Baseline (RTR)' com o desempenho real.
        decimals: Número de casas decimais para formatação (padrão: 3).
        formatted: Se True, retorna strings "média ± desvio". Se False, retorna valores numéricos.

    Returns:
        pd.DataFrame indexado por gerador (com opcional Baseline RTR na primeira linha)
        e colunas correspondentes aos classificadores.
    """
    if 'classifier' not in utility_df.columns:
        df = get_expanded_utility_df(utility_df)
    else:
        df = utility_df

    if classifiers is None:
        present_clf = df['classifier'].unique()
        classifiers = [c for c in CLASSIFIER_ORDER if c in present_clf]
        classifiers += [c for c in present_clf if c not in classifiers]

    present_gens = df['generator'].unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gens]
    generators += [g for g in present_gens if g not in generators]

    metrics = [metric] if isinstance(metric, str) else list(metric)

    rows = []
    row_labels = []

    # 1. Linha do Baseline RTR (treinado com dados reais)
    if include_rtr_baseline:
        row = {}
        for clf in classifiers:
            clf_sub = df[df['classifier'] == clf]
            for m in metrics:
                col_name = f'rtr_{m}'
                if col_name in clf_sub.columns and len(clf_sub) > 0:
                    rtr_mean = float(clf_sub[col_name].astype(float).mean())
                    rtr_std = float(clf_sub[col_name].astype(float).std(ddof=1)) if len(clf_sub) > 1 else 0.0
                    if np.isnan(rtr_std):
                        rtr_std = 0.0
                else:
                    rtr_mean, rtr_std = np.nan, np.nan

                key = (clf, m) if len(metrics) > 1 else clf
                if formatted:
                    if np.isnan(rtr_mean):
                        row[key] = '-'
                    elif rtr_std == 0.0:
                        row[key] = f"{rtr_mean:.{decimals}f}"
                    else:
                        row[key] = f"{rtr_mean:.{decimals}f} ± {rtr_std:.{decimals}f}"
                else:
                    row[(key, 'mean') if isinstance(key, tuple) else f"{key}_mean"] = rtr_mean
                    row[(key, 'std') if isinstance(key, tuple) else f"{key}_std"] = rtr_std

        rows.append(row)
        row_labels.append('Baseline (RTR)')

    # 2. Linhas dos Geradores (cenário TSTR)
    for gen in generators:
        row = {}
        for clf in classifiers:
            sub = df[(df['generator'] == gen) & (df['classifier'] == clf)]
            for m in metrics:
                col_name = f'tstr_{m}'
                if col_name in sub.columns and len(sub) > 0:
                    mean_val = float(sub[col_name].astype(float).mean())
                    std_val = float(sub[col_name].astype(float).std(ddof=1)) if len(sub) > 1 else 0.0
                    if np.isnan(std_val):
                        std_val = 0.0
                else:
                    mean_val, std_val = np.nan, np.nan

                key = (clf, m) if len(metrics) > 1 else clf
                if formatted:
                    if np.isnan(mean_val):
                        row[key] = '-'
                    else:
                        row[key] = f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
                else:
                    row[(key, 'mean') if isinstance(key, tuple) else f"{key}_mean"] = mean_val
                    row[(key, 'std') if isinstance(key, tuple) else f"{key}_std"] = std_val

        rows.append(row)
        row_labels.append(gen)

    table_df = pd.DataFrame(rows, index=row_labels)
    table_df.index.name = 'Modelo'
    if len(metrics) > 1:
        table_df.columns = pd.MultiIndex.from_tuples(table_df.columns, names=['Classificador', 'Métrica'])
    return table_df


def get_privacy_table(
    privacy_df: pd.DataFrame,
    attacks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    decimals: int = 3,
    formatted: bool = True,
) -> pd.DataFrame:
    """
    Constrói um DataFrame pandas com o resumo das métricas de privacidade
    (AUC-ROC e TPR@FPR) para cada ataque de MIA e gerador (média ± desvio padrão entre as rodadas).

    Args:
        privacy_df: DataFrame retornado por get_all_privacy_results ou get_expanded_privacy_df.
        attacks: Lista de ataques a incluir (padrão: ['FlowMiaGan', 'Domias', 'DCR']).
        metrics: Lista de métricas a incluir (padrão: ['auc', 'tpr@fpr']).
        decimals: Número de casas decimais para formatação (padrão: 3).
        formatted: Se True, retorna strings "média ± desvio". Se False, retorna valores numéricos.

    Returns:
        pd.DataFrame indexado por gerador com colunas em MultiIndex (Ataque, Métrica).
    """
    if 'mia_attack' not in privacy_df.columns:
        df = get_expanded_privacy_df(privacy_df)
    else:
        df = privacy_df

    if attacks is None:
        present_att = df['mia_attack'].unique()
        attacks = [a for a in MIA_ATTACK_ORDER if a in present_att]
        attacks += [a for a in present_att if a not in attacks]

    if metrics is None:
        metrics = ['auc', 'tpr@fpr']

    present_gens = df['generator'].unique()
    generators = [g for g in GENERATOR_ORDER if g in present_gens]
    generators += [g for g in present_gens if g not in generators]

    metric_labels = {'auc': 'AUC-ROC', 'tpr@fpr': 'TPR@FPR=0.10'}

    rows = []
    for gen in generators:
        row = {}
        for att in attacks:
            for m in metrics:
                sub = df[(df['generator'] == gen) & (df['mia_attack'] == att)]
                if len(sub) > 0 and m in sub.columns:
                    mean_val = float(sub[m].astype(float).mean())
                    std_val = float(sub[m].astype(float).std(ddof=1)) if len(sub) > 1 else 0.0
                    if np.isnan(std_val):
                        std_val = 0.0
                else:
                    mean_val, std_val = np.nan, np.nan

                m_label = metric_labels.get(m, m)
                key = (att, m_label)
                if formatted:
                    if np.isnan(mean_val):
                        row[key] = '-'
                    else:
                        row[key] = f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
                else:
                    row[(att, m_label, 'mean')] = mean_val
                    row[(att, m_label, 'std')] = std_val

        rows.append(row)

    table_df = pd.DataFrame(rows, index=generators)
    table_df.index.name = 'Gerador'
    table_df.columns = pd.MultiIndex.from_tuples(table_df.columns, names=['Ataque', 'Métrica'])
    return table_df


# ==============================================================================
# Funções para Análise de Domain Shift (Experimento 2)
# ==============================================================================

GENERATOR_MARKERS = {'CTGAN': 'o', 'TVAE': 's', 'TabDDPM': '^', 'Tabula': 'D'}


def plot_domain_shift_evolution(
    data_by_shift: Dict[float, pd.DataFrame],
    shift_labels: Optional[Dict[float, str]] = None,
    metric: str = "auc",
    attacks: Optional[List[str]] = None,
    generators: Optional[List[str]] = None,
    separate_plots: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = (0.35, 0.82),
    with_caption: bool = False,
    save_dir: Optional[str] = None,
    save_prefix: str = "alabama_shift_evol",
    save_format: str = "png",
    dpi: int = 300,
    show: bool = True,
):
    """
    Plota a evolução da métrica de privacidade ('auc' ou 'tpr@fpr') em função do
    Domain Shift (Adversarial Validation ROC-AUC), calculando corretamente a
    média ± desvio padrão entre as sementes (exp_idx).

    Args:
        data_by_shift: Dicionário {valor_shift: df_privacidade_expandido}.
        shift_labels: Dicionário opcional {valor_shift: nome_estado} (ex.: 'Alabama', 'Geórgia', 'Colorado').
        metric: 'auc' (padrão) ou 'tpr@fpr'.
        attacks: Lista de ataques a plotar (padrão: ['FlowMiaGan', 'Domias', 'DCR']).
        generators: Lista de geradores (padrão: ordem padrão presente nos dados).
        separate_plots: Se True, gera uma figura independente por ataque (ideal para subfigure no LaTeX).
        figsize: Dimensões da figura ((6, 5) para separate_plots=True ou (16, 4.8) para False).
        ylim: Limites do eixo Y.
        with_caption: Se True, exibe título no gráfico.
        save_dir: Pasta opcional para salvar as figuras.
    """
    x_shifts = sorted(data_by_shift.keys())
    sample_df = data_by_shift[x_shifts[0]]
    if "mia_attack" not in sample_df.columns:
        data_by_shift = {k: get_expanded_privacy_df(v) for k, v in data_by_shift.items()}
        sample_df = data_by_shift[x_shifts[0]]

    if attacks is None:
        present_att = sample_df["mia_attack"].unique()
        target_attacks = [a for a in MIA_ATTACK_ORDER if a in present_att]
        target_attacks += [a for a in present_att if a not in target_attacks]
    else:
        target_attacks = list(attacks)

    if generators is None:
        present_gen = sample_df["generator"].unique()
        target_generators = [g for g in GENERATOR_ORDER if g in present_gen]
        target_generators += [g for g in present_gen if g not in target_generators]
    else:
        target_generators = list(generators)

    default_names = ["Alabama", "Geórgia", "Colorado"]
    xtick_labels = []
    for i, x in enumerate(x_shifts):
        if shift_labels and x in shift_labels:
            lbl = shift_labels[x]
        elif i < len(default_names):
            lbl = default_names[i]
        else:
            lbl = ""
        xtick_labels.append(f"{x:.2f}\n({lbl})" if lbl else f"{x:.2f}")

    baseline_val = 0.50 if metric == "auc" else 0.10
    baseline_label = f"Aleatório ({baseline_val:.2f})"
    ylabel = "AUC-ROC do Ataque" if metric == "auc" else "TPR@FPR = 0.10"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def _plot_on_ax(ax, att_name, is_first_col: bool = True):
        ax.axhline(
            baseline_val,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=baseline_label,
            zorder=2,
        )

        for gen in target_generators:
            means = []
            stds = []
            for x in x_shifts:
                sub = data_by_shift[x]
                vals = sub[(sub["mia_attack"] == att_name) & (sub["generator"] == gen)][metric].astype(float)
                if len(vals) > 0:
                    m_val = float(vals.mean())
                    s_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    if np.isnan(s_val):
                        s_val = 0.0
                else:
                    m_val, s_val = np.nan, 0.0
                means.append(m_val)
                stds.append(s_val)

            means_arr = np.array(means)
            stds_arr = np.array(stds)
            color = GENERATOR_COLORS.get(gen)
            marker = GENERATOR_MARKERS.get(gen, "o")

            ax.plot(
                x_shifts,
                means_arr,
                label=gen,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
                zorder=3,
            )
            if np.any(stds_arr > 0):
                ax.fill_between(
                    x_shifts,
                    np.maximum(0.0, means_arr - stds_arr),
                    np.minimum(1.0, means_arr + stds_arr),
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

        ax.set_xlabel("Deslocamento de Distribuição (AUC-ROC)", fontsize=10)
        if is_first_col:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_shifts)
        ax.set_xticklabels(xtick_labels, fontsize=9.5)
        ax.set_xlim(min(x_shifts) - 0.03, max(x_shifts) + 0.03)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if separate_plots:
        results = []
        fig_size = figsize or (6, 5)
        for att in target_attacks:
            fig, ax = plt.subplots(figsize=fig_size)
            _plot_on_ax(ax, att, is_first_col=True)
            if with_caption:
                ax.set_title(f"Evolução com Domain Shift - Ataque: {att}", fontsize=11, pad=10)
            fig.tight_layout()
            if save_dir is not None:
                safe_att = att.lower().replace(" ", "_")
                out_path = os.path.join(save_dir, f"{save_prefix}_{safe_att}.{save_format}")
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            results.append((fig, ax))
        return results
    else:
        fig_size = figsize or (16, 4.8)
        fig, axes = plt.subplots(1, len(target_attacks), figsize=fig_size, sharey=True)
        axes = np.atleast_1d(axes)
        for idx, (ax, att) in enumerate(zip(axes, target_attacks)):
            _plot_on_ax(ax, att, is_first_col=(idx == 0))
            ax.set_title(f"Ataque: {att}", fontsize=11, pad=10)
        if with_caption:
            fig.suptitle(
                "Sensibilidade dos Modelos Generativos ao Domain Shift por Ataque de MIA",
                fontsize=13,
                y=1.02,
            )
        fig.tight_layout()
        if save_dir is not None:
            out_path = os.path.join(save_dir, f"{save_prefix}_combined.{save_format}")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes