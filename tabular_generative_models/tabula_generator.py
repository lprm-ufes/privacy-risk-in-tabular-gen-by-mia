import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import torch
import numpy as np

from tabular_generative_models.tabular_generator_base import TabularGeneratorBase
from tabular_generative_models.tabula.tabula import Tabula


class TabulaGenerator(TabularGeneratorBase):
    """
    Gerador tabular baseado no modelo Tabula (Language Model para dados tabulares).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        num_cols: list,
        cat_cols: list,
        target_col: str = None,
    ):
        super().__init__(data, num_cols, cat_cols, target_col)
        self.file_extension = ""
        self.model = None

    def _fit(self, params: Dict[str, Any], seed: int) -> None:
        # Configurar seeds do PyTorch e NumPy
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model_params = params.copy()
        llm = model_params.pop("llm", "distilgpt2")
        epochs = model_params.pop("epochs", 100)
        batch_size = model_params.pop("batch_size", 32)
        custom_exp_dir = model_params.pop("experiment_dir", None)

        is_temp_dir = False
        if custom_exp_dir:
            exp_dir = str(custom_exp_dir)
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
        else:
            exp_dir = tempfile.mkdtemp(prefix="tabula_train_")
            is_temp_dir = True

        try:
            self.model = Tabula(
                llm=llm,
                experiment_dir=exp_dir,
                epochs=epochs,
                batch_size=batch_size,
                categorical_columns=self.cat_cols,
                **model_params,
            )
            self.model.fit(self.data)
        finally:
            if is_temp_dir and Path(exp_dir).exists():
                shutil.rmtree(exp_dir, ignore_errors=True)

    def _generate(self, count: int, seed: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("O modelo Tabula precisa ser treinado antes da geração.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Calcula automaticamente o max_length baseado nas linhas reais do dataset
        sample_rows = self.data.head(min(50, len(self.data)))
        row_lengths = []
        for _, row in sample_rows.iterrows():
            text = ", ".join(f"{col} {str(val).strip()}" for col, val in row.items())
            tokens = self.model.tokenizer(text)["input_ids"]
            row_lengths.append(len(tokens))

        base_length = max(row_lengths) if row_lengths else len(self.data.columns) * 15
        # Margem de segurança de 30% + 20 tokens para variações de valores durante amostragem
        calculated_max_length = int(base_length * 1.3) + 20

        # Respeita o limite máximo de contexto do modelo se definido
        model_limit = getattr(self.model.tokenizer, "model_max_length", 1024)
        if model_limit and model_limit < 100000:
            calculated_max_length = min(calculated_max_length, model_limit)

        print(f"[Tabula] max_length calculado automaticamente: {calculated_max_length} tokens.")
        df_synthetic = self.model.sample(
            n_samples=count,
            device=device,
            k=10,
            max_length=calculated_max_length,
        )
        return df_synthetic

    def save(self, path: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("O modelo precisa ser treinado antes de ser salvo.")
        self.model.save(str(path))

