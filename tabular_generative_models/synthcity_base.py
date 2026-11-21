from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import synthcity.utils.dataframe as sc_dataframe
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file

from tabular_generative_models.tabular_generator_base import TabularGeneratorBase

# Apenas colunas com tipo inteiro podem ser tratadas como categóricas. Evita o erro
# que colunas contínuas possam ser classificadas como categóricas causando
# erro no CUDA.
def _integer_discrete_columns(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False
) -> list:
    return [
        (col, cnt) if return_counts else col
        for col, vals in dataframe.items()
        for cnt in [vals.nunique()]
        if pd.api.types.is_integer_dtype(vals.dtype) and cnt <= max_classes
    ]

sc_dataframe.discrete_columns = _integer_discrete_columns


class SynthcityGeneratorBase(TabularGeneratorBase):
    """
    Classe base para modelos geradores suportados pela biblioteca Synthcity.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        target_col: Optional[str] = None,
    ):
        super().__init__(data, num_cols, cat_cols, target_col)
        self.plugin = None
        self.loader = self._prepare_dataloader()

    def _prepare_dataloader(self) -> GenericDataLoader:
        df = self.data.copy()
        if self.num_cols:
            df[self.num_cols] = df[self.num_cols].apply(pd.to_numeric)
        for col in self.cat_cols:
            df[col] = df[col].astype("category")

        return GenericDataLoader(
            df,
            target_column=self.target_col,
            sensitive_columns=[],
            discrete_features=self.cat_cols,
        )

    def _generate(self, count: int, seed: int) -> pd.DataFrame:
        if self.plugin is None:
            raise RuntimeError("O modelo precisa ser treinado antes da geração.")
        synthetic_result = self.plugin.generate(count, random_state=seed)
        return synthetic_result.dataframe()

    def save(self, path: Path | str) -> None:
        if self.plugin is None:
            raise RuntimeError("O modelo precisa ser treinado antes de ser salvo.")
        save_to_file(path, self.plugin)

