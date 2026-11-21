from abc import ABC, abstractmethod
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class TabularGeneratorBase(ABC):
    """
    Classe base abstrata para modelos geradores de dados sintéticos tabulares.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        target_col: Optional[str] = None,
    ):
        """
        Inicializa o gerador tabular com os dados e metadados comuns do dataset.

        Args:
            data: DataFrame contendo os dados reais de treino.
            num_cols: Lista de nomes das colunas numéricas.
            cat_cols: Lista de nomes das colunas categóricas.
            target_col: Nome da coluna alvo/label (opcional).
        """
        self.data = data.copy()
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.target_col = target_col
        self.file_extension: str = ".pkl"

    @abstractmethod
    def _fit(self, params: Dict[str, Any], seed: int) -> None:
        """
        Método interno para treinar o modelo com os hiperparâmetros fornecidos.

        Args:
            params: Dicionário contendo os hiperparâmetros específicos do modelo.
            seed: Semente aleatória para reprodutibilidade.
        """
        pass

    @abstractmethod
    def _generate(self, count: int, seed: int) -> pd.DataFrame:
        """
        Método interno para gerar amostras sintéticas.

        Args:
            count: Quantidade de amostras a serem geradas.
            seed: Semente aleatória para amostragem.

        Returns:
            DataFrame pandas com os dados gerados.
        """
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """
        Persiste o modelo treinado no caminho informado.

        Args:
            path: Caminho (arquivo ou diretório) para salvar o modelo.
        """
        pass

    def fit_and_generate(
        self,
        params: Dict[str, Any],
        count: int,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Orquestra o treinamento e a geração de dados sintéticos, cronometrando
        os tempos de execução.

        Args:
            params: Dicionário de hiperparâmetros para o treino.
            count: Número de linhas sintéticas a serem geradas.
            seed: Semente aleatória utilizada.

        Returns:
            Tuple contendo:
                - df_synthetic: pandas DataFrame com os dados gerados.
                - train_time: Tempo de execução do treinamento (em segundos).
                - generate_time: Tempo de execução da geração (em segundos).
        """
        start_time = time.time()
        self._fit(params, seed)
        train_time = time.time() - start_time

        start_time = time.time()
        df_synthetic = self._generate(count, seed)
        generate_time = time.time() - start_time

        return df_synthetic, train_time, generate_time
