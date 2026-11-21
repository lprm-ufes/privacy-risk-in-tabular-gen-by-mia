from typing import Any, Dict
from synthcity.plugins import Plugins

from tabular_generative_models.synthcity_base import SynthcityGeneratorBase


class TVAEGenerator(SynthcityGeneratorBase):
    """
    Gerador tabular baseado no modelo TVAE (via Synthcity).
    """

    def _fit(self, params: Dict[str, Any], seed: int) -> None:
        default_params = dict(
            n_iter=500,
            lr=2e-4,
            weight_decay=1e-3,
            batch_size=64,
            random_state=seed,
        )
        default_params.update(params)

        self.plugin = Plugins().get("tvae", **default_params)
        self.plugin.fit(self.loader)