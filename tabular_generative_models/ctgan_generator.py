from typing import Any, Dict
from synthcity.plugins import Plugins

from tabular_generative_models.synthcity_base import SynthcityGeneratorBase


class CTGANGenerator(SynthcityGeneratorBase):
    """
    Gerador tabular baseado no modelo CTGAN (via Synthcity).
    """

    def _fit(self, params: Dict[str, Any], seed: int) -> None:
        default_params = dict(
            n_iter=2000,
            lr=1e-3,
            weight_decay=1e-3,
            batch_size=500,
            random_state=seed,
            generator_n_layers_hidden=2,
            generator_n_units_hidden=500,
            generator_nonlin="relu",
            discriminator_n_layers_hidden=2,
            discriminator_n_units_hidden=500,
            discriminator_nonlin="leaky_relu",
            patience = 5
        )
        default_params.update(params)

        self.plugin = Plugins().get("ctgan", **default_params)
        self.plugin.fit(self.loader)