from typing import Any, Dict
from synthcity.plugins import Plugins
import synthcity.plugins.core.models.tabular_ddpm as tabular_ddpm
from synthcity.plugins.core.models.tabular_ddpm.gaussian_multinomial_diffsuion import (
    GaussianMultinomialDiffusion,
)

from tabular_generative_models.synthcity_base import SynthcityGeneratorBase

# Clipa o valor de x0 para impedir que os altos erros da rede nos 
# passos iniciais explodam (problema que ocorre na implementação do synthcity) 
# os valores e prejudiquem o restante da geração dos dados.
DDPM_X0_CLIP = 5.2
_orig_gaussian_p_mean_variance = GaussianMultinomialDiffusion.gaussian_p_mean_variance

def _clipped_gaussian_p_mean_variance(self, model_output, x, t, model_kwargs=None):
    out = _orig_gaussian_p_mean_variance(self, model_output, x, t, model_kwargs)
    if not self.training:
        x0 = out["pred_xstart"].clamp(-DDPM_X0_CLIP, DDPM_X0_CLIP)
        out["mean"], _, _ = self.gaussian_q_posterior_mean_variance(x_start=x0, x_t=x, t=t)
        out["pred_xstart"] = x0
    return out

GaussianMultinomialDiffusion.gaussian_p_mean_variance = _clipped_gaussian_p_mean_variance

class TabDDPMGenerator(SynthcityGeneratorBase):
    """
    Gerador tabular baseado no modelo TabDDPM (via Synthcity).
    """

    def _fit(self, params: Dict[str, Any], seed: int) -> None:
        default_params = dict(
            is_classification=True,
            n_iter=1000,
            lr=1e-4,
            weight_decay=1e-4,
            batch_size=500,
            random_state=seed,
            num_timesteps=1000,
            dim_embed=128,
            log_interval=100,
        )
        default_params.update(params)

        self.plugin = Plugins().get("ddpm", **default_params)
        self.plugin.fit(self.loader)