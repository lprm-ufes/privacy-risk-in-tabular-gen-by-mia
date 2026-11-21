from tabular_generative_models.tabular_generator_base import TabularGeneratorBase
from tabular_generative_models.ctgan_generator import CTGANGenerator
from tabular_generative_models.tvae_generator import TVAEGenerator
from tabular_generative_models.tab_ddpm_generator import TabDDPMGenerator
from tabular_generative_models.tabula_generator import TabulaGenerator

MODEL_REGISTRY = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
    "ddpm": TabDDPMGenerator,
    "tab_ddpm": TabDDPMGenerator,
    "tabula": TabulaGenerator,
}


def get_model_generator(model_name: str, data, num_cols, cat_cols, target_col=None) -> TabularGeneratorBase:
    """
    Retorna uma instância inicializada do gerador tabular correspondente.
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' é inválido. Modelos disponíveis: {list(MODEL_REGISTRY.keys())}"
        )
    generator_cls = MODEL_REGISTRY[model_name_lower]
    return generator_cls(data=data, num_cols=num_cols, cat_cols=cat_cols, target_col=target_col)


__all__ = [
    "TabularGeneratorBase",
    "CTGANGenerator",
    "TVAEGenerator",
    "TabDDPMGenerator",
    "TabulaGenerator",
    "get_model_generator",
]

