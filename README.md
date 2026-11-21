# Avaliação de Riscos de Privacidade em Modelos Generativos Tabulares via Ataques de Inferência de Membros

**Alan Teixeira da Costa**<br>
Orientador: **Vinícius Fernandes Soares Mota**<br>
Universidade Federal do Espírito Santo (UFES) — Trabalho de Conclusão de Curso, 2026

---

## Resumo

Este repositório reúne o código utilizado nos experimentos do Trabalho de Conclusão de Curso *"Avaliação de Riscos de Privacidade em Modelos Generativos Tabulares via Ataques de Inferência de Membros"*. Ele contém os pipelines de geração de dados sintéticos com os modelos CTGAN, TVAE, TabDDPM e Tabula, e os pipelines de avaliação desses dados quanto à utilidade preditiva, à fidelidade estatística e à privacidade. A privacidade é medida pelos ataques de inferência de membros FlowMIA-GAN, DOMIAS e DCR.

O código está organizado em dois experimentos. O primeiro compara os geradores nos domínios de saúde, redes de computadores e finanças. O segundo analisa como o deslocamento de distribuição no conjunto de não membros de referência afeta a avaliação de privacidade. O repositório inclui ainda os scripts de execução, as configurações de cada experimento, os ambientes conda e os notebooks que geram os gráficos e as tabelas do trabalho.

## Experimentos

### Experimento 1 — Comparação entre geradores

Os modelos CTGAN, TVAE, TabDDPM e Tabula são treinados e avaliados quanto à utilidade, à fidelidade e à privacidade nos datasets Breast Cancer (saúde), CIDDS e TON_IoT (redes de computadores) e Default of Credit Card Clients (finanças). Em todos eles, o conjunto de referência (não membros) vem da mesma distribuição dos dados de treinamento.

- Configurações e resultados: `results/<modelo>/<dataset>/`
- Análise: [`generators_analysis.ipynb`](generators_analysis.ipynb)

### Experimento 2 — Deslocamento de distribuição no conjunto de referência

Os modelos CTGAN, TVAE e TabDDPM são treinados com dados censitários do ACS Income (folktables) do Alabama. Os ataques são avaliados com conjuntos de referência de três origens: Alabama (sem deslocamento), Geórgia e Colorado.

- Configurações e resultados: `results_domain_shift/<modelo>/alabama_income/`
- Análise: [`mia_analysis.ipynb`](mia_analysis.ipynb)

As distribuições dos datasets estão em [`datasets_distribution.ipynb`](datasets_distribution.ipynb).

## Como reproduzir

### Ambiente

```bash
# Experimentos 1 e 2
conda env create -f environment/environment.yml        # cria o ambiente SynthEnvironment

# Notebooks (.ipynb)
conda env create -f environment/environment-notebook.yml   # cria o ambiente JupyterEnv
```

> Se encontrar problemas ao executar em uma RTX 5090, consulte [`environment/5090_torun.txt`](environment/5090_torun.txt).

### Experimento 1

```bash
./synthetic_run_all_datasets.sh <modelo>     # gera os dados sintéticos
./evaluation_run_all_datasets.sh <modelo>    # gera os resultados
```

`<modelo>`: `CTGAN`, `TVAE`, `TabDDPM` ou `Tabula`.

### Experimento 2

```bash
./domain_shift_synth.sh    # gera os dados sintéticos
./domain_shift_eval.sh     # gera os resultados
```

### Análises

Com o ambiente `JupyterEnv` ativo, execute os notebooks `datasets_distribution.ipynb`, `generators_analysis.ipynb` e `mia_analysis.ipynb`.

## Créditos

Este repositório incorpora código, com alterações, dos seguintes projetos:

- **Tabula** — https://github.com/zhao-zilong/Tabula (em `tabular_generative_models/tabula/`)
- **FlowMIA** — https://github.com/guilherme751/FlowMIA (em `flowmia/`)

As implementações do CTGAN, do TVAE e do TabDDPM vêm da biblioteca [synthcity](https://github.com/vanderschaarlab/synthcity).
