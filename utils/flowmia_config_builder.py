class FlowMiaConfigBuilder:
    def __init__(self, member_path: str, non_member_path: str, synth_path: str, 
                 categorical_cols: list, numerical_cols: list, label_col: str):
        
        self._member_path = member_path
        self._non_member_path = non_member_path
        self._synth_path = synth_path
        self._categorical_cols = categorical_cols
        self._numerical_cols = numerical_cols
        self._label_col = label_col

        self._ip_cols = []
        
        self._test_path = None
        self._test_size = None
        
        self._batch_size = 50
        self._num_epochs = 30
        self._fcheckpoint = 100
        self._save_path = './'
        self._use_wgan = True
        self._seed = None

    def with_ip_columns(self, ip_columns: list):
        self._ip_cols = ip_columns
        return self

    def with_test(self, test_path: str, test_size: int):
        self._test_path = test_path
        self._test_size = test_size
        return self

    def with_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        return self

    def with_num_epochs(self, num_epochs: int):
        self._num_epochs = num_epochs
        return self

    def with_checkpoint_frequency(self, frequency: int):
        self._fcheckpoint = frequency
        return self
    
    def with_save_path(self, save_path: str):
        self._save_path = save_path
        return self

    def with_seed(self, seed: int):
        self._seed = seed
        return self

    def with_use_wgan(self, use_wgan: bool):
        self._use_wgan = use_wgan
        return self

    def build(self):
        return {
                'member_path': self._member_path, # path dos membros
                'non_member_path': self._non_member_path, # path dos não-membros
                'synth_path': self._synth_path, # path dos sintéticos
                'test_path': self._test_path, # path do teste
                'test_size': self._test_size,
                'categorical_cols': self._categorical_cols, # colunas categóricas
                'numerical_cols': self._numerical_cols, #colunas numéricas
                'ip_cols': self._ip_cols, # colunas de ip
                'label_col': self._label_col, # nome da coluna do rótulo 
                'batch_size': self._batch_size, # número de amostrar por lote
                'num_epochs': self._num_epochs, # número de épocas
                'fcheckpoint': self._fcheckpoint, # frequência para salvar o checkpoint
                'save_path': self._save_path,
                'use_wgan': self._use_wgan, # se deve usar WGAN ou GAN tradicional
                'seed': self._seed
            }