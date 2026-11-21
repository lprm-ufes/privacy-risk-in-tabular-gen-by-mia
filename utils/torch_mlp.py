import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class _MLPModule(nn.Module):
    def __init__(self, in_features: int, hidden_layer_sizes: tuple[int, ...] | list[int], num_classes: int):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible MLP Classifier running on PyTorch with CUDA acceleration.
    Serves as a drop-in replacement for sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=300,
        batch_size=50,
        lr=1e-3,
        random_state=None,
        device=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = device

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)

        device = self._resolve_device()

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        num_classes = len(self.classes_)

        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        elif hasattr(X, "toarray"):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)

        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_encoded, dtype=torch.long)

        if isinstance(self.hidden_layer_sizes, int):
            hidden_layers = [self.hidden_layer_sizes]
        else:
            hidden_layers = list(self.hidden_layer_sizes)

        self.model_ = _MLPModule(
            in_features=X_arr.shape[1],
            hidden_layer_sizes=hidden_layers,
            num_classes=num_classes,
        ).to(device)

        batch_sz = self.batch_size
        if batch_sz == "auto" or batch_sz is None:
            batch_sz = min(200, len(X_arr))
        else:
            batch_sz = int(batch_sz)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.max_iter):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        del dataset, dataloader, X_tensor, y_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def predict_proba(self, X):
        device = self._resolve_device()
        self.model_.eval()

        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        elif hasattr(X, "toarray"):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)

        X_tensor = torch.as_tensor(X_arr, dtype=torch.float32)
        eval_batch_size = max(256, self.batch_size if isinstance(self.batch_size, int) else 256)
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=eval_batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(device)
                outputs = self.model_(batch_x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probas.append(probs)

        return np.concatenate(probas, axis=0)

