# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/vinicius-souza-lima/mvp_ml/blob/main/mvp_ml.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

# %%
from pathlib import Path
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# %% [markdown]
# ## Problema de Regressão

# %%
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# %%
path_chest_train = Path(path) / "chest_xray" / "chest_xray" / "train"


# %% [markdown]
# ## Problema de Classificação (Algoritmos Clássicos)
#
# ### Criação do Dataset simplificado a partir das imagens
#
# Objetivo: Classificar imagem de raio x de paciente em saudável, pneumonia bacteriana ou pneumonia virótica

# %%
class Dataset:
    def __init__(
        self,
        path_dir: str,
        file: str,
        url: str,
        resolution: tuple[int, int] = (128, 128),
    ):
        self.data = None
        self.path_dir = path_dir
        self.url = url
        self.resolution = resolution

    def load_csv(self, **kwargs):
        """
        Função para carregar os dados, verifica se dataset já foi baixado, baixando-o de acordo
        com a necessidade
        """
        file_path = Path(self.path_dir)
        if not file_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            if self.url is None:
                raise ValueError("Url não informada")
            urllib.request.urlretrieve(self.url, file_path)
        self.data = pd.read_csv(self.path_dir, **kwargs)

    def convert_toarray(
        self,
        target_values: list[str],
        fallback: str | None = None,
    ):
        dir = Path(self.path_dir)
        files = dir.glob("*")
        imgs = []
        targets = []
        for f in files:
            imgs.append(
                np.array(
                    Image.open(str(f))  # Lê a imagem
                    .convert("L")  # Converte para escala de cinza
                    .resize(
                        self.resolution, Image.Resampling.LANCZOS
                    )  # Redimensiona a imagem
                ).flatten()  # Formata a matriz como array
            )
            targets.append(
                next(
                    (cat for cat in f.name.split("_") if cat in target_values),
                    fallback,  # procura categoria no nome do arquivo
                )
            )

        return np.stack(imgs), np.array(targets)


# %%
resolution = (128, 128)
# X_chest,y_chest = convert_toarray("./datasets/chest_xray/imgs",["virus","bacteria"],"normal",resolution)

# %%
# for i,array in enumerate(np.array_split(X_chest,5)):
# np.save(f"datasets/chest_xray/imgs_array/X_chest_{i}",array)
# np.save("datasets/chest_xray/imgs_array/y_chest",y_chest)

# %%
X_chest = []
for file in sorted(Path("datasets/chest_xray/imgs_array/X_chest/").glob("*")):
    X_chest.append(np.load(str(file)))
X_chest = np.vstack(X_chest)
y_chest = np.load("datasets/chest_xray/imgs_array/y_chest.npy")

# %%
X_chest.shape

# %%
plt.imshow(X_chest[4000].reshape(resolution))
plt.axis("off")
plt.show()


# %% [markdown]
# ### Data Augmentation

# %%
class Augmenter(BaseEstimator, TransformerMixin):
    def __init__(
        self, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1
    ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        np.random.normal(0, 1, X.shape)


# %% [markdown]
# ### Selecionar modelo

# %%
from sklearn.model_selection import KFold


X_chest_train, X_chest_test, y_chest_train, y_chest_test = train_test_split(
    X_chest, y_chest, test_size=0.2, random_state=42, stratify=y_chest
)

num_particoes = 5
kfold = KFold(n_splits=num_particoes, shuffle=True, random_state=42)

# %%
np.random.seed(42)
results = []


models = {
    "NB": GaussianNB(),  # 5 sec
    "KNN": KNeighborsClassifier(),  # 12 sec
    "RF": RandomForestClassifier(),  # +- 3 min
    "CART": DecisionTreeClassifier(),  # +- 7 min
}

for name, model in models.items():
    cv_results = cross_val_score(
        model, X_chest_train, y_chest_train, cv=kfold, scoring="accuracy"
    )
    results.append(cv_results)
    np.save("results.npy", results)

    print(f"{name}: Média: {cv_results.mean():.2f}, std:{cv_results.std():.3f}")

# %%
sns.set_theme()
names = ["NB", "KNN", "RF", "CART"]
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Comparação da acurácia dos modelos")
ax.set_xticklabels(names)
sns.boxplot(results);

# %% [markdown]
# Assim foi escolhido Random Forest para prosseguir na otimização de hiperparâmetros

# %% [markdown]
# ## Otimização de Hiperparâmetros

# %%
model = RandomForestClassifier()

# %%
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 2, 5, 10],
    "min_samples_split": [2, 4, 5],
    "min_samples_leaf": [2, 4, 5],
}

random_search = RandomizedSearchCV(
    model, param_grid, cv=3, scoring="accuracy", n_iter=5, n_jobs=2, random_state=42
)

# %%
random_search.fit(X_chest_train, y_chest_train)

# %%
random_search.best_score_

# %%
final_model = random_search.best_estimator_

# %%
joblib.dump(final_model, "final_model.joblib")

# %% [markdown]
# # Avaliação dos Resultados

# %%

# %% [markdown]
# ## Problema de Visão Computacional (Deep Learning)
#
# Objetivo: Treinar rede neural que classifique os tipos de tumores e segmente na imagem o local em que ele aparece
#
# ## Problema de Processamento de Linguagem Natural
