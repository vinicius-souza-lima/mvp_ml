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
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib


# %% [markdown]
# ## Problema de Regressão

# %%

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
                    .resize(self.resolution, Image.Resampling.LANCZOS)
                ).flatten()  # Redimensiona a imagem
            )  # Formata a matriz como array
            targets.append(
                next(
                    (cat for cat in f.name.split("_") if cat in target_values), fallback
                )
            )  # procura categoria no nome do arquivo

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

# %%
X_chest_train, X_chest_test, y_chest_train, y_chest_test = train_test_split(
    X_chest, y_chest, test_size=0.2, random_state=42, stratify=y_chest
)

# %%
# lr_clf = LogisticRegression(max_iter=1000,random_state=42)
# lr_clf.fit(X_chest_train,y_chest_train)

# %%
# joblib.dump(lr_clf,"Logistic_Regression.joblib")

# %%
lr_clf = joblib.load("Logistic_Regression.joblib")

# %%
# y_chest_pred = cross_val_predict(lr_clf,X_chest_train,y_chest_train,cv=3)
# np.save("y_chest_pred",y_chest_pred)

# %% [markdown]
# # Selecionar modelo

# %%
y_chest_pred = np.load("y_chest_pred.npy")

# %%
cm_chest = confusion_matrix(y_chest_train, y_chest_pred)

# %%
cm_chest

# %%
display(precision_score(y_chest_train, y_chest_pred, average=None))
display(precision_score(y_chest_train, y_chest_pred, average="macro"))
display(precision_score(y_chest_train, y_chest_pred, average="micro"))
display(precision_score(y_chest_train, y_chest_pred, average="weighted"))

# %%
display(recall_score(y_chest_train, y_chest_pred, average=None))
display(recall_score(y_chest_train, y_chest_pred, average="macro"))
display(recall_score(y_chest_train, y_chest_pred, average="micro"))
display(recall_score(y_chest_train, y_chest_pred, average="weighted"))

# %%
display(f1_score(y_chest_train, y_chest_pred, average=None))
display(f1_score(y_chest_train, y_chest_pred, average="macro"))
display(f1_score(y_chest_train, y_chest_pred, average="micro"))
display(f1_score(y_chest_train, y_chest_pred, average="weighted"))

# %% [markdown]
# ## Problema de Visão Computacional (Deep Learning)
#
# Objetivo: Treinar rede neural que classifique os tipos de tumores e segmente na imagem o local em que ele aparece
#
# ## Problema de Processamento de Linguagem Natural
