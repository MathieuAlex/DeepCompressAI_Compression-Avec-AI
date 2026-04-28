# Compression d'Images par Intelligence Artificielle

## Introduction

Étudiant-chercheur passionné par les algorithmes et l'intelligence artificielle, j'ai initié ce projet de compression d'images par IA afin d'explorer concrètement l'intersection entre théorie mathématique et apprentissage profond. Ce projet me tient particulièrement à cœur car il incarne ce qui me fascine le plus : concevoir des algorithmes intelligents capables de résoudre des problèmes complexes du monde réel. La compression d'images avec perte, à travers les réseaux de neurones, en est un exemple parfait — alliant optimisation, théorie de l'information et deep learning.

Le projet s'appuie sur deux articles fondateurs de Ballé (2018) : *"Joint Autoregressive and Hierarchical Priors for Learned Image Compression"* et *"Variational Image Compression with a Scale Hyperprior"*. Ces travaux sont reconnus comme des références incontournables dans l'application du deep learning à la compression d'images. Le défi central de la compression avec perte réside dans l'équilibre entre la qualité visuelle (mesurée par la distorsion) et le poids du fichier (mesuré par le débit binaire).

## Résultats & Conclusions

Le modèle a été entraîné sur **2 millions d'itérations** à l'aide d'un GPU NVIDIA RTX 4090, sur une durée d'environ 48 heures. Tout au long de l'entraînement, j'ai suivi les performances en temps réel via **Weights & Biases (Wandb)**. Les phases d'entraînement et de validation ont été réalisées sur le dataset **CLIC2021**, référence standard dans le domaine de la compression d'images.

Voici les métriques clés obtenues sur le jeu de validation CLIC2021 :

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Erreur Quadratique Moyenne (MSE)** | 16.52 | Différence moyenne au carré entre l'image originale et l'image compressée — plus c'est bas, mieux c'est |
| **Bits Par Pixel (bpp)** | 1.83 | Nombre moyen de bits utilisés pour encoder chaque pixel — plus c'est bas, mieux c'est |
| **Taux de compression implicite** | 13.12 : 1 | Le fichier est compressé plus de 13 fois sans perte de qualité significative |

Ces résultats surpassent les performances des techniques de compression standard telles que JPEG et WebP, ce qui confirme le potentiel des réseaux de neurones avancés pour la compression d'images.

Les courbes d'entraînement ci-dessous illustrent l'évolution de la **MSE** (perte de distorsion) et du **bpp** (perte de débit) tout au long du processus :

![courbe_mse](images/Validation-2.png)
![courbe_bpp](images/Validation-3.png)

Les images suivantes montrent concrètement les effets de l'algorithme de compression par rapport aux méthodes classiques :

![comparaison_train](images/Train-4.png)
![comparaison_validation](images/Validation-4.png)

## Comment Utiliser le Code

Pour tester le modèle de compression sur vos propres images, suivez ces étapes :

1. **Activer le chargement du modèle**
   Dans le script, passez le flag `load_model` de `False` à `True` pour utiliser le modèle pré-entraîné.

2. **Spécifier le chemin du modèle**
   Mettez à jour le script pour pointer vers le modèle sauvegardé après 2 millions d'itérations.

3. **Préparer l'image d'entrée**
   Assurez-vous que le chemin vers l'image d'entrée est correctement renseigné dans le script.

4. **Résultat**
   L'image décompressée (`xhat`) sera sauvegardée automatiquement. L'efficacité de la compression peut être calculée via `bpp_total`, qui représente le nombre total de bits par pixel : `bpp_total × hauteur × largeur` de l'image d'entrée.

## Prérequis

Avant de lancer le projet, assurez-vous d'avoir installé les éléments suivants :

- `PyTorch` — Pour la création et l'entraînement du modèle
- `Weights & Biases (wandb)` — Pour le suivi des expériences et la visualisation des résultats
- `CUDA` — Optionnel, pour l'accélération GPU si disponible
- `tqdm` — Pour l'affichage des barres de progression pendant l'entraînement et la validation

## Installation

Installez les dépendances Python requises avec la commande suivante :

```bash
pip install torch wandb tqdm
```

## Sources

- Ballé, J. (2018). **"Joint Autoregressive and Hierarchical Priors for Learned Image Compression"**. [Accéder à l'article](https://arxiv.org/pdf/1809.02736)
- Ballé, J. (2018). **"Variational Image Compression with a Scale Hyperprior"**. [Accéder à l'article](https://arxiv.org/pdf/1802.01436v2)

Outils et ressources utilisés dans ce projet :

- **[Weights & Biases (Wandb)](https://wandb.com)** — Outil de suivi des expériences, utilisé pour monitorer l'entraînement et la validation du modèle
- **[PyTorch](https://pytorch.org)** — Framework de deep learning utilisé pour l'implémentation du modèle
- **[Dataset CLIC2021](https://www.compression.cc/2021/)** — Dataset utilisé pour l'entraînement et la validation
