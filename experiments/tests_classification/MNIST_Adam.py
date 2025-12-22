# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 02:07:12 2025

@author: Romain
"""

# -*- coding: utf-8 -*-
"""
Benchmark Statistique : Adam (20 runs)
Mesure: Moyenne et Écart-type sur Accuracy et Temps
"""
import torch.nn as nn

import time
import torch 
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from torchvision import datasets, transforms

# Assurez-vous que ces imports fonctionnent avec votre structure de dossier
from NeuroEvolution import NeuroOptimizer 
from NeuroEvolution.layer_classes import Conv2dCfg, FlattenCfg, LinearCfg,MaxPool2dCfg, DropoutCfg

# --- CONFIGURATION ---
N_RUNS = 20          # Nombre d'itérations pour les statistiques
OPTIMIZER = "Adam"   # On ne teste que Adam
EPOCHS = 70          # Nombre d'époques par entraînement
POPULATION = 30      
EPOCHS_MODEL=20
def get_data_digits():
    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) # (N, 1, 8, 8)
    y = torch.tensor(digits.target, dtype=torch.long)
    return "Digits (8x8)", X, y

def get_data_fashion_mnist():
    try:
        print("Chargement FashionMNIST...", end="\r")
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        
        # On prend un sous-ensemble pour que le benchmark 20x ne soit pas trop long
        # Vous pouvez augmenter 2000 à 10000 ou plus si vous avez du temps/GPU
        indices = torch.randperm(len(dataset))[:2000] 
        X = dataset.data[indices].float().unsqueeze(1) / 255.0 
        y = dataset.targets[indices].long()
        return "FashionMNIST (28x28)", X, y
    except Exception as e:
        print(f"\nImpossible de charger FashionMNIST ({e}). On passe.")
        return None

def evaluate_model(model, X, y):
    """Retourne (Accuracy, Inference_Time_ms)"""
    model.eval()
    if X.is_cuda: torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
    
    if X.is_cuda: torch.cuda.synchronize()
    inf_time = time.time() - start
    
    correct = (preds == y).sum().item()
    acc = correct / y.size(0)
    
    return acc, inf_time * 1000 

# --- PRÉPARATION DES DATASETS ---
DATASETS = [
    get_data_digits(),
    get_data_fashion_mnist() 
]
DATASETS = [d for d in DATASETS if d is not None]

results = []

print(f"\n{'='*100}")
print(f"BENCHMARK STATISTIQUE : {OPTIMIZER} sur {N_RUNS} ITÉRATIONS")
print(f"{'='*100}\n")

for data_name, X, y in DATASETS:
    print(f" Dataset: {data_name} | Shape: {X.shape}")
    
 
    LAYERS= [
        # --- BLOC 1 : Extraction de caractéristiques primaires (Bords, Lignes) ---
        # Entrée : 28x28x1
        Conv2dCfg(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        # Réduction : 28x28 -> 14x14
        MaxPool2dCfg(kernel_size=2, stride=2),
        
        # --- BLOC 2 : Extraction de formes complexes (Boucles, Coins) ---
        # Entrée : 14x14x32 -> Sortie : 14x14x64 (On augmente la profondeur quand l'image rétrécit)
        Conv2dCfg(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        # Réduction : 14x14 -> 7x7
        MaxPool2dCfg(kernel_size=2, stride=2),
    
        # --- CLASSIFICATION ---
        FlattenCfg(),
        
        # Couche dense intermédiaire (Le "cerveau" qui interprète les formes)
        # Calcul auto : 7 * 7 * 64 = 3136 entrées
        LinearCfg(in_features=(X.shape[2] // 4) * (X.shape[3] // 4) * 64, out_features=256,activation=nn.ReLU),
        
        # Dropout pour éviter que le NAS ne crée un modèle qui apprend par cœur
        DropoutCfg(p=0.25),
        
        # Sortie finale (10 classes pour FashionMNIST)
        LinearCfg(in_features=256, out_features=10, activation=None) 
    ]

    # Listes pour stocker les résultats de chaque run
    acc_scores = []
    train_times = []
    inf_times = []

    print(f" Lancement de {N_RUNS} runs pour {OPTIMIZER}...")

    for i in range(N_RUNS):
        # On recrée l'objet NeuroOptimizer à chaque boucle pour réinitialiser le split Train/Test
        # Note: Si random_state est fixé à 42 dans votre classe, le split sera identique,
        # mais l'initialisation des poids du réseau (DynamicNet) sera différente.
        neuro = NeuroOptimizer(X, y, task="classification", Layers=list(LAYERS))
        
        start_train = time.time()
        
        # Entraînement des poids (Mode Fixe)
        #model_fixed = neuro.search_weights(
        #    optimizer_name=OPTIMIZER, 
        #    epochs=EPOCHS,          
        #    population=POPULATION,    
        #    verbose=False
        #)
        
        model=neuro.search_model( epochs=EPOCHS_MODEL, train_time=EPOCHS_MODEL*60, optimizer_name_weights='Adam', 
                         epochs_weights=EPOCHS,
                         verbose=False, verbose_weights=False, time_importance=None)
        
        train_time = time.time() - start_train
        acc, inf_ms = evaluate_model(model, neuro.X_test, neuro.y_test)
        # Stockage
        acc_scores.append(acc)
        train_times.append(train_time)
        inf_times.append(inf_ms)

        print(f"   -> Run {i+1}/{N_RUNS} : Acc={acc:.2%} | Time={train_time:.2f}s", end="\r")

    print(f"\n   Terminé pour {data_name}.")

    # --- CALCUL DES STATISTIQUES ---
    results.append({
        "Dataset": data_name,
        "Algo": OPTIMIZER,
        "Runs": N_RUNS,
        # Accuracy
        "Acc Mean": np.mean(acc_scores),
        "Acc Std": np.std(acc_scores),
        "Acc Max": np.max(acc_scores),
        # Training Time
        "Train Time Mean (s)": np.mean(train_times),
        "Train Time Std (s)": np.std(train_times),
        # Inference Time
        "Inf Time Mean (ms)": np.mean(inf_times)
    })

print("\n\n")
print("="*120)
print(" RÉSULTATS STATISTIQUES FINAUX")
print("="*120)

df = pd.DataFrame(results)

# Fonction de formatage pour l'affichage (Moyenne ± Ecart-type)
def format_result(row):
    return f"{row['Acc Mean']:.2%} ± {row['Acc Std']:.2%}"

df['Accuracy (Mean ± Std)'] = df.apply(format_result, axis=1)

# Sélection des colonnes pertinentes pour l'affichage
display_cols = ["Dataset", "Algo", "Runs", "Accuracy (Mean ± Std)", "Acc Max", "Train Time Mean (s)", "Inf Time Mean (ms)"]
final_df = df[display_cols]

print(final_df.to_string(index=False))
print("\n" + "="*120)