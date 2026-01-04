# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 00:58:25 2026

@author: Romain
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler


from optimizer import NeuroOptimizer 
from NeuroEvolution.layer_classes import Conv2dCfg, FlattenCfg, LinearCfg

def run_benchmark(task_name, X, y, layers_cfg, task_type="classification", n_runs=3, epochs=5):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {task_name}")
    print(f"{'='*60}")
    
    accuracies = [] # Ou MSE pour régression
    train_times = []
    inf_times = []
    
    for i in range(n_runs):
        print(f" > Run {i+1}/{n_runs}...", end=" ", flush=True)
        
        # 1. Instantiation
        optimizer = NeuroOptimizer(
            X, y, 
            Layers=layers_cfg, 
            task=task_type
        )
        
        # 2. Mesure temps d'entrainement (NAS + Poids)
        t0 = time.time()
        # On utilise une recherche légère pour le benchmark (Adam seulement ici pour la vitesse)
        best_model = optimizer.search_model(
            epochs=2, # Peu d'itérations NAS pour le test
            train_time=60, 
            optimizer_name_weights='Adam',
            epochs_weights=epochs, 
            verbose=False
        )
        t_train = time.time() - t0
        train_times.append(t_train)
        
        # 3. Evaluation et Temps d'inférence
        # evaluate_model retourne (loss, latency_seconds)
        # On refait une eval manuelle pour avoir l'accuracy ou la MSE précise sur le test set
        val_score = optimizer.evaluate(best_model, dataset="test", verbose=False)
        
        # Pour récupérer le temps d'inférence précis calculé par la méthode interne
        _, t_inf = best_model.evaluate_model(optimizer.X_test, optimizer.y_test, verbose=False)
        
        # Note: evaluate retourne -Acc pour classification (minimisation), on remet en positif
        score = -val_score if task_type == "classification" else val_score
        accuracies.append(score)
        inf_times.append(t_inf * 1000) # ms
        
        print(f"Done. Score: {score:.4f} | Train: {t_train:.2f}s | Inf: {t_inf*1000:.2f}ms")

    # Calcul des moyennes
    avg_score = np.mean(accuracies)
    avg_train = np.mean(train_times)
    avg_inf = np.mean(inf_times)
    
    return {
        "Task": task_name,
        "Avg Score": avg_score,
        "Avg Train Time (s)": avg_train,
        "Avg Inference (ms)": avg_inf
    }

if __name__ == "__main__":
    results = []
    
    # ---------------------------------------------------------
    # SCENARIO 1: Classification Linéaire (Dense)
    # ---------------------------------------------------------
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_cls = StandardScaler().fit_transform(X_cls)
    
    # Architecture Simple: Input(20) -> Linear(16) -> ReLU -> Linear(2)
    layers_linear = [
        LinearCfg(in_features=20, out_features=16, activation=nn.ReLU),
        LinearCfg(in_features=16, out_features=2, activation=None)
    ]
    
    res_lin = run_benchmark("Linear Classification", X_cls, y_cls, layers_linear, "classification", n_runs=3)
    results.append(res_lin)

    # ---------------------------------------------------------
    # SCENARIO 2: Classification CNN (Image Synthétique 1x8x8)
    # ---------------------------------------------------------
    # On génère 64 features qu'on reshape en images 8x8
    X_cnn, y_cnn = make_classification(n_samples=1000, n_features=64, n_informative=40, n_classes=3, random_state=42)
    X_cnn = StandardScaler().fit_transform(X_cnn)
    # Reshape en (N, Channels, Height, Width) -> (1000, 1, 8, 8)
    X_cnn = X_cnn.reshape(-1, 1, 8, 8)
    
    # Architecture CNN: Conv -> MaxPool -> Flatten -> Linear
    layers_cnn = [
        Conv2dCfg(in_channels=1, out_channels=8, kernel_size=3, padding=1, activation=nn.ReLU),
        MaxPool2dCfg(2, 2, 0),
        FlattenCfg(),
        LinearCfg(in_features=None, out_features=3, activation=None) # LazyLinear devinera in_features
    ]
    
    res_cnn = run_benchmark("CNN Classification (8x8)", X_cnn, y_cnn, layers_cnn, "classification", n_runs=3)
    results.append(res_cnn)

    # ---------------------------------------------------------
    # SCENARIO 3: Régression
    # ---------------------------------------------------------
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_reg = StandardScaler().fit_transform(X_reg)
    
    # Architecture Regression
    layers_reg = [
        LinearCfg(in_features=20, out_features=32, activation=nn.ReLU),
        LinearCfg(in_features=32, out_features=16, activation=nn.ReLU),
        LinearCfg(in_features=16, out_features=1, activation=None)
    ]
    
    res_reg = run_benchmark("Regression (MSE)", X_reg, y_reg, layers_reg, "regression", n_runs=3)
    results.append(res_reg)

    # ---------------------------------------------------------
    # AFFICHAGE DU RAPPORT FINAL
    # ---------------------------------------------------------
    print("\n\n")
    print("#"*80)
    print(f"{'FINAL BENCHMARK REPORT':^80}")
    print("#"*80)
    print(f"{'TASK':<30} | {'METRIC (Acc/MSE)':<15} | {'TRAIN TIME (s)':<15} | {'INFERENCE (ms)':<15}")
    print("-" * 80)
    
    for r in results:
        metric_label = "MSE" if "Regression" in r['Task'] else "Accuracy"
        print(f"{r['Task']:<30} | {r['Avg Score']:<15.4f} | {r['Avg Train Time (s)']:<15.4f} | {r['Avg Inference (ms)']:<15.4f}")
    print("-" * 80)