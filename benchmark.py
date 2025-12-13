# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:34:18 2025

@author: Romain
"""

import time
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from main import  NeuroOptimizer
import torch
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Import de ta librairie (assure-toi que les classes sont accessibles)
# from main import NeuroOptimizer, Benchmark (si tu as séparé les fichiers)

# --- CLASSE UTILITAIRE DE BENCHMARK ---
from thop import profile # Nécessite pip install thop

class Benchmark:
    @staticmethod
    def measure_efficiency(model, input_shape, device='cpu'):
        """
        Mesure FLOPs, Paramètres et Latence.
        """
        model.to(device)
        model.eval()
        
        # 1. Dummy Input
        dummy_input = torch.randn(input_shape).to(device)
        
        # 2. FLOPs & Params
        try:
            flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        except Exception:
            flops, params = 0, 0 # Fallback si thop plante sur certaines architectures

        # 3. Latence (Moyenne sur 100 runs)
        # Warmup
        for _ in range(10): _ = model(dummy_input)
        
        start_time = time.time()
        runs = 100
        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
        end_time = time.time()
        
        latency_ms = ((end_time - start_time) / runs) * 1000
        
        return {
            "FLOPs (M)": round(flops / 1e6, 4),
            "Params (k)": round(params / 1e3, 2),
            "Latency (ms)": round(latency_ms, 4)
        }

# --- FONCTION PRINCIPALE DE TEST ---
def run_uci_benchmark():
    print("\n" + "="*80)
    print("🔬 INITIALISATION DU BENCHMARK UCI (Iris, Wine, Cancer)")
    print("="*80)

    # 1. Définition des Datasets
    datasets = [
        ("Iris", load_iris()),
        ("Wine", load_wine()),
        ("Breast Cancer", load_breast_cancer())
    ]

    results_table = []
    for opt in NeuroOptimizer.get_available_optimizers():
        # 2. Boucle sur chaque Dataset
        for name, data in datasets:
            print(f"\n👉 Traitement du dataset : {name.upper()}")
            X, y = data.data, data.target
            
            # Standardisation (CRITIQUE pour les réseaux de neurones)
            # Note : NeuroOptimizer fait déjà un split, mais on scale tout ici pour simplifier
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
    
            # 3. Initialisation de TON Optimiseur
            # On choisit un algo rapide pour le test (ex: PSO ou GWO)
            algo_choice = opt
            print(f"   Algo: {algo_choice} | Input Features: {X.shape[1]} | Classes: {len(np.unique(y))}")
            
            neuro_opt = NeuroOptimizer(X_scaled, y, task="classification")
            
            # 4. Lancement de la recherche (Search Weights)
            # On met peu d'époques/pop pour que le test soit rapide (Demo)
            start_train = time.time()
            #model = neuro_opt.search_weights(optimizer_name=algo_choice, epochs=30, population=20)
            model = neuro_opt.search_model(optimizer_name_weights=algo_choice, epochs=30)
            train_time = time.time() - start_train
    
            # 5. Évaluation Précision (Accuracy)
            # On recrée les tensors de test pour l'évaluation finale
            X_tensor = neuro_opt.X_test
            y_tensor = neuro_opt.y_test
            
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                _, preds = torch.max(logits, 1)
                acc = accuracy_score(y_tensor.numpy(), preds.numpy())
    
            # 6. Benchmark Hardware (FLOPs/Latence)
            # Input shape = (1 sample, n_features)
            metrics = Benchmark.measure_efficiency(model, (1, X.shape[1]))
    
            print(f" Terminé. Accuracy: {acc*100:.2f}%")
    
            # Stockage des résultats
            results_table.append({
                "Dataset": name,
                "Algorithm": algo_choice,
                "Accuracy (%)": round(acc * 100, 2),
                "Train Time (s)": round(train_time, 2),
                "Latency (ms)": metrics["Latency (ms)"],
                "Params (k)": metrics["Params (k)"],
                "FLOPs (M)": metrics["FLOPs (M)"]
            })

    # --- AFFICHAGE DU RAPPORT FINAL ---
    print("\n\n")
    print("="*100)
    print("Benchmarks")
    print("="*100)
    
    # Utilisation de Pandas pour un affichage joli (si dispo), sinon print classique
    try:
        df = pd.DataFrame(results_table)
        print(df.to_string(index=False))
    except ImportError:
        # Fallback si pandas n'est pas installé
        print(f"{'Dataset':<15} | {'Acc (%)':<10} | {'Lat (ms)':<10} | {'Params (k)':<10}")
        print("-" * 55)
        for row in results_table:
            print(f"{row['Dataset']:<15} | {row['Accuracy (%)']:<10} | {row['Latency (ms)']:<10} | {row['Params (k)']:<10}")
    
    print("="*100 + "\n")

if __name__ == "__main__":
    run_uci_benchmark()