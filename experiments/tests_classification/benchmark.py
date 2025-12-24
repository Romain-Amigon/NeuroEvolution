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
from NeuroEvolution import  NeuroOptimizer
import torch
from NeuroEvolution.layer_classes import Conv2dCfg,DropoutCfg,FlattenCfg,LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score


from thop import profile 

class Benchmark:
    @staticmethod
    def measure_efficiency(model, input_shape, device='cpu'):
        """
        Mesure FLOPs, Paramètres et Latence.
        """
        model.to(device)
        model.eval()
        
        dummy_input = torch.randn(input_shape).to(device)

        try:
            flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        except Exception:
            flops, params = 0, 0 

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

def run_uci_benchmark():
    print("\n" + "="*80)
    print("🔬 INITIALISATION DU BENCHMARK UCI (Iris, Wine, Cancer)")
    print("="*80)

    datasets = [
        ("Iris", load_iris()),
        ("Wine", load_wine()),
        ("Breast Cancer", load_breast_cancer())
    ]

    results_table = []
    for opt in range(1) :#NeuroOptimizer.get_available_optimizers():
        for name, data in datasets:
            print(f"\n👉 Traitement du dataset : {name.upper()}")
            X, y = data.data, data.target

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            algo_choice = opt
            print(f"   Algo: {algo_choice} | Input Features: {X.shape[1]} | Classes: {len(np.unique(y))}")
            
            neuro_opt = NeuroOptimizer(X_scaled, y, task="classification")
            

            start_train = time.time()
            #model = neuro_opt.search_model(optimizer_name_weights=algo_choice, epochs=30)
            model = neuro_opt.search_model(
                hybrid=['GWO','Adam'],  hybrid_epochs=[10,10],
                epochs=10,                   
                train_time=60,             
                epochs_weights=10,          
                population_weights=20, 
               
            )
            train_time = time.time() - start_train
    

            X_tensor = neuro_opt.X_test
            y_tensor = neuro_opt.y_test
            
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                _, preds = torch.max(logits, 1)
                acc = accuracy_score(y_tensor.numpy(), preds.numpy())
    

            metrics = Benchmark.measure_efficiency(model, (1, X.shape[1]))
    
            print(f" Terminé. Accuracy: {acc*100:.2f}%")
    
            results_table.append({
                "Dataset": name,
                "Algorithm": algo_choice,
                "Accuracy (%)": round(acc * 100, 2),
                "Train Time (s)": round(train_time, 2),
                "Latency (ms)": metrics["Latency (ms)"],
                "Params (k)": metrics["Params (k)"],
                "FLOPs (M)": metrics["FLOPs (M)"]
            })

    print("\n\n")
    print("="*100)
    print("Benchmarks")
    print("="*100)
    
    try:
        df = pd.DataFrame(results_table)
        print(df.to_string(index=False))
    except ImportError:
        print(f"{'Dataset':<15} | {'Acc (%)':<10} | {'Lat (ms)':<10} | {'Params (k)':<10}")
        print("-" * 55)
        for row in results_table:
            print(f"{row['Dataset']:<15} | {row['Accuracy (%)']:<10} | {row['Latency (ms)']:<10} | {row['Params (k)']:<10}")
    
    print("="*100 + "\n")

if __name__ == "__main__":
    run_uci_benchmark()