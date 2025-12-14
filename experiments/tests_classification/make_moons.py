# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:41:22 2025
@author: Romain
"""

from NeuroEvolution import  NeuroOptimizer
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons


def time_importance(accuracy, inference_time):

    return accuracy + (inference_time * 10)

print("\n\n")
print("="*100)
print("BENCHMARK: make_moons (n=2000, noise=0.3) | 20 Runs Average")
print("="*100)

if __name__ == "__main__":
    Res = {opt: [] for opt in NeuroOptimizer.get_available_optimizers()}
    
    N_RUNS = 20
    
    for i in range(N_RUNS):
        print(f"\nRun {i+1}/{N_RUNS}...")
        
        X, y = make_moons(n_samples=2000, noise=0.3)
        X_tensor_final = torch.tensor(X, dtype=torch.float32)
        y_tensor_final = torch.tensor(y, dtype=torch.float32)

        for opt in NeuroOptimizer.get_available_optimizers():
            print(f"   > Testing {opt}...", end="\r")
            
            neuro_opt = NeuroOptimizer(X, y, task="classification")

            model = neuro_opt.search_model(
                optimizer_name_weights=opt, 
                epochs=10,                   
                train_time=60,             
                epochs_weights=10,          
                population_weights=20, 
               
            )
            
            with torch.no_grad():
                model.eval()
                start = time.time()
                logits = model(X_tensor_final)
                inf_time = time.time() - start
                
                _, predictions = torch.max(logits, 1)
                acc = accuracy_score(y, predictions.numpy())
                
                Res[opt].append((acc, inf_time))
            
            print(f"  {opt} : Acc={acc:.2%} | Time={inf_time*1000:.2f}ms")

    print("\n\n")
    print("="*100)
    print(f"{'ALGORITHM':<15} | {'AVG ACCURACY':<15} | {'STD DEV':<10} | {'AVG INF TIME (ms)':<20} | {'BEST ACC':<10}")
    print("-" * 100)
    

    final_stats = []
    for opt, values in Res.items():
        if not values: continue
        
        accs = [v[0] * 100 for v in values]
        times = [v[1] * 1000 for v in values] # En ms
        
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        avg_time = np.mean(times)
        best_acc = np.max(accs)
        
        final_stats.append((opt, avg_acc, std_acc, avg_time, best_acc))

    final_stats.sort(key=lambda x: x[1], reverse=True)

    for opt, avg_acc, std_acc, avg_time, best_acc in final_stats:
        print(f"{opt:<15} | {avg_acc:7.2f}%        | ±{std_acc:4.2f}%   | {avg_time:10.4f} ms        | {best_acc:6.2f}%")
    
    print("="*100)