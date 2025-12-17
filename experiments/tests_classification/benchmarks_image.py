# -*- coding: utf-8 -*-
"""
Benchmark Multi-Datasets : NeuroEvolution vs Adam
Mesure: Accuracy, Training Time, Inference Time
"""

import time
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from NeuroEvolution import NeuroOptimizer 
from NeuroEvolution.layer_classes import Conv2dCfg, FlattenCfg, LinearCfg



def get_data_digits():

    digits = load_digits()
    X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) # (N, 1, 8, 8)
    y = torch.tensor(digits.target, dtype=torch.long)
    return "Digits (8x8)", X, y

def get_data_fashion_mnist():

    try:
        print("Téléchargement FashionMNIST...", end="\r")
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        

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




DATASETS = [
    get_data_digits(),
    get_data_fashion_mnist() 
]

DATASETS = [d for d in DATASETS if d is not None]




results = []

print(f"\n{'='*100}")
print(f"BENCHMARK CNN MULTI-DATASETS")
print(f"{'='*100}\n")

for data_name, X, y in DATASETS:
    print(f" Dataset: {data_name} | Shape: {X.shape}")
    LAYERS = [
        Conv2dCfg(1, 8, 3, padding=1),
        Conv2dCfg(8, 8, 3, padding=1),
        Conv2dCfg(8, 8, 3, padding=1),
        FlattenCfg(),
        LinearCfg(X.shape[2]*X.shape[3]*X.shape[1]*8, 10, None) 
    ]
    for opt in NeuroOptimizer.get_available_optimizers():
        print(f" Testing {opt}...", end="\r")
        

        neuro= NeuroOptimizer(X, y, task="classification", Layers=list(LAYERS))
        
        start_train = time.time()
        model_fixed = neuro.search_weights(
            optimizer_name=opt, 
            epochs=30,         
            population=30,    
            verbose=False
        )
        train_time_fixed = time.time() - start_train

        acc_fixed, inf_fixed = evaluate_model(model_fixed, neuro.X_test, neuro.y_test)

        results.append({
            "Dataset": data_name,
            "Algo": opt,
            "Mode": "Weights (Fixed)",
            "Accuracy": acc_fixed,
            "Train Time (s)": train_time_fixed,
            "Inf Time (ms)": inf_fixed
        })

        neuro_nas = NeuroOptimizer(X, y, task="classification", Layers=list(LAYERS))
        
        start_train = time.time()
        model_nas = neuro_nas.search_model(
            optimizer_name_weights=opt,
            epochs=5,           
            train_time=120,     
            epochs_weights=30, 
            population_weights=30,
            verbose=False
        )
        train_time_nas = time.time() - start_train
        

        acc_nas, inf_nas = evaluate_model(model_nas, neuro_nas.X_test, neuro_nas.y_test)

        results.append({
            "Dataset": data_name,
            "Algo": opt,
            "Mode": "NAS (Evolved)",
            "Accuracy": acc_nas,
            "Train Time (s)": train_time_nas,
            "Inf Time (ms)": inf_nas
        })
        
        print(f"  {opt} Done. (Fixed: {acc_fixed:.1%} | NAS: {acc_nas:.1%})")




print("\n\n")
print("="*100)
print(" RÉSULTATS DU BENCHMARK")
print("="*100)

df = pd.DataFrame(results)

df = df.sort_values(by=["Dataset", "Algo", "Mode"])


print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

print("\n" + "="*100)
