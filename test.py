# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:12:01 2025

@author: Romain
"""

from main import  NeuroOptimizer
import torch
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    from sklearn.datasets import make_classification,make_blobs
    import matplotlib.pyplot as plt 
    

    X, y = make_classification(n_samples=2000, n_features=50, n_informative=5, n_classes=4)
    #X, y = make_blobs(n_samples=200)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    Layers = [
        LinearCfg(50, 16, nn.ReLU),
        *[LinearCfg(16, 16, nn.ReLU) for _ in range(5)],
        LinearCfg(16, 4, None),
    ]
    
    neuro_opt = NeuroOptimizer(X, y, task="classification", Layers = Layers)
    model = neuro_opt.search_model(optimizer_name_weights='Adam', epochs=2000,  train_time=10*60, epochs_weights=30, population_weights=50)
    """
    Res={}
    
    for opt in NeuroOptimizer.get_available_optimizers():
        neuro_opt = NeuroOptimizer(X, y, task="classification")
        
    
        model = neuro_opt.search_weights(optimizer_name=opt, epochs=50, population=50)
    
        
    
        with torch.no_grad():
            logits = model(X_tensor)
            _, predictions = torch.max(logits, 1)
            test_loss = accuracy_score(predictions, y_tensor)
            Res[opt]=test_loss
    print(Res)
    """
    with torch.no_grad():
        logits = model(X_tensor)
        _, predictions = torch.max(logits, 1)
        test_loss = accuracy_score(predictions, y_tensor)
    
    print(test_loss)
    predictions = predictions.numpy()
    
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
        
        plt.subplot(1, 2, 2)
        plt.title("Neuro-Evolution Predictions")
        plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', edgecolor='k')
        
        plt.show()