# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:12:01 2025

@author: Romain
"""

from NeuroEvolution import  NeuroOptimizer
import torch
from NeuroEvolution.layer_classes import Conv2dCfg,DropoutCfg,FlattenCfg,LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time

def  time_importance(loss, time):
    return loss+time*10

if __name__ == "__main__":
    from sklearn.datasets import make_classification,make_blobs, make_moons
    import matplotlib.pyplot as plt 
    

    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5, n_classes=4)
    #X, y = make_blobs(n_samples=200)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    

    """
    model = neuro_opt.search_linear_model(optimizer_name_weights='GWO', epochs=20,  train_time=10*60, epochs_weights=20, population_weights=20)
   
    
    activation=nn.Tanh
    Layers = [
        LinearCfg(2,32,activation),
        LinearCfg(32,32,activation),
        LinearCfg(32, 2, None)
    ] """

    
    Res={}
    neuro_opt = NeuroOptimizer(X, y, task="classification")
    for opt in range(10):
        
        
    
        #model = neuro_opt.search_weights(optimizer_name=opt, epochs=50, population=50)
        start=time.time()
        model= neuro_opt.search_model(
            hybrid=['GWO','Adam'],  hybrid_epochs=[10,10],
            epochs=10,                   
            train_time=60,             
            epochs_weights=10,          
            population_weights=20,  time_importance=time_importance
           
        )
                                    
        train_time=time.time()-start
        with torch.no_grad():
            start=time.time()
            logits = model(X_tensor)
            inf_time=time.time()-start
            _, predictions = torch.max(logits, 1)
            test_loss = accuracy_score(predictions, y_tensor)
            Res[opt]=(test_loss,inf_time,train_time)
            print((opt,test_loss,inf_time,train_time))
    print(Res)
    
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