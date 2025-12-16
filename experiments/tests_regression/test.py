import numpy as np
from NeuroEvolution import  NeuroOptimizer
import torch
from NeuroEvolution.layer_classes import Conv2dCfg,DropoutCfg,FlattenCfg,LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt 
    

    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = torch.cos( x) + torch.randn(100,1)*0.1
    
    xtest= torch.linspace(0, 1, 100).unsqueeze(1)
    
    Layers = [
        LinearCfg(1, 32, nn.Tanh),
        LinearCfg(32, 32, nn.Tanh),
        LinearCfg(32, 1, None)
    ]
    neuro_opt = NeuroOptimizer(x, y, task="regression", Layers=Layers, activation=nn.Tanh)
    model = neuro_opt.search_linear_model(optimizer_name_weights='Adam', epochs=50,  train_time=10*60,
                                   epochs_weights=200, population_weights=20,
                                   verbose=True)
    """
    Res={}
    neuro_opt = NeuroOptimizer(X, y, task="classification")
    for opt in NeuroOptimizer.get_available_optimizers():
        
        
    
        #model = neuro_opt.search_weights(optimizer_name=opt, epochs=50, population=50)
    
        model=neuro_opt.search_linear_model(optimizer_name_weights='Adam', epochs=200,  train_time=10*60,
                                     epochs_weights=30, population_weights=50,
                                     time_importance=time_importance)
    
        with torch.no_grad():
            start=time.time()
            logits = model(X_tensor)
            inf_time=time.time()-start
            _, predictions = torch.max(logits, 1)
            test_loss = accuracy_score(predictions, y_tensor)
            Res[opt]=(test_loss,inf_time)
    print(Res)
    """
    with torch.no_grad():
        pred = model(xtest)
        
    
    
    plt.figure(figsize=(10, 5))
    

    plt.scatter(x,y)
    plt.plot(xtest,pred,'r')
    

    plt.show()