import torch
import time
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from NeuroEvolution import  NeuroOptimizer

def reg_time_importance(mse, time_taken):
    """
    On veut minimiser le MSE.
    On ajoute une pénalité si le modèle est lent.
    """
    # Si le temps > 10ms, on pénalise fortement
    time_penalty = time_taken * 10.0
    return mse + time_penalty

def run_regression_benchmark():
    print("\n" + "="*100)
    print("BENCHMARK REGRESSION : Adam vs Métaheuristiques")
    print("="*100)


    datasets = []
    

    cal_X, cal_y = fetch_california_housing(return_X_y=True)
    datasets.append(("California Housing (2k)", cal_X[:2000], cal_y[:2000]))


    dia_X, dia_y = load_diabetes(return_X_y=True)
    datasets.append(("Diabetes", dia_X, dia_y))


    fri_X, fri_y = make_friedman1(n_samples=1000, n_features=10, noise=0.1)
    datasets.append(("Friedman Non-Linear", fri_X, fri_y))

    results_table = []


    for data_name, X, y in datasets:
        print(f"\nTraitement : {data_name} | Input Shape: {X.shape}")
        

        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        
        

        neuro_opt = NeuroOptimizer(X_scaled, y, task="regression")


        for algo in  NeuroOptimizer.get_available_optimizers():
            print(f"  Optimisation avec {algo}...", end="\r")
            

            
            
            start_global = time.time()

            best_model = neuro_opt.search_model(
                optimizer_name_weights=algo,
                epochs=15,                 
                train_time=60,            
                epochs_weights=30,        
                population_weights=20,    
                time_importance=reg_time_importance,
                verbose=False
            )
            
            train_duration = time.time() - start_global


            best_model.eval()
            with torch.no_grad():
                X_test = neuro_opt.X_test
                y_test = neuro_opt.y_test.numpy() 
                
                start_inf = time.time()
                preds = best_model(X_test).numpy()
                inference_time = time.time() - start_inf
                
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

            print(f"  {algo:<5} | R²: {r2:.4f} | MSE: {mse:.4f} | Train: {train_duration:.1f}s")

            results_table.append({
                "Dataset": data_name,
                "Algorithm": algo,
                "R² Score": r2,      
                "MSE": mse,          
                "Inf Time (ms)": inference_time * 1000
            })

    
    print("\n\n")
    print("="*105)
    print(f"{'DATASET':<25} | {'ALGO':<10} | {'R² SCORE (Max 1.0)':<20} | {'MSE':<15} | {'INF TIME (ms)':<15}")
    print("-" * 105)
    
    for row in results_table:
        
        r2_display = f"{row['R² Score']:.4f}"
        
        print(f"{row['Dataset']:<25} | {row['Algorithm']:<10} | {r2_display:<20} | {row['MSE']:<15.4f} | {row['Inf Time (ms)']:<15.4f}")
    
    print("="*105)

if __name__ == "__main__":
    run_regression_benchmark()