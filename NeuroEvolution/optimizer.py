import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mealpy import FloatVar
from mealpy.swarm_based import GWO, PSO, WOA, ABC, SMO, HHO
from mealpy.bio_based import SMA
from mealpy.evolutionary_based import GA, DE

import random as rd
import copy

from .layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg

class DynamicNet(nn.Module):
    """
    A PyTorch Neural Network module that dynamically builds its architecture 
    based on a provided list of layer configurations.
    """
    def __init__(self, layers_cfg: list):
        super().__init__()
        layers = []
        for cfg in layers_cfg:
            if isinstance(cfg, LinearCfg):
                layers.append(nn.Linear(cfg.in_features, cfg.out_features))
                if cfg.activation:
                    layers.append(cfg.activation())
            elif isinstance(cfg, Conv2dCfg):
                layers.append(nn.Conv2d(cfg.in_channels, cfg.out_channels, 
                                        cfg.kernel_size, cfg.stride, cfg.padding))
                if cfg.activation:
                    layers.append(cfg.activation())
            elif isinstance(cfg, DropoutCfg):
                layers.append(nn.Dropout(p=cfg.p))
            elif isinstance(cfg, FlattenCfg):
                layers.append(nn.Flatten(start_dim=cfg.start_dim))
            elif isinstance(cfg, MaxPool2dCfg):
                layers.append(nn.MaxPool2d(kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding,  ceil_mode=cfg.ceil_mode) )
            elif isinstance(cfg, GlobalAvgPoolCfg):
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def flatten_weights(self, to_numpy=True, device=None):
        vec = parameters_to_vector(self.parameters())
        if to_numpy:
            return vec.detach().cpu().numpy()
        return vec.to(device) if device is not None else vec

    def load_flattened_weights(self, flat_weights):
        if isinstance(flat_weights, np.ndarray):
            flat_weights = torch.as_tensor(flat_weights, dtype=torch.float32)
        
        device = next(self.parameters()).device
        flat_weights = flat_weights.to(device)
        
        try:
            vector_to_parameters(flat_weights, self.parameters())
        except RuntimeError:
            pass

    def evaluate_model(self, X, y, loss_fn=nn.MSELoss(), n_warmup=3, n_runs=20, verbose=False):
        model = self.net
        model.eval()

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        
        if next(model.parameters()).device.type != device:
            model = model.to(device)

        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = model(X)
            loss_value = loss_fn(pred, y).item()

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(X)
            if use_cuda:
                torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(X)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        inference_time = float(np.median(times))

        if verbose:
            print(
                f"Loss: {loss_value:.6f} | "
                f"Inference time (median): {inference_time*1000:.3f} ms | "
                f"Input: {tuple(X.shape)}"
            )

        return loss_value, inference_time


class NeuroOptimizer:
    """
    A controller class that manages data preparation and uses metaheuristic algorithms 
    (or standard Adam) to optimize the weights of a neural network.
    """
    def __init__(self, X, y, Layers=None, task="classification", inference_time=float('inf'), activation=nn.ReLU):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = torch.as_tensor(self.X_train, dtype=torch.float32).to(self.device)
        self.X_test  = torch.as_tensor(self.X_test, dtype=torch.float32).to(self.device)

        self.task = task
        if task == "regression":
            self.output_dim = 1
            self.y_train = torch.as_tensor(self.y_train, dtype=torch.float32).view(-1, 1).to(self.device)
            self.y_test  = torch.as_tensor(self.y_test, dtype=torch.float32).view(-1, 1).to(self.device)
            self.criterion = nn.MSELoss()
        else: 
            self.classes = len(np.unique(y))
            self.output_dim = self.classes
            self.y_train = torch.as_tensor(self.y_train, dtype=torch.long).to(self.device)
            self.y_test  = torch.as_tensor(self.y_test, dtype=torch.long).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            
        self.activation = activation
        self.n_features = X.shape[1]

        if Layers is None:
            if isinstance(self.n_features, int):
                self.Layers = [
                    LinearCfg(self.n_features, 16, nn.ReLU),
                    LinearCfg(16, self.output_dim, None) 
                ]
            else:
                self.Layers = [
                    Conv2dCfg(1, 8, 3, padding=1),
                    FlattenCfg(),
                    LinearCfg(1, self.output_dim, None)
                ]
        else:
            self.Layers = Layers
        
        self.inference_time = inference_time
        
        # Singleton model for fitness function to avoid re-instantiation
        self.shared_model = DynamicNet(layers_cfg=self.Layers)
        self.shared_model.to(self.device)

    @staticmethod
    def print_available_optimizers():
        algos = {
            "Adam": {"name": "Adaptive Moment Estimation", "strength": "Gradient-based (Backprop)."},
            "GWO":  {"name": "Grey Wolf Optimizer", "strength": "Balanced. Good general purpose."},
            "PSO":  {"name": "Particle Swarm Optimization", "strength": "Fast convergence."},
            "DE":   {"name": "Differential Evolution", "strength": "Robust for noisy functions."},
            "WOA":  {"name": "Whale Optimization Algorithm", "strength": "Spiral search escapes local minima."},
            "GA":   {"name": "Genetic Algorithm", "strength": "Classic evolutionary approach."},
            "ABC":  {"name": "Artificial Bee Colony", "strength": "Strong local search."},
            "SMO":  {"name": "Spider Monkey Optimization", "strength": "Wide exploration."},
            "SMA":  {"name": "Slime Mould Algorithm", "strength": "Adaptive weights."},
            "HHO":  {"name": "Harris Hawks Optimization", "strength": "Cooperative chasing."}
        }
        print("\n" + "="*110)
        print(f"{'CODE':<10} | {'FULL NAME':<30} | {'STRENGTHS / BEST USE CASE'}")
        print("="*110)
        for code, info in algos.items():
            print(f"{code:<10} | {info['name']:<30} | {info['strength']}")
        print("="*110 + "\n")

    def evaluate(self, model, verbose=False, time_importance=None):
        if next(model.parameters()).device.type != self.device:
            model = model.to(self.device)
            
        # warmup
        start = time.time()
        with torch.no_grad():
            outputs = model(self.X_test)
        inference_time = time.time() - start

        if self.task == "classification":
            _, predicted = torch.max(outputs.data, 1)
            acc = float(accuracy_score(self.y_test.cpu(), predicted.cpu()))

            if verbose:
                print(f"   [Eval] Acc: {acc*100:.2f}% | Time: {inference_time*1000:.4f}ms")

            if time_importance:
                return time_importance(acc, inference_time)
            return -acc 

        else: # Regression
            mse_loss = self.criterion(outputs, self.y_test).item()

            if verbose:
                print(f"   [Eval] MSE: {mse_loss:.4f} | Time: {inference_time*1000:.4f}ms")

            if time_importance:
                return time_importance(mse_loss, inference_time)
            return mse_loss 

    @staticmethod
    def get_available_optimizers():
        return ["Adam", "GWO", "PSO", "DE", "WOA", "GA", "ABC", "SMO", "SMA", "HHO"]

    def fitness_function(self, solution):
        # Use shared model instead of creating new one
        try:
            self.shared_model.load_flattened_weights(solution)
        except Exception:
            return 9999.0 

        self.shared_model.eval()
        
        # Mini-batching for performance
        batch_size = 1024
        if len(self.X_train) > batch_size:
            indices = torch.randint(0, len(self.X_train), (batch_size,))
            X_batch = self.X_train[indices]
            y_batch = self.y_train[indices]
        else:
            X_batch, y_batch = self.X_train, self.y_train

        with torch.no_grad():
            y_pred = self.shared_model(X_batch) 
            loss = self.criterion(y_pred, y_batch)
        return loss.item()

    def search_weights(self, optimizer_name='GWO', epochs=20, population=30, learning_rate=0.01, verbose=False):
        
        # Ensure shared model architecture matches current config
        self.shared_model = DynamicNet(layers_cfg=self.Layers).to(self.device)

        if optimizer_name == "Adam":
            if verbose: print(f"Starting Gradient Descent (Adam) for {epochs} epochs...")
            model = DynamicNet(layers_cfg=self.Layers).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.train() 
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Batching for Adam as well
                batch_size = 1024
                if len(self.X_train) > batch_size:
                    indices = torch.randint(0, len(self.X_train), (batch_size,))
                    X_batch = self.X_train[indices]
                    y_batch = self.y_train[indices]
                else:
                    X_batch, y_batch = self.X_train, self.y_train
                
                y_pred = model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            if verbose: print(f"Finished! Final Train Loss: {loss.item():.4f}")
            return model 
        
        dummy_model = DynamicNet(layers_cfg=self.Layers).to(self.device)
        n_params = dummy_model.count_parameters()

        if verbose: print(f"Architecture defined. Number of weights to optimize: {n_params}")

        if n_params > 5000:
            print("WARNING: Above 5000 parameters, swarm algorithms converge very poorly.")

        lb = [-1.0] * n_params
        ub = [ 1.0] * n_params

        problem = {
            "obj_func": self.fitness_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min",
            "verbose": False,          
            "log_to": None,            
            "save_population": False,
        }

        term_dict = {
           "max_early_stop": 25 
        }

        if optimizer_name == "GWO":
            model_opt = GWO.RW_GWO(epoch=epochs, pop_size=population)
        elif optimizer_name == "PSO":
            model_opt = PSO.C_PSO(epoch=epochs, pop_size=population)
        elif optimizer_name == "DE":
            model_opt = DE.JADE(epoch=epochs, pop_size=population) 
        elif optimizer_name == "WOA":
            model_opt = WOA.OriginalWOA(epoch=epochs, pop_size=population)
        elif optimizer_name == "GA":
            model_opt = GA.BaseGA(epoch=epochs, pop_size=population)
        elif optimizer_name == "ABC":
            model_opt = ABC.OriginalABC(epoch=epochs, pop_size=population)
        elif optimizer_name == "SMO": 
            model_opt = SMO.DevSMO(epoch=epochs, pop_size=population)
        elif optimizer_name == "SMA": 
            model_opt = SMA.OriginalSMA(epoch=epochs, pop_size=population)
        elif optimizer_name == "HHO": 
            model_opt = HHO.OriginalHHO(epoch=epochs, pop_size=population)
        else:
            print(f"Algorithm {optimizer_name} unknown. Fallback to GWO.")
            model_opt = GWO.OriginalGWO(epoch=epochs, pop_size=population)

        if verbose: print(f"Starting Neuro-evolution ({optimizer_name})...")
        best_agent = model_opt.solve(problem, termination=term_dict)

        best_position = best_agent.solution
        best_fitness = best_agent.target.fitness

        if verbose: print(f"Finished! Best Train Loss: {best_fitness:.4f}")

        dummy_model.load_flattened_weights(best_position)
        return dummy_model

    def _reconnect_layers(self, layers):
        new_layers = []
        
        if isinstance(self.n_features, tuple) or isinstance(self.n_features, list):
            dummy_input = torch.zeros(1, *self.n_features)
        else:
            dummy_input = torch.zeros(1, self.n_features)
            
        # Helper to avoid connecting layers that don't match dimensions
        try:
            for i, cfg in enumerate(layers):
                if isinstance(cfg, Conv2dCfg):
                    cfg.in_channels = dummy_input.shape[1] 
                    layer = nn.Conv2d(cfg.in_channels, cfg.out_channels, 
                                      cfg.kernel_size, cfg.stride, cfg.padding)
                    dummy_input = layer(dummy_input)
                    new_layers.append(cfg)

                elif isinstance(cfg, LinearCfg):
                    if len(dummy_input.shape) > 2:
                        flat_cfg = FlattenCfg()
                        dummy_input = torch.flatten(dummy_input, 1)
                        new_layers.append(flat_cfg)

                    cfg.in_features = dummy_input.shape[1]
                    layer = nn.Linear(cfg.in_features, cfg.out_features)
                    dummy_input = layer(dummy_input)
                    new_layers.append(cfg)

                elif isinstance(cfg, FlattenCfg):
                    dummy_input = torch.flatten(dummy_input, 1)
                    new_layers.append(cfg)

                elif isinstance(cfg, DropoutCfg):
                    new_layers.append(cfg)

                elif isinstance(cfg, MaxPool2dCfg):
                    layer = nn.MaxPool2d(kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding, ceil_mode=cfg.ceil_mode)
                    dummy_input = layer(dummy_input)
                    new_layers.append(cfg)
                
                elif isinstance(cfg, GlobalAvgPoolCfg):
                    if dummy_input.ndim < 4: continue 
                    layer = nn.AdaptiveAvgPool2d((1, 1))
                    dummy_input = layer(dummy_input)
                    dummy_input = torch.flatten(dummy_input, 1) 
                    new_layers.append(cfg)
        except Exception:
            pass # Skip invalid layer configurations

        return new_layers

    def hybrid_search(self, train_time=float("inf"), optimizers=['Adam'], epochs=[10],
                      populations=20, learning_rate=0.01, verbose=False):
        if len(optimizers) != len(epochs):
            print('ERROR : optimizers and epochs not same length')
            return
        
        current_model = DynamicNet(layers_cfg=self.Layers).to(self.device)
        self.shared_model = current_model # Sync shared model

        for i in range(len(optimizers)):
            optimizer_name = optimizers[i]
            ep = epochs[i]
            
            if optimizer_name == "Adam":
                if verbose: print(f"Starting Gradient Descent (Adam) for {ep} epochs...")
                optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
                current_model.train() 
                for epoch in range(ep):
                    optimizer.zero_grad()
                    
                    batch_size = 1024
                    if len(self.X_train) > batch_size:
                        indices = torch.randint(0, len(self.X_train), (batch_size,))
                        X_batch = self.X_train[indices]
                        y_batch = self.y_train[indices]
                    else:
                        X_batch, y_batch = self.X_train, self.y_train
                        
                    y_pred = current_model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
            else:
                n_params = current_model.count_parameters()
                lb = [-1.0] * n_params
                ub = [ 1.0] * n_params
                
                problem = {
                    "obj_func": self.fitness_function,
                    "bounds": FloatVar(lb=lb, ub=ub),
                    "minmax": "min",
                    "verbose": False,          
                    "log_to": None,            
                    "save_population": False,
                }
                
                # Update shared model with current weights before starting swarm
                self.shared_model.load_state_dict(current_model.state_dict())
                
                if optimizer_name == "GWO":
                    model_opt = GWO.RW_GWO(epoch=ep, pop_size=populations)
                elif optimizer_name == "PSO":
                    model_opt = PSO.C_PSO(epoch=ep, pop_size=populations)
                elif optimizer_name == "DE":
                    model_opt = DE.JADE(epoch=ep, pop_size=populations) 
                elif optimizer_name == "WOA":
                    model_opt = WOA.OriginalWOA(epoch=ep, pop_size=populations)
                elif optimizer_name == "GA":
                    model_opt = GA.BaseGA(epoch=ep, pop_size=populations)
                elif optimizer_name == "ABC":
                    model_opt = ABC.OriginalABC(epoch=ep, pop_size=populations)
                elif optimizer_name == "SMO": 
                    model_opt = SMO.DevSMO(epoch=ep, pop_size=populations)
                elif optimizer_name == "SMA": 
                    model_opt = SMA.OriginalSMA(epoch=ep, pop_size=populations)
                elif optimizer_name == "HHO": 
                    model_opt = HHO.OriginalHHO(epoch=ep, pop_size=populations)
                else:
                    model_opt = GWO.OriginalGWO(epoch=ep, pop_size=populations)

                best_agent = model_opt.solve(problem)
                current_model.load_flattened_weights(best_agent.solution)
                if verbose: print(f"   [{optimizer_name}] Best Fitness: {best_agent.target.fitness:.4f}")

        return current_model

    def search_model(self, epochs=10, train_time=300, optimizer_name_weights='GWO', accuracy_target=0.99, hybrid=[], hybrid_epochs=[],
                     epochs_weights=10, population_weights=20, learning_rate_weights=0.01,
                     verbose=False, verbose_weights=False, time_importance=None):

        START = time.time()
        if verbose: print(f"\n Démarrage de la recherche d'architecture (NAS)...")

        if verbose: print("  -> Évaluation de l'architecture de départ...")

        if len(hybrid) > 0:
             start_model = self.hybrid_search(
                 optimizers=hybrid, 
                 epochs=hybrid_epochs, 
                 populations=population_weights,
                 learning_rate=learning_rate_weights, 
                 verbose=verbose_weights
             )
        else:
             start_model = self.search_weights(
                 optimizer_name=optimizer_name_weights, 
                 epochs=epochs_weights, 
                 population=population_weights,
                 learning_rate=learning_rate_weights,
                 verbose=verbose_weights
             )

        best_score = self.evaluate(start_model , time_importance=time_importance)
        best_model = start_model
        best_layers = copy.deepcopy(self.Layers) 

        if verbose: print(f"  -> Score initial : {best_score:.4f}")

        new_layers = copy.deepcopy(self.Layers) 
        ITER = 0

        while ITER < epochs and (time.time() - START) < train_time and best_score > -accuracy_target:
            ITER += 1
            if verbose: print(f"\n[NAS Iteration {ITER}/{epochs}] Tentative de mutation...")

            if rd.random() < 0.6: new_layers = copy.deepcopy(best_layers)
            else: new_layers = copy.deepcopy(new_layers) 

            mutation_type = rd.choice(["change_neurons", "add_layer", "remove_layer"])
            modifiable_indices = [i for i, l in enumerate(new_layers[:-1]) if isinstance(l, (LinearCfg, Conv2dCfg))]

            if not modifiable_indices and mutation_type != "add_layer": continue

            mutated = False
            if mutation_type == "change_neurons" and modifiable_indices:
                idx = rd.choice(modifiable_indices)
                layer = new_layers[idx]
                if isinstance(layer, LinearCfg):
                    noise = rd.randint(-16, 16)
                    new_val = max(4, layer.out_features + noise)
                    if new_val != layer.out_features:
                        new_layers[idx].out_features = new_val
                        if verbose: print(f"  Action: Linear {idx} -> {new_val} neurones")
                        mutated = True
                elif isinstance(layer, Conv2dCfg):
                    if rd.random() < 0.5:
                        noise = rd.choice([-2, 2]) 
                        new_k = max(1, layer.kernel_size + noise)
                        if new_k != layer.kernel_size:
                            new_layers[idx].kernel_size = new_k
                            new_layers[idx].padding = new_k // 2 
                            if verbose: print(f"  Action: Conv {idx} -> Kernel {new_k}x{new_k}")
                            mutated = True
                    else:
                        noise = rd.choice([-8, 8, 16]) 
                        new_ch = max(4, layer.out_channels + noise)
                        if new_ch != layer.out_channels:
                            new_layers[idx].out_channels = new_ch
                            if verbose: print(f"  Action: Conv {idx} -> {new_ch} Channels")
                            mutated = True

            elif mutation_type == "add_layer":
                insert_idx = rd.randint(0, len(new_layers) - 1)
                if insert_idx < len(new_layers) and isinstance(new_layers[insert_idx], Conv2dCfg):
                    new_layer = copy.copy(new_layers[insert_idx])
                else:
                    new_layer = LinearCfg(in_features=1, out_features=32, activation=self.activation)
                new_layers.insert(insert_idx, new_layer)
                if verbose: print(f"  Action: Ajout couche à l'index {insert_idx}")
                mutated = True

            elif mutation_type == "remove_layer" and len(modifiable_indices) > 1:
                idx = rd.choice(modifiable_indices)
                del new_layers[idx]
                if verbose: print(f"  Action: Suppression de la couche {idx}")
                mutated = True

            if not mutated: continue

            new_layers = self._reconnect_layers(new_layers)
            
            temp_optimizer = NeuroOptimizer(self.X_train.cpu().numpy(), self.y_train.cpu().numpy(), 
                                            Layers=new_layers, task=self.task)
            
            # Explicitly set device for temp optimizer
            temp_optimizer.device = self.device
            temp_optimizer.X_train = temp_optimizer.X_train.to(self.device)
            temp_optimizer.y_train = temp_optimizer.y_train.to(self.device)
            
            try:
                if len(hybrid) > 0:
                     temp_model = temp_optimizer.hybrid_search(
                         optimizers=hybrid, epochs=hybrid_epochs, 
                         populations=population_weights, 
                         learning_rate=learning_rate_weights, 
                         verbose=verbose_weights
                     )
                else:
                     temp_model = temp_optimizer.search_weights(
                         optimizer_name=optimizer_name_weights, 
                         epochs=epochs_weights, 
                         population=population_weights,
                         learning_rate=learning_rate_weights, 
                         verbose=verbose_weights
                     )

                new_score = self.evaluate(temp_model, time_importance=time_importance)
                if verbose: print(f"  -> Nouveau Score : {new_score:.4f} (Best: {best_score:.4f})")

                if new_score < best_score:
                    if verbose: print(" AMÉLIORATION !")
                    best_score = new_score
                    best_model = temp_model
                    best_layers = new_layers
                    self.Layers = best_layers
                else:
                    if verbose: print(" Rejeté.")

            except Exception as e:
                if verbose: print(f"   Crash architecture : {e}")

        print(f"\nFin du NAS. Meilleur Score : {best_score:.4f}")
        return best_model