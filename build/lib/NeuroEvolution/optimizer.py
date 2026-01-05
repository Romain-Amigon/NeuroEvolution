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

from .layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg


class ResidualWrapper(nn.Module):
    def __init__(self, sub_layers_module, use_projection=False, in_channels=0, out_channels=0):
        super().__init__()
        self.net = sub_layers_module
        self.use_projection = use_projection
        self.projection = None
        

        if use_projection and in_channels != out_channels:

            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif in_channels != out_channels:

             pass

    def forward(self, x):
        identity = x
        out = self.net(x)
        
        if self.projection is not None:
            identity = self.projection(identity)

        if identity.shape != out.shape:
            return out 
            
        return out + identity

class DynamicNet(nn.Module):
    """
    A PyTorch Neural Network module that dynamically builds its architecture 
    based on a provided list of layer configurations.

    This class serves as a flexible wrapper to instantiate models with varying
    structures (Linear, Conv2d, Pooling, etc.) on the fly, which is essential
    for Neural Architecture Search (NAS).

    Args:
        layers_cfg (list): A list of configuration objects (e.g., LinearCfg, Conv2dCfg)
                           defining the sequence of layers.
    """
    def __init__(self,layers_cfg: list, input_shape: tuple = None):
        super().__init__()
        if input_shape is not None:
            self.layers_cfg = self._reconnect_layers(layers_cfg, input_shape)
        else:
            self.layers_cfg = layers_cfg

        self.net = self._build_sequential(self.layers_cfg)
        
    def _build_sequential(self, cfgs):
        layers = []
        for cfg in cfgs:
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
            elif isinstance(cfg, BatchNorm1dCfg):
                layers.append(nn.BatchNorm1d(cfg.num_features))
                
            elif isinstance(cfg, BatchNorm2dCfg):
                layers.append(nn.BatchNorm2d(cfg.num_features))

            elif isinstance(cfg, ResBlockCfg):
                inner_seq = self._build_sequential(cfg.sub_layers)
                

                in_ch = 0
                out_ch = 0
                if len(cfg.sub_layers) > 0:
                    first = cfg.sub_layers[0]
                    last = cfg.sub_layers[-1]
                    if hasattr(first, 'in_channels'): in_ch = first.in_channels
                    elif hasattr(first, 'in_features'): in_ch = first.in_features 
                    
                    if hasattr(last, 'out_channels'): out_ch = last.out_channels
                    elif hasattr(last, 'out_features'): out_ch = last.out_features

                wrapper = ResidualWrapper(inner_seq, cfg.use_projection, in_ch, out_ch)
                layers.append(wrapper)
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the sequential network.
        """
        return self.net(x)

    def count_parameters(self):
        """
        Counts the total number of trainable parameters in the network.

        Returns:
            int: The total number of elements in all parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def flatten_weights(self, to_numpy=True, device=None):
        """
        Flattens all network parameters into a single vector.

        Args:
            to_numpy (bool, optional): If True, returns a NumPy array. If False,
                                       returns a PyTorch tensor. Defaults to True.
            device (torch.device, optional): The target device for the tensor if
                                             returning a tensor. Defaults to None.

        Returns:
            np.ndarray or torch.Tensor: A 1D array/tensor containing all model weights.
        """
        vec = parameters_to_vector(self.parameters())
        if to_numpy:
            return vec.detach().cpu().numpy()
        return vec.to(device) if device is not None else vec

    def load_flattened_weights(self, flat_weights):
        """
        Loads a flat vector of weights into the network's parameters.

        Args:
            flat_weights (np.ndarray or torch.Tensor): A 1D array or tensor representing
                                                       the weights to load.
        """
        if isinstance(flat_weights, np.ndarray):
            flat_weights = torch.as_tensor(flat_weights, dtype=torch.float32)
        
        device = next(self.parameters()).device
        flat_weights = flat_weights.to(device)
        
        try:
            vector_to_parameters(flat_weights, self.parameters())
        except RuntimeError:
            pass

    def evaluate_model(self, X, y, loss_fn=nn.MSELoss(), n_warmup=3, n_runs=20, verbose=False):
        """
        Evaluates the model on a given dataset, measuring loss and inference latency.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target labels or values.
            loss_fn (nn.Module, optional): The loss function to use. Defaults to nn.MSELoss().
            n_warmup (int, optional): Number of warm-up runs for timing. Defaults to 3.
            n_runs (int, optional): Number of runs to calculate median inference time. Defaults to 20.
            verbose (bool, optional): If True, prints evaluation results. Defaults to False.

        Returns:
            tuple: A tuple containing (loss_value, inference_time_in_seconds).
        """
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
    
    def _reconnect_layers(self, layers, input_shape):
        
        dummy_input = torch.zeros(1, *input_shape)
        
        def process_recursive(cfg_list, current_tensor):
            processed = []
            x = current_tensor
            
            for original_cfg in cfg_list:

                import copy
                cfg = copy.deepcopy(original_cfg)
                
                try:
                    if isinstance(cfg, Conv2dCfg):
                        cfg.in_channels = x.shape[1]
                        layer = nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, BatchNorm2dCfg):
                        cfg.num_features = x.shape[1]
                        layer = nn.BatchNorm2d(cfg.num_features)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, LinearCfg):
                        if len(x.shape) > 2:
                            flat_cfg = FlattenCfg()
                            processed.append(flat_cfg)
                            x = torch.flatten(x, 1)
                        
                        cfg.in_features = x.shape[1]
                        layer = nn.Linear(cfg.in_features, cfg.out_features)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, ResBlockCfg):
 
                        inner_cfgs, inner_out = process_recursive(cfg.sub_layers, x)
                        cfg.sub_layers = inner_cfgs
                        
                        x = inner_out
                        processed.append(cfg)
                    
                    elif isinstance(cfg, GlobalAvgPoolCfg):
                        x = nn.AdaptiveAvgPool2d((1, 1))(x)
                        x = torch.flatten(x, 1)
                        processed.append(cfg)
                    
                    elif isinstance(cfg, FlattenCfg):
                        x = torch.flatten(x, cfg.start_dim)
                        processed.append(cfg)

                    else:
                        processed.append(cfg)
                        
                except Exception as e:
                    print(f"Warning:{cfg}: {e}")
                    pass
            
            return processed, x

        new_layers, _ = process_recursive(layers, dummy_input)
        return new_layers


class NeuroOptimizer:
    """
    A controller class that manages data preparation and uses metaheuristic algorithms 
    (or standard Adam) to optimize the weights of a neural network and search for
    optimal architectures (Neuro-evolution / NAS).

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels or values.
        Layers (list, optional): Initial list of layer configurations. Defaults to None.
        task (str, optional): The type of task, either 'classification' or 'regression'. 
                              Defaults to "classification".
        inference_time (float, optional): Maximum allowed inference time constraint. 
                                          Defaults to infinity.
        activation (nn.Module, optional): Default activation function class. Defaults to nn.ReLU.
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
        """
        Prints a table of available optimization algorithms and their strengths.
        """
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
        """
        Evaluates the model performance on the test set.

        Args:
            model (DynamicNet): The neural network model to evaluate.
            verbose (bool, optional): If True, prints accuracy/loss and time. Defaults to False.
            time_importance (callable, optional): A function that weighs the trade-off between 
                                                  performance and inference time. Defaults to None.

        Returns:
            float: The evaluation score. For classification, returns negative accuracy (for minimization) 
                   unless time_importance is used. For regression, returns MSE.
        """
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
        """
        Returns a list of available optimizer codes.

        Returns:
            list: List of strings representing optimizer codes.
        """
        return ["Adam", "GWO", "PSO", "DE", "WOA", "GA", "ABC", "SMO", "SMA", "HHO"]

    def fitness_function(self, solution):
        """
        Calculates the fitness of a specific set of weights (solution).
        
        This method is used by the swarm intelligence algorithms. It loads weights
        into the shared model and calculates loss on a mini-batch.

        Args:
            solution (np.ndarray): The flat vector of weights to evaluate.

        Returns:
            float: The loss value (fitness) to be minimized.
        """
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
        """
        Optimizes the weights of the current network architecture using a specified algorithm.

        Args:
            optimizer_name (str, optional): The name of the optimization algorithm ('Adam', 'GWO', etc.). 
                                            Defaults to 'GWO'.
            epochs (int, optional): Number of iterations/epochs. Defaults to 20.
            population (int, optional): Population size for swarm algorithms. Defaults to 30.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.01.
            verbose (bool, optional): If True, prints progress. Defaults to False.

        Returns:
            DynamicNet: The model with optimized weights.
        """
        
        self.shared_model = DynamicNet(layers_cfg=self.Layers).to(self.device)

        if optimizer_name == "Adam":
            if verbose: print(f"Starting Gradient Descent (Adam) for {epochs} epochs...")
            model = DynamicNet(layers_cfg=self.Layers).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model.train() 
            for epoch in range(epochs):
                optimizer.zero_grad()
                
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
        """
        Re-calculates input/output dimensions for a list of layers to ensure connectivity.

        This method performs a forward pass with dummy data to dynamically determine
        tensor shapes (e.g., after a Conv2d or Flatten layer) and updates the
        configuration objects accordingly.

        Args:
            layers (list): A list of layer configuration objects.

        Returns:
            list: A new list of layer configurations with corrected dimensions.
        """
        new_layers = []
        
        if isinstance(self.n_features, (tuple, list)):
            dummy_input = torch.zeros(1, *self.n_features)
        else:
            dummy_input = torch.zeros(1, self.n_features)
            
        def process_list(cfg_list, current_input):
            processed_list = []
            local_input = current_input
            
            for cfg in cfg_list:
                try:
                    if isinstance(cfg, BatchNorm1dCfg):
                        cfg.num_features = local_input.shape[1]
                        layer = nn.BatchNorm1d(cfg.num_features)
                        local_input = layer(local_input)
                        processed_list.append(cfg)

                    elif isinstance(cfg, BatchNorm2dCfg):
                        cfg.num_features = local_input.shape[1]
                        layer = nn.BatchNorm2d(cfg.num_features)
                        local_input = layer(local_input)
                        processed_list.append(cfg)

                    elif isinstance(cfg, ResBlockCfg):

                        inner_cfgs, inner_out = process_list(cfg.sub_layers, local_input)
                        cfg.sub_layers = inner_cfgs
                        
                        local_input = inner_out 
                        
                        processed_list.append(cfg)


                    elif isinstance(cfg, Conv2dCfg):
                        cfg.in_channels = local_input.shape[1]
                        layer = nn.Conv2d(cfg.in_channels, cfg.out_channels, 
                                          cfg.kernel_size, cfg.stride, cfg.padding)
                        local_input = layer(local_input)
                        processed_list.append(cfg)

                    elif isinstance(cfg, LinearCfg):
                        if len(local_input.shape) > 2:

                            flat = FlattenCfg()
                            local_input = torch.flatten(local_input, 1)
                            processed_list.append(flat)
                        
                        cfg.in_features = local_input.shape[1]
                        layer = nn.Linear(cfg.in_features, cfg.out_features)
                        local_input = layer(local_input)
                        processed_list.append(cfg)

                    elif isinstance(cfg, (FlattenCfg, MaxPool2dCfg, GlobalAvgPoolCfg, DropoutCfg)):
                        if isinstance(cfg, FlattenCfg): l = nn.Flatten(start_dim=cfg.start_dim)
                        elif isinstance(cfg, MaxPool2dCfg): l = nn.MaxPool2d(cfg.kernel_size, cfg.stride, cfg.padding, cfg.dilation, cfg.ceil_mode)
                        elif isinstance(cfg, GlobalAvgPoolCfg): l = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
                        elif isinstance(cfg, DropoutCfg): l = nn.Dropout(cfg.p)
                        
                        local_input = l(local_input)
                        processed_list.append(cfg)

                except Exception as e:
                    pass
            
            return processed_list, local_input

        # Lancement du processus
        new_layers, _ = process_list(layers, dummy_input)
        return new_layers

    def hybrid_search(self, train_time=float("inf"), optimizers=['Adam'], epochs=[10], 
                      populations=20, learning_rate=0.01, verbose=False):
        """
        Performs a hybrid weight optimization using a sequence of different algorithms.

        Args:
            train_time (float, optional): Max training time (unused in current logic but reserved). 
                                          Defaults to infinity.
            optimizers (list, optional): List of optimizer names to run sequentially. 
                                         Defaults to ['Adam'].
            epochs (list, optional): List of epochs corresponding to each optimizer. 
                                     Defaults to [10].
            populations (int, optional): Population size for swarm algorithms. Defaults to 20.
            learning_rate (float, optional): Learning rate for Adam. Defaults to 0.01.
            verbose (bool, optional): If True, prints progress. Defaults to False.

        Returns:
            DynamicNet: The model optimized by the sequence of algorithms.
        """
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

    def search_model(self, epochs=10, train_time=float('inf'), optimizer_name_weights='GWO', accuracy_target=1, hybrid=[], hybrid_epochs=[], 
                     epochs_weights=10, population_weights=20, learning_rate_weights=0.01, 
                     verbose=False, verbose_weights=False, time_importance=None):
        """
        Executes the Neural Architecture Search (NAS) loop using hill-climbing mutations.

        This method iteratively mutates the network structure (changing neurons, adding/removing layers),
        re-trains weights, and evaluates performance to find the best architecture.

        Args:
            epochs (int, optional): Maximum number of NAS iterations. Defaults to 10.
            train_time (int, optional): Maximum total runtime in seconds. Defaults to 300.
            optimizer_name_weights (str, optional): Algorithm to optimize weights during NAS steps. 
                                                    Defaults to 'GWO'.
            accuracy_target (float, optional): Target accuracy to stop search early. Defaults to 0.99.
            hybrid (list, optional): List of optimizers for hybrid weight training. Defaults to [].
            hybrid_epochs (list, optional): Epochs for hybrid training. Defaults to [].
            epochs_weights (int, optional): Epochs for single optimizer training. Defaults to 10.
            population_weights (int, optional): Population size for swarm weight optimization. 
                                                Defaults to 20.
            learning_rate_weights (float, optional): Learning rate for gradient descent steps. 
                                                     Defaults to 0.01.
            verbose (bool, optional): If True, prints NAS progress. Defaults to False.
            verbose_weights (bool, optional): If True, prints weight optimization details. 
                                              Defaults to False.
            time_importance (callable, optional): Metric to weight accuracy vs latency. Defaults to None.

        Returns:
            DynamicNet: The best found model architecture and weights.
        """

        START = time.time()
        if verbose: print(f"\n Start model search (NAS)...")

        if verbose: print("  -> Evaluating initial architecture...")

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

        if verbose: print(f"  -> Score : {best_score:.4f}")

        new_layers = copy.deepcopy(self.Layers) 
        ITER = 0

        while ITER < epochs and (time.time() - START) < train_time and best_score > -accuracy_target:
            ITER += 1
            if verbose: print(f"\n[NAS Iteration {ITER}/{epochs}] Attempting mutation...")

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
                        if verbose: print(f"  Action: Linear {idx} -> {new_val} neurons")
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
                if len(new_layers) > 0:
                    insert_idx = rd.randint(0, len(new_layers) - 1)
                else:
                    insert_idx = 0
                if insert_idx < len(new_layers) and isinstance(new_layers[insert_idx], Conv2dCfg):
                    new_layer = copy.copy(new_layers[insert_idx])
                else:
                    new_layer = LinearCfg(in_features=1, out_features=32, activation=self.activation)
                new_layers.insert(insert_idx, new_layer)
                if verbose: print(f"  Action: Adding layer at index {insert_idx}")
                mutated = True

            elif mutation_type == "remove_layer" and len(modifiable_indices) > 1:
                idx = rd.choice(modifiable_indices)
                del new_layers[idx]
                if verbose: print(f"  Action: Removing layer {idx}")
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
                if verbose: print(f"  -> New Score : {new_score:.4f} (Best: {best_score:.4f})")

                if new_score < best_score:
                    if verbose: print(" IMPROVEMENT !")
                    best_score = new_score
                    best_model = temp_model
                    best_layers = new_layers
                    self.Layers = best_layers
                else:
                    if verbose: print(" Rejected.")

            except Exception as e:
                if verbose: print(f"   Architecture crash : {e}")

        print(f"\nNAS Finished. Best Score : {best_score:.4f}")
        return best_model
    

    def _transfer_weights(self,parent_model, child_model):
        """
        Transfers the parent's weight to the child
        """
        parent_state = parent_model.state_dict()
        child_state = child_model.state_dict()
        
        new_state = {}
        for name, param in child_state.items():
            if name in parent_state:
                if parent_state[name].shape == param.shape:
                    new_state[name] = parent_state[name]
                else:

                    new_state[name] = param
            else:
                new_state[name] = param
                
        child_model.load_state_dict(new_state)
        return child_model
    
    def _mutate_layers_config(self, current_layers):

        new_layers = copy.deepcopy(current_layers)
        
        mutation_type = rd.choice(["change_neurons", "add_layer", "remove_layer"])
        modifiable_indices = [i for i, l in enumerate(new_layers[:-1]) if isinstance(l, (LinearCfg, Conv2dCfg))]

        if not modifiable_indices and mutation_type != "add_layer":
            return new_layers

        if mutation_type == "change_neurons" and modifiable_indices:
            idx = rd.choice(modifiable_indices)
            layer = new_layers[idx]
            
            if isinstance(layer, LinearCfg):
                noise = rd.randint(-16, 16)
                new_val = max(4, layer.out_features + noise)
                layer.out_features = new_val
                
            elif isinstance(layer, Conv2dCfg):
                if rd.random() < 0.5: # Changer Kernel
                    noise = rd.choice([-2, 2]) 
                    new_k = max(1, layer.kernel_size + noise)
                    layer.kernel_size = new_k
                    layer.padding = new_k // 2 
                else: # Changer Channels
                    noise = rd.choice([-8, 8, 16]) 
                    new_ch = max(4, layer.out_channels + noise)
                    layer.out_channels = new_ch

        elif mutation_type == "add_layer":
            insert_idx = rd.randint(0, len(new_layers) - 1) if new_layers else 0
            
            if insert_idx < len(new_layers) and isinstance(new_layers[insert_idx], Conv2dCfg):
                new_layer = copy.copy(new_layers[insert_idx])
            else:
                new_layer = LinearCfg(in_features=1, out_features=32, activation=self.activation)
            
            new_layers.insert(insert_idx, new_layer)

        elif mutation_type == "remove_layer" and len(modifiable_indices) > 1:
            idx = rd.choice(modifiable_indices)
            del new_layers[idx]

        return new_layers
    
    def search_model_evolutionary(self, epochs=10, train_time=float('inf'), population_size=10,
                                  optimizer_name_weights='Adam', accuracy_target=1.0, 
                                  hybrid=[], hybrid_epochs=[], epochs_weights=10, population_weights=20, 
                                  learning_rate_weights=0.01, verbose=False, verbose_weights=False, 
                                  time_importance=None):
        """
        Executes a Neural Architecture Search (NAS) using a Genetic Algorithm.
        
        This method evolves a population of neural network architectures over several generations.
        For each architecture candidate, it creates a temporary optimizer instance to train 
        and evaluate its weights (using either standard Adam or Swarm Intelligence).
        It employs elitism and weight inheritance to accelerate convergence.

        Args:
            epochs (int): Number of NAS generations (iterations of the genetic algorithm).
            train_time (int): Maximum total execution time in seconds (safety stop).
            optimizer_name_weights (str): The optimizer used for weight training (e.g., 'Adam', 'GWO').
            accuracy_target (float): The target accuracy (0.0 to 1.0) to stop the search early if reached.
            hybrid (list[str]): List of optimizers for hybrid training sequence (e.g., ['Adam', 'GWO']).
            hybrid_epochs (list[int]): Corresponding epochs for each optimizer in the hybrid list.
            epochs_weights (int): Number of training epochs for each candidate model (Proxy Task).
            population_weights (int): Size of the population (number of candidate models).
            learning_rate_weights (float): Learning rate for the weight optimizer.
            verbose (bool): If True, prints high-level NAS progress.
            verbose_weights (bool): If True, prints detailed weight training logs.
            time_importance (callable, optional): A function to weigh performance vs inference time.

        Returns:
            DynamicNet: The best found model architecture with optimized weights.
        """
        
        START_TIME = time.time()
        
        if verbose:
            print(f"\n[NAS-Evolutionary] Start config:")
            print(f"  -> Generations (epochs): {epochs}")
            print(f"  -> Population Size: {population_weights}")
            print(f"  -> Training per model (epochs_weights): {epochs_weights}")
            print(f"  -> Time Limit: {train_time}s")

        population = []
        for _ in range(population_size):
            initial_layers = self._reconnect_layers(copy.deepcopy(self.Layers))
            model = DynamicNet(initial_layers, input_shape=self.X_train.shape[1:]).to(self.device)
            population.append(model)

        best_global_model = None
        best_global_score = -float('inf')

        for gen in range(epochs):
            if (time.time() - START_TIME) > train_time:
                if verbose: print(f"\n[NAS-Evolutionary] Time limit reached ({train_time}s). Stopping.")
                break

            if verbose: print(f"\n--- Generation {gen+1}/{epochs} ---")
            
            scores = []
            trained_population = [] 

            for i, model in enumerate(population):
                if (time.time() - START_TIME) > train_time: break


                current_layers_cfg = model.layers_cfg


                temp_optimizer = NeuroOptimizer(self.X_train.cpu().numpy(), self.y_train.cpu().numpy(), 
                                                Layers=current_layers_cfg, task=self.task)
                
                temp_optimizer.device = self.device
                temp_optimizer.X_train = temp_optimizer.X_train.to(self.device)
                temp_optimizer.y_train = temp_optimizer.y_train.to(self.device)
                

                try:
                    temp_optimizer.shared_model.load_state_dict(model.state_dict())
                except Exception:
                    pass 


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

                score = self.evaluate(temp_model, time_importance=time_importance)
                scores.append(score)
                trained_population.append(temp_model)
                
                if score > best_global_score:
                    best_global_score = score
                    best_global_model = copy.deepcopy(temp_model)
                    if verbose: print(f"  [Gen {gen+1} - Model {i}] New Best Score: {score:.4f}")

            if abs(best_global_score) >= accuracy_target: 
                 if self.task == "classification" and time_importance is None: 
                     if verbose: print(f"\n[NAS-Evolutionary] Target accuracy reached ({accuracy_target}). Stopping.")
                     break
            
            if (time.time() - START_TIME) > train_time: break

            sorted_indices = np.argsort(scores)[::-1]
            
            elites_indices = sorted_indices[:2]
            elites = [trained_population[i] for i in elites_indices]
            
            if verbose and len(scores) > 0: 
                top_score = scores[sorted_indices[0]]
                print(f"  Best of Gen {gen+1}: {top_score:.4f}")

            new_population = []
            
            for elite in elites:
                new_population.append(copy.deepcopy(elite))
            
            while len(new_population) < population_weights:
                if len(sorted_indices) > 0:
                    parent_idx = rd.choice(sorted_indices[:max(1, len(sorted_indices)//2)])
                    parent_model = trained_population[parent_idx]
                else:
                    parent_model = trained_population[0]

                child_layers_cfg = self._mutate_layers_config(parent_model.layers_cfg)
                
                try:
                    child_layers_cfg = self._reconnect_layers(child_layers_cfg) 
                except Exception:
                    child_layers_cfg = copy.deepcopy(parent_model.layers_cfg)

                child_model = DynamicNet(child_layers_cfg, input_shape=self.X_train.shape[1:]).to(self.device)
                
                child_model = self._transfer_weights(parent_model, child_model)
                
                new_population.append(child_model)
            
            population = new_population

        if verbose: print(f"\n[NAS-Evolutionary] Finished. Best Global Score: {best_global_score:.4f}")
        
        if best_global_model is not None:
            self.Layers = best_global_model.layers_cfg
            return best_global_model
        else:
            return population[0]