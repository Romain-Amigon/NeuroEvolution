import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mealpy import FloatVar

from mealpy.swarm_based import GWO, PSO, WOA, ABC, SMO, HHO, SSA

import random as rd

from mealpy.bio_based import SMA 
from mealpy.evolutionary_based import GA, DE
from mealpy.physics_based import SA

from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg

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

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def flatten_weights(self):
        return torch.cat([p.detach().flatten() for p in self.parameters()])

    def load_flattened_weights(self, flat_weights_numpy):
        flat_weights = torch.as_tensor(flat_weights_numpy, dtype=torch.float32)
        idx = 0
        with torch.no_grad(): 
            for p in self.parameters():
                num = p.numel()
                if idx + num > len(flat_weights):
                    raise ValueError("Weight vector is too short!")
                
                block = flat_weights[idx : idx + num]
                p.data.copy_(block.view_as(p))
                idx += num

class NeuroOptimizer:
    """
    A controller class that manages data preparation and uses metaheuristic algorithms 
    (or standard Adam) to optimize the weights of a neural network.
    """
    def __init__(self, X, y, Layers=None, task="classification", inference_time=float('inf')):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test  = torch.tensor(self.X_test, dtype=torch.float32)
        
        self.task = task
        if task == "regression":
            self.output_dim = 1
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
            self.y_test  = torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(1)
            self.criterion = nn.MSELoss()
        else: 
            self.classes = len(np.unique(y))
            self.output_dim = self.classes
            self.y_train = torch.tensor(self.y_train, dtype=torch.long)
            self.y_test  = torch.tensor(self.y_test, dtype=torch.long)
            self.criterion = nn.CrossEntropyLoss()

        self.n_features = X.shape[1]

        if Layers is None:
            self.Layers = [
                LinearCfg(self.n_features, 16, nn.ReLU),
                LinearCfg(16, self.output_dim, None) 
            ]
        else:
            self.Layers = Layers
            
        self.inference_time=inference_time

    @staticmethod
    def print_available_optimizers():
        """
        Prints a catalog of all available metaheuristic algorithms supported by this library.
        """
        algos = {
            "Adam": {"name": "Adaptive Moment Estimation", "strength": "Gradient-based (Backprop). The industry standard baseline."},
            "GWO":  {"name": "Grey Wolf Optimizer", "strength": "Balanced. Good general purpose."},
            "PSO":  {"name": "Particle Swarm Optimization", "strength": "Fast convergence. Good for simple landscapes."},
            "DE":   {"name": "Differential Evolution", "strength": "Robust. Excellent for complex/noisy functions."},
            "WOA":  {"name": "Whale Optimization Algorithm", "strength": "Spiral search helps escape local minima."},
            "GA":   {"name": "Genetic Algorithm", "strength": "Classic evolutionary approach. Very robust."},
            "ABC":  {"name": "Artificial Bee Colony", "strength": "Strong local search (exploitation)."},
            "SMO":  {"name": "Spider Monkey Optimization", "strength": "Fission-Fusion social structure. Great for wide exploration."},
            "SMA":  {"name": "Slime Mould Algorithm", "strength": "Adaptive weights based on fitness. High precision."},
            "HHO":  {"name": "Harris Hawks Optimization", "strength": "Cooperative chasing. Excellent balance exploration/exploitation."}
        }

        print("\n" + "="*110)
        print(f"{'CODE':<10} | {'FULL NAME':<30} | {'STRENGTHS / BEST USE CASE'}")
        print("="*110)
        for code, info in algos.items():
            print(f"{code:<10} | {info['name']:<30} | {info['strength']}")
        print("="*110 + "\n")


    def evaluate(self, model , verbose=False):
        with torch.no_grad():
            outputs = model(self.X_test)
            if self.task == "classification":
                _, predicted = torch.max(outputs.data, 1)
                acc = accuracy_score(self.y_test, predicted)
                if verbose :print(f"Accuracy: {acc*100:.2f}%")
                return -acc
            else:
                test_loss = self.criterion(outputs, self.y_test)
                if verbose :print(f"{self.criterion} : {test_loss.item():.4f}")
                return test_loss

    @staticmethod
    def get_available_optimizers():
        return ["Adam", "GWO", "PSO", "DE", "WOA", "GA", "ABC", "SMO", "SMA", "HHO"]

    def fitness_function(self, solution):
        model = DynamicNet(layers_cfg=self.Layers)
        try:
            model.load_flattened_weights(solution)
        except Exception:
            return 9999.0 

        model.eval()
        with torch.no_grad():
            y_pred = model(self.X_train) 
            loss = self.criterion(y_pred, self.y_train)
        return loss.item()

    def search_weights(self, optimizer_name='GWO', epochs=20, population=30, learning_rate=0.01):
        """
        Executes the optimization process using the specified algorithm.
        Supports: Adam, GWO, PSO, DE, WOA, GA, ABC, SMO, SMA, HHO.
        
        Args:
            learning_rate (float): Only used if optimizer_name is 'Adam'.
        """
        
        # --- CAS SPÉCIAL : ADAM (Gradient Descent) ---
        if optimizer_name == "Adam":
            print(f"Starting Gradient Descent (Adam) for {epochs} epochs...")
            model = DynamicNet(layers_cfg=self.Layers)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            model.train() # Mode entraînement pour Adam
            for epoch in range(epochs):
                optimizer.zero_grad()
                y_pred = model(self.X_train)
                loss = self.criterion(y_pred, self.y_train)
                loss.backward()
                optimizer.step()
                
                # Optional: print log occasionally
                # if epoch % 10 == 0: print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
            print(f"Finished! Final Train Loss: {loss.item():.4f}")
            return model # On retourne directement le modèle entraîné

        # --- CAS GÉNÉRAL : MÉTAHEURISTIQUES (Mealpy) ---
        dummy_model = DynamicNet(layers_cfg=self.Layers)
        n_params = dummy_model.count_parameters()
        
        print(f"Architecture defined. Number of weights to optimize: {n_params}")
        if n_params > 5000:
            print("WARNING: Above 5000 parameters, swarm algorithms converge very poorly.")

        lb = [-1.0] * n_params
        ub = [ 1.0] * n_params
        
        problem = {
            "obj_func": self.fitness_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min",
            "verbose": False # Mis à False pour alléger la console lors de la boucle
        }
        
        term_dict = {
           "max_early_stop": 15  # after 30 epochs, if the global best doesn't improve then we stop the program
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
            print(f"❌ Algorithm {optimizer_name} unknown. Fallback to GWO.")
            model_opt = GWO.OriginalGWO(epoch=epochs, pop_size=population)

        print(f"Starting Neuro-evolution ({optimizer_name})...")
        best_agent = model_opt.solve(problem, termination=term_dict)
        
        best_position = best_agent.solution
        best_fitness = best_agent.target.fitness
        
        print(f"Finished! Best Train Loss: {best_fitness:.4f}")
        
        # On charge les meilleurs poids dans le modèle
        dummy_model.load_flattened_weights(best_position)
        return dummy_model
    
    




    def _reconnect_layers(self, layers):
        """
        Fonction utilitaire critique : parcourt la liste des layers
        et s'assure que in_features de la couche N correspond à out_features de la couche N-1.
        """
        current_input = self.n_features
        
        for cfg in layers:
            if isinstance(cfg, LinearCfg):
                # On force l'entrée à correspondre à la sortie précédente
                cfg.in_features = current_input
                # La sortie de cette couche devient l'entrée de la prochaine
                current_input = cfg.out_features
            
            # (Tu pourras ajouter ici la logique pour Conv2d si besoin)
            
        return layers

    def search_model(self, epochs=10, train_time=300, optimizer_name_weights='GWO', 
                     epochs_weights=10, population_weights=20, learning_rate_weights=0.01):
        """
        Neural Architecture Search (NAS) basique type Hill-Climbing.
        Modifie aléatoirement l'architecture et garde les changements qui améliorent la performance.
        """
        import random
        import copy
        
        START = time.time()
        print(f"\n🚀 Démarrage de la recherche d'architecture (NAS)...")

        # 1. Optimisation initiale (Baseline)
        print("  -> Évaluation de l'architecture de départ...")
        # On utilise search_weights sur l'objet actuel
        start_model = self.search_weights(optimizer_name=optimizer_name_weights, 
                                        epochs=epochs_weights, 
                                        population=population_weights)
        
        best_score = self.evaluate(start_model)
        best_model = start_model
        # CRITIQUE : Deepcopy pour ne pas modifier l'original par erreur
        best_layers = copy.deepcopy(self.Layers) 

        print(f"  -> Score initial (Loss) : {best_score:.4f}")
        new_layers=copy.deepcopy(self.Layers) 
        ITER = 0
        while ITER < epochs and (time.time() - START) < train_time:
            ITER += 1
            print(f"\n[NAS Iteration {ITER}/{epochs}] Tentative de mutation...")
            
            # On part de la meilleure config connue
            if rd.random()<0.6: new_layers = copy.deepcopy(best_layers)
            elif rd.random()<0.5: new_layers = copy.deepcopy(new_layers)
            else: new_layers = copy.deepcopy(self.Layers)
            
            # CHOIX DE LA MUTATION
            mutation_type = random.choice(["change_neurons", "add_layer", "remove_layer"])
            
            # On filtre pour ne garder que les couches Linéaires modifiables (on exclut souvent la dernière couche de sortie)
            linear_indices = [i for i, l in enumerate(new_layers[:-1]) if isinstance(l, LinearCfg)]
            
            mutated = False

            # --- CAS 1 : Changer le nombre de neurones ---
            if mutation_type == "change_neurons" and len(linear_indices) > 0:
                idx = random.choice(linear_indices)
                noise = random.randint(-16, 16)
                new_val = new_layers[idx].out_features + noise
                # Garde-fou : pas moins de 4 neurones
                if new_val > 4:
                    new_layers[idx].out_features = new_val
                    print(f"  Action: Modification couche {idx} -> {new_val} neurones")
                    mutated = True

            # --- CAS 2 : Ajouter une couche ---
            elif mutation_type == "add_layer":
                # On insère une couche aléatoire au milieu
                insert_idx = random.randint(0, len(new_layers) - 1)
                new_neurons = random.randint(16, 64)
                # On crée une couche "tampon", les dimensions seront corrigées par _reconnect_layers
                new_layer = LinearCfg(in_features=1, out_features=new_neurons, activation=nn.ReLU)
                new_layers.insert(insert_idx, new_layer)
                print(f"  Action: Ajout d'une couche de {new_neurons} neurones à l'index {insert_idx}")
                mutated = True

            # --- CAS 3 : Supprimer une couche ---
            elif mutation_type == "remove_layer" and len(linear_indices) > 1:
                # On ne supprime pas s'il ne reste qu'une seule couche cachée
                idx = random.choice(linear_indices)
                del new_layers[idx]
                print(f"  Action: Suppression de la couche {idx}")
                mutated = True

            if not mutated:
                print("  (Pas de mutation valide trouvée, on passe)")
                continue

            # CRITIQUE : On répare les connexions (in_features == out_features précédent)
            new_layers = self._reconnect_layers(new_layers)

            # 2. Évaluation de la nouvelle architecture
            # On crée une instance temporaire juste pour tester cette config
            # Attention à bien passer 'task' (syntaxe corrigée: task=self.task)
            temp_optimizer = NeuroOptimizer(self.X_train.numpy(), self.y_train.numpy(), 
                                          Layers=new_layers, task=self.task)
            
            # On optimise les poids de cette nouvelle architecture
            # On réduit un peu les époques pour aller plus vite pendant la recherche NAS ?
            try:
                temp_model = temp_optimizer.search_weights(optimizer_name=optimizer_name_weights, 
                                                         epochs=epochs_weights, 
                                                         population=population_weights)
                
                # Évaluation
                new_score = self.evaluate(temp_model)
                print(f"  -> Nouveau Score : {new_score:.4f} (Meilleur : {best_score:.4f})")

                # 3. Acceptation ou Rejet (Hill Climbing)
                if new_score < best_score:
                    print(" AMÉLIORATION ! Architecture adoptée.")
                    best_score = new_score
                    best_model = temp_model
                    best_layers = new_layers
                    # On met à jour l'objet actuel pour qu'il garde la meilleure config
                    self.Layers = best_layers
                else:
                    print("Rejeté.")
            
            except Exception as e:
                print(f" Crash architecture invalide : {e}")

        print(f"\nFin du NAS. Meilleur Score : {best_score:.4f}")
        return best_model
        
        
        
        

if __name__ == "__main__":
    from sklearn.datasets import make_classification,make_blobs
    import matplotlib.pyplot as plt 
    

    X, y = make_classification(n_samples=2000, n_features=50, n_informative=5, n_classes=4)
    #X, y = make_blobs(n_samples=200)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    neuro_opt = NeuroOptimizer(X, y, task="classification")
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