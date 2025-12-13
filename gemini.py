import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from mealpy import FloatVar, CEM

# --- Importations des moteurs externes ---
# MEALPY pour l'optimisation par essaims (ex: Grey Wolf)
from mealpy.swarm_based import GWO 
# MLxtend pour l'architecture de Stacking
from mlxtend.classifier import StackingClassifier
# Skorch pour rendre PyTorch compatible avec MLxtend
from skorch import NeuralNetClassifier
# THOP pour mesurer les FLOPs (coût de calcul théorique)
from thop import profile 

from layer_classes import Conv2dCfg, DropoutCfg,FlattenCfg, LinearCfg



class DynamicNet(nn.Module):
    def __init__(self, n_features=None, n_classes=None, n_layers=1, n_neurons=32, layers_cfg=None):
        super().__init__()
        
        layers = []
        
        # CAS 1 : Construction automatique via l'optimiseur (int)
        if layers_cfg is None:
            input_dim = n_features
            
            for _ in range(n_layers):
                layers.append(nn.Linear(input_dim, n_neurons))
                layers.append(nn.ReLU())
                input_dim = n_neurons # La sortie devient l'entrée de la suivante
            
            layers.append(nn.Linear(input_dim, n_classes))
            
        # CAS 2 : Construction via ta liste de configs personnalisée 
        else:
            for cfg in layers_cfg:
                if isinstance(cfg, LinearCfg):
                    layers.append(nn.Linear(cfg.in_features, cfg.out_features))
                    layers.append(cfg.activation())
                elif isinstance(cfg, Conv2dCfg):
                    layers.append(nn.Conv2d(cfg.in_channels, cfg.out_channels, 
                                          cfg.kernel_size, cfg.stride, cfg.padding))
                    layers.append(cfg.activation())
                elif isinstance(cfg, DropoutCfg):
                    layers.append(nn.Dropout(p=cfg.p))
                elif isinstance(cfg, FlattenCfg):
                    layers.append(nn.Flatten(start_dim=cfg.start_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class NeuroStackOptimizer:
    def __init__(self, X, y, efficiency_weight=0.3, latency_budget_ms=50):
        """
        Args:
            efficiency_weight (float): Importance du temps vs précision (0.0 = précision pure).
            latency_budget_ms (float): Limite stricte de temps en millisecondes.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        self.efficiency_weight = efficiency_weight
        self.latency_budget = latency_budget_ms / 1000.0 # conversion en secondes

    def _decode_solution(self, solution):
        """
        Traduit le vecteur de nombres de MEALPY en hyperparamètres concrets.
        Format du vecteur solution (exemple):
        [n_layers_model1, n_neurons_model1, n_layers_model2, n_neurons_model2, lr]
        """
        # On cast en entiers car on ne peut pas avoir 2.5 couches
        params = {
            'm1_layers': int(solution),
            'm1_neurons': int(solution[1]),
            'm2_layers': int(solution[2]),
            'm2_neurons': int(solution[3]),
            'lr': solution[4]
        }
        return params

    def _create_skorch_model(self, layers, neurons, lr):
        """Wrappe le modèle PyTorch dans Skorch pour le rendre compatible MLxtend"""
        return NeuralNetClassifier(
            DynamicNet,
            module__n_features=self.n_features,
            module__n_classes=self.n_classes,
            module__n_layers=layers,
            module__n_neurons=neurons,
            max_epochs=5, # Peu d'époques pour la recherche (speed up)
            lr=lr,
            optimizer=torch.optim.Adam,
            verbose=0,
            train_split=None # On gère le split nous-mêmes
        )

    def fitness_function(self, solution):
        """
        C'est ici que la magie opère.
        MEALPY appelle cette fonction pour tester une 'solution' (une configuration).
        """
        params = self._decode_solution(solution)
        
        # 1. Construction des modèles de base (Architecture en série / Ensemble)
        net1 = self._create_skorch_model(params['m1_layers'], params['m1_neurons'], params['lr'])
        net2 = self._create_skorch_model(params['m2_layers'], params['m2_neurons'], params['lr'])
        
        # Le méta-learner (celui qui décide à la fin)
        # On utilise une régression logistique simple ou un petit réseau
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression()

        # 2. Assemblage avec MLxtend (Stacking)
        # Les sorties de net1 et net2 deviennent les entrées du meta_learner
        clf_stack = StackingClassifier(
            classifiers=[net1, net2],
            meta_classifier=meta_learner, 
            use_probas=True,
            average_probas=False
        )

        # 3. Mesure de l'Efficacité (Hardware constraints)
        start_time = time.time()
        
        # On entraine sur une fraction des données pour aller vite lors de la recherche
        # Attention: il faut convertir en float32 pour PyTorch
        X_train_t = self.X_train.astype(np.float32)
        y_train_t = self.y_train.astype(np.int64)
        
        try:
            clf_stack.fit(X_train_t, y_train_t)
        except Exception as e:
            return 100.0 # Pénalité max si le modèle crash (architecture invalide)

        # Inférence (c'est souvent ça qu'on veut optimiser)
        latency = (time.time() - start_time) / len(X_train_t)
        
        # 4. Calcul du Score Composite
        # On veut MAXIMISER la précision, donc MINIMISER l'erreur (1 - acc)
        # On veut MINIMISER la latence
        
        acc = accuracy_score(self.y_test, clf_stack.predict(self.X_test.astype(np.float32)))
        error = 1.0 - acc
        
        # GARDIEN D'EFFICACITÉ : Si trop lent, on rejette violemment
        if latency > self.latency_budget:
            return 10.0 + latency # Enorme pénalité

        # Fonction objectif à minimiser (compromis efficacité/temps)
        score = (1 - self.efficiency_weight) * error + self.efficiency_weight * latency
        
        return score

    def search(self, optimizer_name="GWO", epochs=10, population=15):
            """
            Lance la recherche avec l'algorithme choisi du panel MEALPY.
            """
            # 1. Définition des bornes (Espace de recherche)
            # lb = Lower Bound (Minimums)
            # [m1_layers, m1_neurons, m2_layers, m2_neurons, lr]
            lb = [1,  16,  1,  16,  0.001]
            
            # ub = Upper Bound (Maximums)
            ub = [5, 128,  5, 128,  0.1]
    
            # 2. Création du dictionnaire problème avec la clé 'bound'
            # Mealpy attend généralement un tuple ou une liste : (lb, ub)
            problem = {
                "obj_func": self.fitness_function,
                "bounds": (lb, ub),  # <--- C'est ici la correction importante
                "minmax": "min",
                "verbose": True     # Optionnel : pour voir les logs d'optimisation
            }
    
            # 3. Choix dynamique de l'algo
            if optimizer_name == "GWO":
                model = GWO.OriginalGWO(epoch=epochs, pop_size=population)
            
            print(f"🚀 Démarrage de l'optimisation avec {optimizer_name}...")
            
            # Lancement de la résolution
            best_position, best_fitness = model.solve(problem)
        
            print(f"✅ Terminé! Meilleure config : {self._decode_solution(best_position)}")
            return self._decode_solution(best_position)
# ============================================================
# 3. Exemple d'utilisation (Demo utilisateur)
# ============================================================
if __name__ == "__main__":
    # Génération de fausses données pour tester
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2)

    # Initialisation de VOTRE librairie
    # "Je veux optimiser un système, mais la vitesse compte pour 50% du score"
    my_optimizer = NeuroStackOptimizer(X, y, efficiency_weight=0.5, latency_budget_ms=20)

    # Lancement de la recherche avec l'algorithme des Loups Gris
    best_params = my_optimizer.search(optimizer_name="GWO")