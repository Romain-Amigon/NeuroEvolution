# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:32:46 2025

@author: Romain
"""

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based import GWO # Ou ton SMO favori
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg
import torch.nn as nn
from sklearn.metrics import accuracy_score
# On réutilise ton DynamicNet (il est parfait pour ça)
from layer_classes import LinearCfg


from main import DynamicNet 

class NeuroRLOptimizer:
    """
    Optimiseur spécialisé pour l'Apprentissage par Renforcement (RL).
    Le réseau apprend à contrôler un agent dans un environnement Gym.
    """
    def __init__(self, env_name="CartPole-v1", layers_config=None):
        self.env_name = env_name
        
        # --- CORRECTION ICI : disable_env_checker=True ---
        # On crée un env juste pour lire les dimensions
        try:
            temp_env = gym.make(env_name, disable_env_checker=True)
        except TypeError:
            # Fallback pour les vieilles versions de gym qui n'ont pas cet argument
            temp_env = gym.make(env_name)
            
        self.input_dim = temp_env.observation_space.shape[0]
        # CartPole a un espace d'action discret (0 ou 1) -> Discrete(2)
        if hasattr(temp_env.action_space, 'n'):
            self.n_actions = temp_env.action_space.n
        else:
            # Fallback pour environnements continus (pas pour CartPole, mais pour robustesse)
            self.n_actions = temp_env.action_space.shape[0]
            
        temp_env.close()

        # Architecture par défaut pour CartPole (Petit réseau suffisant)
        if layers_config is None:
            self.layers_config = [
                LinearCfg(self.input_dim, 16, nn.ReLU),
                LinearCfg(16, self.n_actions, None) # Pas d'activation sur la sortie (Logits)
            ]
        else:
            self.layers_config = layers_config

    def run_episode(self, model, render=False):
        """
        Joue UNE partie complète avec le modèle donné.
        Retourne la récompense totale (Score).
        """
        # Si on veut voir le jeu, on change le mode de rendu
        if render:
            # render_mode='human' est la nouvelle syntaxe gymnasium
            env = gym.make(self.env_name, render_mode="human", disable_env_checker=True)
        else:
            env = gym.make(self.env_name, disable_env_checker=True)

        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        model.eval()
        with torch.no_grad():
            while not done:
                # 1. Le réseau voit l'état (obs)
                state_tensor = torch.FloatTensor(obs).unsqueeze(0) # [1, input_dim]
                
                # 2. Le réseau décide (forward)
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item() # On prend l'action la plus forte
                
                # 3. On applique l'action dans le jeu
                # Gymnasium renvoie 5 valeurs (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, _ = env.step(action)
                
                total_reward += reward
                done = terminated or truncated

        env.close()
        return total_reward

    def fitness_function(self, solution):
        """
        Fonction objectif pour Mealpy.
        Plus le score est haut, mieux c'est.
        Mealpy MINIMISE, donc on retourne -Score.
        """
        model = DynamicNet(layers_cfg=self.layers_config)
        try:
            model.load_flattened_weights(solution)
        except:
            return 9999.0 # Pénalité crash

        # Pour avoir une estimation robuste, on peut faire la moyenne de 3 parties
        # (car le RL peut avoir une part d'aléatoire)
        scores = [self.run_episode(model) for _ in range(3)]
        avg_score = sum(scores) / len(scores)
        
        # On veut MAXIMISER le score, donc on retourne -score pour Mealpy
        return -avg_score

    def search(self, optimizer_name="GWO", epochs=10, population=20):
        dummy_model = DynamicNet(layers_cfg=self.layers_config)
        n_params = dummy_model.count_parameters()
        
        print(f"🎮 RL Task: {self.env_name} | Weights to optimize: {n_params}")

        # Bornes des poids [-1, 1] ou plus large [-5, 5] pour RL
        lb = [-1.0] * n_params
        ub = [ 1.0] * n_params
        
        problem = {
            "obj_func": self.fitness_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min", # On minimise l'opposé du score
            "verbose": True
        }

        # Choix de l'algo (SMO est souvent très bon en RL pour l'exploration)
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

        print(f"🚀 Training Agent with {optimizer_name}...")
        best_agent = model_opt.solve(problem)
        
        print(f"🏆 Best Score Reached: {-best_agent.target.fitness}") # On remet le signe positif
        
        # On renvoie le modèle avec les meilleurs poids
        dummy_model.load_flattened_weights(best_agent.solution)
        return dummy_model

# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    # 1. Création de l'optimiseur pour CartPole
    # CartPole : Tenir un bâton en équilibre. Score max = 500.
    rl_opt = NeuroRLOptimizer(env_name="CartPole-v1")
    
    # 2. Entraînement (Neuro-Evolution)
    # 20 loups pendant 10 époques suffisent souvent pour CartPole
    best_model = rl_opt.search(optimizer_name="GWO", epochs=10, population=20)
    
    # 3. DÉMO VISUELLE
    print("\n🍿 Lancement de la démo visuelle...")
    rl_opt.run_episode(best_model, render=True)