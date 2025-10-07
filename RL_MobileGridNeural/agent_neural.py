# agent_neural.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random

class QNetwork(nn.Module):
    """Réseau de neurones pour approximer la fonction Q"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NeuralQLearning:
    """Q-Learning avec approximation par réseau de neurones"""

    def __init__(self, env, alpha=0.001, gamma=0.9, epsilon=1.0, hidden_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        
        # Configuration du réseau
        self.input_size = 2  # (x, y) de l'agent
        self.hidden_size = hidden_size
        self.output_size = 4  # 4 actions possibles
        
        # Créer le réseau de neurones
        self.q_network = QNetwork(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Replay buffer pour stabiliser l'apprentissage
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 32
        
        # Politique extraite (pour affichage)
        self.policy = defaultdict(int)
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Métriques d'exploration
        self.state_visit_count = defaultdict(int)
        self.action_count = defaultdict(lambda: defaultdict(int))
        
        # Initialiser la politique par défaut
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                state = (x, y)
                self.policy[state] = np.random.randint(0, 4)

    def state_to_tensor(self, state):
        """Convertir un état (x, y) en tensor PyTorch"""
        return torch.FloatTensor(state).unsqueeze(0)
    
    def get_q_values(self, state):
        """Obtenir les Q-values pour un état donné"""
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.squeeze().numpy()
    
    def epsilon_greedy_policy(self, state):
        """Politique epsilon-greedy améliorée"""
        if np.random.random() < self.epsilon:
            # Exploration intelligente - favoriser les actions moins explorées
            action_counts = [self.action_count[state][a] for a in range(4)]
            min_count = min(action_counts)
            least_explored = [a for a in range(4) if action_counts[a] == min_count]
            action = np.random.choice(least_explored)
        else:
            # Exploitation - choisir la meilleure action selon le réseau
            q_values = self.get_q_values(state)
            action = np.argmax(q_values)
        
        self.state_visit_count[state] += 1
        self.action_count[state][action] += 1
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocker une transition dans le replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_batch(self):
        """Échantillonner un batch du replay buffer"""
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def update_network(self):
        """Mettre à jour le réseau avec un batch du replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.sample_batch()
        
        # Calculer les Q-values actuelles
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculer les Q-values cibles
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calculer la perte et mettre à jour
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_policy_and_q_dict(self):
        """Mettre à jour la politique et le dictionnaire Q pour l'affichage"""
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                state = (x, y)
                q_values = self.get_q_values(state)
                
                # Mettre à jour le dictionnaire Q pour l'affichage
                for a in range(4):
                    self.Q[state][a] = q_values[a]
                
                # Mettre à jour la politique
                self.policy[state] = np.argmax(q_values)

    def train(self, num_episodes=1000, verbose=True, max_steps_per_episode=200):
        """Entraîner l'agent avec le réseau de neurones"""
        if verbose:
            print("Démarrage de Q-LEARNING avec RÉSEAU DE NEURONES...")
            print("=" * 60)
            print(f"Architecture: {self.input_size} -> {self.hidden_size} -> {self.hidden_size} -> {self.output_size}")
            print(f"Replay Buffer: {self.replay_buffer.maxlen} transitions")
            print("=" * 60)

        episode_rewards = []
        episode_steps = []
        success_rate = []
        losses = []
        
        stagnation_counter = 0
        last_avg_reward = -float('inf')

        for episode_num in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            episode_loss = []
            
            visited_states_episode = set()
            consecutive_repeats = 0
            
            for step in range(max_steps_per_episode):
                # Choisir une action
                action = self.epsilon_greedy_policy(state)
                
                # Exécuter l'action
                next_state, reward, done = self.env.step(action)
                
                # Pénalité pour boucles
                if next_state in visited_states_episode:
                    consecutive_repeats += 1
                    if consecutive_repeats > 3:
                        reward -= 0.5
                else:
                    consecutive_repeats = 0
                    visited_states_episode.add(next_state)
                
                total_reward += reward
                steps += 1
                
                # Stocker la transition
                self.store_transition(state, action, reward, next_state, float(done))
                
                # Mettre à jour le réseau
                loss = self.update_network()
                if loss > 0:
                    episode_loss.append(loss)
                
                state = next_state
                
                if done:
                    break
            
            # Mettre à jour la politique pour l'affichage
            if episode_num % 10 == 0:
                self.update_policy_and_q_dict()
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Calcul du taux de succès
            if episode_num >= 99:
                recent_success = sum(1 for i in range(episode_num - 99, episode_num + 1) 
                                   if episode_rewards[i] > 0)
                success_rate.append(recent_success / 100)
            else:
                success_rate.append(0)
            
            # Détection de stagnation
            if episode_num > 0 and episode_num % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                
                if avg_reward <= last_avg_reward + 0.5:
                    stagnation_counter += 1
                    
                    if stagnation_counter >= 3:
                        self.epsilon = min(0.3, self.epsilon * 1.5)
                        if verbose:
                            print(f"  ⚠️ Stagnation détectée - Augmentation exploration (ε={self.epsilon:.3f})")
                        stagnation_counter = 0
                else:
                    stagnation_counter = 0
                
                last_avg_reward = avg_reward
            
            # Décroissance epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if verbose and (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                curr_success_rate = success_rate[-1] if success_rate else 0
                print(f"Épisode {episode_num + 1}: "
                      f"Récompense = {avg_reward:.2f}, "
                      f"Loss = {avg_loss:.4f}, "
                      f"Succès = {curr_success_rate*100:.1f}%, "
                      f"ε = {self.epsilon:.3f}")
        
        # Mise à jour finale de la politique
        self.update_policy_and_q_dict()
        
        if verbose:
            print("\nEntraînement terminé!")
            print(f"Transitions stockées: {len(self.replay_buffer)}")

        return episode_rewards, episode_steps, success_rate