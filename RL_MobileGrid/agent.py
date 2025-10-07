# agent.py

import numpy as np
from collections import defaultdict

class AdaptiveQLearning:
    """Q-Learning adaptatif pour environnement dynamique"""

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = defaultdict(int)
        
        self.state_visit_count = defaultdict(int)
        self.action_count = defaultdict(lambda: defaultdict(int))
        
        # CORRECTION: Bonus d'exploration pour encourager la diversité
        self.exploration_bonus = defaultdict(lambda: defaultdict(lambda: 0.5))

        # Initialisation complète (pour un affichage propre du rendu)
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                state = (x, y)
                for action in range(4):  # 4 actions (0, 1, 2, 3)
                    self.Q[state][action] = 0.0
                self.policy[state] = np.random.randint(0, 4)

    def epsilon_greedy_policy(self, state):
        """
        CORRECTION: Politique epsilon-greedy améliorée avec bonus d'exploration
        """
        if np.random.random() < self.epsilon:
            # CORRECTION: Exploration intelligente - favoriser les actions moins explorées
            action_counts = [self.action_count[state][a] for a in range(4)]
            min_count = min(action_counts)
            
            # Actions les moins explorées
            least_explored = [a for a in range(4) if action_counts[a] == min_count]
            action = np.random.choice(least_explored)
        else:
            # CORRECTION: Exploitation avec bonus d'exploration
            q_with_bonus = []
            for a in range(4):
                base_q = self.Q[state][a]
                bonus = self.exploration_bonus[state][a]
                # Réduire le bonus au fil du temps
                bonus *= 0.999
                self.exploration_bonus[state][a] = bonus
                q_with_bonus.append(base_q + bonus)
            
            action = np.argmax(q_with_bonus)
        
        self.state_visit_count[state] += 1
        self.action_count[state][action] += 1
        
        return action

    def train(self, num_episodes=1000, verbose=True, max_steps_per_episode=200):
        """
        CORRECTION: Entraîner avec détection de stagnation et redémarrage
        """
        if verbose:
            print("Démarrage de Q-LEARNING ADAPTATIF (Environnement Mobile)...")
            print("=" * 60)

        episode_rewards = []
        episode_steps = []
        success_rate = []
        
        # CORRECTION: Métriques pour détecter la stagnation
        stagnation_counter = 0
        last_avg_reward = -float('inf')

        for episode_num in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # CORRECTION: Historique des états visités dans cet épisode
            visited_states_episode = set()
            consecutive_repeats = 0
            
            for step in range(max_steps_per_episode):
                # Choisir une action avec epsilon-greedy
                action = self.epsilon_greedy_policy(state)

                next_state, reward, done = self.env.step(action)
                
                # CORRECTION: Pénalité supplémentaire si on revisite trop le même état
                if next_state in visited_states_episode:
                    consecutive_repeats += 1
                    if consecutive_repeats > 3:
                        reward -= 0.5  # Pénalité pour boucle locale
                else:
                    consecutive_repeats = 0
                    visited_states_episode.add(next_state)
                
                total_reward += reward
                steps += 1

                # CORRECTION: Mise à jour Q-Learning avec learning rate adaptatif
                visit_count = self.state_visit_count[state]
                adaptive_alpha = self.alpha / (1 + visit_count * 0.001)  # Décroissance lente
                
                next_q_values = [self.Q[next_state][a] for a in range(4)]
                best_next_q = np.max(next_q_values) 
                
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += adaptive_alpha * td_error

                state = next_state

                if done:
                    break

            # Mettre à jour la politique (greedy)
            for x in range(self.env.grid_size):
                for y in range(self.env.grid_size):
                    s = (x, y)
                    q_values = [self.Q[s][a] for a in range(4)]
                    self.policy[s] = np.argmax(q_values)

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            # Calcul du taux de succès
            if episode_num >= 99:
                recent_success = sum(1 for i in range(episode_num - 99, episode_num + 1) 
                                   if episode_rewards[i] > 0)
                success_rate.append(recent_success / 100)
            else:
                success_rate.append(0)

            # CORRECTION: Détection de stagnation et ajustement
            if episode_num > 0 and episode_num % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                
                # Si pas d'amélioration
                if avg_reward <= last_avg_reward + 0.5:
                    stagnation_counter += 1
                    
                    # CORRECTION: Augmenter l'exploration si stagnation
                    if stagnation_counter >= 3:
                        self.epsilon = min(0.3, self.epsilon * 1.5)
                        # Réinitialiser les bonus d'exploration
                        for s in self.exploration_bonus:
                            for a in range(4):
                                self.exploration_bonus[s][a] = 0.5
                        
                        if verbose:
                            print(f"  ⚠️ Stagnation détectée - Augmentation exploration (ε={self.epsilon:.3f})")
                        stagnation_counter = 0
                else:
                    stagnation_counter = 0
                
                last_avg_reward = avg_reward

            # Décroissance epsilon (plus lente pour mieux explorer)
            self.epsilon = max(0.05, self.epsilon * 0.997)

            if verbose and (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                curr_success_rate = success_rate[-1] if success_rate else 0
                print(f"Épisode {episode_num + 1}: "
                      f"Récompense moy. = {avg_reward:.2f}, "
                      f"Taux succès = {curr_success_rate*100:.1f}%, "
                      f"Epsilon = {self.epsilon:.3f}")

        if verbose:
            print("\nEntraînement terminé!")

        return episode_rewards, episode_steps, success_rate