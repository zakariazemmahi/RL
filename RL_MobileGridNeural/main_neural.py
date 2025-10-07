import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import time
from environment import MobileGridWorld
from agent_neural import NeuralQLearning

# ====================================================================
# FONCTIONS DE CONFIGURATION (identiques)
# ====================================================================

def get_user_configuration():
    """Demander à l'utilisateur s'il veut entrer les paramètres ou les générer aléatoirement"""
    print("\n" + "=" * 60)
    print("CONFIGURATION DE L'ENVIRONNEMENT MOBILE")
    print("=" * 60)
    
    while True:
        choice = input("\nVoulez-vous:\n  1. Entrer les paramètres manuellement\n  2. Générer aléatoirement\nVotre choix (1 ou 2): ").strip()
        
        if choice == "1":
            return configure_manually()
        elif choice == "2":
            return configure_randomly()
        else:
            print("❌ Choix invalide! Veuillez entrer 1 ou 2.")

def configure_manually():
    """Configuration manuelle par l'utilisateur"""
    print("\n--- Configuration Manuelle ---")
    
    while True:
        try:
            grid_size = int(input("\nTaille de la grille (ex: 5 pour 5x5): "))
            if grid_size < 3:
                print("❌ La grille doit être au moins 3x3!")
                continue
            break
        except ValueError:
            print("❌ Veuillez entrer un nombre entier!")
    
    print(f"\nPosition initiale (entre 0 et {grid_size-1}):")
    while True:
        try:
            start_x = int(input(f"  Position X (0-{grid_size-1}): "))
            start_y = int(input(f"  Position Y (0-{grid_size-1}): "))
            if 0 <= start_x < grid_size and 0 <= start_y < grid_size:
                start_pos = (start_x, start_y)
                break
            else:
                print(f"❌ Les coordonnées doivent être entre 0 et {grid_size-1}!")
        except ValueError:
            print("❌ Veuillez entrer des nombres entiers!")
    
    while True:
        try:
            num_goals = int(input(f"\nNombre de goals (1-5): "))
            if 1 <= num_goals <= 5:
                break
            else:
                print(f"❌ Le nombre de goals doit être entre 1 et 5!")
        except ValueError:
            print("❌ Veuillez entrer un nombre entier!")
    
    goal_positions = []
    print(f"\nEntrez les positions INITIALES des {num_goals} goal(s) mobiles:")
    for i in range(num_goals):
        while True:
            try:
                print(f"\n  Goal {i+1}:")
                goal_x = int(input(f"    Position X (0-{grid_size-1}): "))
                goal_y = int(input(f"    Position Y (0-{grid_size-1}): "))
                goal_pos = (goal_x, goal_y)
                
                if not (0 <= goal_x < grid_size and 0 <= goal_y < grid_size):
                    print(f"    ❌ Les coordonnées doivent être entre 0 et {grid_size-1}!")
                elif goal_pos == start_pos:
                    print("    ❌ Le goal ne peut pas être à la position de départ!")
                elif goal_pos in goal_positions:
                    print("    ❌ Cette position est déjà un goal!")
                else:
                    goal_positions.append(goal_pos)
                    break
            except ValueError:
                print("    ❌ Veuillez entrer des nombres entiers!")
    
    while True:
        try:
            goal_prob = float(input(f"\nProbabilité de mouvement des goals (0.0-1.0, recommandé 0.3): "))
            if 0.0 <= goal_prob <= 1.0:
                break
            else:
                print("❌ La probabilité doit être entre 0.0 et 1.0!")
        except ValueError:
            print("❌ Veuillez entrer un nombre décimal!")
    
    obstacles = []
    while True:
        try:
            num_obstacles = int(input(f"\nNombre d'obstacles mobiles (0-{grid_size*grid_size//4}): "))
            if 0 <= num_obstacles <= grid_size*grid_size//4:
                break
            else:
                print(f"❌ Le nombre d'obstacles doit être entre 0 et {grid_size*grid_size//4}!")
        except ValueError:
            print("❌ Veuillez entrer un nombre entier!")
    
    if num_obstacles > 0:
        print(f"\nEntrez les positions INITIALES des {num_obstacles} obstacles mobiles:")
        for i in range(num_obstacles):
            while True:
                try:
                    print(f"\n  Obstacle {i+1}:")
                    obs_x = int(input(f"    Position X (0-{grid_size-1}): "))
                    obs_y = int(input(f"    Position Y (0-{grid_size-1}): "))
                    obs_pos = (obs_x, obs_y)
                    
                    if not (0 <= obs_x < grid_size and 0 <= obs_y < grid_size):
                        print(f"    ❌ Les coordonnées doivent être entre 0 et {grid_size-1}!")
                    elif obs_pos == start_pos:
                        print("    ❌ L'obstacle ne peut pas être à la position de départ!")
                    elif obs_pos in goal_positions:
                        print("    ❌ L'obstacle ne peut pas être à la position d'un goal!")
                    elif obs_pos in obstacles:
                        print("    ❌ Cette position est déjà un obstacle!")
                    else:
                        obstacles.append(obs_pos)
                        break
                except ValueError:
                    print("    ❌ Veuillez entrer des nombres entiers!")
        
        while True:
            try:
                obs_prob = float(input(f"\nProbabilité de mouvement des obstacles (0.0-1.0, recommandé 0.2): "))
                if 0.0 <= obs_prob <= 1.0:
                    break
                else:
                    print("❌ La probabilité doit être entre 0.0 et 1.0!")
            except ValueError:
                print("❌ Veuillez entrer un nombre décimal!")
    else:
        obs_prob = 0.0
    
    return grid_size, start_pos, goal_positions, obstacles, goal_prob, obs_prob

def configure_randomly():
    """Configuration aléatoire"""
    print("\n--- Génération Aléatoire ---")
    
    grid_size = np.random.randint(6, 10)
    print(f"✓ Taille de la grille: {grid_size}x{grid_size}")
    
    start_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    print(f"✓ Position initiale: {start_pos}")
    
    num_goals = np.random.randint(1, 4)
    print(f"✓ Nombre de goals: {num_goals}")
    
    goal_positions = []
    attempts = 0
    while len(goal_positions) < num_goals and attempts < 1000:
        goal_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        if goal_pos != start_pos and goal_pos not in goal_positions:
            goal_positions.append(goal_pos)
        attempts += 1
    
    print(f"✓ Positions initiales des goals: {goal_positions}")
    
    goal_prob = np.random.uniform(0.2, 0.4)
    print(f"✓ Probabilité de mouvement des goals: {goal_prob:.2f}")
    
    num_obstacles = np.random.randint(grid_size, grid_size * 2)
    
    obstacles = []
    attempts = 0
    while len(obstacles) < num_obstacles and attempts < 1000:
        obs_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        if obs_pos != start_pos and obs_pos not in goal_positions and obs_pos not in obstacles:
            obstacles.append(obs_pos)
        attempts += 1
    
    print(f"✓ Nombre d'obstacles: {len(obstacles)}")
    
    obs_prob = np.random.uniform(0.1, 0.3)
    print(f"✓ Probabilité de mouvement des obstacles: {obs_prob:.2f}")
    
    return grid_size, start_pos, goal_positions, obstacles, goal_prob, obs_prob


# ====================================================================
# FONCTION D'ENTRAÎNEMENT
# ====================================================================

def train_neural_agent(agent, num_episodes=1000):
    """
    Entraîne l'agent neural avec Q-Learning
    """
    print("\n" + "=" * 60)
    print("PHASE D'ENTRAÎNEMENT DU RÉSEAU DE NEURONES")
    print("=" * 60)
    
    # Utiliser la méthode train() de l'agent
    episode_rewards, episode_steps, success_rate = agent.train(
        num_episodes=num_episodes, 
        verbose=True
    )
    
    # Afficher les courbes d'apprentissage
    plot_training_curves(episode_rewards, success_rate, agent.epsilon_min)
    
    return episode_rewards, success_rate


def plot_training_curves(rewards_history, success_history, epsilon_final, window=50):
    """
    Affiche les courbes d'apprentissage
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Récompenses par épisode
    ax = axes[0, 0]
    ax.plot(rewards_history, alpha=0.3, color='blue', label='Récompense brute')
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards_history)), moving_avg, 
                color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense Totale')
    ax.set_title('Évolution des Récompenses')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Taux de succès
    ax = axes[0, 1]
    if len(success_history) > 0:
        success_rate_percent = [s * 100 for s in success_history]
        ax.plot(success_rate_percent, color='green', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Taux de Succès (%)')
    ax.set_title(f'Taux de Succès (Moyenne mobile 100 épisodes)')
    ax.set_ylim([0, 105])
    ax.grid(alpha=0.3)
    
    # 3. Distribution des récompenses
    ax = axes[1, 0]
    ax.hist(rewards_history, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rewards_history), color='red', linestyle='--', 
               linewidth=2, label=f'Moyenne: {np.mean(rewards_history):.2f}')
    ax.set_xlabel('Récompense Totale')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des Récompenses')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Statistiques finales
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    STATISTIQUES D'ENTRAÎNEMENT
    
    Nombre d'épisodes: {len(rewards_history)}
    
    Récompense moyenne: {np.mean(rewards_history):.2f}
    Récompense max: {np.max(rewards_history):.2f}
    Récompense min: {np.min(rewards_history):.2f}
    
    Taux de succès final: {success_history[-1]*100:.1f}%
    
    Epsilon final: {epsilon_final:.3f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ====================================================================
# FONCTIONS DE SIMULATION ET DE COMPARAISON
# ====================================================================

def execute_policy_and_render(env, agent, max_steps=100, pause_time=0.4):
    """Simulation avec rendu graphique"""
    print("\n" + "=" * 60)
    print("SIMULATION DES MOUVEMENTS DE L'AGENT (Neural Network)")
    print("=" * 60)

    state = env.reset()
    total_reward = 0
    action_names = ["Haut", "Bas", "Gauche", "Droite"]
    
    position_sequence = []
    loop_detected = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()

    def draw_grid_and_entities(ax, is_policy_view=False):
        ax.clear() 
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for i in range(env.grid_size + 1):
            ax.plot([0, env.grid_size], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, env.grid_size], 'k-', linewidth=1)

        if not is_policy_view:
            title = f'Environnement Mobile (Goal atteint: {len(env.goals_reached) > 0}) - Pas: {step+1}'
            if loop_detected:
                title += " ⚠️ BOUCLE DÉTECTÉE"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            for obs_x, obs_y in env.obstacles:
                rect = plt.Rectangle((obs_x, obs_y), 1, 1, color='gray', alpha=0.7, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(obs_x + 0.5, obs_y + 0.5, '↻', ha='center', va='center', fontsize=20, color='white', weight='bold')

            for goal_pos in env.goal_positions:
                color = 'lightgreen' if goal_pos in env.goals_reached else 'red'
                edge_color = 'darkgreen' if goal_pos in env.goals_reached else 'darkred'
                
                circle = plt.Circle((goal_pos[0] + 0.5, goal_pos[1] + 0.5), 0.35, color=color, alpha=0.3)
                ax.add_patch(circle)
                ax.plot(goal_pos[0] + 0.5, goal_pos[1] + 0.5, marker='*', markersize=30, color=color, markeredgecolor=edge_color, markeredgewidth=2)

            ax.plot(env.agent_pos[0] + 0.5, env.agent_pos[1] + 0.5, 'o', markersize=20, color='blue', markeredgecolor='darkblue', markeredgewidth=2)
            
        else:
            ax.set_title('Q-Values & Policy (Neural Network)', fontsize=12, fontweight='bold')
            for x in range(env.grid_size):
                for y in range(env.grid_size):
                    s = (x, y)
                    q_values = [agent.Q[s][a] for a in range(4)]
                    avg_q = np.mean(q_values) if s in agent.Q else 0.0

                    color_intensity = min(1.0, max(0.0, (avg_q + 5) / 15))
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='green', alpha=color_intensity * 0.3))
                    
                    ax.text(x + 0.5, y + 0.5, f'{avg_q:.1f}', ha='center', va='center', fontsize=9, weight='bold')

                    if s in agent.policy:
                        action_policy = agent.policy[s]
                        dx, dy = 0, 0
                        if action_policy == 0: dy = -0.3
                        elif action_policy == 1: dy = 0.3
                        elif action_policy == 2: dx = -0.3
                        elif action_policy == 3: dx = 0.3

                        ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.15, head_length=0.1, fc='cyan', ec='blue', linewidth=1.5)

    try:
        for step in range(max_steps):
            position_sequence.append(state)
            if len(position_sequence) > 8:
                recent = position_sequence[-8:]
                if recent.count(state) >= 4:
                    loop_detected = True
                    print(f"\n⚠️ BOUCLE DÉTECTÉE au pas {step + 1}! Position répétée: {state}")
                    action = np.random.randint(0, 4)
                    print(f"   → Action aléatoire forcée: {action_names[action]}")
                else:
                    loop_detected = False
                    action = agent.policy.get(state, np.random.randint(0, 4))
            else:
                action = agent.policy.get(state, np.random.randint(0, 4))

            print(f"Pas {step + 1}: Pos={state}, Action={action_names[action]}")

            state, reward, done = env.step(action)
            total_reward += reward

            draw_grid_and_entities(ax1, is_policy_view=False)
            draw_grid_and_entities(ax2, is_policy_view=True)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(pause_time)

            if done:
                print(f"\n🎉 OBJECTIF ATTEINT en {step + 1} pas!")
                plt.pause(2.0)
                break

    except Exception as e:
        print(f"Erreur lors de la simulation: {e}")
        
    finally:
        plt.ioff()
        plt.show(block=False)
        
    if not done:
        print(f"\n⚠️ Limite de pas atteinte. Simulation terminée sans succès.")

    print("=" * 60)


def compare_mobility_strategies(env, agent, num_episodes=50):
    """Compare la politique Neural avec une politique Aléatoire"""
    print("\n" + "=" * 60)
    print("COMPARAISON DE STRATÉGIES (Neural Network vs Aléatoire)")
    print("=" * 60)
    
    success_neural = 0
    rewards_neural = []
    
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            action = agent.policy.get(state, np.random.randint(0, 4))
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                success_neural += 1
                break
        rewards_neural.append(total_reward)
        
    avg_reward_neural = np.mean(rewards_neural)
    success_rate_neural = success_neural / num_episodes * 100
    
    print(f"\nPOLITIQUE NEURAL NETWORK ({num_episodes} épisodes):")
    print(f"  - Taux de succès: {success_neural}/{num_episodes} ({success_rate_neural:.1f}%)")
    print(f"  - Récompense moyenne: {avg_reward_neural:.2f}")

    success_random = 0
    rewards_random = []
    
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            action = np.random.randint(0, 4) 
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                success_random += 1
                break
        rewards_random.append(total_reward)
        
    avg_reward_random = np.mean(rewards_random)
    success_rate_random = success_random / num_episodes * 100

    print(f"\nPOLITIQUE ALÉATOIRE ({num_episodes} épisodes):")
    print(f"  - Taux de succès: {success_random}/{num_episodes} ({success_rate_random:.1f}%)")
    print(f"  - Récompense moyenne: {avg_reward_random:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    strategies = ['Neural Q-Learning', 'Aléatoire']
    avg_rewards = [avg_reward_neural, avg_reward_random]
    success_rates = [success_rate_neural, success_rate_random]

    ax.bar(strategies, avg_rewards, color=['purple', 'gray'], alpha=0.7)
    ax.set_ylabel('Récompense Moyenne Totale', fontsize=12)
    ax.set_title('Comparaison de Performance des Politiques', 
                 fontsize=14, fontweight='bold')
    
    for i, rate in enumerate(success_rates):
        ax.text(i, avg_rewards[i] + 0.5, f'{rate:.1f}% succès', ha='center', 
                color='black', fontsize=10, weight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


# ====================================================================
# FONCTION PRINCIPALE (CORRIGÉE!)
# ====================================================================

def main():
    """Programme principal avec entraînement du réseau de neurones"""
    
    # 1. Configuration de l'environnement
    grid_size, start_pos, goal_positions, obstacles, goal_prob, obs_prob = get_user_configuration()
    
    # 2. Création de l'environnement
    env = MobileGridWorld(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_positions=goal_positions,
        obstacles=obstacles,
        goal_move_prob=goal_prob,
        obstacle_move_prob=obs_prob
    )
    
    print("\n✓ Environnement créé avec succès!")
    
    # 3. Demander le nombre d'épisodes d'entraînement
    while True:
        try:
            num_episodes = int(input("\nNombre d'épisodes d'entraînement (recommandé: 1000-5000): "))
            if num_episodes > 0:
                break
            else:
                print("❌ Le nombre d'épisodes doit être positif!")
        except ValueError:
            print("❌ Veuillez entrer un nombre entier!")
    
    # 4. CORRECTION: Création de l'agent neural AVEC l'environnement
    agent = NeuralQLearning(
        env=env,
        alpha=0.001,
        gamma=0.9,
        epsilon=1.0,
        hidden_size=64
    )
    
    print("\n✓ Agent neural créé avec succès!")
    
    # 5. ENTRAÎNEMENT
    train_neural_agent(agent, num_episodes=num_episodes)
    
    # 6. Demander si l'utilisateur veut voir la simulation
    while True:
        simulate = input("\nVoulez-vous voir la simulation graphique? (o/n): ").strip().lower()
        if simulate in ['o', 'oui', 'y', 'yes']:
            execute_policy_and_render(env, agent)
            break
        elif simulate in ['n', 'non', 'no']:
            break
        else:
            print("❌ Réponse invalide! Veuillez entrer 'o' ou 'n'.")
    
    # 7. Demander si l'utilisateur veut la comparaison
    while True:
        compare = input("\nVoulez-vous comparer avec une politique aléatoire? (o/n): ").strip().lower()
        if compare in ['o', 'oui', 'y', 'yes']:
            compare_mobility_strategies(env, agent, num_episodes=50)
            break
        elif compare in ['n', 'non', 'no']:
            break
        else:
            print("❌ Réponse invalide! Veuillez entrer 'o' ou 'n'.")
    
    print("\n" + "=" * 60)
    print("PROGRAMME TERMINÉ")
    print("=" * 60)
    print("\nFermez les fenêtres matplotlib pour quitter.")
    plt.show()


if __name__ == "__main__":
    main()
