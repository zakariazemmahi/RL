# main.py
import matplotlib
matplotlib.use('TkAgg')  # Backend pour affichage interactif

import numpy as np
import matplotlib.pyplot as plt
import time
from environment import MobileGridWorld
from agent import AdaptiveQLearning

# ====================================================================
# 1. FONCTIONS DE CONFIGURATION
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
    
    # Taille de la grille
    while True:
        try:
            grid_size = int(input("\nTaille de la grille (ex: 5 pour 5x5): "))
            if grid_size < 3:
                print("❌ La grille doit être au moins 3x3!")
                continue
            break
        except ValueError:
            print("❌ Veuillez entrer un nombre entier!")
    
    # Position initiale
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
    
    # Configuration des goals
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
    
    # Probabilité de mouvement des goals
    while True:
        try:
            goal_prob = float(input(f"\nProbabilité de mouvement des goals (0.0-1.0, recommandé 0.3): "))
            if 0.0 <= goal_prob <= 1.0:
                break
            else:
                print("❌ La probabilité doit être entre 0.0 et 1.0!")
        except ValueError:
            print("❌ Veuillez entrer un nombre décimal!")
    
    # Configuration des obstacles
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
        
        # Probabilité de mouvement des obstacles
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
# 2. FONCTIONS DE SIMULATION ET DE COMPARAISON
# ====================================================================

def execute_policy_and_render(env, agent, max_steps=100, pause_time=0.4):
    """
    CORRECTION: Ajout de détection de boucle pendant la simulation
    """
    print("\n" + "=" * 60)
    print("SIMULATION DES MOUVEMENTS DE L'AGENT")
    print("=" * 60)

    state = env.reset()
    total_reward = 0
    action_names = ["Haut", "Bas", "Gauche", "Droite"]
    
    # CORRECTION: Détection de boucle pendant simulation
    position_sequence = []
    loop_detected = False

    # Créer la figure une seule fois
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()  # Mode interactif

    def draw_grid_and_entities(ax, is_policy_view=False):
        """Fonction interne pour dessiner le contenu de la grille"""
        ax.clear() 
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Grille
        for i in range(env.grid_size + 1):
            ax.plot([0, env.grid_size], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, env.grid_size], 'k-', linewidth=1)

        if not is_policy_view:
            title = f'Environnement Mobile (Goal atteint: {len(env.goals_reached) > 0}) - Pas: {step+1}'
            if loop_detected:
                title += " ⚠️ BOUCLE DÉTECTÉE"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Obstacles
            for obs_x, obs_y in env.obstacles:
                rect = plt.Rectangle((obs_x, obs_y), 1, 1, color='gray', alpha=0.7, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(obs_x + 0.5, obs_y + 0.5, '↻', ha='center', va='center', fontsize=20, color='white', weight='bold')

            # Goals
            for goal_pos in env.goal_positions:
                color = 'lightgreen' if goal_pos in env.goals_reached else 'red'
                edge_color = 'darkgreen' if goal_pos in env.goals_reached else 'darkred'
                
                circle = plt.Circle((goal_pos[0] + 0.5, goal_pos[1] + 0.5), 0.35, color=color, alpha=0.3)
                ax.add_patch(circle)
                ax.plot(goal_pos[0] + 0.5, goal_pos[1] + 0.5, marker='*', markersize=30, color=color, markeredgecolor=edge_color, markeredgewidth=2)

            # Agent
            ax.plot(env.agent_pos[0] + 0.5, env.agent_pos[1] + 0.5, 'o', markersize=20, color='blue', markeredgecolor='darkblue', markeredgewidth=2)
            
        else:  # Vue Q-values et Policy
            ax.set_title('Q-Values & Policy Adaptative', fontsize=12, fontweight='bold')
            for x in range(env.grid_size):
                for y in range(env.grid_size):
                    s = (x, y)
                    q_values = [agent.Q[s][a] for a in range(4)]
                    avg_q = np.mean(q_values) if s in agent.Q else 0.0

                    # Coloration
                    color_intensity = min(1.0, max(0.0, (avg_q + 5) / 15))
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='green', alpha=color_intensity * 0.3))
                    
                    # Texte Q-values
                    ax.text(x + 0.5, y + 0.5, f'{avg_q:.1f}', ha='center', va='center', fontsize=9, weight='bold')

                    # Flèches de politique
                    if s in agent.policy:
                        action_policy = agent.policy[s]
                        dx, dy = 0, 0
                        if action_policy == 0: dy = -0.3  # Haut
                        elif action_policy == 1: dy = 0.3   # Bas
                        elif action_policy == 2: dx = -0.3  # Gauche
                        elif action_policy == 3: dx = 0.3   # Droite

                        ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.15, head_length=0.1, fc='cyan', ec='blue', linewidth=1.5)

    try:
        for step in range(max_steps):
            # CORRECTION: Vérifier si on est dans une boucle
            position_sequence.append(state)
            if len(position_sequence) > 8:
                recent = position_sequence[-8:]
                if recent.count(state) >= 4:
                    loop_detected = True
                    print(f"\n⚠️ BOUCLE DÉTECTÉE au pas {step + 1}! Position répétée: {state}")
                    # Forcer une action aléatoire pour sortir de la boucle
                    action = np.random.randint(0, 4)
                    print(f"   → Action aléatoire forcée: {action_names[action]}")
                else:
                    loop_detected = False
                    action = agent.policy.get(state, np.random.randint(0, 4))
            else:
                action = agent.policy.get(state, np.random.randint(0, 4))

            print(f"Pas {step + 1}: Pos={state}, Action={action_names[action]}")

            # Exécuter l'étape
            state, reward, done = env.step(action)
            total_reward += reward

            # Mettre à jour le rendu
            draw_grid_and_entities(ax1, is_policy_view=False)
            draw_grid_and_entities(ax2, is_policy_view=True)
            
            # Mise à jour graphique
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
    """Compare la politique Adaptative apprise avec une politique Aléatoire."""
    print("\n" + "=" * 60)
    print("COMPARAISON DE STRATÉGIES (Adaptative vs Aléatoire)")
    print("=" * 60)
    
    # Évaluation de la Politique Adaptative Apprise
    success_adaptive = 0
    rewards_adaptive = []
    
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            action = agent.policy.get(state, np.random.randint(0, 4))
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                success_adaptive += 1
                break
        rewards_adaptive.append(total_reward)
        
    avg_reward_adaptive = np.mean(rewards_adaptive)
    success_rate_adaptive = success_adaptive / num_episodes * 100
    
    print(f"\nPOLITIQUE ADAPTATIVE ({num_episodes} épisodes):")
    print(f"  - Taux de succès: {success_adaptive}/{num_episodes} ({success_rate_adaptive:.1f}%)")
    print(f"  - Récompense moyenne: {avg_reward_adaptive:.2f}")

    # Évaluation de la Politique Aléatoire
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

    # Visualisation du Test
    fig, ax = plt.subplots(figsize=(8, 6))
    strategies = ['Adaptative Q-Learning', 'Aléatoire']
    avg_rewards = [avg_reward_adaptive, avg_reward_random]
    success_rates = [success_rate_adaptive, success_rate_random]

    ax.bar(strategies, avg_rewards, color=['blue', 'gray'], alpha=0.7)
    ax.set_ylabel('Récompense Moyenne Totale', fontsize=12)
    ax.set_title('Comparaison de Performance des Politiques', 
                 fontsize=14, fontweight='bold')
    
    for i, rate in enumerate(success_rates):
        ax.text(i, avg_rewards[i] + 0.5, f'{rate:.1f}% succès', ha='center', 
                color='black', fontsize=10, weight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    print("\nConclusion du test affichée dans le graphique.")


# ====================================================================
# 3. PROGRAMME PRINCIPAL
# ====================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("Q-LEARNING ADAPTATIF - Environnement Dynamique Mobile")
    print("=" * 60)

    # Obtenir la configuration
    try:
        GRID_SIZE, START_POSITION, GOAL_POSITIONS, OBSTACLES, GOAL_MOVE_PROB, OBS_MOVE_PROB = get_user_configuration()
    except Exception:
        print("\nErreur de configuration, utilisation des paramètres par défaut.")
        GRID_SIZE, START_POSITION, GOAL_POSITIONS, OBSTACLES, GOAL_MOVE_PROB, OBS_MOVE_PROB = 5, (0, 0), [(4, 4)], [(2, 2)], 0.3, 0.2

    # Créer l'environnement mobile
    env = MobileGridWorld(
        grid_size=GRID_SIZE,
        start_pos=START_POSITION,
        goal_positions=GOAL_POSITIONS,
        obstacles=OBSTACLES,
        goal_move_prob=GOAL_MOVE_PROB,
        obstacle_move_prob=OBS_MOVE_PROB
    )

    # CORRECTION: Hyperparamètres optimisés pour éviter les boucles
    agent = AdaptiveQLearning(env, alpha=0.3, gamma=0.95, epsilon=1.0)

    # ENTRAÎNEMENT
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60)
    episode_rewards, episode_steps, success_rate = agent.train(num_episodes=2000, max_steps_per_episode=200)

    # Affichage des courbes d'apprentissage
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Courbe 1: Évolution des Récompenses
    ax1.plot(episode_rewards, alpha=0.3, label='Récompense par épisode', color='blue')
    if len(episode_rewards) >= 100:
        ax1.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), 
                 label='Moyenne mobile (100 épisodes)', linewidth=2, color='red')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense Totale')
    ax1.set_title('1. Évolution des Récompenses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Courbe 2: Évolution du Nombre de Pas
    ax2.plot(episode_steps, alpha=0.3, label='Pas par épisode', color='orange')
    if len(episode_steps) >= 100:
        ax2.plot(np.convolve(episode_steps, np.ones(100)/100, mode='valid'), 
                 label='Moyenne mobile (100 épisodes)', linewidth=2, color='green')
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Nombre de Pas')
    ax2.set_title("2. Évolution du Nombre de Pas")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Courbe 3: Évolution du Taux de Succès
    ax3.plot(success_rate, linewidth=2, color='purple')
    ax3.axhline(y=0.7, color='g', linestyle='--', label='Objectif 70%')
    ax3.set_xlabel('Épisode')
    ax3.set_ylabel('Taux de Succès (100 derniers épisodes)')
    ax3.set_title('3. Évolution du Taux de Succès')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Courbe 4: Distribution des Récompenses
    ax4.hist(episode_rewards[-100:], bins=20, alpha=0.7, color='cyan', edgecolor='black')
    ax4.axvline(x=np.mean(episode_rewards[-100:]), color='r', linestyle='--', 
                linewidth=2, label=f'Moyenne: {np.mean(episode_rewards[-100:]):.2f}')
    ax4.set_xlabel('Récompense')
    ax4.set_ylabel('Fréquence')
    ax4.set_title('4. Distribution des Récompenses (100 derniers épisodes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

    # SIMULATION DU MOUVEMENT DE L'AGENT (Post-entraînement)
    execute_policy_and_render(env, agent, max_steps=100, pause_time=0.4) 

    # ANALYSE FINALE (Heatmap)
    print("\n" + "=" * 60)
    print("ANALYSE DES PERFORMANCES ET POLITIQUE APPRISES")
    print("=" * 60)

    # Visualisation de la heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    q_value_grid = np.zeros((env.grid_size, env.grid_size))
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            state = (x, y)
            q_values = [agent.Q[state][a] for a in range(4)]
            q_value_grid[y, x] = np.mean(q_values)
    
    im = ax.imshow(q_value_grid, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(np.arange(env.grid_size))
    ax.set_yticks(np.arange(env.grid_size))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Heatmap des Q-Values Moyennes par État', fontsize=14, fontweight='bold')
    
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            ax.text(x, y, f'{q_value_grid[y, x]:.1f}', ha="center", va="center", 
                   color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Q-Value Moyenne')
    plt.tight_layout()
    plt.show(block=False)
    
    # Test de robustesse
    compare_mobility_strategies(env, agent, num_episodes=50)
    
    # Attendre à la fin pour fermer toutes les fenêtres
    print("\n" + "=" * 60)
    print("Programme terminé. Fermez les fenêtres pour quitter.")
    print("=" * 60)
    plt.show()
