# environment.py

import numpy as np

class MobileGridWorld:
    """Environnement GridWorld avec goals et obstacles mobiles"""

    def __init__(self, grid_size=5, start_pos=(0, 0), goal_positions=None, obstacles=None,
                 goal_move_prob=0.3, obstacle_move_prob=0.2):
        """
        Initialiser l'environnement
        
        Args:
            grid_size: Taille de la grille (int)
            start_pos: Position initiale de l'agent (tuple)
            goal_positions: Liste des positions des buts [(x1,y1), (x2,y2), ...]
            obstacles: Liste des positions d'obstacles [(x1,y1), (x2,y2), ...]
            goal_move_prob: Probabilité de mouvement des goals à chaque step
            obstacle_move_prob: Probabilité de mouvement des obstacles à chaque step
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.initial_goal_positions = goal_positions if goal_positions is not None else [(4, 4)]
        self.initial_obstacles = obstacles if obstacles is not None else []
        self.goal_move_prob = goal_move_prob
        self.obstacle_move_prob = obstacle_move_prob
        
        # Positions actuelles (mobiles)
        self.goal_positions = list(self.initial_goal_positions)
        self.obstacles = list(self.initial_obstacles)
        self.agent_pos = self.start_pos
        self.goals_reached = set()
        
        # CORRECTION: Historique des positions pour détecter les boucles
        self.position_history = []
        self.max_history = 20  # Garder les 20 dernières positions
        
        # Validation des positions initiales
        self._validate_positions()

    def _validate_positions(self):
        """Valider que les positions sont dans la grille et cohérentes"""
        def is_valid(pos):
            return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        
        if not is_valid(self.start_pos):
            raise ValueError(f"Position initiale {self.start_pos} hors de la grille!")
        
        if len(self.initial_goal_positions) == 0:
            raise ValueError("Au moins un goal doit être défini!")
        
        for goal_pos in self.initial_goal_positions:
            if not is_valid(goal_pos):
                raise ValueError(f"Position du but {goal_pos} hors de la grille!")
        
        for obs in self.initial_obstacles:
            if not is_valid(obs):
                raise ValueError(f"Obstacle {obs} hors de la grille!")

    def _move_entity(self, pos):
        """Déplacer une entité (goal ou obstacle) d'une case aléatoirement"""
        x, y = pos
        # Choisir une direction aléatoire (0=Haut, 1=Bas, 2=Gauche, 3=Droite, 4=Rester)
        direction = np.random.randint(0, 5)
        
        new_x, new_y = x, y
        if direction == 0:  # Haut
            new_y = max(0, y - 1)
        elif direction == 1:  # Bas
            new_y = min(self.grid_size - 1, y + 1)
        elif direction == 2:  # Gauche
            new_x = max(0, x - 1)
        elif direction == 3:  # Droite
            new_x = min(self.grid_size - 1, x + 1)
        # direction == 4: reste sur place
        
        return (new_x, new_y)

    def _move_goals(self):
        """Déplacer les goals avec une certaine probabilité"""
        new_goal_positions = []
        for goal_pos in self.goal_positions:
            if np.random.random() < self.goal_move_prob:
                # Bouger le goal
                new_pos = self._move_entity(goal_pos)
                # Éviter de bouger sur l'agent ou un obstacle
                attempts = 0
                while (new_pos == self.agent_pos or new_pos in self.obstacles) and attempts < 10:
                    new_pos = self._move_entity(goal_pos)
                    attempts += 1
                new_goal_positions.append(new_pos)
            else:
                new_goal_positions.append(goal_pos)
        self.goal_positions = new_goal_positions

    def _move_obstacles(self):
        """Déplacer les obstacles avec une certaine probabilité"""
        new_obstacles = []
        for obs_pos in self.obstacles:
            if np.random.random() < self.obstacle_move_prob:
                # Bouger l'obstacle
                new_pos = self._move_entity(obs_pos)
                # Éviter de bouger sur l'agent ou un goal
                attempts = 0
                while (new_pos == self.agent_pos or new_pos in self.goal_positions) and attempts < 10:
                    new_pos = self._move_entity(obs_pos)
                    attempts += 1
                new_obstacles.append(new_pos)
            else:
                new_obstacles.append(obs_pos)
        self.obstacles = new_obstacles

    def _is_in_loop(self):
        """
        CORRECTION: Détecte si l'agent est piégé dans une boucle
        Retourne True si la position actuelle apparaît trop souvent récemment
        """
        if len(self.position_history) < 8:
            return False
        
        # Compter les occurrences de la position actuelle dans l'historique récent
        recent_positions = self.position_history[-8:]
        count = recent_positions.count(self.agent_pos)
        
        # Si la même position apparaît plus de 3 fois sur les 8 derniers pas
        return count >= 3

    def reset(self):
        """Réinitialiser l'environnement"""
        self.agent_pos = self.start_pos
        self.goal_positions = list(self.initial_goal_positions)
        self.obstacles = list(self.initial_obstacles)
        self.goals_reached = set()
        self.position_history = []  # CORRECTION: Réinitialiser l'historique
        return self.agent_pos

    def step(self, action):
        """
        Actions: 0=Haut, 1=Bas, 2=Gauche, 3=Droite
        
        CORRECTION: Ajout de pénalité pour les boucles infinies
        """
        x, y = self.agent_pos

        # Calculer la nouvelle position de l'agent
        new_x, new_y = x, y
        if action == 0:  # Haut
            new_y = max(0, y - 1)
        elif action == 1:  # Bas
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # Gauche
            new_x = max(0, x - 1)
        elif action == 3:  # Droite
            new_x = min(self.grid_size - 1, x + 1)

        # Vérifier si la nouvelle position est un obstacle
        if (new_x, new_y) in self.obstacles:
            # L'agent reste à sa position actuelle
            reward = -1  # Pénalité pour avoir heurté un obstacle
        else:
            # L'agent se déplace
            self.agent_pos = (new_x, new_y)
            
            # CORRECTION: Ajouter la position à l'historique
            self.position_history.append(self.agent_pos)
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # CORRECTION: Pénalité si l'agent est dans une boucle
            if self._is_in_loop():
                reward = -2.0  # Forte pénalité pour tourner en rond
            # Vérifier si l'agent a atteint UN goal (AVANT que les goals bougent)
            elif self.agent_pos in self.goal_positions and self.agent_pos not in self.goals_reached:
                self.goals_reached.add(self.agent_pos)
                reward = 10  # Récompense pour atteindre un goal
            else:
                reward = -0.1  # Coût de déplacement normal

        # Déplacer les goals et obstacles APRÈS le mouvement de l'agent
        self._move_goals()
        self._move_obstacles()

        # L'épisode est terminé dès qu'UN goal est atteint
        done = len(self.goals_reached) > 0

        return self.agent_pos, reward, done