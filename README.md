# Q-Learning Adaptatif - Environnement Mobile GridWorld

## 📝 Description

Implémentation d'un agent Q-Learning adaptatif capable d'apprendre à naviguer dans un environnement dynamique avec des objectifs (goals) et des obstacles mobiles. Ce projet démontre l'apprentissage par renforcement dans un contexte non-stationnaire.

## ✨ Caractéristiques

- **Environnement dynamique** : Goals et obstacles se déplacent aléatoirement
- **Q-Learning adaptatif** : L'agent apprend à s'adapter aux changements
- **Détection de boucles** : Évite que l'agent tourne en rond indéfiniment
- **Visualisation en temps réel** : Animation des mouvements de l'agent
- **Analyse des performances** : Graphiques d'apprentissage et heatmaps
- **Configuration flexible** : Paramètres manuels ou génération aléatoire

## 🚀 Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de packages Python)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Lancer le programme

```bash
python main.py
```

### Options de configuration

Au démarrage, vous avez deux options :

1. **Configuration manuelle** : Définir manuellement la taille de la grille, positions des goals, obstacles, etc.
2. **Génération aléatoire** : Laisser le programme générer un environnement aléatoire

### Exemple d'exécution

```
CONFIGURATION DE L'ENVIRONNEMENT MOBILE
Voulez-vous:
  1. Entrer les paramètres manuellement
  2. Générer aléatoirement
Votre choix (1 ou 2): 2
```

## 📊 Fonctionnalités

### 1. Entraînement de l'agent
- Apprentissage par Q-Learning sur plusieurs épisodes
- Adaptation dynamique du taux d'exploration (epsilon)
- Détection automatique de stagnation

### 2. Visualisations
- **Animation en temps réel** : Mouvements de l'agent, goals et obstacles
- **Politique apprise** : Affichage des Q-values et directions optimales
- **Courbes d'apprentissage** :
  - Évolution des récompenses
  - Nombre de pas par épisode
  - Taux de succès
  - Distribution des récompenses
- **Heatmap** : Visualisation des Q-values moyennes par état

### 3. Comparaison de stratégies
- Politique Q-Learning vs Politique aléatoire
- Évaluation sur plusieurs épisodes

## 🧠 Algorithme

### Q-Learning Adaptatif

L'algorithme utilise plusieurs techniques pour gérer l'environnement dynamique :

1. **Learning rate adaptatif** : Décroît avec le nombre de visites
2. **Exploration intelligente** : Favorise les actions moins explorées
3. **Bonus d'exploration** : Encourage la découverte de nouveaux états
4. **Détection de boucles** : Pénalise les comportements répétitifs
5. **Réinitialisation adaptative** : Augmente l'exploration si stagnation détectée

### Récompenses

- **+10** : Atteindre un goal
- **-0.1** : Coût de déplacement normal
- **-1** : Collision avec un obstacle
- **-2** : Détection de boucle (tourner en rond)
- **-0.5** : Revisiter trop souvent le même état

## 📁 Structure du projet

```
QLearning-Mobile-GridWorld/
│
├── environment.py       # Classe MobileGridWorld
├── agent.py            # Classe AdaptiveQLearning
├── main.py             # Programme principal
├── README.md           # Documentation
├── requirements.txt    # Dépendances
└── .gitignore         # Fichiers à ignorer
```

## 🔧 Paramètres configurables

### Environnement
- `grid_size` : Taille de la grille (ex: 5 pour 5x5)
- `start_pos` : Position initiale de l'agent
- `goal_positions` : Liste des positions des goals
- `obstacles` : Liste des positions des obstacles
- `goal_move_prob` : Probabilité de mouvement des goals (0.0-1.0)
- `obstacle_move_prob` : Probabilité de mouvement des obstacles (0.0-1.0)

### Agent Q-Learning
- `alpha` : Taux d'apprentissage (défaut: 0.3)
- `gamma` : Facteur de discount (défaut: 0.95)
- `epsilon` : Taux d'exploration initial (défaut: 1.0)
- `num_episodes` : Nombre d'épisodes d'entraînement (défaut: 2000)
- `max_steps_per_episode` : Limite de pas par épisode (défaut: 200)

## 📈 Résultats attendus

Après l'entraînement, l'agent devrait :
- ✅ Atteindre un taux de succès > 70%
- ✅ Réduire progressivement le nombre de pas nécessaires
- ✅ Éviter les boucles infinies
- ✅ S'adapter aux mouvements des goals et obstacles

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

Votre Nom - [Votre GitHub](https://github.com/votre-username)

## 🙏 Remerciements

- Inspiré par les principes du Reinforcement Learning
- Basé sur l'algorithme Q-Learning de Watkins & Dayan (1992)

---

**Note** : Ce projet est à but éducatif pour comprendre l'apprentissage par renforcement dans des environnements dynamiques.
