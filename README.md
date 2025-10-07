# 🤖 Mobile GridWorld - Projet d'Apprentissage par Renforcement

## 📋 Description du Projet

Ce projet implémente deux approches d'apprentissage par renforcement pour résoudre un problème de navigation dans une grille mobile avec obstacles et objectifs dynamiques :

1. **RL_MobileGrid** : Q-Learning classique avec table Q
2. **RL_MobileGridNeural** : Q-Learning avec approximation par réseau de neurones (Deep Q-Learning)

---

## 📁 Structure du Projet

```
├── RL_MobileGrid/              # Approche classique (Q-Learning)
│   ├── agent.py                # Agent avec table Q
│   ├── environment.py          # Environnement GridWorld
│   └── main.py                 # Programme principal
│
├── RL_MobileGridNeural/        # Approche Deep Learning
│   ├── agent_neural.py         # Agent avec réseau de neurones
│   ├── environment.py          # Environnement GridWorld
│   └── main_neural.py          # Programme principal
│
└── README.md                   # Ce fichier
```

---

## 🔧 Installation et Prérequis

### 1. **Prérequis**
- Python 3.8 ou supérieur
- Visual Studio Code (recommandé)

### 2. **Installation des dépendances**

Ouvrez un terminal dans le dossier du projet et exécutez :

```bash
pip install numpy matplotlib torch
```

**Détail des bibliothèques :**
- `numpy` : Calculs numériques
- `matplotlib` : Visualisation graphique
- `torch` : PyTorch pour le réseau de neurones (uniquement pour RL_MobileGridNeural)

---

## 🚀 Utilisation

### **Option 1 : Q-Learning Classique** (Table Q)

#### Étapes d'exécution dans VS Code :

1. **Ouvrir le dossier** `RL_MobileGrid` dans VS Code
2. **Ouvrir le terminal** (Ctrl + ù ou Terminal → Nouveau Terminal)
3. **Exécuter le programme** :
   ```bash
   python main.py
   ```

#### Déroulement du programme :

```
========================================
CONFIGURATION DE L'ENVIRONNEMENT MOBILE
========================================

Voulez-vous:
  1. Entrer les paramètres manuellement
  2. Générer aléatoirement
Votre choix (1 ou 2):
```

**✅ Choix 1 : Configuration Manuelle**
- Vous devrez entrer :
  - Taille de la grille (ex: 5 pour 5×5)
  - Position initiale de l'agent (x, y)
  - Nombre et positions des goals mobiles
  - Probabilité de mouvement des goals (recommandé : 0.3)
  - Nombre et positions des obstacles mobiles
  - Probabilité de mouvement des obstacles (recommandé : 0.2)

**✅ Choix 2 : Configuration Aléatoire**
- Le programme génère automatiquement :
  - Une grille de taille aléatoire (6×6 à 9×9)
  - 1 à 3 goals mobiles
  - Plusieurs obstacles mobiles
  - Probabilités de mouvement optimales

**Ensuite :**
```
Nombre d'épisodes d'entraînement (recommandé: 1000-3000):
```
- Entrez le nombre d'épisodes (ex: 2000)
- L'entraînement commence automatiquement
- Des statistiques s'affichent tous les 100 épisodes

**À la fin de l'entraînement :**
- Une fenêtre matplotlib s'ouvre avec 4 graphiques :
  - Évolution des récompenses
  - Taux de succès
  - Décroissance de l'exploration (epsilon)
  - Distribution des récompenses

- Le programme propose ensuite :
  ```
  Voulez-vous voir la simulation graphique? (o/n):
  ```
  - Tapez `o` pour voir l'agent naviguer en temps réel
  - Tapez `n` pour passer

- Puis :
  ```
  Voulez-vous comparer avec une politique aléatoire? (o/n):
  ```
  - Tapez `o` pour voir une comparaison de performance
  - Tapez `n` pour terminer

---

### **Option 2 : Deep Q-Learning** (Réseau de Neurones)

#### Étapes d'exécution dans VS Code :

1. **Ouvrir le dossier** `RL_MobileGridNeural` dans VS Code
2. **Ouvrir le terminal** (Ctrl + ù ou Terminal → Nouveau Terminal)
3. **Exécuter le programme** :
   ```bash
   python main_neural.py
   ```

#### Déroulement du programme :

Le programme suit **exactement les mêmes étapes** que l'Option 1, mais avec des différences importantes :

**🔹 Architecture du réseau de neurones :**
```
========================================
PHASE D'ENTRAÎNEMENT DU RÉSEAU DE NEURONES
========================================
Architecture: 2 -> 64 -> 64 -> 4
Replay Buffer: 5000 transitions
```

**🔹 Différences clés :**
- Utilise un **réseau de neurones** au lieu d'une table Q
- Emploie un **replay buffer** pour stabiliser l'apprentissage
- Convergence généralement **plus rapide** et **plus robuste**
- Meilleure généralisation à de grandes grilles

**🔹 Options de visualisation :**
Après l'entraînement, vous avez le choix :

1. **Voir la simulation graphique :**
   ```
   Voulez-vous voir la simulation graphique? (o/n):
   ```
   - `o` : Affiche l'agent en action avec :
     - **Vue gauche** : Environnement en temps réel
     - **Vue droite** : Q-Values et politique apprise
   - `n` : Passer directement à la suite

2. **Comparer avec une politique aléatoire :**
   ```
   Voulez-vous comparer avec une politique aléatoire? (o/n):
   ```
   - `o` : Graphique comparatif des performances
   - `n` : Terminer le programme

---

## 📊 Interprétation des Résultats

### **Courbes d'apprentissage :**

1. **Évolution des Récompenses**
   - Doit augmenter au fil des épisodes
   - La moyenne mobile (rouge) doit converger vers une valeur positive

2. **Taux de Succès**
   - Doit tendre vers 100% (ou proche)
   - Indique le pourcentage d'épisodes où le goal est atteint

3. **Décroissance d'Epsilon**
   - Montre la transition exploration → exploitation
   - Doit diminuer progressivement de 1.0 vers 0.05

4. **Distribution des Récompenses**
   - Doit montrer une concentration vers des valeurs élevées
   - Peu de récompenses négatives en fin d'entraînement

### **Simulation Graphique :**

**Légende :**
- 🔵 **Point bleu** : Agent
- ⭐ **Étoile rouge** : Goal non atteint
- ⭐ **Étoile verte** : Goal atteint
- ⬛ **Carré gris** : Obstacle mobile
- 🎯 **Flèches cyan** : Politique apprise (vue droite)
- 🟢 **Intensité verte** : Q-Values moyennes (vue droite)

---

## ⚙️ Paramètres Recommandés

### **Pour une grille 5×5 :**
- Épisodes : 1000-2000
- Goals : 1-2
- Obstacles : 3-5
- Goal move prob : 0.3
- Obstacle move prob : 0.2

### **Pour une grille 8×8 ou plus grande :**
- Épisodes : 3000-5000 (classique) ou 2000-3000 (neural)
- Goals : 2-3
- Obstacles : 8-15
- Goal move prob : 0.2-0.4
- Obstacle move prob : 0.1-0.3

---

## 🎯 Objectifs Pédagogiques

Ce projet permet de :
1. ✅ Comparer Q-Learning classique vs. Deep Q-Learning
2. ✅ Observer l'impact de l'exploration (epsilon-greedy)
3. ✅ Comprendre le replay buffer et la stabilisation
4. ✅ Visualiser l'apprentissage en temps réel
5. ✅ Gérer des environnements dynamiques (goals et obstacles mobiles)

---

## 🐛 Dépannage

### **Problème : "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch
```

### **Problème : Les fenêtres matplotlib ne s'affichent pas**
- Vérifiez que matplotlib est installé : `pip install matplotlib`
- Sur certains systèmes, ajoutez : `pip install PyQt5`

### **Problème : L'agent ne converge pas**
- Augmentez le nombre d'épisodes
- Réduisez la probabilité de mouvement des obstacles
- Augmentez la taille de la grille si elle est trop petite

### **Problème : "RuntimeError: CUDA out of memory"** (Neural)
- Réduisez la taille du replay buffer (dans `agent_neural.py`)
- Utilisez CPU au lieu de GPU (PyTorch le fait automatiquement si nécessaire)

---

## 📝 Notes pour le Professeur

### **Temps d'exécution estimé :**
- Configuration : 1-2 minutes
- Entraînement (1000 épisodes) : 2-5 minutes
- Simulation graphique : 30 secondes - 2 minutes

### **Points d'évaluation suggérés :**
1. ✅ Convergence de l'algorithme
2. ✅ Qualité de la politique apprise
3. ✅ Comparaison des deux approches
4. ✅ Gestion des environnements dynamiques
5. ✅ Clarté du code et documentation

### **Extensions possibles :**
- Ajouter des récompenses intermédiaires (shaped rewards)
- Implémenter Double DQN ou Dueling DQN
- Tester avec des grilles plus grandes (15×15)
- Ajouter plusieurs types d'obstacles avec comportements différents

---

## 👥 Auteurs

- **Projet réalisé dans le cadre du cours d'Apprentissage par Renforcement**
- **Université : [Votre Université]**
- **Date : Octobre 2025**

---

## 📄 Licence

Ce projet est à usage éducatif uniquement.

---

## 📧 Contact

Pour toute question concernant ce projet, veuillez contacter :
- Email : zakariaezemmahi@gmail.com
- GitHub : zakariazemmahi

---

**Bon entraînement ! 🚀🤖**
