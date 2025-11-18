import numpy as np
import pickle

class QLearningAgent:
    """
    Agente que aprende usando el algoritmo Q-Learning.
    
    Parámetros:
    - n_states: número de estados en el entorno
    - n_actions: número de acciones posibles
    - learning_rate (α): qué tan rápido aprende (0-1)
    - discount_factor (γ): importancia de recompensas futuras (0-1)
    - epsilon: probabilidad de exploración (0-1)
    - epsilon_decay: tasa de decaimiento de epsilon
    - epsilon_min: valor mínimo de epsilon
    """
    
    def __init__(self, n_states, n_actions, 
                 learning_rate=0.1, 
                 discount_factor=0.95, 
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate  # α
        self.discount_factor = discount_factor  # γ
        self.epsilon = epsilon  # ε
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializar Q-table con valores pequeños aleatorios
        self.q_table = np.random.uniform(low=-1, high=1, 
                                         size=(n_states, n_actions))
    
    def choose_action(self, state):
        """
        Selecciona una acción usando estrategia ε-greedy:
        - Con probabilidad ε: exploración (acción aleatoria)
        - Con probabilidad (1-ε): explotación (mejor acción conocida)
        """
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(0, self.n_actions)
        else:
            # Explotación: mejor acción según Q-table
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        
        if done:
            # Si es estado terminal, no hay valor futuro
            target_q = reward
        else:
            # Calcular el valor objetivo con la ecuación de Bellman
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Actualizar Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Reduce gradualmente la exploración"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_best_action(self, state):
        """Retorna la mejor acción para un estado dado (sin exploración)"""
        return np.argmax(self.q_table[state])
    
    def save(self, filepath):
        """Guarda el agente entrenado"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Carga un agente previamente entrenado"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']