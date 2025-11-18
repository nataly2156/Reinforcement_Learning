import numpy as np

class GridWorld:
    """
    Entorno GridWorld simple para Reinforcement Learning.
    El agente debe navegar desde el inicio hasta la meta evitando obstáculos.
    """
    
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 0: Arriba, 1: Derecha, 2: Abajo, 3: Izquierda
        
        # Posiciones especiales
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # Posiciones de obstáculos
        
        # Estado actual
        self.current_pos = self.start_pos
        
        # Direcciones de movimiento
        self.actions = {
            0: (-1, 0),  # Arriba
            1: (0, 1),   # Derecha
            2: (1, 0),   # Abajo
            3: (0, -1)   # Izquierda
        }
        
    def reset(self):
        """Reinicia el entorno al estado inicial"""
        self.current_pos = self.start_pos
        return self._pos_to_state(self.current_pos)
    
    def _pos_to_state(self, pos):
        """Convierte posición (fila, col) a número de estado"""
        return pos[0] * self.grid_size + pos[1]
    
    def _state_to_pos(self, state):
        """Convierte número de estado a posición (fila, col)"""
        return (state // self.grid_size, state % self.grid_size)
    
    def step(self, action):
        """
        Ejecuta una acción y retorna: (nuevo_estado, recompensa, terminado)
        """
        # Calcular nueva posición
        delta = self.actions[action]
        new_pos = (self.current_pos[0] + delta[0], 
                   self.current_pos[1] + delta[1])
        
        # Verificar límites del grid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            # Verificar si no es obstáculo
            if new_pos not in self.obstacles:
                self.current_pos = new_pos
        
        # Calcular recompensa
        reward = self._get_reward()
        
        # Verificar si terminó
        done = self.current_pos == self.goal_pos
        
        # Retornar estado, recompensa y si terminó
        state = self._pos_to_state(self.current_pos)
        return state, reward, done
    
    def _get_reward(self):
        """Calcula la recompensa según la posición actual"""
        if self.current_pos == self.goal_pos:
            return 100  # Gran recompensa por llegar a la meta
        elif self.current_pos in self.obstacles:
            return -10  # Penalización por obstáculo
        else:
            return -1  # Pequeña penalización por cada paso (incentiva eficiencia)
    
    def render(self):
        """Muestra el estado actual del entorno"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '·'
        
        # Marcar obstáculos
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        # Marcar meta
        grid[self.goal_pos] = 'G'
        
        # Marcar agente
        grid[self.current_pos] = 'A'
        
        print("\n" + "="*20)
        for row in grid:
            print(" ".join(row))
        print("="*20 + "\n")
    
    def get_grid_representation(self):
        """Retorna una representación del grid para visualización"""
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # 0: vacío, 1: obstáculo, 2: meta, 3: agente
        for obs in self.obstacles:
            grid[obs] = 1
        
        grid[self.goal_pos] = 2
        grid[self.current_pos] = 3
        
        return grid.tolist()