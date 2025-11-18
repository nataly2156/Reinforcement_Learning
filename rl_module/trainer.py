import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend para guardar sin mostrar
import matplotlib.pyplot as plt
from datetime import datetime

class RLTrainer:
    """
    Clase para entrenar y evaluar agentes de Reinforcement Learning
    """
    
    def __init__(self, environment, agent):
        self.env = environment
        self.agent = agent
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilons': []
        }
    
    def train(self, n_episodes=1000, max_steps=100, verbose=False):
        """
        Entrena al agente por un número de episodios
        
        Args:
            n_episodes: número de episodios de entrenamiento
            max_steps: máximo de pasos por episodio
            verbose: si mostrar progreso
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Elegir acción
                action = self.agent.choose_action(state)
                
                # Ejecutar acción
                next_state, reward, done = self.env.step(action)
                
                # Aprender
                self.agent.learn(state, action, reward, next_state, done)
                
                # Actualizar estado y contadores
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Reducir exploración
            self.agent.decay_epsilon()
            
            # Guardar métricas
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilons'].append(self.agent.epsilon)
            
            # Mostrar progreso
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                print(f"Episodio {episode + 1}/{n_episodes} - "
                      f"Recompensa promedio: {avg_reward:.2f} - "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        return self.training_history
    
    def test(self, n_episodes=10, render=False):
        """
        Prueba al agente entrenado (sin aprendizaje)
        """
        test_rewards = []
        test_steps = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            if render:
                print(f"\n--- Episodio de prueba {episode + 1} ---")
                self.env.render()
            
            while steps < 100:
                # Usar política aprendida (sin exploración)
                action = self.agent.get_best_action(state)
                next_state, reward, done = self.env.step(action)
                
                if render:
                    self.env.render()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            test_steps.append(steps)
        
        return {
            'avg_reward': np.mean(test_rewards),
            'avg_steps': np.mean(test_steps),
            'rewards': test_rewards,
            'steps': test_steps
        }
    
    def plot_training_progress(self, save_path='static/images/training_progress.png'):
        """
        Genera gráficos del progreso de entrenamiento
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico 1: Recompensas por episodio
        axes[0, 0].plot(self.training_history['episodes'], 
                       self.training_history['rewards'], 
                       alpha=0.3, label='Recompensa por episodio')
        
        # Media móvil de 100 episodios
        window = 100
        if len(self.training_history['rewards']) >= window:
            moving_avg = np.convolve(self.training_history['rewards'], 
                                    np.ones(window)/window, 
                                    mode='valid')
            axes[0, 0].plot(range(window-1, len(self.training_history['episodes'])), 
                          moving_avg, 
                          'r-', 
                          linewidth=2, 
                          label=f'Media móvil ({window} ep.)')
        
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa Total')
        axes[0, 0].set_title('Evolución de la Recompensa')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Número de pasos por episodio
        axes[0, 1].plot(self.training_history['episodes'], 
                       self.training_history['steps'], 
                       'g-', 
                       alpha=0.5)
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Número de Pasos')
        axes[0, 1].set_title('Eficiencia del Agente')
        axes[0, 1].grid(True, alpha=0.3)