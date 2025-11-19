from flask import Flask, render_template, jsonify, request
import os
from rl_module import GridWorld, QLearningAgent, RLTrainer

app = Flask(__name__)

# Variables globales para mantener el estado
env = None
agent = None
trainer = None
training_completed = False

# Crear carpeta para imágenes si no existe
os.makedirs('static/images', exist_ok=True)

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/rl/conceptos')
def rl_conceptos():
    """Página de conceptos básicos de RL"""
    return render_template('rl_conceptos.html')

@app.route('/rl/practico')
def rl_practico():
    """Página del caso práctico"""
    return render_template('rl_practico.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_environment():
    """Inicializa el entorno y el agente"""
    global env, agent, trainer, training_completed
    
    try:
        # Obtener parámetros de la solicitud
        data = request.get_json()
        grid_size = data.get('grid_size', 5)
        learning_rate = data.get('learning_rate', 0.1)
        discount_factor = data.get('discount_factor', 0.95)
        epsilon = data.get('epsilon', 1.0)
        
        # Crear entorno y agente
        env = GridWorld(grid_size=grid_size)
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
        trainer = RLTrainer(env, agent)
        training_completed = False
        
        return jsonify({
            'success': True,
            'message': 'Entorno inicializado correctamente',
            'grid': env.get_grid_representation(),
            'n_states': env.n_states,
            'n_actions': env.n_actions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error al inicializar: {str(e)}'
        }), 500

@app.route('/api/train', methods=['POST'])
def train_agent():
    """Entrena al agente"""
    global trainer, training_completed
    
    if trainer is None:
        return jsonify({
            'success': False,
            'message': 'Debe inicializar el entorno primero'
        }), 400
    
    try:
        # Obtener parámetros
        data = request.get_json()
        n_episodes = data.get('n_episodes', 1000)
        
        # Entrenar
        history = trainer.train(n_episodes=n_episodes, verbose=True)
        
        # Generar gráficos
        progress_path = trainer.plot_training_progress()
        policy_path = trainer.visualize_policy()
        
        # Guardar modelo
        agent.save('static/trained_agent.pkl')
        
        training_completed = True
        
        # Calcular estadísticas finales
        avg_last_100 = sum(history['rewards'][-100:]) / min(100, len(history['rewards']))
        
        return jsonify({
            'success': True,
            'message': 'Entrenamiento completado',
            'final_epsilon': agent.epsilon,
            'avg_reward_last_100': avg_last_100,
            'total_episodes': len(history['episodes']),
            'progress_image': '/static/images/training_progress.png',  
            'policy_image': '/static/images/policy.png'                
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error durante el entrenamiento: {str(e)}'
        }), 500

@app.route('/api/test', methods=['POST'])
def test_agent():
    """Prueba al agente entrenado"""
    global trainer, training_completed
    
    if trainer is None or not training_completed:
        return jsonify({
            'success': False,
            'message': 'Debe entrenar al agente primero'
        }), 400
    
    try:
        # Probar agente
        results = trainer.test(n_episodes=10)
        
        return jsonify({
            'success': True,
            'avg_reward': results['avg_reward'],
            'avg_steps': results['avg_steps'],
            'message': f'Prueba completada: {results["avg_reward"]:.2f} recompensa promedio'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error durante la prueba: {str(e)}'
        }), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_episode():
    """Simula un episodio completo y retorna la trayectoria"""
    global env, agent, training_completed
    
    if env is None or agent is None or not training_completed:
        return jsonify({
            'success': False,
            'message': 'Debe entrenar al agente primero'
        }), 400
    
    try:
        trajectory = []
        state = env.reset()
        trajectory.append(env.get_grid_representation())
        
        steps = 0
        total_reward = 0
        max_steps = 50
        
        while steps < max_steps:
            action = agent.get_best_action(state)
            next_state, reward, done = env.step(action)
            
            trajectory.append(env.get_grid_representation())
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return jsonify({
            'success': True,
            'trajectory': trajectory,
            'total_reward': total_reward,
            'steps': steps,
            'reached_goal': done
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error durante la simulación: {str(e)}'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_training():
    """Reinicia el entrenamiento"""
    global env, agent, trainer, training_completed
    
    env = None
    agent = None
    trainer = None
    training_completed = False
    
    return jsonify({
        'success': True,
        'message': 'Sistema reiniciado correctamente'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)