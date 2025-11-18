# Proyecto de Aprendizaje por Refuerzo - GridWorld Q-Learning

Aplicación web interactiva que implementa un agente inteligente mediante el algoritmo Q-Learning para resolver un problema de navegación en un entorno tipo GridWorld. El proyecto muestra de forma clara cómo un agente aprende una política óptima a través de la exploración y la interacción continua con un entorno discreto.


## *Descripción del Entorno*

El proyecto utiliza un entorno tipo GridWorld, una cuadrícula donde un agente debe desplazarse desde una posición inicial hacia una meta evitando obstáculos. Las características principales del entorno son:

- Grid de tamaño configurable (por defecto 5x5).
- Posición inicial ubicada en la esquina superior izquierda.
- Meta ubicada en la esquina inferior derecha.
- Obstáculos distribuidos en posiciones fijas dentro del grid.
- Acciones permitidas: moverse arriba, abajo, izquierda y derecha.
- Restricciones: el agente no puede salir del grid y no puede atravesar obstáculos.
- Función de recompensa basada en penalizar pasos innecesarios, castigar choques y premiar alcanzar la meta.

Este entorno permite observar con claridad la evolución del aprendizaje del agente y la forma en que decide sus trayectorias.


## *Descripción del Algoritmo Utilizado*

El agente implementa el algoritmo Q-Learning, un método de aprendizaje por refuerzo off-policy que aprende el valor óptimo de cada par estado-acción mediante interacción repetida con el entorno.

Aspectos clave del algoritmo:

- Uso de una política epsilon-greedy para equilibrar exploración y explotación.
- Actualización de la tabla Q basada en la ecuación de Bellman.
- Aprendizaje libre de modelo: no requiere conocer dinámicas internas del entorno.
- Uso de hiperparámetros como la tasa de aprendizaje, factor de descuento, valor inicial de epsilon y su decaimiento.
- Entrenamiento mediante episodios completos que inician desde el estado inicial y finalizan al llegar a la meta o agotar los pasos permitidos.

El algoritmo ajusta gradualmente los valores Q hasta converger hacia una política cada vez más eficiente.


## *Descripción del Comportamiento Obtenido*

Durante el entrenamiento se observó la progresión del agente en su capacidad de navegación:

- En los primeros episodios predomina la exploración y el agente realiza movimientos aleatorios, con frecuentes choques contra obstáculos.
- A medida que el epsilon disminuye, el agente identifica trayectorias favorables y reduce el número de pasos necesarios para alcanzar la meta.
- En etapas avanzadas del entrenamiento la política se estabiliza, logrando comportamientos consistentes y cercanos a la ruta óptima.
- Las recompensas acumuladas aumentan progresivamente y el número de pasos decrece, evidenciando la mejora en la calidad de las decisiones.
- Finalmente, el agente aprende una ruta eficiente evitando obstáculos y alcanzando la meta de forma estable.

Este comportamiento demuestra la efectividad del algoritmo Q-Learning para adquirir estrategias de navegación óptimas en entornos discretos como GridWorld.


*¡Gracias por explorar nuestro proyecto de Aprendizaje por Refuerzo! 🤖*
