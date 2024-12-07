import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from deap import base, creator, tools, algorithms

# Data Preparation
def prepare_data(filepath):
    # Load and preprocess the data
    data = pd.read_csv(filepath)
    data = data.dropna()  # Drop missing values

    # Convert categorical data to numeric
    encoder = LabelEncoder()
    data['weather_conditions'] = encoder.fit_transform(data['weather_conditions'])

    features = data[['vehicle_counts', 'average_speeds', 'weather_conditions', 'signal_timings']]
    target = data['congestion_level']

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Traffic Prediction with Neural Networks
def build_and_train_model(X_train, y_train, X_test, y_test):
    # Build the neural network model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    return model

# Signal Optimization with Genetic Algorithms
def genetic_algorithm_optimization(predicted_congestion):
    # Define a fitness function
    def fitness(individual):
        return sum(individual),  # Example fitness function, returning a tuple

    # DEAP setup for genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    # Run the genetic algorithm
    population = toolbox.population(n=300)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=None, halloffame=None, verbose=True)

    # Return the best individual
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

# Dynamic Routing with Swarm Intelligence
class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # Going back to the start
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

# Integrate All Components
def run_itms(filepath):
    X_train, X_test, y_train, y_test = prepare_data(filepath)
    model = build_and_train_model(X_train, y_train, X_test, y_test)

    predicted_congestion = model.predict(X_test)
    optimized_signals = genetic_algorithm_optimization(predicted_congestion)

    distances = np.array([
        [np.inf, 2, 2, 5, 7],
        [2, np.inf, 4, 8, 2],
        [2, 4, np.inf, 1, 3],
        [5, 8, 1, np.inf, 2],
        [7, 2, 3, 2, np.inf]
    ])

    ant_colony = AntColony(distances, 3, 1, 10, 0.95, alpha=1, beta=2)
    dynamic_routes = ant_colony.run()

    return optimized_signals, dynamic_routes

# Execute the ITMS
optimized_signals, dynamic_routes = run_itms('traffic_data.csv')
print('Optimized Signal Timings:', optimized_signals)
print('Dynamic Routes:', dynamic_routes)
