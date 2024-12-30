import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for y = 2x^2 + 3x + 5
np.random.seed(42)
x = np.linspace(-10, 10, 100)
true_a, true_b, true_c = 2, 3, 5
y = true_a * x**2 + true_b * x + true_c + np.random.normal(0, 10, size=x.shape)

# Genetic Algorithm Parameters
POP_SIZE = 50
GENS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Objective function to minimize
def fitness_function(coeffs):
    a, b, c = coeffs
    predictions = a * x**2 + b * x + c
    error = np.mean((y - predictions) ** 2)
    return error

# Initialize population
def initialize_population(size):
    return np.random.uniform(-10, 10, (size, 3))

# Selection: Tournament Selection
def select_parents(population, fitnesses):
    idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
    return population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]

# Crossover: Single-Point Crossover
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Mutation: Random Perturbation
def mutate(individual):
    if np.random.rand() < MUTATION_RATE:
        idx = np.random.randint(len(individual))
        individual[idx] += np.random.normal()
    return individual

# Main Genetic Algorithm Loop
population = initialize_population(POP_SIZE)
best_fit_per_gen = []

for generation in range(GENS):
    fitnesses = np.array([fitness_function(ind) for ind in population])
    new_population = []

    for _ in range(POP_SIZE // 2):
        # Select parents
        parent1 = select_parents(population, fitnesses)
        parent2 = select_parents(population, fitnesses)
        # Apply crossover
        child1, child2 = crossover(parent1, parent2)
        # Apply mutation
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    population = np.array(new_population)
    best_fit_per_gen.append(np.min(fitnesses))

    if generation % 10 == 0:
        print(f"Generation {generation}, Best Fitness: {np.min(fitnesses)}")

# Extract the best solution
best_solution = population[np.argmin([fitness_function(ind) for ind in population])]
print("\nBest Solution Found:")
print(f"a = {best_solution[0]:.4f}, b = {best_solution[1]:.4f}, c = {best_solution[2]:.4f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'b.', label="True Data")
plt.plot(x, best_solution[0] * x**2 + best_solution[1] * x + best_solution[2], 'r-', label="GA Fit")
plt.legend()
plt.title("Symbol Regression with Genetic Algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot fitness progression
plt.figure(figsize=(10, 5))
plt.plot(best_fit_per_gen, 'g-')
plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()
