import numpy as np
from client_moodle import get_errors, submit

team_secret_key = 'I22KGMKf3ZqtxxvxklykgAlk1dQZvVqhgfZT1i8NWjOgBC4ntl'

def cal_pop_fitness(pop):
    fitness = [get_errors(team_secret_key, list(i))[0] for i in pop]
    print(fitness)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = 99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover


def distort(vector):
    return np.add(vector, np.random.uniform(low=-1.0, high=1.0, size=(len(vector))))

num_weights = 11

sol_per_pop = 8

num_parents_mating = 4

pop_size = (sol_per_pop, num_weights)

overfit_model = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

new_population = np.array([distort(overfit_model) for i in range(sol_per_pop)])

print(new_population)
print(new_population.shape)

num_generations = 5

for generation in range(num_generations):
    print("Generation : ", generation)
    fitness = cal_pop_fitness(new_population)

    parents = select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    offspring_crossover = crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    offspring_mutation = mutation(offspring_crossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    print("Fitness so far : ")
    print(fitness)

fitness = cal_pop_fitness(new_population)
best_match_idx = np.where(fitness == np.min(fitness))[0][0]

weights_vector = new_population[best_match_idx, :]

print("weights_vector")
print(weights_vector)
print("Best solution fitness : ", fitness[best_match_idx])

#submit stuff

submit(team_secret_key, list(weights_vector))
