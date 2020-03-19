import numpy
from client_moodle import get_errors, submit

team_secret_key = 'I22KGMKf3ZqtxxvxklykgAlk1dQZvVqhgfZT1i8NWjOgBC4ntl'

def cal_pop_fitness(pop):
    fitness = [get_errors(team_secret_key, list(i))[0] for i in pop]
    print(fitness)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = 99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover


num_weights = 11

sol_per_pop = 8

num_parents_mating = 4

pop_size = (sol_per_pop, num_weights)

new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

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
best_match_idx = numpy.where(fitness == numpy.min(fitness))

weights_vector = new_population[best_match_idx, :]

print("weights_vector")
print(weights_vector)
print("Best solution fitness : ", fitness[best_match_idx])
