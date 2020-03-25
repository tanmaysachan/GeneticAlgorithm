import numpy as np
from client_moodle import get_errors, submit

team_secret_key = 'I22KGMKf3ZqtxxvxklykgAlk1dQZvVqhgfZT1i8NWjOgBC4ntl'
# team_secret_key = 'iYL7HCll9GU7hDiil3sQKqnJJX107Zs2YD4dLDeQWzdWpMInCr' # Fiza - team10


def cal_pop_fitness(pop):
    fitness = [get_errors(team_secret_key, list(i)) for i in pop]
    # fitness = [[i,2*i] for i in range(len(pop))]
    fitness = [abs(i[0]-i[1]) for i in fitness]
    # fitness = [i[0]*0.3+i[1] for i in fitness]
    # print(fitness)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    # can change to column stack then argsort and split firse
    # looks cleaner
    # but slower (?)
    for parent_num in range(num_parents):
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, : ] = pop[min_fitness_idx, : ]
        fitness[min_fitness_idx] = 1e1000
    # print(parents)
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        parent1_idx = np.random.randint(0,parents.shape[0])
        parent2_idx = np.random.randint(0,parents.shape[0])
        crossover_point = np.random.randint(1,offspring_size[0])
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        while(True):
            mutate_index = np.random.randint(0,10)
            cur = offspring_crossover[idx, mutate_index]
            random_value = np.random.uniform(-1,1, 1)
            if abs((offspring_crossover[idx, mutate_index] * random_value)[0]) <= 10:
                offspring_crossover[idx, mutate_index] = offspring_crossover[idx, mutate_index] * random_value
                break
            else:
                continue
    return offspring_crossover

def error(error_value):
    print(error_value,sum(error_value))


def distort(vector):
    to_ret = []
    for i in vector:
        to_ret.append(i + np.random.uniform(low=-1*(i/500), high=i/500, size=1)[0])

    return np.array(to_ret)

num_weights = 11

population = 40

num_parents_mating = 15

fil = open('lulli.txt', 'r')

model = fil.readline()
model = model.strip('[]').split(',')
model = [float(i) for i in model]
error(get_errors(team_secret_key, model))
prev_error = cal_pop_fitness([model])
print(prev_error)
#model = [ 0.00000000e+00,  1.28200354e-01, -6.05800043e+00,  5.29444159e-02,
#  3.63051580e-02,  7.99636168e-05, -5.97183727e-05, -1.33975300e-07,
#  3.54504234e-08,  4.36850525e-11, -6.90589558e-12]

new_population = np.array([distort(model) for i in range(population)])
# generations to train for
num_generations = 20

for generation in range(num_generations):
    print("Generation : ", generation)
    fitness = cal_pop_fitness(new_population)

    parents = select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    offspring_crossover = crossover(parents, (population-parents.shape[0], num_weights))

    offspring_mutation = mutation(offspring_crossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    print("Fitness so far : ")
    print(fitness)


fitness = cal_pop_fitness(new_population)
best_match_idx = np.where(fitness == np.min(fitness))[0][0]

print('all weights')
print(new_population)

weights_vector = new_population[best_match_idx, :]

print("weights_vector")
print(weights_vector)
print("Best solution fitness : ", fitness[best_match_idx])


print("errors:")
error(get_errors(team_secret_key, list(weights_vector)))
new_error = cal_pop_fitness([weights_vector])
submit(team_secret_key, list(weights_vector))

# submit stuff
if new_error<prev_error:
    print('BETTER :):):):):)')
    file = open("lulli.txt", 'w+')
    file.write(str(list(weights_vector)))