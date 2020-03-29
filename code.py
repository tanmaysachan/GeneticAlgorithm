import numpy as np
from client_moodle import get_errors, submit
import json
team_secret_key = 'I22KGMKf3ZqtxxvxklykgAlk1dQZvVqhgfZT1i8NWjOgBC4ntl'


def cal_pop_fitness(pop):
    print('Getting errors')
    errors = [get_errors(team_secret_key, list(i[0:11])) for i in pop]
    print('Errors Done')
    # print(pop.shape)
    # errors = [[i,2*i] for i in range(len(pop))]
    fitness = [abs(i[0]**2-i[1]**2) for i in errors]
    # fitness = [i[0]*0.3+i[1] for i in fitness]
    pop = np.column_stack((pop,errors,fitness))
    pop = pop[np.argsort(pop[:,-1])]
    return pop


def select_mating_pool(pop,num_parents):
    return pop[:num_parents]


def crossover(parents, offspring_size):
    print(offspring_size,parents[0].shape)
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = np.random.randint(0,parents.shape[0])
        parent2_idx = parent1_idx
        while parent1_idx==parent2_idx:
            parent2_idx = np.random.randint(0,parents.shape[0])
        fit1, fit2 = parents[parent1_idx][-1], parents[parent2_idx][-1]
        for i in range(2,11):
            
        crossover_point = np.random.randint(2,11)
        # print(parent1_idx,parent2_idx,crossover_point)
        # changed to proportional to fitness
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        while(True):
            mutate_index = np.random.randint(0,10)
            cur = offspring_crossover[idx, mutate_index]
            random_value = np.random.uniform(-1,1,1) / (10**(mutate_index+1))
            if abs((offspring_crossover[idx, mutate_index] + random_value)[0]) <= 10:
                offspring_crossover[idx, mutate_index] = offspring_crossover[idx, mutate_index] + random_value[0]
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

population = 50

num_parents_mating = 30

# fil = open('lulli.txt', 'r')

# model = fil.readline()
# model = model.strip('[]').split(',')
# model = [float(i) for i in model]
# error(get_errors(team_secret_key, model))
# prev_error = cal_pop_fitness([model])
# print(prev_error)
model = [ 0.00000000e+00,  1.28200354e-01, -6.05800043e+00,  5.29444159e-02,
 3.63051580e-02,  7.99636168e-05, -5.97183727e-05, -1.33975300e-07,
 3.54504234e-08,  4.36850525e-11, -6.90589558e-12]

# new_population = np.array([distort(model) for i in range(1)])
# # generations to train for
# pop_err_fit = cal_pop_fitness(new_population)
# print(pop_err_fit)





with open('cur_best.json','r+') as f:
    data = json.loads(f.read())

pop_err_fit = np.asarray([ i['Population'] for i in data['GA'])
num_generations = 50

for generation in range(num_generations):
    print("Generation : ", generation)

    parents = select_mating_pool(pop_err_fit,num_parents_mating)
    print(parents.shape)

    offspring_crossover = crossover(parents, (population-parents.shape[0], num_weights + 3))
    print(offspring_crossover.shape)
    offspring_mutation = mutation(offspring_crossover)

    pop_err_fit[0:parents.shape[0], :] = parents
    pop_err_fit[parents.shape[0]:, :] = offspring_mutation
    
    pop_err_fit = cal_pop_fitness(pop_err_fit[:,:-3])
    # exit(0)
    with open('data.json','r+') as f:
        data = json.loads(f.read())
    print("Fitness so far : ")
    with open('data.json','w+') as f:
        # data = {}
        # data['GA'] = []
        for i in pop_err_fit:
            temp = {}
            temp['Generation'] = generation
            temp['Population'] = i.tolist()
            data['GA'].append(temp)
        json.dump(data,f,indent=4)
    if generation == num_generations - 1:
        with open('cur_best.json','w+') as f:
            data = {}
            data['GA'] = []
            for i in pop_err_fit:
                temp = {}
                temp['Generation'] = generation
                temp['Population'] = i.tolist()
                data['GA'].append(temp)
            json.dump(data,f,indent=4)
    for i in pop_err_fit[:5]:
        print('Submitting')
        submit(team_secret_key, list(i[:11]))
# exit(0)

# # fitness = cal_pop_fitness(new_population)
# best_match_idx = np.where(fitness == np.min(fitness))[0][0]

# print('all weights')
# # print(new_population)

# weights_vector = new_population[best_match_idx, :]

# print("weights_vector")
# # print(weights_vector)
# print("Best solution fitness : ", fitness[best_match_idx])


# print("errors:")
# # error(get_errors(team_secret_key, list(weights_vector)))
# new_error = cal_pop_fitness([weights_vector])
# # submit(team_secret_key, list(weights_vector))

# # submit stuff
# if new_error<prev_error:
#     print('BETTER :):):):):)')
#     file = open("lulli.txt", 'w+')
#     file.write(str(list(weights_vector)))