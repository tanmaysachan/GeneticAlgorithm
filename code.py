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
    # fitness = [abs(i[0]**2-3*i[1]**2) for i in errors]
    # fitness = [abs(i[0]-i[1]) for i in errors]   
    fitness = [ i[1] for i in errors ]
    pop = np.column_stack((pop,errors,fitness))
    pop = pop[np.argsort(pop[:,-1])]
    return pop


def select_mating_pool(pop,num_parents):
    return pop[:num_parents]


def crossover(parents, offspring_size):
    print(offspring_size,parents[0].shape)
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = k
        parent2_idx = parent1_idx
        while parent1_idx==parent2_idx:
            parent2_idx = np.random.randint(0,parents.shape[0])
        print(parent1_idx,parent2_idx)
        fit1, fit2 = parents[parent1_idx][-1], parents[parent2_idx][-1]
        par1prob,par2prob = (fit2/(fit1+fit2)) , fit1/(fit1+fit2)
        for i in range(0,11):
            chance = np.random.uniform(0,1)
            if chance<= par1prob:
                print("Less")
                parent2_idx = parent1_idx
            offspring[k][i] = parents[parent2_idx][i]
    return offspring


def mutation(offspring_crossover):
    # print(offspring_crossover)
    for idx in range(offspring_crossover.shape[0]):
        while(True):
            mutate_index = np.random.randint(0,11)
            cur = offspring_crossover[idx, mutate_index]
            # random_value = np.random.uniform(-1,1,1) / (10**(mutate_index+1))
            random_value = np.random.uniform(low = -1,high = 1,size = 1)
            if mutate_index == 5:
                random_value = 1e-5
            if mutate_index == 2:
                random_value = 1e-2
            # print(offspring_crossover[idx])
            if abs((offspring_crossover[idx, mutate_index] * random_value)) <= 10:
                offspring_crossover[idx, mutate_index] = offspring_crossover[idx, mutate_index] * random_value
                if mutate_index == 2:
                    print(offspring_crossover[idx])
                break
            else:
                continue
    return offspring_crossover

def error(error_value):
    print(error_value,sum(error_value))


def distort(vector):
    to_ret = vector.copy()
    index = np.random.randint(0,11)
    dir = np.random.randint(0,1)
    if dir:
        len = np.random.randint(1,11-index)
        for i in range(index,index+len):
            to_ret[i] = to_ret[i] * np.random.uniform(-10,10,1)[0]
    else:
        len = np.random.randint(0,index+1)
        for i in range(index-len+1,index):
            to_ret[i] = to_ret[i] * np.random.uniform(-10,10,1)[0]
    for i in to_ret:
        assert(i<=10 and i>=-10)
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

march22= [
    0.0,
     0.1297979015027513,
     -6.0280794462463305,
     0.05317328444376371,
     0.036421100399254704,
     8.042570953326977e-05,
     -5.936482143965488e-05,
     -1.338587966566886e-07,
     3.536700609248182e-08,
     4.4090933182860225e-11,
     -6.9518533768844116e-12]
    
# new_population = np.array([distort(march22) for i in range(500)])
# print(new_population.shape)
# pop_err_fit = cal_pop_fitness(new_population)
# for i in pop_err_fit[:20]:
#     print('Sumbitting')
#     submit(team_secret_key,
#   list(i[:11]))
# with open('22ndinit.json','w+') as f:
#     data = {}
#     data['GA'] = []
#     for i in pop_err_fit:
#         temp = {}
#         temp['Generation'] = 0
#         temp['Population'] = i.tolist()
#         data['GA'].append(temp)
#     json.dump(data,f,indent=4)
# randomgoodvector = [ np.asarray(i) for i in randomgoodvector]
# pop_err_fit = cal_pop_fitness(randomgoodvector)
# pop_err_fit = [ i.tolist() for i in pop_err_fit]
# print(pop_err_fit)
# # print(submit(team_secret_key,march22))
# exit(0)
# with open('22ndmarchinit.json','w+') as f:
#     data = {}
#     data['GA'] = []
#     for i in pop_err_fit:
#         temp = {}
#         temp['Generation'] = 0
#         temp['Population'] = i
#         data['GA'].append(temp)
# generations to train for
#     json.dump(data,f,indent=4)
# print(pop_err_fit)





with open('cur_best.json','r+') as f:
    data = json.loads(f.read())

pop_err_fit = np.asarray([ i['Population'] for i in data['GA']])
num_generations = 1
for generation in range(num_generations):
    print("Generation : ", generation)

    parents = select_mating_pool(pop_err_fit,num_parents_mating)

    offspring_crossover = crossover(parents, (population-parents.shape[0], num_weights + 3))
    offspring_mutation = mutation(offspring_crossover)

    pop_err_fit[0:parents.shape[0], :] = parents
    pop_err_fit[parents.shape[0]:, :] = offspring_mutation
    # print(offspring_mutation,parents)
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
        print("Writing to cur_best")
        with open('cur_best2.json','w+') as f:
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
exit(0)

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