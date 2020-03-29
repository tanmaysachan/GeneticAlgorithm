from client_moodle import get_errors, submit
import numpy as np
import random 
# from key import SECRET_KEY 
import json
import os

POPULATION_SIZE = 100
VECTOR_SIZE = 11
MATING_POOL_SIZE = 10
FILE_NAME = 'JSON/shradha.json'
first_parent = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

fieldNames = ['Generation','Vector','Train Error','Validation Error', 'Fitness']

def where_json(fileName):
    return os.path.exists(fileName)

def write_json(data, filename=FILE_NAME): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent = 4) 

def initial_population():
    first_population = [np.copy(first_parent) for i in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE-1):
        index = random.randint(0,10)
        first_population[i][index] = random.uniform(-10,10)

    return first_population

def calculate_fitness(population):
    fitness = np.empty((POPULATION_SIZE, 3))

    for i in range(POPULATION_SIZE):
        # error = get_errors(SECRET_KEY, list(population[i]))
        error = [1, 1]
        fitness[i][0] = error[0]
        fitness[i][1] = error[1]
        fitness[i][2] = error[0]*0.7 + error[1]
        # population_errors[i] = error
        # fitness[i] = 6

    pop_fit = np.column_stack((population, fitness))
    pop_fit = pop_fit[np.argsort(pop_fit[:,-1])]
    return pop_fit

def create_mating_pool(population_fitness):
    population_fitness = population_fitness[np.argsort(population_fitness[:,-1])]
    mating_pool = population_fitness[:MATING_POOL_SIZE]
    return mating_pool

def mutation(child):
    mutation_index = random.randint(0, VECTOR_SIZE-1)
    child[mutation_index] = random.uniform(-10,10)
    return child

def crossover(parent1, parent2):

    child = np.empty(VECTOR_SIZE)
    crossover_point = random.randint(0, VECTOR_SIZE)
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    return child

def create_children(mating_pool):
    mating_pool = mating_pool[:, :-3]
    children = []
    for i in range(POPULATION_SIZE):
        parent1 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        parent2 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        child = crossover(parent1, parent2)
        child = mutation(child)
        children.append(child)

    return children 

def new_generation(parents_fitness, children):
    children_fitness = calculate_fitness(children)
    children_fitness = children_fitness[:90]
    generation = np.concatenate((parents_fitness, children_fitness))
    return generation



def main():
    # population = initial_population()
    # population_fitness = calculate_fitness(population)
    # population_fitness = # LOAD FROM CSV
    with open('clean_36.json','r+') as f:
        population_fitness = json.loads(f.read())['population']
    # print(population_and_fitness[0])
    num_generations = 5

    # if where_json(FILE_NAME):
    #     with open(FILE_NAME) as json_file:
    #         data = json.load(json_file)
    #         population = [dict_item["Vector"] for dict_item in data["Storage"][-100:]]
    #         train = [dict_item["Train Error"] for dict_item in data["Storage"][-100:]]
    #         valid = [dict_item["Validation Error"] for dict_item in data["Storage"][-100:]]
    #         fitness = [dict_item["Fitness"] for dict_item in data["Storage"][-100:]]

    #         population_fitness = np.column_stack((population, train, valid, fitness))
    # else:
    #     data = {"Storage": []}
    #     with open(FILE_NAME, 'w') as writeObj:
    #         json.dump(data, writeObj)

    for generation in range(num_generations):   

        mating_pool = create_mating_pool(population_fitness)
        children = create_children(mating_pool)
        population_fitness = new_generation(mating_pool, children)

        fitness = population_fitness[:, -3:] 
        population = population_fitness[:, :-3]      
        
        for i in range(POPULATION_SIZE):
            # if i == 0 or i == 1 or i == 2:
            #     submit_status = submit(SECRET_KEY, population[i])
            #     assert "submitted" in submit_status
            with open(FILE_NAME) as json_file:
                data = json.load(json_file)
                temporary = data["Storage"]
                
                rowDict = { "Generation": generation, 
                            "Vector": population[i].tolist(), 
                            "Train Error": fitness[i][0], 
                            "Validation Error": fitness[i][1],
                            "Fitness": fitness[i][2]}
                temporary.append(rowDict)
            write_json(data)

if _name_ == '_main_':
    main()