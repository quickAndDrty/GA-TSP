from City import *
import random
import math

class TSP:

    def __init__(self, n):
        self.n = n
        self.dist = [[]]
        self.cities = []
        self.noPop = 0

    def eucliadinanDistance(self, a, b):
        return round(math.sqrt( (b.getX()-a.getX())*(b.getX()-a.getX()) + (b.getY()-a.getY())*(b.getY()-a.getY()) ), 2)

    def calculateDistances(self):
        self.dist = [[0 for j in range(self.n+1)] for i in range(self.n+1)]
        for i in range (self.n):
            for j in range (i+1, self.n-1):
                self.dist[i + 1][j + 1] = self.eucliadinanDistance(self.cities[i], self.cities[j])
                self.dist[j + 1][i + 1] = self.eucliadinanDistance(self.cities[i], self.cities[j])
        #for checking distances
        #self.printResults()

    def readFromFile(self):
        #f = open ("lessValues.txt", "r")
        f = open("values.txt", "r")

        for i in range (6):
            buffer = f.readline()

        self.n = 0
        for buffer in f:
            elements = buffer.split()
            if ( len(elements) == 3 ):
                self.cities.append( City(elements[0], elements[1], elements[2]))
                self.n = self.n + 1

        self.calculateDistances()

    def generateSolution(self):
        c = [i for i in range(1, self.n + 1)]
        random.shuffle(c)
        return c

    def fitness(self, tour):
        fit = 0
        #print("ruta este ", tour)
        for i in range (0, self.n-1):
            fit = fit + self.dist[tour[i]][tour[i+1]]
        fit = fit + self.dist[tour[self.n - 1]][tour[0]]
        return fit

    def randomPopulation(self, N):
        pop = []
        for i in range (0, math.ceil((2 * N)/10)):
            pop.append(self.simulatedAnnealing(1000, 0.001, 0.999, 10))
        for i in range (0, math.ceil((8 * N)/10)):
            pop.append(self.generateSolution())

        return pop[:N]

    def tournamentSelection(self, population):
        k = 20
        bestCanditate = self.generateSolution()
        for i in range (k):
            candidate = population[random.randint(0, int(self.noPop) - 1)]
            if (self.fitness(candidate) < self.fitness(bestCanditate)):
                bestCanditate = candidate[:]
        return bestCanditate

    def orderXover(self, parent1, parent2):
        index1 = random.randint(1, len(parent1) - 1)
        index2 = random.randint(1, len(parent1) - 1)
        while (index1 == index2 ):
            index2 = random.randint(1, len(parent1))
        if ( index1 > index2):
            index1, index2 = index2, index1
        #print(index1, " ", index2)
        child1 = [0]*len(parent1)
        child2 = [0] * len(parent1)
        for i in range (index1, index2):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        #print(child1, " ", child2)
        candidate1 = parent2[index2:]
        candidate1.extend(parent2[:index2])
        candidate2 = parent1[index2:]
        candidate2.extend(parent1[:index2])

        start = index2
        for i in range (len(candidate1)):
            if candidate1[i] not in child1:
                child1[start % len(candidate1)] = candidate1[i]
                start = start + 1
            #print(child1)
        start = index2
        for i in range(len(candidate2)):
            if candidate2[i] not in child2:
                child2[start % len(candidate2)] = candidate2[i]
                start = start + 1
        #print(child1, " ", child2)

        return child1, child2

    def sortPopulation(self, population):
        population.sort(key=lambda x : self.fitness(x))
        print(population)
        for i in population:
            print(self.fitness(i))

    def avgFitness(self, population):
        avg = 0
        for i in population:
            avg = avg + self.fitness(i)
        return avg/self.noPop

    def randomNeighbor(self, c):
        x = random.randint(0, int(self.n) - 1)
        y = random.randint(0, int(self.n) - 1)
        #different values
        while (x == y):
            x = random.randint(0, int(self.n) - 1)
            y = random.randint(0, int(self.n) - 1)
        #print (" index ", x, " ", y)
        c[x], c[y] = c[y], c[x]
        return c

    def simulatedAnnealing(self, Tmax, Tmin, alpha, maxIter ):

        temp = Tmax #set initial temp
        chromosome = self.generateSolution()

        while ( temp > Tmin):
            k = 1
            neighbor = self.randomNeighbor(chromosome[:])
            delta = int(self.fitness(neighbor) - self.fitness(chromosome))
            if (delta < 0):
                chromosome = neighbor
            else:
                if ( random.random() < math.exp( -delta/temp) ):
                    chromosome = neighbor

            while (k < maxIter):
                neighbor = self.randomNeighbor(chromosome[:])
                delta = int(self.fitness(neighbor) - self.fitness(chromosome))
                if (delta < 0):
                    chromosome = neighbor
                else:
                    if (random.random() < math.exp(-delta / temp)):
                        chromosome = neighbor
                k = k + 1

            #cool system
            temp = alpha * temp

        return chromosome[:]

    def run(self, noPop, noGen):
        t = 0 #current generation

        population = self.randomPopulation(noPop)
        print(population)
        self.noPop = noPop
        best = None
        avg = 0

        while ( t < noGen ):
            #choose parents
            parent1 = self.tournamentSelection(population)
            parent2 = self.tournamentSelection(population)
            #while (parent1  == parent2):
                #parent2 = self.tournamentSelection(population)
            #print(parent1, " ", self.fitness(parent1))
            #print(parent2, " ", self.fitness(parent2))

            child1, child2 = self.orderXover(parent1, parent2)
            #print(child1, " ", self.fitness(child1), " ", child2, " ", self.fitness(child2))

            #population.append(parent1)
            #population.append(parent2)

            #simulated annealing for every new generation
            #newChromosome = self.simulatedAnnealing(1000, 0.001, 0.999, 10)
            #population.append(newChromosome)
            self.sortPopulation(population)
            population = population[:self.noPop - 2]
            population.append(child1)
            population.append(child2)
            random.shuffle(population)

            avg = avg + self.avgFitness(population)
            t = t + 1

        self.sortPopulation(population)
        best = self.fitness(population[0])
        avg = avg/noGen
        return best, avg



