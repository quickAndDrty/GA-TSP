from TSP import *


def main():
    print("hello home")

    tsp = TSP(0)
    tsp.readFromFile()
    globalBest = 5000
    globalAvg = 0
    N = 10
    M = 50
    for i in range (10):
        best, avg = tsp.run(N, M)
        if (best < globalBest):
            globalBest = best
        globalAvg = globalAvg + avg


    print ("results: ", globalBest, " ", globalAvg/10)







if __name__ == '__main__':
    main()