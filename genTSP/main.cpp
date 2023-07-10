#include "src/graph/graph.h"
#include "src/tsp/sequentialTSP.h"
#include "src/tsp/parallelTSP.h"
#include "src/tsp/fastFlowTSP.h"
#include <fstream>

int main(int argc, char *argv[]) {
    if (argc < 7) {
        std::cout << "Usage is " << argv[0]
                  << " numVertices chromosomesNumber generationNumber crossoverRate mutationRate workersNumber [seed]"
                  << std::endl;
        return (-1);
    }

    int numVertices = std::atoi(argv[1]);
    int chromosomesNumber = std::atoi(argv[2]);
    int generationNumber = std::atoi(argv[3]);
    double crossoverRate = std::atof(argv[4]);
    double mutationRate = std::atof(argv[5]);
    int workerNumber = std::atoi(argv[6]);
    int seed = 0;

    if (argv[7]) {
        seed = std::atoi(argv[7]);
    }


    Graph<int> graph(numVertices, 1, 10);
    TSPSequential<int> tspSequential(graph);
    TSPParallel<int> tspParallel(graph);
    TSPFastFlow<int> tspFastFlow(graph);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        std::cout << "Unable to determine the number of threads supported by the system." << std::endl;
    } else {
        std::cout << "Number of available threads: " << numThreads << std::endl;
    }
    std::cout << std::endl;


    // Create a file stream for writing
    //std::ofstream outputFile("execution_times.txt");

    // Perform sequential computation
    auto start = std::chrono::system_clock::now();
    tspSequential.Run(chromosomesNumber, generationNumber, mutationRate, crossoverRate, workerNumber, seed);
    auto end = std::chrono::system_clock::now();
    std::chrono::milliseconds sequentialTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sequential time: " << sequentialTime.count() << "ms" << std::endl << std::endl;

    // Save sequential execution time to file
    //outputFile << "Sequential time: " << sequentialTime.count() << "ms" << std::endl;

    // Perform parallel computation (native threads)
    auto startParallel = std::chrono::system_clock::now();
    tspParallel.Run(chromosomesNumber, generationNumber, mutationRate, crossoverRate, workerNumber, seed);
    auto endParallel = std::chrono::system_clock::now();
    std::chrono::milliseconds parallelTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            endParallel - startParallel);
    std::cout << "Parallel time (native threads): " << parallelTime.count() << "ms" << std::endl << std::endl;

    // Save parallel execution time to file
    //outputFile << "Parallel time (native threads): " << parallelTime.count() << "ms" << std::endl;

    // Perform parallel computation (fastFlow)
    auto startParallelFF = std::chrono::system_clock::now();
    tspFastFlow.Run(chromosomesNumber, generationNumber, mutationRate, crossoverRate, workerNumber, seed);
    auto endParallelFF = std::chrono::system_clock::now();
    std::chrono::milliseconds parallelFFTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            endParallelFF - startParallelFF);
    std::cout << "Parallel time (fastFlow): " << parallelFFTime.count() << "ms" << std::endl;

    // Save fastFlow parallel execution time to file
    //outputFile << "Parallel time (fastFlow): " << parallelFFTime.count() << "ms" << std::endl;
    
    // Close the output file
    //outputFile.close();

    return 0;
}
