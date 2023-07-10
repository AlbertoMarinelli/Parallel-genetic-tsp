//
// Created by alber on 13/06/2023.
//

#ifndef GENTSP_PARALLELTSP_H
#define GENTSP_PARALLELTSP_H

#include "geneticAlgorithm.h"
#include "../graph/graph.h"
#include <iostream>
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <fstream>
#include <utility>

template<typename T>
class TSPParallel : public GeneticAlgorithm {
private:
    std::random_device rd;
    std::mt19937 gen{rd()};
    Graph<T> &graph;
    std::vector<std::pair<double, std::vector<int>>> population;


    void generatePopulation(int chromosomeNumber, int numVertices, int workers);

    void evaluate(double &evaluationsAverage, int k, int workers);

    void fitness(double &evaluationsAverage);

    void selection(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation);

    void crossover(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation, double crossoverRate,
                   int workers);

    void mutation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation, double mutationRate);

    void printPopulation();

    void printIntermediatePopulation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation);

    void printBestSolution();

    void printBestSolution_andPath();

public:
    explicit TSPParallel(Graph<T> &graph);

    void Run(
            int chromosomeNumber,
            int generationNumber,
            double mutationRate,
            double crossoverRate,
            int workers,
            int seed) override;

};

template<typename T>
void TSPParallel<T>::Run(int chromosomeNumber, int generationNumber, double mutationRate, double crossoverRate,
                         int workers, int seed) {
    double evaluationsAverage = 0;
    std::vector<std::pair<double, std::vector<int>>> intermediatePopulation;

    if (seed) {
        gen.seed(seed);
    }

    population.reserve(chromosomeNumber);

    //std::ofstream outputFile("bestSolution_parallel(native).txt"); // Create an output file stream

    generatePopulation(chromosomeNumber, graph.getNumVertices(), workers);
    //printPopulation();

    for (size_t generation = 1; generation <= generationNumber; generation++) {
        evaluate(evaluationsAverage, chromosomeNumber, workers);

        //printBestSolution();
        //double bestFitness = population[0].first;
        //outputFile << bestFitness << std::endl; // Write the value to the file

        fitness(evaluationsAverage);
        selection(intermediatePopulation);
        crossover(intermediatePopulation, crossoverRate, workers);
        mutation(intermediatePopulation, mutationRate);

        evaluationsAverage = 0;

        // Swap the contents of population and intermediatePopulation
        population.swap(intermediatePopulation);
    }

    //Print best solution and path for test purposes
    //evaluate(evaluationsAverage, chromosomeNumber, workers);
    //printBestSolution_andPath();
    //outputFile.close(); // Close the output file stream
    //printBestSolution();
}

template<typename T>
TSPParallel<T>::TSPParallel(Graph<T> &graph): graph(graph) {}

template<typename T>
void TSPParallel<T>::generatePopulation(int chromosomeNumber, int numVertices, int workers) {
    auto start = std::chrono::system_clock::now();
    population.clear(); // Clear the previous population if any

    // Create a vector with values from 0 to numVertices
    std::vector<int> sequence(numVertices);
    std::iota(sequence.begin(), sequence.end(), 0);

    // Define the function to generate a portion of the chromosomes
    auto generateChromosomes = [&](int start, int end) {
        std::vector<std::pair<double, std::vector<int>>> partialPopulation;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> shuffledSequence(sequence);

        for (int i = start; i < end; ++i) {
            std::shuffle(shuffledSequence.begin(), shuffledSequence.end(), gen);
            partialPopulation.emplace_back(0, shuffledSequence);
        }

        return partialPopulation;
    };

    // Calculate the number of chromosomes each worker will generate
    int chromosomesPerWorker = chromosomeNumber / workers;
    int extraChromosomes = chromosomeNumber % workers;

    // Create a vector to store the futures
    std::vector<std::future<std::vector<std::pair<double, std::vector<int>>>>> futures;

    // Create and launch the worker tasks using async
    for (int i = 0; i < workers; ++i) {
        int start_par = i * chromosomesPerWorker;
        int end = start_par + chromosomesPerWorker;
        if (i == workers - 1) {
            end += extraChromosomes;  // Add any extra chromosomes to the last worker
        }

        futures.emplace_back(std::async(std::launch::async, generateChromosomes, start_par, end));
    }

    // Collect the partial results from the futures and update the population
    for (auto& future : futures) {
        std::vector<std::pair<double, std::vector<int>>> partialPopulation = future.get();
        population.insert(population.end(), partialPopulation.begin(), partialPopulation.end());
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "Generation parallel time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


template<typename T>
void TSPParallel<T>::evaluate(double& evaluationsAverage, int k, int workers) {
    auto start = std::chrono::system_clock::now();
    int chromosomeSize = graph.getNumVertices();

    // Calculate chromosome scores in parallel
    std::vector<std::future<std::vector<double>>> futures;

    auto calculateChromosomeScore = [&](int start, int end) -> std::vector<double> {
        std::vector<double> scores(end - start);
        for (int i = start; i < end; ++i) {
            auto& individual = population[i];
            double chromosomeScoreCurrIt = 0;
            for (size_t j = 0; j < chromosomeSize - 1; ++j) {
                chromosomeScoreCurrIt += graph.getWeight(individual.second[j], individual.second[j + 1]);
            }
            chromosomeScoreCurrIt += graph.getWeight(individual.second[0], individual.second[chromosomeSize - 1]);
            scores[i - start] = chromosomeScoreCurrIt;
        }
        return scores;
    };

    // Calculate the number of chromosomes each worker will evaluate
    int chromosomesPerWorker = population.size() / workers;
    int extraChromosomes = population.size() % workers;

    // Create and launch the worker tasks
    for (int i = 0; i < workers; ++i) {
        int start_par = i * chromosomesPerWorker;
        int end = start_par + chromosomesPerWorker;
        if (i == workers - 1) {
            end += extraChromosomes;  // Add any extra chromosomes to the last worker
        }
        futures.emplace_back(std::async(std::launch::async, calculateChromosomeScore, start_par, end));
    }

    // Wait for all tasks to finish and retrieve the chromosome scores
    std::vector<double> chromosomeScores;
    for (auto& future : futures) {
        auto scores = future.get();
        chromosomeScores.insert(chromosomeScores.end(), scores.begin(), scores.end());
    }

    // Update chromosome scores in the population
    for (int i = 0; i < population.size(); ++i) {
        population[i].first = chromosomeScores[i];
    }

    // Sort population by key value
    auto compareByKeyValue = [](const std::pair<double, std::vector<int>>& a,
                                const std::pair<double, std::vector<int>>& b) {
        return a.first < b.first;
    };
    std::sort(population.begin(), population.end(), compareByKeyValue);

    // Calculate evaluation average on top k elements
    double totalScore = 0;
    int numElements = std::min(k, static_cast<int>(population.size()));
    for (int i = 0; i < numElements; ++i) {
        totalScore += population[i].first;
    }

    evaluationsAverage = totalScore / numElements;
    population.resize(numElements);

    auto end = std::chrono::system_clock::now();
    std::cout << "Evaluation parallel time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


//Fitness function
template<typename T>
void TSPParallel<T>::fitness(double &evaluationsAverage) {
    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < population.size(); ++i) {
        auto &individual = population[i];
        individual.first /= evaluationsAverage;
    }
    auto end = std::chrono::system_clock::now();
    //std::cout << "Fitness sequential time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

template<typename T>
void TSPParallel<T>::selection(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto &individual: population) {
        int numCopies = static_cast<int>(individual.first); // Get the integer part of the key

        for (int i = 0; i < numCopies; ++i) {
            intermediatePopulation.push_back(
                    {individual.first, individual.second}); // Copy the individual to intermediatePopulation
        }

        double fractionalPart = individual.first - numCopies; // Get the fractional part of the key

        if (distribution(gen) < fractionalPart) {
            intermediatePopulation.push_back(
                    {individual.first, individual.second}); // Place an additional copy based on the fractional part
        }
    }
    auto end = std::chrono::system_clock::now();
    //std::cout << "Selection sequential time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


template<typename T>
void TSPParallel<T>::crossover(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation,
                               double crossoverRate, int workers) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<> prob(0.0, 1.0);

    std::vector<std::future<void>> futures(workers);

    // Function to be executed by each worker
    auto workerFunction = [&](int start, int end) {
        // Each worker operates on a range of tasks
        auto secondElementSize = intermediatePopulation[0].second.size();
        int twentyPercent = static_cast<int>(secondElementSize * 0.2);
        for (int i = start; i < end; i+=2) {
            if (i + 1 < end) {
                // Perform crossover with a certain probability
                if (prob(gen) < crossoverRate) {
                    std::vector<int> &vec1 = intermediatePopulation[i].second;
                    std::vector<int> &vec2 = intermediatePopulation[i+1].second;

                    // Choose a random index as the starting point for the cycle
                    std::uniform_int_distribution<> dis(0, vec1.size() - twentyPercent-1);
                    int startIndex = dis(gen);

                    int endIndex = startIndex+twentyPercent;

                    for (int j=startIndex; j<endIndex; j++){

                        // Find the position of the number in the vector
                        auto it = std::find(vec1.begin(), vec1.end(), vec2[j]);
                        // Find the position of the number in the vector
                        auto it_2 = std::find(vec2.begin(), vec2.end(), vec1[j]);
                        // Calculate the index using the distance between it and vec1.begin()
                        int index_1 = std::distance(vec1.begin(), it);

                        // Swap the number with the index
                        std::swap(vec1[index_1], vec1[j]);

                        // Calculate the index using the distance between it and vec1.begin()
                        int index_2 = std::distance(vec2.begin(), it_2);
                        // Swap the number with the index 0
                        std::swap(vec2[index_2], vec2[j]);
                    }
                }
            }
        }
    };

    // Calculate the number of chromosomes each worker will evaluate
    int chromosomesPerWorker = intermediatePopulation.size() / workers;
    int extraChromosomes = intermediatePopulation.size() % workers;

    // Create and launch the worker tasks
    for (int i = 0; i < workers; ++i) {
        int start_par = i * chromosomesPerWorker;
        int end_par= start_par + chromosomesPerWorker;

        if (i == workers - 1) {
            end_par += extraChromosomes;  // Add any extra chromosomes to the last worker
        }
        // Each worker performs its tasks using the workerFunction
        futures[i] = std::async(std::launch::async, workerFunction, start_par, end_par);
    }


    // Wait for all worker tasks to complete
    for (auto &future: futures) {
        future.wait();
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover parallel time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}


//Mutation
template<typename T>
void TSPParallel<T>::mutation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation,
                              double mutationRate) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Iterate over each individual in the intermediate population
    for (auto &individual: intermediatePopulation) {
        if (dis(gen) < mutationRate) { // Check if the randomly generated number is less than the mutation rate
            int size = individual.second.size();
            if (size > 1) { // Perform mutation if the chromosome has more than one element
                std::uniform_int_distribution<> indexDis(0, size - 1);
                int index1 = indexDis(gen);
                int index2 = indexDis(gen);

                std::swap(individual.second[index1], individual.second[index2]);
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    //std::cout << "Mutation sequential time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Print population
template<typename T>
void TSPParallel<T>::printPopulation() {
    for (const auto &individual: population) {
        std::cout << "Key: " << individual.first << ", Values: ";
        for (const auto &value: individual.second) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

//Print population
template<typename T>
void
TSPParallel<T>::printIntermediatePopulation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation) {
    for (const auto &individual: intermediatePopulation) {
        std::cout << "Key: " << individual.first << ", Values: ";
        for (const auto &value: individual.second) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void TSPParallel<T>::printBestSolution() {
    std::cout << "Best solution cost: " << population[0].first << std::endl;
}

template<typename T>
void TSPParallel<T>::printBestSolution_andPath() {
    int chromosomeSize = graph.getNumVertices();
    for (int i = 0; i < 1 && i < population.size(); ++i) {
        const auto &individual = population[i];
        std::cout << "Chromosome " << i << ": Score = " << individual.first << ", Path = ";
        for (const auto &vertex: individual.second) {
            std::cout << vertex << " ";
        }
        std::cout << std::endl;
    }
    if (!population.empty()) {
        auto &firstIndividual = population[0];
        double chromosomeScoreCurrIt = 0;
        for (size_t j = 0; j < chromosomeSize - 1; ++j) {
            chromosomeScoreCurrIt += graph.getWeight(firstIndividual.second[j], firstIndividual.second[j + 1]);
        }
        chromosomeScoreCurrIt += graph.getWeight(firstIndividual.second[0], firstIndividual.second[chromosomeSize - 1]);
        std::cout << "Calculated cost of solution: " << chromosomeScoreCurrIt;
    }
}

#endif //GENTSP_PARALLELTSP_H
