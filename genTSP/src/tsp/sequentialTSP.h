#ifndef GENTSP_SEQUENTIALTSP_H
#define GENTSP_SEQUENTIALTSP_H

#include "geneticAlgorithm.h"
#include "../graph/graph.h"
#include <iostream>
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <utility>

template<typename T>
class TSPSequential : public GeneticAlgorithm {
private:
    std::random_device rd;
    std::mt19937 gen{rd()};
    Graph<T> &graph;
    std::vector<std::pair<double, std::vector<int>>> population;

    void generatePopulation(int chromosomeNumber, int numVertices);

    void evaluate(double &evaluationsAverage, int k);

    void fitness(double &evaluationsAverage);

    void selection(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation);

    void crossover(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation, double crossoverRate);

    void mutation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation, double mutationRate);

    void printPopulation();

    void printIntermediatePopulation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation);

    void printBestSolution();

    void printBestSolution_andPath();

public:
    explicit TSPSequential(Graph<T> &graph);

    void Run(
            int chromosomeNumber,
            int generationNumber,
            double mutationRate,
            double crossoverRate,
            int workers,
            int seed) override;
};

template<typename T>
void TSPSequential<T>::Run(int chromosomeNumber, int generationNumber, double mutationRate, double crossoverRate,
                           int workers, int seed) {
    double evaluationsAverage = 0;
    std::vector<std::pair<double, std::vector<int>>> intermediatePopulation;

    if (seed) {
        gen.seed(seed);
    }

    //std::ofstream outputFile("bestSolution_sequential.txt"); // Create an output file stream

    population.reserve(chromosomeNumber);
    generatePopulation(chromosomeNumber, graph.getNumVertices());
    //printPopulation();

    for (size_t generation = 1; generation <= generationNumber; generation++) {
        evaluate(evaluationsAverage, chromosomeNumber);
        std::cout<<generation<<std::endl;
        printBestSolution();
        //Save best solution
        //double bestFitness = population[0].first;
        //outputFile << bestFitness << std::endl; // Write the value to the file
        //printBestSolution();
        fitness(evaluationsAverage);
        selection(intermediatePopulation);
        crossover(intermediatePopulation, crossoverRate);
        mutation(intermediatePopulation, mutationRate);

        evaluationsAverage = 0;

        // Swap the contents of population and intermediatePopulation
        population.swap(intermediatePopulation);
    }

    //Print best solution and path for test purposes
    //evaluate(evaluationsAverage, chromosomeNumber);
    //printBestSolution_andPath();
    //printBestSolution();

    //outputFile.close(); // Close the output file stream
}

template<typename T>
TSPSequential<T>::TSPSequential(Graph<T> &graph): graph(graph) {}

template<typename T>
void TSPSequential<T>::generatePopulation(int chromosomeNumber, int numVertices) {
    auto start = std::chrono::system_clock::now();
    population.clear(); // Clear the previous population if any
    double keyValue=0;

    // Create a vector with values from 0 to numVertices
    std::vector<int> sequence(numVertices);
    std::iota(sequence.begin(), sequence.end(), 0);

    // Generate copies of the shuffled sequence with different key values of probability
    for (int i = 0; i < chromosomeNumber; ++i) {
        std::shuffle(sequence.begin(), sequence.end(), gen);
        population.emplace_back(keyValue, sequence);
    }
    auto end = std::chrono::system_clock::now();
    //std::cout << "Generation sequential time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Evaluate population
template<typename T>
void TSPSequential<T>::evaluate(double &evaluationsAverage, int k) {
    auto start = std::chrono::system_clock::now();
    double chromosomeScoreCurrIt = 0;
    int chromosomeSize = graph.getNumVertices();

    // Calculate chromosome scores
    for (auto &individual: population) {
        chromosomeScoreCurrIt = 0;
        for (size_t j = 0; j < chromosomeSize - 1; ++j) {
            chromosomeScoreCurrIt += graph.getWeight(individual.second[j], individual.second[j + 1]);
        }
        chromosomeScoreCurrIt += graph.getWeight(individual.second[0], individual.second[chromosomeSize -
                                                                                         1]); //Connect the path of evaluated nodes
        individual.first = chromosomeScoreCurrIt;
    }

    // Sort population by key value
    auto compareByKeyValue = [](const std::pair<double, std::vector<int>> &a,
                                const std::pair<double, std::vector<int>> &b) {
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
    //std::cout << "Evaluation sequential time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Fitness function
template<typename T>
void TSPSequential<T>::fitness(double &evaluationsAverage) {
    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < population.size(); ++i) {
        auto &individual = population[i];
        individual.first /= evaluationsAverage;
    }
    auto end = std::chrono::system_clock::now();
    //std::cout << "Fitness sequential time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

template<typename T>
void TSPSequential<T>::selection(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto &individual: population) {
        int numCopies = static_cast<int>(individual.first); // Get the integer part of the key

        for (int i = 0; i < numCopies; ++i) { //Place one copies of element equals to integer part
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
void TSPSequential<T>::crossover(std::vector<std::pair<double, std::vector<int>>>& intermediatePopulation, double crossoverRate) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<> prob(0.0, 1.0); // Initialize a uniform distribution for probability generation
    auto secondElementSize = intermediatePopulation[0].second.size();
    int twentyPercent = static_cast<int>(secondElementSize * 0.2);

    // Iterate over the intermediate population by pairs
    for (int i = 0; i < intermediatePopulation.size(); i += 2) {
        auto& individual1 = intermediatePopulation[i]; // Retrieve the first individual
        if (i + 1 < intermediatePopulation.size()) { // Check if there is another individual available for pairing
            auto& individual2 = intermediatePopulation[i + 1]; // Retrieve the second individual

            // Check if crossover should be performed based on the crossover rate
            if (prob(gen) < crossoverRate) {
                std::vector<int>& vec1 = individual1.second;
                std::vector<int>& vec2 = individual2.second;

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
    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover sequential time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Mutation
template<typename T>
void TSPSequential<T>::mutation(std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation,
                                double mutationRate) {
    auto start = std::chrono::system_clock::now();
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Iterate over each individual in the intermediate population
    for (auto &individual: intermediatePopulation) {
        if (dis(gen) < mutationRate) { // Check if the randomly generated number is less than the mutation rate
            int size = individual.second.size();
            if (size > 1) {
                std::uniform_int_distribution<> indexDis(0, size - 1);
                int index1 = indexDis(gen); // Generate a random index
                int index2 = indexDis(gen); // Generate another random index

                // Swap the elements at the randomly generated indices.
                std::swap(individual.second[index1], individual.second[index2]);
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Mutation sequential time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

//Print population
template<typename T>
void TSPSequential<T>::printPopulation() {
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
void TSPSequential<T>::printIntermediatePopulation(
        std::vector<std::pair<double, std::vector<int>>> &intermediatePopulation) {
    for (const auto &individual: intermediatePopulation) {
        std::cout << "Key: " << individual.first << ", Values: ";
        for (const auto &value: individual.second) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void TSPSequential<T>::printBestSolution() {
    std::cout << "Best solution cost: " << population[0].first << std::endl;
}

template<typename T>
void TSPSequential<T>::printBestSolution_andPath() {
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
#endif //GENTSP_SEQUENTIALTSP_H
