#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <vector>
#include <random>
#include <optional>

// Undirected, completely connected and weighted graph
template <typename T>
class Graph {
public:
    Graph(int numVertices, int minWeight = 1, int maxWeight = 100,std::optional<int> seed = std::nullopt)
            : numVertices_(numVertices), minWeight_(minWeight), maxWeight_(maxWeight) {
        generateRandomGraph(seed);
    }

    void generateRandomGraph(std::optional<int> seed = std::nullopt) {
        // Resize the adjacency matrix to match the number of vertices,
        // initializing all values to 0.
        adjacencyMatrix_.resize(numVertices_, std::vector<T>(numVertices_, 0));

        // Create a random device for generating a seed if not provided.
        std::random_device rd;
        // Create a random number generator using the provided seed or the random device.
        std::mt19937 gen(seed.has_value() ? seed.value() : rd());

        // Create a uniform distribution for generating weights between minWeight_ and maxWeight_ (inclusive).
        std::uniform_int_distribution<T> weightDistribution(minWeight_, maxWeight_);

        // Iterate over each pair of vertices in the graph
        for (int i = 0; i < numVertices_; ++i) {
            for (int j = i + 1; j < numVertices_; ++j) {
                // Generate a random weight using the weight distribution and the random number generator
                T weight = weightDistribution(gen);
                // Assign the same weight to the edges (i, j) and (j, i) since the graph is undirected
                adjacencyMatrix_[i][j] = weight;
                adjacencyMatrix_[j][i] = weight;
            }
        }
    }

    void printGraph() const {
        for (int i = 0; i < numVertices_; ++i) {
            for (int j = 0; j < numVertices_; ++j) {
                std::cout << adjacencyMatrix_[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    int getNumVertices() const {
        return numVertices_;
    }

    T getWeight(int vertex1, int vertex2) const {
        if (vertex1 < 0 || vertex1 >= numVertices_ || vertex2 < 0 || vertex2 >= numVertices_) {
            throw std::out_of_range("Invalid vertex index");
        }

        return adjacencyMatrix_[vertex1][vertex2];
    }

private:
    int numVertices_;
    T minWeight_;
    T maxWeight_;
    std::vector<std::vector<T>> adjacencyMatrix_;
};

#endif  // GRAPH_H