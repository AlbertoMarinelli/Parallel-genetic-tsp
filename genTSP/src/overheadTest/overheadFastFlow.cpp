#include <ff/ff.hpp>
#include <ff/farm.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <optional>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <utility>
#include <thread>


struct GenericWorker : ff::ff_node_t<std::pair<int, int>> {
private:

public:
    std::pair<int, int>* svc(std::pair<int, int>* chunk) override {
        int start = chunk->first; // Start index of the chunk to process
        int end = chunk->second; // End index of the chunk to process
        delete chunk;
        // Simulate some work
        //std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Adjust the sleep time to represent your actual task
        this->ff_send_out(new std::pair<int, int>(start, end)); // Send the processed chunk to the next stage
        return this->GO_ON;
    }
};

struct ChunksEmitter : ff::ff_node_t<std::pair<int, int>> {
private:
    std::vector<std::pair<int, int>> &chunks;  // Reference to a vector of pairs
    int generationsNumber; // Number of generations to emit
public:
    explicit ChunksEmitter(std::vector<std::pair<int, int>> &chunks, int generationsNumber_)
            : chunks(chunks), generationsNumber(generationsNumber_) {}

    std::pair<int, int> *svc(std::pair<int, int> *currentGeneration) override {
        if (generationsNumber <= 0) { // If no more generations to emit
            return this->EOS; // End of stream signal
        }
        generationsNumber--;
        std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> &chunk) {
            this->ff_send_out(new std::pair<int, int>(chunk.first, chunk.second)); // Send each chunk to the next stage
        });
        return this->GO_ON; // Continue processing
    }
};

struct ChunksCollector : ff::ff_node_t<std::pair<int, int>> {
private:
    const int chunksNumber; // Total number of chunks to be collected
    int currentChunksNumber; // Number of remaining chunks to be collected
public:
    explicit ChunksCollector(int chunks_number) : chunksNumber(chunks_number), currentChunksNumber(chunks_number) {}

    std::pair<int, int>* svc(std::pair<int, int>* chunk) override {
        currentChunksNumber--;

        // if all the chunks have been computed, then proceed to the next stage
        if (currentChunksNumber == 0) {
            currentChunksNumber = chunksNumber;
            this->ff_send_out(chunk);
        } else {
            delete chunk; // Delete the chunk after processing
        }
        return this->GO_ON; // Continue processing next chunks
    }
};

void setupComputation(int &workersNumber, int chromosomeNumber, std::vector<std::pair<int, int>> &chunks) {
    // Adjust the number of workers based on the available hardware concurrency
    workersNumber =
            (workersNumber < std::thread::hardware_concurrency() - 1) ? workersNumber :
            std::thread::hardware_concurrency()
            - 1;

    // Ensure that the number of workers does not exceed the number of chromosomes
    if (workersNumber > chromosomeNumber) {
        workersNumber = chromosomeNumber;
    }

    // Clear and resize the 'chunks' vector
    chunks.clear();
    chunks.resize(workersNumber);

    // Calculate the number of chromosomes each worker will generate
    int chromosomesPerWorker = chromosomeNumber / workersNumber;
    int extraChromosomes = chromosomeNumber % workersNumber;

    int index = 0;

    // Assign chromosome ranges
    std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> &chunk) {
        chunk.first = index*chromosomesPerWorker;
        chunk.second=chunk.first+chromosomesPerWorker;
        index += 1;
    });

    // Adjust the range of the last worker to account for extra chromosomes
    if (!chunks.empty()) {
        std::pair<int, int>& lastElement = chunks.back();
        lastElement.second += extraChromosomes; // Modify .second
    }
}

int main(int argc, char *argv[]) {
    if (argc < 1) {
        std::cout << "Usage is " << argv[0]
                  << " num_workers [population_size]"
                  << std::endl;
        return (-1);
    }

    int population_size = 5000;
    int num_workers = std::atoi(argv[1]);
    if (argv[2]) {
        population_size = std::atoi(argv[2]);
    }

    std::vector<std::pair<int, int>> chunks;

    setupComputation(num_workers, population_size, chunks);

    // Create ChunksEmitter and ChunksCollector for generating chromosomes
    ChunksEmitter generateChromosomesEmitter(chunks, 1);
    ChunksCollector generateChromosomesCollector(chunks.size());

    // Create generationWorkers for chromosome generation
    std::vector<std::unique_ptr<ff::ff_node>> generationWorkers;
    for (int i = 0; i < num_workers; i++) {
        generationWorkers.emplace_back(new GenericWorker());
    }

    // Create the ff_Farm for chromosome generation, using the generationWorkers, emitter, and collector
    ff::ff_Farm<std::pair<int, int>> creationFarm(std::move(generationWorkers), generateChromosomesEmitter,
                                                  generateChromosomesCollector);
    creationFarm.wrap_around();

    auto start = std::chrono::high_resolution_clock::now();
    // Run the chromosome generation farm and check for errors
    if (creationFarm.run_and_wait_end() < 0) {
        std::cerr << "farm Evaluation execution interrupted!" << std::endl;
        return 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_with_tasks = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


    std::cout << "Overhead: " << duration_with_tasks.count() << " microseconds" << std::endl;

    return 0;
}
