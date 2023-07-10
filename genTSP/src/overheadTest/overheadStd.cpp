#include <iostream>
#include <chrono>
#include <future>
#include <vector>
#include <thread>

void task() {
    // Simulate some work
    //std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

int main(int argc, char *argv[]) {
    if (argc < 1) {
        std::cout << "Usage is " << argv[0]
                  << " num_workers"
                  << std::endl;
        return (-1);
    }

    int workers = std::atoi(argv[1]);
    std::vector<std::future<void>> futures;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < workers; ++i) {
        futures.push_back(std::async(std::launch::async, task));
    }

    // Wait for all the threads to finish
    for (auto& future : futures) {
        future.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Total time taken: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
