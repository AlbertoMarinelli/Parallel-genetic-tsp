//
// Created by alber on 10/06/2023.
//

#ifndef GENTSP_GENETICALGORITHM_H
#define GENTSP_GENETICALGORITHM_H

#include <iostream>
#include <vector>
#include <functional>
#include <chrono>

//! let the user decide the precision of intermediate results
using precision = double;

class GeneticAlgorithm {
public:
    virtual ~GeneticAlgorithm() = default;
    virtual void Run(int chromosomeNumber,
                     int generationNumber,
                     double mutationRate,
                     double crossoverRate,
                     int workers,
                     int seed) = 0;
};
#endif //GENTSP_GENETICALGORITHM_H
