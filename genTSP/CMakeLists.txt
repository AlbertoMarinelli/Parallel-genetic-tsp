cmake_minimum_required(VERSION 3.25)
project(genTSP)

add_executable(genTSP main.cpp src/graph/graph.h src/tsp/geneticAlgorithm.h src/tsp/sequentialTSP.h src/tsp/parallelTSP.h src/tsp/fastFlowTSP.h src/tsp/ffTSP.h src/overheadTest/overheadStd.cpp src/overheadTest/fastFlowTSP_overhead.h main_ffpf.cpp)