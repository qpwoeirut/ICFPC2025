#!/bin/zsh

g++-14 -std=c++20 -O3 -funroll-loops -mtune=native -march=native -Wall solve.cpp -o temp_solve.out
>&2 echo "Compiled!"
time ./temp_solve.out
rm temp_solve.out
