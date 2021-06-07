#include "su2Element.hpp"

#ifndef SU2_5CELLElEMENT_HPP
#define SU2_5CELLElEMENT_HPP

#define C5_ETA 0.5590169943749474241022934

class su2_5CellElement : public su2Element {
public:
  __device__ __host__ su2_5CellElement() : su2Element(){};

  __device__ __host__ su2_5CellElement(double *el)
      : su2Element(el){

        };

  __device__ __host__ void renormalize(){};

  su2_5CellElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 4);
    return randomize(dist(gen));
  };

  __device__ su2_5CellElement randomize(double delta,

                                        CUDA_RAND_STATE_TYPE *state) {
    int n = 5;
    while (n == 5) {
      double t = curand_uniform_double(state) * 5;
      n = (int)t;
    }
    return randomize(n);
  };

protected:
  __device__ __host__ su2_5CellElement randomize(int direction) {
    double vertices[5][4] = {{1, 0, 0, 0},
                             {-0.25, C5_ETA, C5_ETA, C5_ETA},
                             {-0.25, -C5_ETA, -C5_ETA, C5_ETA},
                             {-0.25, -C5_ETA, C5_ETA, -C5_ETA},
                             {-0.25, C5_ETA, -C5_ETA, -C5_ETA}};

    return su2_5CellElement(&vertices[direction][0]);
  };
};

#endif /*SU2_5CELLElEMENT_HPP*/
