#include "su2Element.hpp"

#ifndef SU2_16CELLElEMENT_HPP
#define SU2_16CELLElEMENT_HPP

class su2_16CellElement : public su2Element {
public:
  __device__ __host__ su2_16CellElement() : su2Element() { nonZeroIndex = 0; };

  __device__ __host__ su2_16CellElement(double *el, int idx) : su2Element(el) {
    nonZeroIndex = idx;
  };

  __device__ __host__ void renormalize(){};

  su2_16CellElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 5);
    return randomize(dist(gen));
  };

  __device__ su2_16CellElement randomize(double delta,

                                         CUDA_RAND_STATE_TYPE *state) {
    int n = 6;
    while (n == 6) {
      double t = curand_uniform_double(state) * 6;
      n = (int)t;
    }
    return randomize(n);
  };

protected:
  __device__ __host__ su2_16CellElement randomize(int direction) {

    int sign = 1 - (2 * (direction & 1));
    int newNonZeroIndex = (nonZeroIndex + 1 + (direction >> 1)) % 4;

    double newElement[4] = {0, 0, 0, 0};
    newElement[newNonZeroIndex] = sign;
    return su2_16CellElement(&newElement[0], sign);
  };
  int nonZeroIndex;
};

#endif /*SU2_16CELLElEMENT_HPP*/
