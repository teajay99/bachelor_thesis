#include "su2Element.hpp"

#ifndef SU2TETElEMENT_HPP
#define SU2TETElEMENT_HPP

#define TET_EPS 1e-8

class su2TetElement : public su2Element {
public:
  __device__ __host__ su2TetElement() : su2Element(){};

  __device__ __host__ su2TetElement(const su2Element &el) {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
  };

  su2TetElement &operator=(const su2Element &el) {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
    return *this;
  };

  __device__ __host__ void renormalize() {
    for (int i = 0; i < 4; i++) {
      if (abs(su2Element::element[i]) < TET_EPS) {
        su2Element::element[i] = 0;
      } else if (roundToTet(&su2Element::element[i], 0.5)) {
      } else if (roundToTet(&su2Element::element[i], 1.0)) {
      } else {
        printf("You just left the Gauge Group [%f,%f,%f,%f]\n",
               su2Element::element[0], su2Element::element[1],
               su2Element::element[2], su2Element::element[3]);
      }
    }
  };

  su2TetElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 7);
    return randomize(dist(gen));
  };

  __device__ su2TetElement randomize(double delta,

                                     CUDA_RAND_STATE_TYPE *state) {
    int n = 8;
    while (n == 8) {
      double t = curand_uniform_double(state) * 8;
      n = (int)t;
    }
    return randomize(n);
  };

protected:
  __device__ __host__ su2TetElement randomize(int direction) {
    double multEl[4] = {0.5, 0, 0, 0};
    multEl[1] = (1 - (2 * (direction & 1))) * 0.5;
    multEl[2] = (1 - (direction & 2)) * 0.5;
    multEl[3] = (1 - ((direction & 4) / 2)) * 0.5;

    return su2TetElement(&multEl[0]) * (*this);
  };

  __host__ __device__ bool roundToTet(double *el, double roundVal) {
    if ((abs(*el) + TET_EPS) > roundVal && (abs(*el) - TET_EPS) < roundVal) {
      *el = ((*el > 0) - (*el < 0)) * roundVal;
      return true;
    }
    return false;
  }
};

#endif /*SU2TETElEMENT_HPP*/
