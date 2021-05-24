#include "su2Element.hpp"

#ifndef SU2IKOELEMENT_HPP
#define SU2IKOELEMENT_HPP

#define IKO_TAU 0.8090169943749474241022934
#define IKO_TAU_PRIME 0.3090169943749474241022934
#define IKO_EPS 1e-8

class su2IkoElement : public su2Element {
public:
  __device__ __host__ su2IkoElement() : su2Element(){};

  __device__ __host__ su2IkoElement(const su2Element &el) {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
  };

  su2IkoElement &operator=(const su2Element &el) {
    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
    return *this;
  };

  __device__ __host__ void renormalize() {
    for (int i = 0; i < 4; i++) {
      if (abs(su2Element::element[i]) < IKO_EPS) {
        su2Element::element[i] = 0;
      } else if (roundToIko(&su2Element::element[i], IKO_TAU)) {
      } else if (roundToIko(&su2Element::element[i], IKO_TAU_PRIME)) {
      } else if (roundToIko(&su2Element::element[i], 0.5)) {
      } else if (roundToIko(&su2Element::element[i], 1.0)) {
      } else {
        printf("You just left the Gauge Group [%f,%f,%f,%f]\n",
               su2Element::element[0], su2Element::element[1],
               su2Element::element[2], su2Element::element[3]);
      }
    }
  };

  su2IkoElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 11);
    return randomize(dist(gen));
  };

  __device__ su2IkoElement randomize(double delta,

                                     CUDA_RAND_STATE_TYPE *state) {
    int n = 12;
    while (n == 12) {
      double t = curand_uniform_double(state) * 12;
      n = (int)t;
    }
    return randomize(n);
  };

protected:
  __device__ __host__ su2IkoElement randomize(int direction) {
    double multEl[4] = {IKO_TAU, 0, 0, 0};
    int signOne = 1 - (2 * (direction & 1));
    int signTwo = 1 - (direction & 2);

    int offset = direction >> 2;

    multEl[1 + ((offset) % 3)] = 0;
    multEl[1 + ((offset + 1) % 3)] = signOne * IKO_TAU_PRIME;
    multEl[1 + ((offset + 2) % 3)] = signTwo * 0.5;

    return su2IkoElement(&multEl[0]) * (*this);
  };

  __host__ __device__ bool roundToIko(double *el, double roundVal) {
    if ((abs(*el) + IKO_EPS) > roundVal && (abs(*el) - IKO_EPS) < roundVal) {
      *el = ((*el > 0) - (*el < 0)) * roundVal;
      return true;
    }
    return false;
  }
};

#endif /*SU2IKOELEMENT_HPP*/