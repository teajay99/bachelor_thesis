#include "su2Element.hpp"

#ifndef SU2LINEARElEMENT_HPP
#define SU2LINEARElEMENT_HPP

#define CUBE_EPS 1e-8

template <bool weighted> class su2LinearElement : public su2Element {
public:
  __device__ __host__ su2LinearElement() : su2LinearElement(0){};

  __device__ __host__ su2LinearElement(int iSubdivs) {
    subdivs = iSubdivs;
    for (int i = 1; i < 4; i++) {
      octCoords[i] = 0;
    }
    octCoords[0] = subdivs;
    projectOctCoords(&octCoords[0], &su2Element::element[0]);
  };

  __device__ __host__ su2LinearElement(int *iOctCoords, int iSubdivs) {
    subdivs = iSubdivs;
    for (int i = 0; i < 4; i++) {
      octCoords[i] = iOctCoords[i];
    }
    projectOctCoords(&octCoords[0], &su2Element::element[0]);
  };

  __device__ __host__ void renormalize() {
    projectOctCoords(&octCoords[0], &su2Element::element[0]);
  };

  su2LinearElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 23);
    bool success = false;
    su2LinearElement out;
    while (!success) {
      out = randomize(dist(gen), &success);
    }
    return out;
  };

  __device__ su2LinearElement randomize(double delta,
                                        CUDA_RAND_STATE_TYPE *state) {
    bool success = false;
    su2LinearElement out;
    while (success == false) {
      int n = 24;
      while ((n == 24)) {
        double t = curand_uniform_double(state) * 24;
        n = (int)t;
      }
      out = randomize(n, &success);
    }
    return out;
  };

  __device__ __host__ double getWeight(){return 1.0;};

protected:
  __device__ __host__ su2LinearElement randomize(int direction, bool *success) {
    int i1 = (6 & direction) >> 1;
    int i2 = (24 & direction) >> 3;

    // Make sure i1 != i2
    if (i2 >= i1) {
      i2++;
    }

    int sign = 1 - (2 * (direction & 1));

    int newOctCoords[4] = {octCoords[0], octCoords[1], octCoords[2],
                           octCoords[3]};
    if ((newOctCoords[i1] == 0) && newOctCoords[i2] == 0) {
      return su2LinearElement(&newOctCoords[0], subdivs);
    } else if (newOctCoords[i1] == 0) {
      newOctCoords[i1] = sign;
      newOctCoords[i2] = ((0 < newOctCoords[i2]) - (newOctCoords[i2] < 0)) *
                         (abs(newOctCoords[i2]) - 1);
    } else if (newOctCoords[i2] == 0) {
      newOctCoords[i2] = sign;
      newOctCoords[i1] = ((0 < newOctCoords[i1]) - (newOctCoords[i1] < 0)) *
                         (abs(newOctCoords[i1]) - 1);
    } else {
      newOctCoords[i1] = ((0 < newOctCoords[i1]) - (newOctCoords[i1] < 0)) *
                         (abs(newOctCoords[i1]) + sign);
      newOctCoords[i2] = ((0 < newOctCoords[i2]) - (newOctCoords[i2] < 0)) *
                         (abs(newOctCoords[i2]) - sign);
    }

    *success = true;


    return su2LinearElement(&newOctCoords[0], subdivs);
  };

  __host__ __device__ bool roundToCoord(double *el, double roundVal) {
    if (abs(roundVal - *el) < CUBE_EPS) {
      *el = roundVal;
      return true;
    }
    return false;
  };

  __device__ __host__ void projectOctCoords(int *in, double *out) {
    double sum = 0;
    for (int i = 0; i < 4; i++) {
      sum += in[i] * in[i];
    }
    sum = sqrt(sum);
    for (int i = 0; i < 4; i++) {
      out[i] = in[i] / sum;
    };
  };

  int octCoords[4];
  int subdivs;
};

template class su2LinearElement<false>;

template <> __host__ __device__ inline double su2LinearElement<true>::getWeight() {
  double weight = 0;

  for (int i = 0; i < 4; i++) {
    weight += octCoords[i] * octCoords[i];
  }

  weight = pow(weight, -1.5);
  return weight;
}


#endif /*SU2LINEARElEMENT_HPP*/
