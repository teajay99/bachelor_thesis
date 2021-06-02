#include "su2Element.hpp"

#ifndef SU2VOLLEYElEMENT_HPP
#define SU2VOLLEYElEMENT_HPP

#define CUBE_EPS 1e-8

class su2VolleyElement : public su2Element {
public:
  __device__ __host__ su2VolleyElement() : su2VolleyElement(0){};

  __device__ __host__ su2VolleyElement(int iSubdivs) {
    subdivs = iSubdivs;
    for (int i = 0; i < 4; i++) {
      cubeCoords[i] = 1;
    }
    projectCubeCoords(&cubeCoords[0], &su2Element::element[0]);
  };

  __device__ __host__ su2VolleyElement(double *iCubeCoords, int iSubdivs) {
    subdivs = iSubdivs;
    for (int i = 0; i < 4; i++) {
      cubeCoords[i] = iCubeCoords[i];
    }
    projectCubeCoords(&cubeCoords[0], &su2Element::element[0]);
  };

  __device__ __host__ void renormalize() {
    double stepWidth = 2.0 / (subdivs + 1.0);
    for (int i = 0; i < 4; i++) {
      bool foundOne = false;
      for (int j = 0; j < subdivs + 2; j++) {
        double val = (stepWidth * j) - 1;
        if (roundToCoord(&cubeCoords[i], val)) {
          foundOne = true;
          break;
        };
      }
      if (foundOne == false) {
        printf("You Left the lattice\n");
      }
    }
    projectCubeCoords(&cubeCoords[0], &su2Element::element[0]);
  };

  su2VolleyElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 7);
    bool succes = false;
    su2VolleyElement out;
    while (!succes) {
      out = randomize(dist(gen), &succes);
    }
    return out;
  };

  __device__ su2VolleyElement randomize(double delta,

                                        CUDA_RAND_STATE_TYPE *state) {
    bool succes = false;
    su2VolleyElement out;
    while (succes == false) {
      int n = 8;
      while ((n == 8)) {
        double t = curand_uniform_double(state) * 8;
        n = (int)t;
      }
      out = randomize(n, &succes);
    }
    return out;
  };

protected:
  __device__ __host__ su2VolleyElement randomize(int direction, bool *succes) {
    int index = (direction & 6) >> 1;

    int sign = 1 - (2 * (direction & 1));

    // printf("Stuff: %d   %d\n", sign, index);

    double stepWidth = 2.0 / (subdivs + 1.0);

    double newCubeCoords[4] = {cubeCoords[0], cubeCoords[1], cubeCoords[2],
                               cubeCoords[3]};
    newCubeCoords[index] += (sign * stepWidth);
    if (abs(newCubeCoords[index]) < 1 + CUBE_EPS) {
      for (int i = 0; i < 4; i++) {
        if (abs(abs(newCubeCoords[i]) - 1) < CUBE_EPS) {

          *succes = true;
          //printf("Success!!!");
        }
      }
    }
    return su2VolleyElement(&newCubeCoords[0], subdivs);
  };

  __host__ __device__ bool roundToCoord(double *el, double roundVal) {
    if (abs(roundVal - *el) < CUBE_EPS) {
      *el = roundVal;
      return true;
    }
    return false;
  };

  __device__ __host__ void projectCubeCoords(double *in, double *out) {
    double sum = 0;
    for (int i = 0; i < 4; i++) {
      sum += in[i] * in[i];
    }
    sum = sqrt(sum);
    for (int i = 0; i < 4; i++) {
      out[i] = in[i] / sum;
    };
  };

  double cubeCoords[4];
  int subdivs;
};

#endif /*SU2VOLLEYElEMENT_HPP*/
