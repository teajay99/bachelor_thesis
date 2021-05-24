#include <iostream>
#include <math.h>
#include <random>

#ifndef SU2ELEMENT_HPP
#define SU2ELEMENT_HPP

#include "config.hpp"
#include <cuda.h>
#include <curand_kernel.h>

/*
#  SU(2) in fundamental rep stored as
#
#  (  array[0] + i* array[1] , array[2] + i* array[3] )
#  ( -array[2] + i* array[3] , array[0] - i* array[1] )
#
*/

class su2Element {

public:
  __device__ __host__ su2Element() {
    element[0] = 1;
    element[1] = 0;
    element[2] = 0;
    element[3] = 0;
  };

  __device__ __host__ su2Element(double *input) {
    for (int i = 0; i < 4; i++) {
      element[i] = input[i];
    }
  }

  __device__ __host__ ~su2Element(){};

  __device__ __host__ double trace() { return 2 * element[0]; };

  friend __device__ __host__ su2Element operator*(const su2Element &e1,
                                                  const su2Element &e2);
  __device__ __host__ su2Element adjoint() {
    double adjoint[4] = {element[0], -element[1], -element[2], -element[3]};
    return su2Element(&adjoint[0]);
  };

  friend std::ostream &operator<<(std::ostream &os, const su2Element &e);

  __device__ __host__ double operator[](int i) const { return element[i]; }
  __device__ __host__ double &operator[](int i) { return element[i]; }

  su2Element randomize(double delta, std::mt19937 &gen) {

    std::normal_distribution<double> normal_dist(0., 1.);
    std::uniform_real_distribution<double> uni_dist(0, M_PI * delta * 2);

    double alpha = uni_dist(gen);

    double pnt[3];
    for (int i = 0; i < 3; i++) {
      pnt[i] = normal_dist(gen);
    }
    return this->randomize(alpha, &pnt[0]);
  };

  __device__ su2Element randomize(double delta, CUDA_RAND_STATE_TYPE *state) {
    double alpha = curand_uniform_double(state) * (M_PI * delta * 2.);
    double pnt[3];
    for (int i = 0; i < 3; i++) {
      pnt[i] = curand_normal_double(state);
    }
    return this->randomize(alpha, &pnt[0]);
  };


  __device__ __host__ void renormalize() {
    double norm = 0;
    for (int i = 0; i < 4; i++) {
      norm += element[i] * element[i];
    }

    norm = sqrt(norm);

    for (int i = 0; i < 4; i++) {
      element[i] /= norm;
    }
  };

protected:
  __device__ __host__ su2Element randomize(double alpha, double *pnt) {
    double norm = 0;
    double sAlpha = sin(alpha);

    for (int i = 0; i < 3; i++) {
      norm += pnt[i] * pnt[i];
    }

    norm = sqrt(norm);
    for (int i = 0; i < 3; i++) {
      pnt[i] /= norm;
      pnt[i] *= sAlpha;
    }
    double coord[4] = {cos(alpha), pnt[0], pnt[1], pnt[2]};

    return su2Element(&coord[0]) * (*this);
  };


  double element[4];
};

inline __device__ __host__ su2Element operator*(const su2Element &e1,
                                                const su2Element &e2) {
  double product[4];
  product[0] =
      (e1[0] * e2[0]) - (e1[1] * e2[1]) - (e1[2] * e2[2]) - (e1[3] * e2[3]);
  product[1] =
      (e1[0] * e2[1]) + (e1[1] * e2[0]) + (e1[2] * e2[3]) - (e1[3] * e2[2]);
  product[2] =
      (e1[0] * e2[2]) - (e1[1] * e2[3]) + (e1[2] * e2[0]) + (e1[3] * e2[1]);
  product[3] =
      (e1[0] * e2[3]) + (e1[1] * e2[2]) - (e1[2] * e2[1]) + (e1[3] * e2[0]);
  return su2Element(&product[0]);
};

inline std::ostream &operator<<(std::ostream &os, const su2Element &e) {
  os << "(" << e[0] << "," << e[1] << ") ";
  os << "(" << e[2] << "," << e[3] << ") " << std::endl;
  os << "(" << -e[2] << "," << e[3] << ") ";
  os << "(" << e[0] << "," << -e[1] << ") " << std::endl;
  return os;
};

#endif
