#include <iostream>
#include <math.h>
#include <random>

#ifndef SU2ELEMENT_HPP
#define SU2ELEMENT_HPP

#include <cuda.h>
#include <curand_kernel.h>

class su2Element {
public:
  __device__ __host__ su2Element();
  __device__ __host__ su2Element(double[4]);
  __device__ __host__ ~su2Element();

  __device__ __host__ friend su2Element operator*(const su2Element &e1,
                                                  const su2Element &e2);
  __device__ __host__ double trace();
  __device__ __host__ su2Element adjoint();

  friend std::ostream &operator<<(std::ostream &os, const su2Element &e);
  __device__ __host__ double operator[](int i) const;
  __device__ __host__ double &operator[](int i);

  su2Element randomize(double delta, std::default_random_engine &gen);
  __device__ su2Element randomize(double delta, curandState_t &state);

  __device__ __host__ void renormalize();

private:
  __device__ __host__ su2Element randomize(double alpha, double *pnt);
  double element[4];
};

#endif
