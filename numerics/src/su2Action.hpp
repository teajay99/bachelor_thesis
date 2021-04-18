#include "customMath.hpp"
#include "su2Element.hpp"

#ifndef SUNACTION_HPP
#define SUNACTION_HPP

template <int dim> class su2Action {
public:
  // Constructor for storing site count and Beta, as well as calculate derived
  // quantities
  __device__ __host__ su2Action(int iLatSize, double iBeta) {
    latSize = iLatSize;
    siteCount = intPow(latSize, dim);
    for (int i = 0; i < dim; i++) {
      basis[i] = intPow(latSize, i);
    }
    beta = iBeta;
  };

  __device__ __host__ ~su2Action(){

  };

  /*
  Indices in the matrices work as
      U_mu(x_0, x_1, x_2...) = filds[dim*(x_0 * 1 + x_1 * n  + x_2 * n^2 + ...)
  + mu] (for n sites per dimension)
  */

  __device__ __host__ double evaluateDelta(su2Element *fields, su2Element newU,
                                           int site, int mu) {
    // return value
    double sum = 0.;
    // Iterating over dimensions
    for (int i = 0; i < dim; i++) {
      if (i != mu) {
        // Evaluating Upper Plaquette
        sum -= this->plaquetteProduct(fields, site, mu, i).trace();
        sum +=
            this->newPlaquetteProduct(fields, newU, true, site, mu, i).trace();

        // Evaluating Lower Plaquette
        sum -= this->plaquetteProduct(fields, getNeighbour(site, i, -1), mu, i)
                   .trace();
        sum += this->newPlaquetteProduct(fields, newU, false,
                                         getNeighbour(site, i, -1), mu, i)
                   .trace();
      }
    }
    return ((-beta) / (2)) * sum;
  };

  // Evaluate Plaquette Product starting from site, in direction mu and nu
  __device__ __host__ su2Element plaquetteProduct(su2Element *fields, int site,
                                                  int mu, int nu) {

    int k0 = (dim * site) + mu;
    int k1 = (dim * getNeighbour(site, mu, 1)) + nu;
    int k2 = (dim * getNeighbour(site, nu, 1)) + mu;
    int k3 = (dim * site) + nu;

    return fields[k0] * fields[k1] * fields[k2].adjoint() *
           fields[k3].adjoint();
  };

  // Evaluate Plaquette Product starting from site, in direction mu and nu
  // Replacing one with a new element
  __device__ __host__ su2Element newPlaquetteProduct(su2Element *fields,
                                                     su2Element newU,
                                                     bool upper, int site,
                                                     int mu, int nu) {

    int k0 = (dim * site) + mu;
    int k1 = (dim * getNeighbour(site, mu, 1)) + nu;
    int k2 = (dim * getNeighbour(site, nu, 1)) + mu;
    int k3 = (dim * site) + nu;

    if (upper) {
      return newU * fields[k1] * fields[k2].adjoint() * fields[k3].adjoint();
    } else {
      return fields[k0] * fields[k1] * newU.adjoint() * fields[k3].adjoint();
    }
  };

  __device__ __host__ int getSiteCount() { return siteCount; };
  __device__ __host__ int getBasis(int i) { return basis[i]; };
  __device__ __host__ int getBeta() { return beta; };
  __device__ __host__ int getLatSize() { return latSize; };
  __device__ __host__ int getDim() { return dim; };

  __device__ __host__ int getIndex(int *coords) {
    int out = 0;
    for (int i = 0; i < dim; i++) {
      out += coords[i] * basis[i];
    }
    return out;
  };

  __device__ __host__ void getCoords(int k, int *coords) {
    for (int i = dim - 1; i >= 0; i--) {
      coords[i] = k / basis[i];
      k %= basis[i];
    };
  };

  __device__ __host__ int getNeighbour(int k, int mu, int dir) {
    int coords[dim];
    this->getCoords(k, &coords[0]);
    coords[mu] = (coords[mu] + latSize + dir) % latSize;
    return getIndex(&coords[0]);
  };

private:
  int siteCount;
  int basis[dim];
  int latSize;
  double beta;
};

#endif
