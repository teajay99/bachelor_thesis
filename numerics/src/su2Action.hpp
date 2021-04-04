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

  // Evaluates the terms in the action dependent on U_mu(loc)
  __device__ __host__ double evaluateDelta(su2Element *fields, int loc,
                                           int mu) {
    // return value
    double sum = 0.;
    // Iterating over dimensions
    for (int i = 0; i < dim; i++) {
      if (i != mu) {
        // Evaluating Upper Plaquette
        su2Element pProd = plaquetteProduct(fields, loc, mu, i);
        sum += pProd.trace();

        // Evaluating Lower Plaquette
        pProd = plaquetteProduct(
            fields, (loc + siteCount - basis[i]) % siteCount, mu, i);
        sum += pProd.trace();
      }
    }
    return ((-beta) / (2)) * sum;
  };

  // Evaluate Plaquette Product starting from loc, in direction mu and nu
  __device__ __host__ su2Element plaquetteProduct(su2Element *fields, int loc,
                                                  int mu, int nu) {

    return fields[(dim * loc) + mu] *
           fields[(dim * ((loc + basis[mu]) % siteCount)) + nu] *
           fields[(dim * ((loc + basis[nu]) % siteCount)) + mu].adjoint() *
           fields[(dim * loc) + nu].adjoint();
  };

  __device__ __host__ int getSiteCount() { return siteCount; };
  __device__ __host__ int getBasis(int i) { return basis[i]; };
  __device__ __host__ int getBeta() { return beta; };
  __device__ __host__ int getLatSize() { return latSize; }
  __device__ __host__ int getDim() { return dim; }

private:
  int siteCount;
  int basis[dim];
  int latSize;
  double beta;
};

#endif
