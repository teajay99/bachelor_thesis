#include "customMath.hpp"
#include "su2Element.hpp"

#ifndef SUNACTION_HPP
#define SUNACTION_HPP

template <int dim> class su2Action {
public:
  // Constructor for storing site count and Beta, as well as calculate derived
  // quantities
  su2Action(int iLatSize, double iBeta) {
    latSize = iLatSize;
    siteCount = intPow(latSize, dim);
    for (int i = 0; i < dim; i++) {
      basis[i] = intPow(latSize, i);
    }
    beta = iBeta;
  };

  ~su2Action(){

  };

  /*
  Indices in the matrices work as
      U_mu(x_0, x_1, x_2...) = filds[dim*(x_0 * 1 + x_1 * n  + x_2 * n^2 + ...)
  + mu] (for n sites per dimension)
  */

  // Evaluates the terms in the action dependent on U_mu(loc)
  double evaluateDelta(su2Element *fields, int loc, int mu) {
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
  su2Element plaquetteProduct(su2Element *fields, int loc, int mu, int nu) {

    return fields[(dim * loc) + mu] *
           fields[(dim * ((loc + basis[mu]) % siteCount)) + nu] *
           fields[(dim * ((loc + basis[nu]) % siteCount)) + mu].adjoint() *
           fields[(dim * loc) + nu].adjoint();
  };

  int getSiteCount() { return siteCount; };
  int getBaisis(int i) { return basis[i]; };
  int getBeta() { return beta; };
  int getLatSize() { return latSize; }
  int getDim() { return dim; }

private:
  int siteCount;
  int basis[dim];
  int latSize;
  double beta;
};

#endif
