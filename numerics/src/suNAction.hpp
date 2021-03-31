
#include "customMath.hpp"
#include <Eigen/Dense>
#include <complex>

#ifndef SUNACTION_HPP
#define SUNACTION_HPP

template <int N, int dim> class suNAction {
public:
  // Constructor for storing site count and Beta, as well as calculate derived
  // quantities
  suNAction(int iSites, double iBeta) {
    sites = iSites;
    siteCount = intPow(sites, dim);
    for (int i = 0; i < dim; i++) {
      basis[i] = intPow(sites, i);
    }
    beta = iBeta;
  };

  ~suNAction(){

  };

  /*
  Indices in the matrices work as
      U_mu(x_0, x_1, x_2...) = filds[dim*(x_0 * 1 + x_1 * n  + x_2 * n^2 + ...)
  + mu] (for n sites per dimension)
  */

  // Evaluates the terms in the action dependent on U_mu(loc)
  double evaluateDelta(Eigen::Matrix<std::complex<double>, N, N> *fields,
                       int loc, int mu) {
    // return value
    double sum = 0.;
    // Iterating over dimensions
    for (int i = 0; i < dim; i++) {
      if (i != mu) {
        // Evaluating Upper Plaquette
        Eigen::Matrix<std::complex<double>, N, N> pProd =
            plaquetteProduct(fields, loc, mu, i);
        sum += (pProd + pProd.adjoint()).trace().real();

        // Evaluating Lower Plaquette
        pProd = plaquetteProduct(
            fields, (loc + siteCount - basis[i]) % siteCount, mu, i);
        sum += (pProd + pProd.adjoint()).trace().real();
      }
    }
    return ((-beta) / (2 * N)) * sum;
  };

  // Evaluate Plaquette Product starting from loc, in direction mu and nu
  Eigen::Matrix<std::complex<double>, N, N>
  plaquetteProduct(Eigen::Matrix<std::complex<double>, N, N> *fields, int loc,
                   int mu, int nu) {

    return fields[(dim * loc) + mu] *
           fields[(dim * ((loc + basis[mu]) % siteCount)) + nu] *
           fields[(dim * ((loc + basis[nu]) % siteCount)) + mu].adjoint() *
           fields[(dim * loc) + nu].adjoint();
  };

private:
  int siteCount;
  int basis[dim];
  int sites;
  double beta;
};

#endif
