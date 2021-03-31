
#include "customMath.hpp"
#include <Eigen/Dense>
#include <complex>

#ifndef SUNACTION_HPP
#define SUNACTION_HPP

template <int N, int dim> class suNAction {
public:
  suNAction(int n, double iBeta) {
    siteCount = intPow(n, dim);
    for (int i = 0; i < dim; i++) {
      basis[i] = intPow(n, i);
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

  double evaluateDelta(Eigen::Matrix<std::complex<double>, N, N> *fields,
                       int loc, int mu) {
    double sum = 0.;
    for (int i = 0; i < dim; i++) {
      if (i != mu) {
        Eigen::Matrix<std::complex<double>, N, N> pProd =
            plaquetteProduct(fields, loc, mu, i);
        sum += (pProd + pProd.adjoint()).trace().real();
        pProd = plaquetteProduct(
            fields, (loc + siteCount - basis[i]) % siteCount, mu, i);
        sum += (pProd + pProd.adjoint()).trace().real();
      }
    }
    return ((-beta) / (2 * N)) * sum;
  };

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
  double beta;
};

#endif
