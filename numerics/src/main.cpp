#include "customMath.hpp"
#include "suNAction.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

Eigen::Matrix2cd getRandomGroupElement(std::default_random_engine &gen,
                                       std::normal_distribution<double> &dist) {

  Eigen::Vector4d pnt;
  for (int i = 0; i < 4; i++) {
    pnt(i) = dist(gen);
  }
  pnt /= pnt.norm();

  Eigen::Matrix2cd out;
  out(0, 0) = std::complex<double>(pnt(0), pnt(1));
  out(0, 1) = std::complex<double>(pnt(2), pnt(3));
  out(1, 0) = std::complex<double>(-pnt(2), pnt(3));
  out(0, 0) = std::complex<double>(pnt(0), -pnt(1));

  return out;
}

int main() {
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist(0., 1.);
  std::uniform_real_distribution<double> uni_dist(0., 1.);

  int sites = 8;
  const int dim = 4;
  int siteCount = intPow(sites, dim);

  suNAction<2, dim> action(sites, 2.0);
  Eigen::Matrix2cd fields[dim * siteCount];

  // Generate hot start
  for (int i = 0; i < dim * siteCount; i++) {
    fields[i] = getRandomGroupElement(generator, normal_dist);
  }

  for (int meas = 0; meas < 500; meas++) {
    for (int iter = 0; iter < 30; iter++) {
      for (int loc = 0; loc < siteCount; loc++) {
        for (int probes = 0; probes < 10; probes++) {
          for (int mu = 0; mu < dim; mu++) {
            double old_val = action.evaluateDelta(&fields[0], loc, mu);
            Eigen::Matrix2cd old_element = fields[(4 * loc) + mu];
            fields[(4 * loc) + mu] =
                getRandomGroupElement(generator, normal_dist);
            double new_val = action.evaluateDelta(&fields[0], loc, mu);

            if ((new_val > old_val) &&
                (uni_dist(generator) > exp(-(new_val - old_val)))) {
              fields[(4 * loc) + mu] = old_element;
            }
          }
        }
      }
    }
    // loop in direction 0 and 1 starting at loc = 0
    Eigen::Matrix2cd loopProd = fields[0] * fields[dim + 1] *
                                fields[dim * dim].adjoint() *
                                fields[1].adjoint();
    double meas_val =
        (1.0 / 4.0) * (loopProd + loopProd.adjoint()).trace().real();
    std::cout << meas_val << std::endl;
  }

  std::cout << "Done!" << std::endl;
}
