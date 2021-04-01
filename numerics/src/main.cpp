#include "customMath.hpp"
#include "suNAction.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <fstream>
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
  out(1, 1) = std::complex<double>(pnt(0), -pnt(1));

  return out;
}

int main() {
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist(0., 1.);
  std::uniform_real_distribution<double> uni_dist(0., 1.);

  // Lattice Sites per Dimension
  int sites = 6;
  // Space-Time Dimensions
  const int dim = 4;
  // Lattice Sites
  int siteCount = intPow(sites, dim);

  // Creating an Action Object
  suNAction<2, dim> action(sites, 2.0);
  // Array to store field configuration
  Eigen::Matrix2cd fields[dim * siteCount];

  // Generate hot start
  for (int i = 0; i < dim * siteCount; i++) {
    fields[i] = getRandomGroupElement(generator, normal_dist);
  }

  std::ofstream outputFile;
  outputFile.open("output.csv");
  outputFile << "single\tplane\thypercube" << std::endl;

  // Iterations over measurements
  for (int meas = 0; meas < 100; meas++) {
    // Metropolissteps per measurement
    for (int iter = 0; iter < 10; iter++) {
      // Iteration over all lattice sites
      for (int loc = 0; loc < siteCount; loc++) {
        // Iteration over all space-time dimensions
        for (int mu = 0; mu < dim; mu++) {
          // Mulitprobing of each site
          for (int probes = 0; probes < 5; probes++) {

            // Evaluates action "around" link Variable U_mu (loc)
            double old_val = action.evaluateDelta(&fields[0], loc, mu);
            Eigen::Matrix2cd old_element = fields[(4 * loc) + mu];
            fields[(4 * loc) + mu] =
                getRandomGroupElement(generator, normal_dist);
            // Evaluating action with new link Variable
            double new_val = action.evaluateDelta(&fields[0], loc, mu);

            // Deciding wether to keep the new link Variable
            if ((new_val > old_val) &&
                (uni_dist(generator) > exp(-(new_val - old_val)))) {
              fields[(4 * loc) + mu] = old_element;
            }
          }
        }
      }
    }

    //##############
    //  Measurements
    //##############

    // W(1,1) averaged a fixed point in space
    double zero_loop = action.plaquetteProduct(fields, 0, 0, 1).trace().real();

    // W(1,1) averaged over the 0,1 plane
    double plane = 0;
    for (int i = 0; i < siteCount; i++) {
      plane += action.plaquetteProduct(fields, i, 0, 1).trace().real();
    }
    plane /= siteCount;

    // W(1,1) averaged over all of space-time
    double all_loops = 0;
    for (int i = 0; i < siteCount; i++) {
      for (int mu = 0; mu < dim; mu++) {
        for (int nu = 0; nu < mu; nu++) {
          all_loops += action.plaquetteProduct(fields, i, mu, nu).trace().real();
        }
      }
    }
    all_loops /= siteCount * dim * (dim - 1) * 0.5;

    // Output to conosle and file
    outputFile << zero_loop << "\t" << plane << "\t" << all_loops << std::endl;
    std::cout << zero_loop << ",  " << plane << ",  " << all_loops << std::endl;
  }

  outputFile.close();
  std::cout << "Done!" << std::endl;
}
