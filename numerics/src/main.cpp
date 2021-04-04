#include <iostream>
#include <fstream>

#include "cxxopts.hpp"
#include "metropolizer.hpp"
#include "su2Action.hpp"

cxxopts::Options getOptions() {
  cxxopts::Options options("SU2-Metropolizer",
                           "Cuda Based implementation of a SU(2) Metropolis "
                           "Monte Carlo Simulation.");

  options.add_options()("h,help", "Show Help")(
      "b,beta", "Value of Beta for Simulation",
      cxxopts::value<double>()->default_value("2.0"))(
      "l,lattice-size", "Size of the lattice (l^4)",
      cxxopts::value<int>()->default_value("8"))(
      "o,output", "Output File Name",
      cxxopts::value<std::string>()->default_value("data.csv"))(
      "m,measurements", "Number of Sweeps",
      cxxopts::value<int>()->default_value("1000"))("c,cold", "Cold Start")(
      "v,verbose", "Verbose output",
      cxxopts::value<bool>()->default_value("false"));

  return options;
}

int main(int argc, char **argv) {
  cxxopts::Options options = getOptions();

  double beta = 0;
  int latSize = 0;
  bool cold = false;
  int measurements = 0;

  try {

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    if (result.count("cold")) {
      cold = true;
    }

    beta = result["beta"].as<double>();
    latSize = result["lattice-size"].as<int>();
    measurements = result["measurements"].as<int>();

  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    std::cout << options.help() << std::endl;
    exit(1);
  }

  su2Action<4> action(latSize, beta);
  metropolizer<4> metro(action, 15, 0.2, cold);


  std::ofstream file;
  for (int i = 0; i < measurements; i++) {
    double plaquette = metro.sweep(50);
    std::cout << i << ") "<< plaquette << std::endl;
    file << i << "\t" << plaquette << std::endl;
  }
}
