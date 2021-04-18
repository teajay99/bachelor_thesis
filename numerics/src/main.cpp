#include <fstream>
#include <iostream>
//#include<sstream>
#include <iomanip>
#include <string>

#include "cudaMetropolizer.hpp"
#include "cxxopts.hpp"
#include "metropolizer.hpp"
#include "su2Action.hpp"

cxxopts::Options getOptions() {
  cxxopts::Options options("SU2-Metropolizer",
                           "Cuda Based implementation of a SU(2) Metropolis "
                           "Monte Carlo Simulation.");

  options.add_options()("h,help",
                        "Show Help")("cuda", "Use NVIDIA GPU for calculation")(
      "b,beta", "Value of Beta for Simulation",
      cxxopts::value<double>()->default_value("2.0"))(
      "d,delta", "Search Radius for Simulation",
      cxxopts::value<double>()->default_value("0.2"))(
      "l,lattice-size", "Size of the lattice (l^4)",
      cxxopts::value<int>()->default_value("8"))(
      "hits", "Hits per site per Measurement",
      cxxopts::value<int>()->default_value("10"))(
      "o,output", "Output File Name",
      cxxopts::value<std::string>()->default_value("data.csv"))(
      "m,measurements", "Number of Sweeps",
      cxxopts::value<int>()->default_value("1000"))("c,cold", "Cold Start")(
      "v,verbose", "Verbose output",
      cxxopts::value<bool>()->default_value("false"));

  return options;
}

void logResults(int i, std::ofstream &file, double plaquette, double hitRate) {
  std::cout << i << " " << std::scientific << std::setw(18)
            << std::setprecision(15) << plaquette << " " << hitRate
            << std::endl;
  file << i << "\t" << std::scientific << std::setw(18) << std::setprecision(15)
       << plaquette << "\t" << hitRate << std::endl;
}

int main(int argc, char **argv) {
  cxxopts::Options options = getOptions();

  double beta = 0;
  int latSize = 0;
  bool cold = false;
  bool useCuda = false;
  int measurements = 0;
  int multiProbe = 0;
  std::string fName;
  double delta = 0;
  try {

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    if (result.count("cold")) {
      cold = true;
    }

    if (result.count("cuda")) {
      useCuda = true;
    }

    beta = result["beta"].as<double>();
    latSize = result["lattice-size"].as<int>();
    measurements = result["measurements"].as<int>();
    fName = result["output"].as<std::string>();
    delta = result["delta"].as<double>();
    multiProbe = result["hits"].as<int>();

  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    std::cout << options.help() << std::endl;
    exit(1);
  }

  su2Action<4> action(latSize, beta);

  std::ofstream file;
  file.open(fName);
  if (useCuda) {
    cudaMetropolizer<4> metro(action, multiProbe, delta, cold);
    for (int i = 0; i < measurements; i++) {
      double plaquette = metro.sweep();
      logResults(i, file, plaquette, metro.getHitRate());
    }
  } else {
    metropolizer<4> metro(action, multiProbe, delta, cold);
    for (int i = 0; i < measurements; i++) {
      double plaquette = metro.sweep();
      logResults(i, file, plaquette, metro.getHitRate());
    }
  }
  file.close();
}
