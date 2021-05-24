#include <fstream>
#include <iostream>
//#include<sstream>
#include <iomanip>
#include <string>

#include "cxxopts.hpp"
#include "executor.hpp"
#include "partitions.hpp"

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
      "m,measurements", "Number of Measurements",
      cxxopts::value<int>()->default_value("1000"))(
      "multi-sweep", "Number of Sweeps per Measurement",
      cxxopts::value<int>()->default_value("1"))(
      "partition-iko", "Use The Ikosaeder Subgroup of SU(2) as a gauge group")(
      "partition-list",
      "Use custom partition provided by an additional List (.csv) File",
      cxxopts::value<std::string>())("c,cold", "Cold Start")(
      "v,verbose", "Verbose output",
      cxxopts::value<bool>()->default_value("false"));

  return options;
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
  int partType = SU2_ELEMENT;
  std::string partFile = "";
  int multiSweep = 0;

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

    if (result.count("partition-iko")) {
      partType = SU2_IKO_ELEMENT;
    } else if (result.count("partition-list")) {
      partFile = result["partition-list"].as<std::string>();
      partType = SU2_LIST_ELEMENT;
    }

    beta = result["beta"].as<double>();
    latSize = result["lattice-size"].as<int>();
    measurements = result["measurements"].as<int>();
    fName = result["output"].as<std::string>();
    delta = result["delta"].as<double>();
    multiProbe = result["hits"].as<int>();
    multiSweep = result["multi-sweep"].as<int>();

  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    std::cout << options.help() << std::endl;
    exit(1);
  }

  if (useCuda && (latSize % 2)) {
    std::cout << "Cuda Support only works with even lattice sizes" << std::endl;
    exit(1);
  }

  // su2Action<4> action(latSize, beta);

  executor<4> exec(latSize, beta, multiProbe, delta, partType, useCuda,
                   partFile);
  exec.initFields(cold);
  exec.run(measurements, multiSweep, fName);
}
