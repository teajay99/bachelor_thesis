#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "su2Action.hpp"
#include "su2Element.hpp"
#include "su2IkoElement.hpp"
#include "su2ListElement.hpp"
//#include "metropolizer.hpp"
#include "cudaMetropolizer.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

template <int dim> class executor {
public:
  executor(int latSize, double beta, int multiProbe, double delta, int partType,
           bool useCuda, std::string partFile);
  ~executor();

  void initFields(bool cold);
  void run(int measurements, std::string outFile);

private:
  template <class su2Type>
  void runMetropolis(int measurements, std::ofstream &outFile);

  void logResults(int i, double plaquette, double hitRate, std::ofstream &file);

  void initListFields(bool cold);

  su2Action<dim> action;
  std::string partFile;
  int multiProbe;
  double delta;
  bool useCuda;
  int partType;
  void *fields;

  //List specific stuff
  su2Element *elementList;
  double *distList;
};

template class executor<4>;

#endif /*EXECUTOR_HPP*/
