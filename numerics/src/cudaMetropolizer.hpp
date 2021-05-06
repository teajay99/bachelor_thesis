#ifndef CUDAMETROPOLIZER_HPP
#define CUDAMETROPOLIZER_HPP

#include "config.hpp"
#include "su2Action.hpp"
#include "su2Element.hpp"
#include "su2IkoElement.hpp"

template <int dim, class su2Type> class cudaMetropolizer {
public:
  cudaMetropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
                   su2Type *fields);
  // cudaMetropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
  //                  bool cold, std::string partFile);
  ~cudaMetropolizer();

  double sweep();

  double partSweep();

  double measurePlaquette();
  double getHitRate();

private:
  int multiProbe;
  int threadCount;
  int blockCount;
  int *hitCounts;
  double delta;
  double hitRate;
  su2Action<dim> action;
  int randStateCount;
  CUDA_RAND_STATE_TYPE *randStates;
  su2Type *fields;
};

// template class cudaMetropolizer<2>;
// template class cudaMetropolizer<3>;
template class cudaMetropolizer<4, su2Element>;
template class cudaMetropolizer<4, su2IkoElement>;
// template class cudaMetropolizer<5>;
// template class cudaMetropolizer<6>;
// template class cudaMetropolizer<7>;
// template class cudaMetropolizer<8>;
// template class cudaMetropolizer<9>;
// template class cudaMetropolizer<10>;

#endif
