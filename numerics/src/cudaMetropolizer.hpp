#ifndef CUDAMETROPOLIZER_HPP
#define CUDAMETROPOLIZER_HPP

#include "su2Action.hpp"
#include "su2Element.hpp"

template <int dim> class cudaMetropolizer {
public:
  cudaMetropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
                   bool cold);
  ~cudaMetropolizer();

  double sweep();
  void measurePlaquette();
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
  curandStateMtgp32 *randStates;
  mtgp32_kernel_params *randStateParams;
  su2Element *fields;
};

template class cudaMetropolizer<2>;
template class cudaMetropolizer<3>;
template class cudaMetropolizer<4>;
template class cudaMetropolizer<5>;
template class cudaMetropolizer<6>;
template class cudaMetropolizer<7>;
template class cudaMetropolizer<8>;
template class cudaMetropolizer<9>;
template class cudaMetropolizer<10>;

#endif
