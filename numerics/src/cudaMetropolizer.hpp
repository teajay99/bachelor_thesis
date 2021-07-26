#ifndef CUDAMETROPOLIZER_HPP
#define CUDAMETROPOLIZER_HPP

#include "config.hpp"
#include "partitions.hpp"
#include "su2Action.hpp"

template <int dim, class su2Type> class cudaMetropolizer {
public:
  cudaMetropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
                   su2Type *fields);
  ~cudaMetropolizer();

  double sweep(int sweeps);

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

template class cudaMetropolizer<4, su2Element>;
template class cudaMetropolizer<4, su2TetElement>;
template class cudaMetropolizer<4, su2OctElement>;
template class cudaMetropolizer<4, su2IcoElement>;
template class cudaMetropolizer<4, su2ListElement>;
template class cudaMetropolizer<4, su2VolleyElement>;
template class cudaMetropolizer<4, su2_5CellElement>;
template class cudaMetropolizer<4, su2_16CellElement>;
template class cudaMetropolizer<4, su2_120CellElement>;

#endif
