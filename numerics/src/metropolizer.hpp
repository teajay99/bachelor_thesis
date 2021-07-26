#ifndef METROPOLIZER_HPP
#define METROPOLIZER_HPP

#include "config.hpp"
#include "partitions.hpp"
#include "su2Action.hpp"
#include <random>

template <int dim, class su2Type> class metropolizer {
public:
  metropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
               su2Type *fields);
  ~metropolizer();

  double sweep(int sweeps);

  double getHitRate();

private:
  std::mt19937 generator;
  int multiProbe;
  double delta;
  double hitRate;
  su2Action<dim> action;
  su2Type *fields;
};

template class metropolizer<4, su2Element>;
template class metropolizer<4, su2TetElement>;
template class metropolizer<4, su2OctElement>;
template class metropolizer<4, su2IcoElement>;
template class metropolizer<4, su2ListElement>;
template class metropolizer<4, su2VolleyElement>;
template class metropolizer<4, su2_5CellElement>;
template class metropolizer<4, su2_16CellElement>;
template class metropolizer<4, su2_120CellElement>;


#endif
