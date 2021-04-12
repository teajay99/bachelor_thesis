#ifndef METROPOLIZER_HPP
#define METROPOLIZER_HPP

#include "su2Action.hpp"
#include "su2Element.hpp"

template <int dim> class metropolizer {
public:
  metropolizer(su2Action<dim> iAction, int iMultiProbe, double iDelta,
               bool cold);
  ~metropolizer();

  double sweep();
  void measurePlaquette();
  double getHitRate();

private:
  std::mt19937 generator;
  int multiProbe;
  int threadCount;
  double delta;
  double hitRate;
  su2Action<dim> action;
  su2Element *fields;
};

template class metropolizer<2>;
template class metropolizer<3>;
template class metropolizer<4>;
template class metropolizer<5>;
template class metropolizer<6>;
template class metropolizer<7>;
template class metropolizer<8>;
template class metropolizer<9>;
template class metropolizer<10>;

#endif
