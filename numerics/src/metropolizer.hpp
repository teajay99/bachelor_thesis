#include "su2Action.hpp"
#include "su2Element.hpp"

template <int dim> class metropolizer {
public:
  metropolizer();
  ~metropolizer();
private:
  int multiProbe;
  int sweepsPerMeasure;
  double delta;
  su2Action<dim> action;
  curandState_t * randStates;
  su2Element * fields;
}
