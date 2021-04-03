#include "metropolizer.hpp"

template <int dim>
__global__ void probe_site(su2Action<dim> act, su2Element *fields,
                           curandState_t *randStates, int multiProbe,
                           double delta, int odd) {

  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int site = 2 * idx;
  int offset = 0;
  for (int i = 0; i < act.getDim(); i++) {
    offset += site / act.getBasis(i);
  }
  site += (offset + odd) % 2;

  for (int mu = 0; mu < act.getDim(); mu++) {
    for (int i = 0; i < multiProbe; i++) {
      // Evaluates action "around" link Variable U_mu (site)
      double oldVal = act.evaluateDelta(&fields[0], site, mu);
      su2Element oldElement = fields[(4 * site) + mu];
      fields[(4 * site) + mu] = oldElement.randomize(delta, randStates[idx]);

      // Evaluating action with new link Variable
      double newVal = act.evaluateDelta(&fields[0], site, mu);

      // Deciding wether to keep the new link Variable
      if ((newVal > oldVal) &&
          (curand_uniform_double(&randStates[idx]) > exp(-(newVal - oldVal)))) {
        fields[(4 * site) + mu] = oldElement;
      }
    }
    fields[(4 * site) + mu].renormalize();
  }
}
