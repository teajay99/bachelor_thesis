#include "metropolizer.hpp"

template <int dim>
metropolizer<dim>::metropolizer(su2Action<dim> iAction, int iMultiProbe,
                                double iDelta, bool cold)
    : action(iAction) {
  delta = iDelta;
  multiProbe = iMultiProbe;

  fields = new su2Element[action.getSiteCount() * dim];

  for (int i = 0; i < action.getSiteCount() * dim; i++) {
    fields[i] = su2Element();
    fields[i].randomize(1, generator);
  }
}

template <int dim> metropolizer<dim>::~metropolizer() { delete[] fields; }

template <int dim> double metropolizer<dim>::sweep() {

  std::uniform_real_distribution<double> uni_dist(0., 1.);

  int hitCount = 0;

  for (int site = 0; site < action.getSiteCount(); site++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * site) + mu;
      for (int i = 0; i < multiProbe; i++) {
        // Evaluates action "around" link Variable U_mu (site)
        double oldVal = action.evaluateDelta(fields, site, mu);
        su2Element oldElement = fields[loc];
        fields[loc] = oldElement.randomize(delta, generator);

        // Evaluating action with new link Variable
        double newVal = action.evaluateDelta(fields, site, mu);

        // Deciding wether to keep the new link Variable
        if ((newVal > oldVal) &&
            (uni_dist(generator) > exp(-(newVal - oldVal)))) {
          fields[loc] = oldElement;
        } else {
          hitCount++;
        }
      }
      fields[loc].renormalize();
    }
  }

  hitRate = (double)hitCount / (action.getSiteCount() * dim * multiProbe);

  double out = 0;
  for (int site = 0; site < action.getSiteCount(); site++) {
    for (int mu = 0; mu < dim; mu++) {
      for (int nu = 0; nu < mu; nu++) {
        out += action.plaquetteProduct(fields, site, mu, nu).trace();
      }
    }
  }
  return out /= action.getSiteCount() * dim * (dim - 1);
}

template <int dim> double metropolizer<dim>::getHitRate() {
  return hitRate;
}
