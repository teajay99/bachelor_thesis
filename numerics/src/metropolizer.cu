#include "metropolizer.hpp"
#include <random>

template <int dim>
metropolizer<dim>::metropolizer(su2Action<dim> iAction, int iMultiProbe,
                                double iDelta, bool cold)
    : action(iAction) {
  delta = iDelta;
  multiProbe = iMultiProbe;

  fields = new su2Element[action.getSiteCount() * dim];

  for (int i = 0; i < action.getSiteCount() * dim; i++) {
    fields[i] = su2Element();
  }
  if (!cold) {
    for (int i = 0; i < action.getSiteCount() * dim; i++) {
      fields[i].randomize(1, generator);
    }
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
        su2Element newElement = fields[loc].randomize(delta, generator);
        double change = action.evaluateDelta(fields, newElement, site, mu);
        if ((change < 0) || (uni_dist(generator) < exp(-change))) {
          fields[loc] = newElement;
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

template <int dim> double metropolizer<dim>::getHitRate() { return hitRate; }
