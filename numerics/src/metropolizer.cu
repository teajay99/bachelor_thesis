#include "metropolizer.hpp"

template <int dim, class su2Type>
metropolizer<dim, su2Type>::metropolizer(su2Action<dim> iAction,
                                         int iMultiProbe, double iDelta,
                                         su2Type *iFields)
    : action(iAction) {
  delta = iDelta;
  multiProbe = iMultiProbe;
  fields = iFields;
}

template <int dim, class su2Type> metropolizer<dim, su2Type>::~metropolizer() {
}

template <int dim, class su2Type>
double metropolizer<dim, su2Type>::sweep(int sweeps) {

  std::uniform_real_distribution<double> uni_dist(0., 1.);

  int hitCount = 0;

  for (int s = 0; s < sweeps; s++) {
    for (int site = 0; site < action.getSiteCount(); site++) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * site) + mu;
        for (int i = 0; i < multiProbe; i++) {
          su2Type newElement = fields[loc].randomize(delta, generator);
          double change = action.evaluateDelta(fields, newElement, site, mu);
          if ((change < 0) || (uni_dist(generator) < exp(-change))) {
            fields[loc] = newElement;
            hitCount++;
          }
        }
        fields[loc].renormalize();
      }
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

template <int dim, class su2Type>
double metropolizer<dim, su2Type>::getHitRate() {
  return hitRate;
}
