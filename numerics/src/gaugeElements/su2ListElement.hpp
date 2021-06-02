#include "discretizer.hpp"
#include "su2Element.hpp"

#ifndef SU2LISTELEMENT_HPP
#define SU2LISTELEMENT_HPP

class su2ListElement : public su2Element {
public:
  __device__ __host__ su2ListElement(discretizer iDisc,
                                     su2Element *iElementList,
                                     double *iDistList) {
    disc = iDisc;
    distList = iDistList;
    elementList = iElementList;
    index = 0;

    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = elementList[index][i];
    }
  };

  __device__ __host__ su2ListElement(int iIndex, discretizer iDisc,
                                     su2Element *iElementList,
                                     double *iDistList) {
    disc = iDisc;
    distList = iDistList;
    elementList = iElementList;
    index = iIndex;

    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = elementList[index][i];
    }
  };

  __device__ __host__ void renormalize(){};

  su2ListElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, disc.getElementCount() - 1);

    int n = 0;

    do {
      n = dist(gen);
    } while (disc.getDistance(distList, n, index) > delta);
    return randomize(n);
  };

  __device__ su2ListElement randomize(double delta,
                                      CUDA_RAND_STATE_TYPE *state) {
    int n = 0;

    do {
      int newInt = disc.getElementCount();
      while (newInt == disc.getElementCount()) {
        double t = curand_uniform_double(state) * disc.getElementCount();
        newInt = (int)t;
      }
      n = newInt;
    } while (disc.getDistance(distList, n, index) > delta);
    return randomize(n);
  };

private:
  __device__ __host__ su2ListElement randomize(int newIdx) {
    return su2ListElement(newIdx, disc, elementList, distList);
  };

  double *distList;
  su2Element *elementList;
  discretizer disc;
  int index;
};

#endif /*SU2ELEMENT_HPP*/
