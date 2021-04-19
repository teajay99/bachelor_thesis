#include "rapidcsv.h"
#include "su2Element.hpp"
#include <fstream>

#ifndef DISCRETIZER_HPP
#define DISCRETIZER_HPP

class discretizer {
public:
  discretizer(std::string fileName) { N = 0; };
  ~discretizer(){};

  int loadElements(std::string fileName, su2Element *elements,
                   double *distances) {

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < j; i++) {
        double sum = 0;
        for (int k = 0; k < 4; k++) {
          sum += elements[i][k] * elements[j][k];
        }
        distances[getDistIndex(i, j)] = acos(sum);
      }
    }

    return 0;
  };

  double getDistance(double *distances, int i, int j) {

    return distances[this->getDistIndex(i, j)];
  };

  int getDistIndex(int i, int j) {
    if (i == j) {
      return 0;
    } else if (i > j) {
      // Fancy swapping algorithm
      i = i + j;
      j = i - j;
      i = i - j;
    };
    int k = ((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1;
    return k;
  };

  int loadElementCount(std::string fileName) {
    rapidcsv::Document doc("", rapidcsv::SeparatorParams("\t"));
  };
  int getElementCount() { return N; };
  int getDistanceCount() { return (N * (N - 1)) / 2; };

private:
  int N;
};

#endif /*DISCRETIZER_HPP*/
