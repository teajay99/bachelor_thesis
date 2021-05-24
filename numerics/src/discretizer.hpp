#include "rapidcsv.h"
#include "su2Element.hpp"

#ifndef DISCRETIZER_HPP
#define DISCRETIZER_HPP

class discretizer {
public:
  __host__ __device__ discretizer() { N = 0; };
  __host__ __device__ ~discretizer(){};

  void loadElements(std::string fileName, su2Element *elements) {
    rapidcsv::Document doc(fileName, rapidcsv::LabelParams(-1, -1),
                           rapidcsv::SeparatorParams('\t'));
    for (int i = 0; i < N; i++) {
      double el[4];
      for (int j = 0; j < 4; j++) {
        el[j] = doc.GetCell<double>(j, i);
      }
      elements[i] = su2Element(&el[0]);
    }
  };

  void loadDistances(su2Element *elements, double *distances) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < j; i++) {
        double sum = 0;
        for (int k = 0; k < 4; k++) {
          sum += elements[i][k] * elements[j][k];
        }
        distances[getDistIndex(i, j)] = acos(sum) / M_PI;
      }
    }
  }

  __host__ __device__ double getDistance(double *distances, int i, int j) {
    if (i == j) {
      return 0;
    } else {
      return distances[this->getDistIndex(i, j)];
    }
  };

  __host__ __device__ int getDistIndex(int i, int j) {
    if (i > j) {
      // Fancy swapping algorithm
      i = i + j;
      j = i - j;
      i = i - j;
    };
    int k = ((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1;
    return k;
  };

  void loadElementCount(std::string fileName) {
    rapidcsv::Document doc(fileName, rapidcsv::LabelParams(-1, -1),
                           rapidcsv::SeparatorParams('\t'));
    N = doc.GetRowCount();
  };
  __host__ __device__ int getElementCount() { return N; };
  __host__ __device__ int getDistanceCount() { return (N * (N - 1)) / 2; };

private:
  int N;
};

#endif /*DISCRETIZER_HPP*/
