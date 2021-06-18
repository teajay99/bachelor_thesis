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

  void loadElementCount(std::string fileName) {
    rapidcsv::Document doc(fileName, rapidcsv::LabelParams(-1, -1),
                           rapidcsv::SeparatorParams('\t'));
    N = doc.GetRowCount();
  };
  __host__ __device__ int getElementCount() { return N; };

private:
  int N;
};

#endif /*DISCRETIZER_HPP*/
