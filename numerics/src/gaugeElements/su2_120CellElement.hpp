#include "su2Element.hpp"

#ifndef SU2_120CELLElEMENT_HPP
#define SU2_120CELLElEMENT_HPP

#define C120_RHO 0.96352549156242113615
#define C120_THETA 0.15450849718747371205

#define IKO_EPS 1e-8

#define C120_ICO_TAU_HALF 0.8090169943749474241022934
#define C120_ICO_TAU_PRIME_HALF 0.3090169943749474241022934

#define C120_ROT_MATRICES                                                      \
  {                                                                            \
    {{C120_RHO, -C120_THETA, -C120_THETA, -C120_THETA},                        \
     {C120_THETA, C120_RHO, C120_THETA, -C120_THETA},                          \
     {C120_THETA, -C120_THETA, C120_RHO, C120_THETA},                          \
     {C120_THETA, C120_THETA, -C120_THETA, C120_RHO}},                         \
                                                                               \
        {{C120_RHO, C120_THETA, -C120_THETA, C120_THETA},                      \
         {-C120_THETA, C120_RHO, -C120_THETA, -C120_THETA},                    \
         {C120_THETA, C120_THETA, C120_RHO, -C120_THETA},                      \
         {-C120_THETA, C120_THETA, C120_THETA, C120_RHO}},                     \
                                                                               \
        {{C120_RHO, -C120_THETA, C120_THETA, C120_THETA},                      \
         {C120_THETA, C120_RHO, -C120_THETA, C120_THETA},                      \
         {-C120_THETA, C120_THETA, C120_RHO, C120_THETA},                      \
         {-C120_THETA, -C120_THETA, -C120_THETA, C120_RHO}},                   \
                                                                               \
    {                                                                          \
      {C120_RHO, C120_THETA, C120_THETA, -C120_THETA},                         \
          {-C120_THETA, C120_RHO, C120_THETA, C120_THETA},                     \
          {-C120_THETA, -C120_THETA, C120_RHO, -C120_THETA}, {                 \
        C120_THETA, -C120_THETA, C120_THETA, C120_RHO                          \
      }                                                                        \
    }                                                                          \
  }


class su2_120CellElement : public su2Element {
public:
  __device__ __host__ su2_120CellElement()
      : rotationMatrices C120_ROT_MATRICES {
    rotIndex = 4;
    icoPos = su2Element();
    this->applyRotation();
  };

  __device__ __host__ su2_120CellElement(su2Element iIcoPos, int iRotIndex)
      : rotationMatrices C120_ROT_MATRICES {
    icoPos = iIcoPos;
    rotIndex = iRotIndex;

    this->applyRotation();
  };

  __device__ __host__ su2_120CellElement(su2Element iIcoPos, int iRotIndex,
                                         double *el)
      : rotationMatrices C120_ROT_MATRICES {
    icoPos = iIcoPos;
    rotIndex = iRotIndex;

    for (int i = 0; i < 4; i++) {
      su2Element::element[i] = el[i];
    }
  };

  __device__ __host__ void renormalize() {
    for (int i = 0; i < 4; i++) {
      if (abs(icoPos[i]) < IKO_EPS) {
        icoPos[i] = 0;
      } else if (this->roundToIco(&icoPos[i], ICO_TAU_HALF)) {
      } else if (this->roundToIco(&icoPos[i], ICO_TAU_PRIME_HALF)) {
      } else if (this->roundToIco(&icoPos[i], 0.5)) {
      } else if (this->roundToIco(&icoPos[i], 1.0)) {
      } else {
        printf("You just left the Gauge Group [%f,%f,%f,%f]\n", icoPos[0],
               icoPos[1], icoPos[2], icoPos[3]);
      }
    }
    this->applyRotation();
  };

  su2_120CellElement randomize(double delta, std::mt19937 &gen) {
    std::uniform_int_distribution<> dist(0, 3);
    return randomize(dist(gen));
  };

  __device__ su2_120CellElement randomize(double delta,
                                          CUDA_RAND_STATE_TYPE *state) {
    int n = 4;
    while (n == 4) {
      double t = curand_uniform_double(state) * 4;
      n = (int)t;
    }
    return randomize(n);
  };

protected:
  __device__ __host__ su2_120CellElement randomize(int direction) {
    su2_120CellElement out;
    // Current Vertex is also a Vertex of the 600 Cell
    // Therefore the Vertex just needs Rotation with the rotMats
    if (rotIndex == 4) {
      out = su2_120CellElement(icoPos, direction);
    } // The current vertex is not on the 600 Cell. This covers the
    // 1/4 chance of it rotating back onto a Vertex of the 600 Cell
    else if (direction == 3) {
      out = su2_120CellElement(icoPos, 4);
    } // The current vertex is not on the 600 Cell. this covers the
    // 3 out of 4 cases, where this vertex transitions to another
    // vertex adjacent to a different vertex of the 600 Cell
    else {
      int newRotIndex = (direction + rotIndex + 1) % 4;

      for (int i = 0; i < 12; i++) {
        su2Element newIcoPos = getIcoMultElement(i) * icoPos;

        double outCoords[4];

        applyRotation(newIcoPos, newRotIndex, &outCoords[0]);

        double cosTheta =
            (outCoords[0] * (*this)[0]) + (outCoords[1] * (*this)[1]) +
            (outCoords[2] * (*this)[2]) + (outCoords[3] * (*this)[3]);
        if (abs(cosTheta - C120_RHO) < IKO_EPS) {
          out = su2_120CellElement(newIcoPos, newRotIndex, &outCoords[0]);
          break;
        }
      }
    }

    // printf("cos(theta): %f\n", (out[0] * (*this)[0]) + (out[1] * (*this)[1]) +
    //                                (out[2] * (*this)[2]) +
    //                                (out[3] * (*this)[3]));
    return out;
  };

  __host__ __device__ bool roundToIco(double *el, double roundVal) {
    if ((abs(*el) + IKO_EPS) > roundVal && (abs(*el) - IKO_EPS) < roundVal) {
      *el = ((*el > 0) - (*el < 0)) * roundVal;
      return true;
    }
    return false;
  }

  __host__ __device__ void applyRotation() {
    applyRotation(icoPos, rotIndex, &su2Element::element[0]);
  };

  __host__ __device__ void applyRotation(su2Element &inp, int &rIndex,
                                         double *out) {
    if (rIndex < 4) {

      for (int i = 0; i < 4; i++) {
        out[i] = 0;
        for (int j = 0; j < 4; j++) {
          out[i] += rotationMatrices[rIndex][i][j] * inp[j];
        }
      }
    } else {
      for (int i = 0; i < 4; i++) {
        out[i] = inp[i];
      }
    }
  };

  __host__ __device__ su2Element getIcoMultElement(int index) {
    double multEl[4] = {C120_ICO_TAU_HALF, 0, 0, 0};
    int signOne = 1 - (2 * (index & 1));
    int signTwo = 1 - (index & 2);

    int offset = index >> 2;

    multEl[1 + ((offset) % 3)] = 0;
    multEl[1 + ((offset + 1) % 3)] = signOne * 0.5;
    multEl[1 + ((offset + 2) % 3)] = signTwo * C120_ICO_TAU_PRIME_HALF;

    return su2Element(&multEl[0]);
  };

  su2Element icoPos;
  int rotIndex;
  double rotationMatrices[4][4][4];

  // rotIndex indicates by which of the four rotation
  // matrices the current vertex is rotated. 4 => no rotation
};

#endif /*SU2_120CELLElEMENT_HPP*/
