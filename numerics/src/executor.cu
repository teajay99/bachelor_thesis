#include "executor.hpp"
#include "partitions.hpp"

template <int dim, class su2Type>
__global__ void kernel_initFields(su2Type *fields, int nMax, bool cold,
                                  int iterations = 1) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  CUDA_RAND_STATE_TYPE state;
  curand_init(42, idx, 0, &state);

  if (idx < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2Type();
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < iterations; i++) {
          fields[loc] = fields[loc].randomize(1.0, &state);
        }
      }
    }
  }
}

template <int dim>
__global__ void kernel_initListFields(su2ListElement *fields, int nMax,
                                      bool cold, discretizer disc,
                                      double *distList,
                                      su2Element *elementList) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  CUDA_RAND_STATE_TYPE state;
  curand_init(42, idx, 0, &state);

  if (idx < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2ListElement(0, disc, elementList, distList);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        fields[loc] = fields[loc].randomize(1.0, &state);
      }
    }
  }
}

template <int dim>
executor<dim>::executor(int iLatSize, double iBeta, int iMultiProbe,
                        double iDelta, int iPartType, bool iUseCuda,
                        std::string iPartFile)
    : action(iLatSize, iBeta) {
  multiProbe = iMultiProbe;
  delta = iDelta;
  useCuda = iUseCuda;
  partType = iPartType;
  partFile = iPartFile;

  int fieldsSize = dim * action.getSiteCount();
  switch (partType) {
  case SU2_ELEMENT:
    fieldsSize *= sizeof(su2Element);
    break;
  case SU2_IKO_ELEMENT:
    fieldsSize *= sizeof(su2IkoElement);
    break;
  case SU2_LIST_ELEMENT:
    fieldsSize *= sizeof(su2ListElement);
    break;
  }

  if (useCuda) {
    cudaMalloc(&fields, fieldsSize);
    printf("Stuff Malloced\n");
  } else {
    // WIP
  }
}
template <int dim> executor<dim>::~executor() {
  if (useCuda) {
    cudaFree(fields);
    cudaFree(elementList);
    cudaFree(distList);
  } else {
    // WIP
  }
}

template <int dim> void executor<dim>::initFields(bool cold) {

  if (useCuda) {
    int blockCount =
        ((action.getSiteCount()) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    switch (partType) {
    case SU2_ELEMENT:
      kernel_initFields<dim, su2Element><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2Element *)fields, action.getSiteCount(), cold);
      break;
    case SU2_IKO_ELEMENT:
      kernel_initFields<dim, su2IkoElement><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2IkoElement *)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_LIST_ELEMENT:
      initListFields(cold);
      break;
    }
  } else {
    // WIP
  }
}
template <int dim>
void executor<dim>::run(int measurements, int multiSweep, std::string outFile) {
  std::ofstream file;
  file.open(outFile);

  switch (partType) {
  case SU2_ELEMENT:
    this->runMetropolis<su2Element>(measurements, multiSweep, file);
    break;
  case SU2_IKO_ELEMENT:
    this->runMetropolis<su2IkoElement>(measurements, multiSweep, file);
    break;

  case SU2_LIST_ELEMENT:
    this->runMetropolis<su2ListElement>(measurements, multiSweep, file);
    break;
  }

  file.close();
}

template <int dim> void executor<dim>::initListFields(bool cold) {
  discretizer disc;
  disc.loadElementCount(partFile);
  int partCount = disc.getElementCount();
  if (partCount == 0) {
    std::cerr << "Partition file '" << partFile << "' could not be read"
              << std::endl;
    exit(1);
  }
  su2Element tmpParts[partCount];
  double tmpDists[disc.getDistanceCount()];
  disc.loadElements(partFile, &tmpParts[0]);
  disc.loadDistances(&tmpParts[0], tmpDists);

  if (useCuda) {
    cudaMalloc(&elementList, sizeof(su2Element) * partCount);
    cudaMemcpy(elementList, &tmpParts[0], sizeof(su2Element) * partCount,
               cudaMemcpyHostToDevice);

    cudaMalloc(&distList, sizeof(double) * disc.getDistanceCount());
    cudaMemcpy(distList, &tmpDists[0], sizeof(double) * disc.getDistanceCount(),
               cudaMemcpyHostToDevice);

    int blockCount =
        ((action.getSiteCount()) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernel_initListFields<dim><<<blockCount, CUDA_BLOCK_SIZE>>>(
        (su2ListElement *)fields, action.getSiteCount(), cold, disc, distList,
        elementList);

  } else { // WIP
  }
}

template <int dim>
template <class su2Type>
void executor<dim>::runMetropolis(int measurements, int multiSweep,
                                  std::ofstream &outFile) {
  if (useCuda) {
    cudaMetropolizer<4, su2Type> metro(action, multiProbe, delta,
                                       (su2Type *)fields);
    for (int i = 0; i < measurements; i++) {
      double plaquette = metro.sweep(multiSweep);
      this->logResults(i, plaquette, metro.getHitRate(), outFile);
    }
  } else {
    // WIP
  }
}

template <int dim>
void executor<dim>::logResults(int i, double plaquette, double hitRate,
                               std::ofstream &file) {
  std::cout << i << " " << std::scientific << std::setw(18)
            << std::setprecision(15) << plaquette << " " << hitRate
            << std::endl;
  file << i << "\t" << std::scientific << std::setw(18) << std::setprecision(15)
       << plaquette << "\t" << hitRate << std::endl;
}
