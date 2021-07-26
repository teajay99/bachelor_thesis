#include "cudaMetropolizer.hpp"
#include "metropolizer.hpp"

#include "executor.hpp"
#include "partitions.hpp"
#include <random>

template <int dim, class su2Type>
__global__ void kernel_initFieldType(su2Type *fields, int nMax, bool cold,
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

template <int dim, class su2Type>
void initFieldType(su2Type *fields, int nMax, bool cold, int iterations = 1) {
  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2Type();
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < iterations; i++) {
          fields[loc] = fields[loc].randomize(1.0, generator);
        }
      }
    }
  }
}

template <int dim>
__global__ void kernel_initVolleyFields(su2VolleyElement *fields, int nMax,
                                        bool cold, int subdivs) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  CUDA_RAND_STATE_TYPE state;
  curand_init(42, idx, 0, &state);

  if (idx < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2VolleyElement(subdivs);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < (subdivs + 2) * 200; i++) {
          fields[loc] = fields[loc].randomize(1.0, &state);
        }
      }
    }
  }
}

template <int dim>
void initVolleyFields(su2VolleyElement *fields, int nMax, bool cold,
                      int subdivs) {

  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2VolleyElement(subdivs);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        for (int i = 0; i < (subdivs + 2) * 200; i++) {
          fields[loc] = fields[loc].randomize(1.0, generator);
        }
      }
    }
  }
}

template <int dim>
__global__ void kernel_initListFields(su2ListElement *fields, int nMax,
                                      bool cold, discretizer disc,
                                      su2Element *elementList) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  CUDA_RAND_STATE_TYPE state;
  curand_init(42, idx, 0, &state);

  if (idx < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2ListElement(0, disc, elementList);
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
void initListFields(su2ListElement *fields, int nMax, bool cold,
                    discretizer disc, su2Element *elementList) {
  std::mt19937 generator;

  for (int idx = 0; idx < nMax; idx++) {
    for (int mu = 0; mu < dim; mu++) {
      int loc = (dim * idx) + mu;
      fields[loc] = su2ListElement(0, disc, elementList);
    }

    if (!cold) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * idx) + mu;
        fields[loc] = fields[loc].randomize(1.0, generator);
      }
    }
  }
}

template <int dim>
executor<dim>::executor(int iLatSize, double iBeta, int iMultiProbe,
                        double iDelta, int iPartType, bool iUseCuda,
                        std::string iPartFile, int iSubdivs)
    : action(iLatSize, iBeta) {
  multiProbe = iMultiProbe;
  delta = iDelta;
  useCuda = iUseCuda;
  partType = iPartType;
  partFile = iPartFile;
  subdivs = iSubdivs;

  int fieldsSize = dim * action.getSiteCount();
  switch (partType) {
  case SU2_ELEMENT:
    fieldsSize *= sizeof(su2Element);
    break;
  case SU2_TET_ELEMENT:
    fieldsSize *= sizeof(su2TetElement);
    break;
  case SU2_OCT_ELEMENT:
    fieldsSize *= sizeof(su2OctElement);
    break;
  case SU2_ICO_ELEMENT:
    fieldsSize *= sizeof(su2IcoElement);
    break;
  case SU2_LIST_ELEMENT:
    fieldsSize *= sizeof(su2ListElement);
    break;
  case SU2_VOLLEY_ELEMENT:
    fieldsSize *= sizeof(su2VolleyElement);
    break;
  case SU2_5_CELL_ELEMENT:
    fieldsSize *= sizeof(su2_5CellElement);
    break;
  case SU2_16_CELL_ELEMENT:
    fieldsSize *= sizeof(su2_16CellElement);
    break;
  case SU2_120_CELL_ELEMENT:
    fieldsSize *= sizeof(su2_120CellElement);
    break;
  }

  if (useCuda) {
    cudaMalloc(&fields, fieldsSize);
  } else {
    fields = malloc(fieldsSize);
  }
}
template <int dim> executor<dim>::~executor() {
  if (useCuda) {
    cudaFree(fields);
    if (SU2_LIST_ELEMENT == partType) {
      cudaFree(elementList);
    }
  } else {
    free(fields);
    if (SU2_LIST_ELEMENT == partType) {
      delete[] elementList;
    }
  }
}

template <int dim> void executor<dim>::initFields(bool cold) {

  if (useCuda) {
    int blockCount =
        ((action.getSiteCount()) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    switch (partType) {
    case SU2_ELEMENT:
      kernel_initFieldType<dim, su2Element><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2Element *)fields, action.getSiteCount(), cold);
      break;
    case SU2_TET_ELEMENT:
      kernel_initFieldType<dim, su2TetElement><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2TetElement *)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_OCT_ELEMENT:
      kernel_initFieldType<dim, su2OctElement><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2OctElement *)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_ICO_ELEMENT:
      kernel_initFieldType<dim, su2IcoElement><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2IcoElement *)fields, action.getSiteCount(), cold, 500);
      break;
    case SU2_LIST_ELEMENT:
      loadListFields(cold);
      break;
    case SU2_VOLLEY_ELEMENT:
      kernel_initVolleyFields<dim><<<blockCount, CUDA_BLOCK_SIZE>>>(
          (su2VolleyElement *)fields, action.getSiteCount(), cold, subdivs);
      break;
    case SU2_5_CELL_ELEMENT:
      kernel_initFieldType<dim, su2_5CellElement>
          <<<blockCount, CUDA_BLOCK_SIZE>>>((su2_5CellElement *)fields,
                                            action.getSiteCount(), cold, 500);
      break;
    case SU2_16_CELL_ELEMENT:
      kernel_initFieldType<dim, su2_16CellElement>
          <<<blockCount, CUDA_BLOCK_SIZE>>>((su2_16CellElement *)fields,
                                            action.getSiteCount(), cold, 500);
      break;
    case SU2_120_CELL_ELEMENT:
      kernel_initFieldType<dim, su2_120CellElement>
          <<<blockCount, CUDA_BLOCK_SIZE>>>((su2_120CellElement *)fields,
                                            action.getSiteCount(), cold, 500);
      break;
    }
  } else {
    switch (partType) {
    case SU2_ELEMENT:
      initFieldType<dim, su2Element>((su2Element *)fields,
                                     action.getSiteCount(), cold);
      break;
    case SU2_TET_ELEMENT:
      initFieldType<dim, su2TetElement>((su2TetElement *)fields,
                                        action.getSiteCount(), cold, 500);
      break;
    case SU2_OCT_ELEMENT:
      initFieldType<dim, su2OctElement>((su2OctElement *)fields,
                                        action.getSiteCount(), cold, 500);
      break;
    case SU2_ICO_ELEMENT:
      initFieldType<dim, su2IcoElement>((su2IcoElement *)fields,
                                        action.getSiteCount(), cold, 500);
      break;
    case SU2_LIST_ELEMENT:
      loadListFields(cold);
      break;
    case SU2_VOLLEY_ELEMENT:
      initVolleyFields<dim>((su2VolleyElement *)fields, action.getSiteCount(),
                            cold, subdivs);
      break;
    case SU2_5_CELL_ELEMENT:
      initFieldType<dim, su2_5CellElement>((su2_5CellElement *)fields,
                                           action.getSiteCount(), cold, 500);
      break;
    case SU2_16_CELL_ELEMENT:
      initFieldType<dim, su2_16CellElement>((su2_16CellElement *)fields,
                                            action.getSiteCount(), cold, 500);
      break;
    case SU2_120_CELL_ELEMENT:
      initFieldType<dim, su2_120CellElement>((su2_120CellElement *)fields,
                                             action.getSiteCount(), cold, 500);
      break;
    }
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
  case SU2_TET_ELEMENT:
    this->runMetropolis<su2TetElement>(measurements, multiSweep, file);
    break;
  case SU2_OCT_ELEMENT:
    this->runMetropolis<su2OctElement>(measurements, multiSweep, file);
    break;
  case SU2_ICO_ELEMENT:
    this->runMetropolis<su2IcoElement>(measurements, multiSweep, file);
    break;
  case SU2_LIST_ELEMENT:
    this->runMetropolis<su2ListElement>(measurements, multiSweep, file);
    break;
  case SU2_VOLLEY_ELEMENT:
    this->runMetropolis<su2VolleyElement>(measurements, multiSweep, file);
    break;
  case SU2_5_CELL_ELEMENT:
    this->runMetropolis<su2_5CellElement>(measurements, multiSweep, file);
    break;
  case SU2_16_CELL_ELEMENT:
    this->runMetropolis<su2_16CellElement>(measurements, multiSweep, file);
    break;
  case SU2_120_CELL_ELEMENT:
    this->runMetropolis<su2_120CellElement>(measurements, multiSweep, file);
    break;
  }

  file.close();
}

template <int dim> void executor<dim>::loadListFields(bool cold) {
  discretizer disc;
  disc.loadElementCount(partFile);
  int partCount = disc.getElementCount();
  if (partCount == 0) {
    std::cerr << "Partition file '" << partFile << "' could not be read"
              << std::endl;
    exit(1);
  }
  su2Element tmpParts[partCount];
  disc.loadElements(partFile, &tmpParts[0]);

  if (useCuda) {
    cudaMalloc(&elementList, sizeof(su2Element) * partCount);
    cudaMemcpy(elementList, &tmpParts[0], sizeof(su2Element) * partCount,
               cudaMemcpyHostToDevice);

    int blockCount =
        ((action.getSiteCount()) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernel_initListFields<dim><<<blockCount, CUDA_BLOCK_SIZE>>>(
        (su2ListElement *)fields, action.getSiteCount(), cold, disc,
        elementList);

  } else {
    elementList = new su2Element[partCount];
    for (int i = 0; i < partCount; i++) {
      elementList[i] = tmpParts[i];
    }
    initListFields<dim>((su2ListElement *)fields, action.getSiteCount(), cold,
                        disc, elementList);
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
    metropolizer<4, su2Type> metro(action, multiProbe, delta,
                                   (su2Type *)fields);
    for (int i = 0; i < measurements; i++) {
      double plaquette = metro.sweep(multiSweep);
      this->logResults(i, plaquette, metro.getHitRate(), outFile);
    }
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
