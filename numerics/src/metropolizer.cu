#include "metropolizer.hpp"
#include <cuda_profiler_api.h>

#define CUDA_BLOCK_SIZE 256

void checkCudaErrors() {
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    cudaProfilerStop();
    exit(1);
  }
}

template <int dim>
__global__ void kernel_probe_site(su2Action<dim> act, su2Element *fields,
                                  curandState_t *randStates, int multiProbe,
                                  double delta, int odd, int mu) {

  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int site = 2 * idx;
  int offset = 0;
  for (int i = 0; i < act.getDim(); i++) {
    offset += site / act.getBasis(i);
  }
  site += (offset + odd) % 2;

  if (site >= act.getSiteCount()) {
    return;
  }

  for (int i = 0; i < multiProbe; i++) {
    int loc = (dim * site) + mu;
    // Evaluates action "around" link Variable U_mu (site)
    double oldVal = act.evaluateDelta(&fields[0], site, mu);
    su2Element oldElement = fields[loc];
    fields[loc] = oldElement.randomize(delta, &randStates[idx]);

    // Evaluating action with new link Variable
    double newVal = act.evaluateDelta(&fields[0], site, mu);

    // Deciding wether to keep the new link Variable
    if ((newVal > oldVal) &&
        (curand_uniform_double(&randStates[idx]) > exp(-(newVal - oldVal)))) {
      fields[loc] = oldElement;
    }
    fields[loc].renormalize();
  }
}

template <int dim>
__global__ void kernel_measurePlaquette(double *sumBuffer, su2Element *fields,
                                        su2Action<dim> action,
                                        int sitesPerThread) {
  const int tid = threadIdx.x;

  sumBuffer[tid] = 0;

  for (int i = 0; i < sitesPerThread; i++) {
    int loc = (sitesPerThread * tid) + i;
    if (loc < action.getSiteCount()) {
      for (int mu = 0; mu < dim; mu++) {
        for (int nu = 0; nu < mu; nu++) {
          sumBuffer[tid] +=
              action.plaquetteProduct(fields, loc, mu, nu).trace();
        }
      }
    }
  }

  int stepSize = 1;
  int activeThreads = CUDA_BLOCK_SIZE / 2;
  __syncthreads();

  while (activeThreads > 0) {
    __syncthreads();
    if (tid < activeThreads) // still alive?
    {
      int fst = tid * stepSize * 2;
      int snd = fst + stepSize;
      sumBuffer[fst] += sumBuffer[snd];
    }

    stepSize *= 2;
    activeThreads /= 2;
    __syncthreads();
  }
}

__global__ void kernel_initIteration(curandState *state, su2Element *fields,
                                     int nMax, int dim) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ((2 * idx) < nMax) {
    curand_init(42, idx, 0, &state[idx]);
    for (int i = 0; i < 2; i++) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * ((2 * idx) + i)) + mu;
        fields[loc] = su2Element();
      }
    }
  }
}

__global__ void kernel_initFieldsHot(curandState *state, su2Element *fields,
                                     int dim, int nMax) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ((2 * idx) < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      for (int i = 0; i < 2; i++) {
        int loc = (dim * ((2 * idx) + i)) + mu;
        fields[loc] = fields[loc].randomize(1.0, &state[idx]);
      }
    }
  }
}

template <int dim>
metropolizer<dim>::metropolizer(su2Action<dim> iAction, int iMultiProbe,
                                double iDelta, bool cold)
    : action(iAction) {
  delta = iDelta;
  blockCount =
      ((action.getSiteCount() / 2) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
  multiProbe = iMultiProbe;

  cudaMalloc(&randStates, sizeof(curandState) * (action.getSiteCount() / 2));
  cudaMalloc(&fields, sizeof(su2Element) * action.getSiteCount() * dim);

  kernel_initIteration<<<blockCount, CUDA_BLOCK_SIZE>>>(
      randStates, fields, action.getSiteCount(), dim);

  kernel_initFieldsHot<<<blockCount, CUDA_BLOCK_SIZE>>>(randStates, fields, dim,
                                                        action.getSiteCount());
}

template <int dim> metropolizer<dim>::~metropolizer() {
  cudaFree(randStates);
  cudaFree(fields);
}

template <int dim> double metropolizer<dim>::sweep(int repeats) {
  for (int r = 0; r < repeats; r++) {
    for (int odd = 0; odd < 2; odd++) {
      for (int mu = 0; mu < dim; mu++) {
        kernel_probe_site<<<blockCount, CUDA_BLOCK_SIZE>>>(
            action, fields, randStates, multiProbe, delta, odd, mu);
      }
    }
  }

  checkCudaErrors();

  int sitesPerThread =
      (action.getSiteCount() + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
  double *sumBuffer;
  cudaMallocManaged(&sumBuffer, sizeof(double) * CUDA_BLOCK_SIZE);
  kernel_measurePlaquette<dim>
      <<<1, CUDA_BLOCK_SIZE>>>(sumBuffer, fields, action, sitesPerThread);

  double out; // = sumBuffer[0];
  cudaMemcpy(&out, sumBuffer, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(sumBuffer);
  out /= action.getSiteCount() * dim * (dim - 1);
  return out;
};
