#include "cudaMetropolizer.hpp"
#include "discretizer.hpp"

#include <cuda_profiler_api.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

template <int dim, class su2Type>
__global__ void kernel_probeSite(su2Action<dim> act, su2Type *fields,
                                 CUDA_RAND_STATE_TYPE *randStates,
                                 int *hitCounts, int multiProbe, double delta,
                                 int odd, int mu) {

  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int site = 2 * idx;
  int offset = 0;
  for (int i = 0; i < dim; i++) {
    offset += site / act.getBasis(i);
  }

  site += ((offset + odd) % 2);

  if (site >= act.getSiteCount()) {
    return;
  }

  int loc = (dim * site) + mu;
  for (int i = 0; i < multiProbe; i++) {
    su2Type newElement = fields[loc].randomize(delta, &randStates[idx]);
    double change =
        act.template evaluateDelta<su2Type>(fields, newElement, site, mu);
    if ((change < 0) ||
        (curand_uniform_double(&randStates[idx]) < exp(-change))) {
      fields[loc] = newElement;
      hitCounts[idx]++;
    }
  }
  fields[loc].renormalize();
}

template <int dim, class su2Type>
__global__ void kernel_measurePlaquette(double *sumBuffer, int *hitBuffer,
                                        su2Type *fields, int *hitCounts,
                                        su2Action<dim> action,
                                        int sitesPerThread) {
  const int tid = threadIdx.x;

  sumBuffer[tid] = 0;
  hitBuffer[tid] = 0;

  for (int i = 0; i < sitesPerThread; i++) {
    int site = (sitesPerThread * tid) + i;
    if (site < action.getSiteCount()) {
      for (int mu = 0; mu < dim; mu++) {
        for (int nu = 0; nu < mu; nu++) {
          sumBuffer[tid] +=
              action.plaquetteProduct(fields, site, mu, nu).trace();
        }
      }
      if (site % 2 == 0) {
        hitBuffer[tid] += hitCounts[site / 2];
        hitCounts[site / 2] = 0;
      }
    }
  }

  int stepSize = 1;
  int activeThreads = CUDA_BLOCK_SIZE / 2;

  while (activeThreads > 0) {
    __syncthreads();
    if (tid < activeThreads) {
      int fst = tid * stepSize * 2;
      int snd = fst + stepSize;
      sumBuffer[fst] += sumBuffer[snd];
      hitBuffer[fst] += hitBuffer[snd];
    }

    stepSize *= 2;
    activeThreads /= 2;
  }
}

template <class su2Type>
__global__ void kernel_initIteration(CUDA_RAND_STATE_TYPE *states,
                                     int *hitCounts, int nMax) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ((2 * idx) < nMax) {
    hitCounts[idx] = 0;
    curand_init(42, idx, 0, &states[idx]);
  }
}

template <class su2Type>
__global__ void kernel_initFieldsHot(CUDA_RAND_STATE_TYPE *states,
                                     su2Type *fields, int dim, int nMax) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ((2 * idx) < nMax) {
    for (int mu = 0; mu < dim; mu++) {
      for (int i = 0; i < 2; i++) {
        int loc = (dim * ((2 * idx) + i)) + mu;
        fields[loc] = fields[loc].randomize(1.0, &states[idx]);
      }
    }
  }
}

template <int dim, class su2Type>
cudaMetropolizer<dim, su2Type>::cudaMetropolizer(su2Action<dim> iAction,
                                                 int iMultiProbe, double iDelta,
                                                 su2Type *iFields)
    : action(iAction) {
  delta = iDelta;
  multiProbe = iMultiProbe;

  blockCount =
      ((action.getSiteCount() / 2) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;


  cudaMalloc(&randStates,
             sizeof(CUDA_RAND_STATE_TYPE) * (action.getSiteCount() / 2));
  cudaMalloc(&hitCounts, sizeof(int) * (action.getSiteCount() / 2));
  fields = iFields;

  kernel_initIteration<su2Type><<<blockCount, CUDA_BLOCK_SIZE>>>(
      randStates, hitCounts, action.getSiteCount());
}

template <int dim, class su2Type>
cudaMetropolizer<dim, su2Type>::~cudaMetropolizer() {
  cudaFree(randStates);
  cudaFree(hitCounts);
}

template <int dim, class su2Type>
double cudaMetropolizer<dim, su2Type>::sweep(int sweeps) {
  for (int i = 0; i < sweeps; i++) {
    for (int odd = 0; odd < 2; odd++) {
      for (int mu = 0; mu < dim; mu++) {
        kernel_probeSite<dim, su2Type><<<blockCount, CUDA_BLOCK_SIZE>>>(
            action, fields, randStates, hitCounts, multiProbe, delta, odd, mu);
      }
    }
  }
  return this->measurePlaquette();
}

template <int dim, class su2Type>
double cudaMetropolizer<dim, su2Type>::measurePlaquette() {
  int sitesPerThread =
      (action.getSiteCount() + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
  double *sumBuffer;
  int *hitBuffer;

  cudaMalloc(&sumBuffer, sizeof(double) * CUDA_BLOCK_SIZE);
  cudaMalloc(&hitBuffer, sizeof(int) * CUDA_BLOCK_SIZE);

  kernel_measurePlaquette<dim, su2Type><<<1, CUDA_BLOCK_SIZE>>>(
      sumBuffer, hitBuffer, fields, hitCounts, action, sitesPerThread);

  double out;
  int hitCount;

  cudaMemcpy(&out, sumBuffer, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&hitCount, hitBuffer, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(sumBuffer);
  cudaFree(hitBuffer);

  hitRate = (double)hitCount / (action.getSiteCount() * dim * multiProbe);
  out /= action.getSiteCount() * dim * (dim - 1);

  return out;
}

template <int dim, class su2Type>
double cudaMetropolizer<dim, su2Type>::getHitRate() {
  return hitRate;
}
