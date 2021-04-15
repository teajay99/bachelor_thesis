#include "cudaMetropolizer.hpp"
#include <cuda_profiler_api.h>

#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define CUDA_CALL(x)                                                           \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
    }                                                                          \
  } while (0)

#define CURAND_CALL(x)                                                         \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
    }                                                                          \
  } while (0)

void checkCudaErrors(int i) {
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error " << i << ": " << cudaGetErrorString(err)
              << std::endl; // add
    cudaProfilerStop();
    exit(1);
  }
}

template <int dim>
__global__ void kernel_probe_site(su2Action<dim> act, su2Element *fields,
                                  CUDA_RAND_STATE_TYPE *randStates,
                                  int *hitCounts, int multiProbe, double delta,
                                  int odd, int mu) {

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

  int loc = (dim * site) + mu;
  for (int i = 0; i < multiProbe; i++) {

    // Evaluates action "around" link Variable U_mu (site)
    double oldVal = act.evaluateDelta(fields, site, mu);
    su2Element oldElement = fields[loc];
    fields[loc] = oldElement.randomize(delta, &randStates[idx]);

    // Evaluating action with new link Variable
    double newVal = act.evaluateDelta(fields, site, mu);

    // Deciding wether to keep the new link Variable
    if ((newVal > oldVal) &&
        (curand_uniform_double(&randStates[idx]) > exp(-(newVal - oldVal)))) {
      fields[loc] = oldElement;
    } else {
      hitCounts[idx]++;
    }
  }
  fields[loc].renormalize();
}

template <int dim>
__global__ void kernel_measurePlaquette(double *sumBuffer, int *hitBuffer,
                                        su2Element *fields, int *hitCounts,
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

__global__ void kernel_initIteration(CUDA_RAND_STATE_TYPE *states,
                                     su2Element *fields, int *hitCounts,
                                     int nMax, int dim) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ((2 * idx) < nMax) {
    hitCounts[idx] = 0;
    curand_init(42, idx, 0, &states[idx]);
    for (int i = 0; i < 2; i++) {
      for (int mu = 0; mu < dim; mu++) {
        int loc = (dim * ((2 * idx) + i)) + mu;
        fields[loc] = su2Element();
      }
    }
  }
}

__global__ void kernel_initFieldsHot(CUDA_RAND_STATE_TYPE *states,
                                     su2Element *fields, int dim, int nMax) {
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

template <int dim>
cudaMetropolizer<dim>::cudaMetropolizer(su2Action<dim> iAction, int iMultiProbe,
                                        double iDelta, bool cold)
    : action(iAction) {
  delta = iDelta;
  blockCount =
      ((action.getSiteCount() / 2) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
  multiProbe = iMultiProbe;

  randStateCount = (blockCount + 200 - 1) / 200;

  cudaMalloc(&randStates,
             sizeof(CUDA_RAND_STATE_TYPE) * (action.getSiteCount() / 2));
  cudaMalloc(&hitCounts, sizeof(int) * (action.getSiteCount() / 2));
  cudaMalloc(&fields, sizeof(su2Element) * action.getSiteCount() * dim);

  kernel_initIteration<<<blockCount, CUDA_BLOCK_SIZE>>>(
      randStates, fields, hitCounts, action.getSiteCount(), dim);

  kernel_initFieldsHot<<<blockCount, CUDA_BLOCK_SIZE>>>(randStates, fields, dim,
                                                        action.getSiteCount());
  checkCudaErrors(2);
}

template <int dim> cudaMetropolizer<dim>::~cudaMetropolizer() {
  cudaFree(randStates);
  cudaFree(fields);
  cudaFree(hitCounts);
}

template <int dim> double cudaMetropolizer<dim>::sweep() {
  for (int odd = 0; odd < 2; odd++) {
    for (int mu = 0; mu < dim; mu++) {
      checkCudaErrors(3);
      kernel_probe_site<<<blockCount, CUDA_BLOCK_SIZE>>>(
          action, fields, randStates, hitCounts, multiProbe, delta, odd, mu);
      checkCudaErrors(1);
    }
  }
  int sitesPerThread =
      (action.getSiteCount() + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
  double *sumBuffer;
  int *hitBuffer;

  cudaMallocManaged(&sumBuffer, sizeof(double) * CUDA_BLOCK_SIZE);
  cudaMallocManaged(&hitBuffer, sizeof(int) * CUDA_BLOCK_SIZE);

  kernel_measurePlaquette<dim><<<1, CUDA_BLOCK_SIZE>>>(
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

template <int dim> double cudaMetropolizer<dim>::getHitRate() {
  return hitRate;
}
