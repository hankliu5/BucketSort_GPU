#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

#define THREADS 512
#ifdef __cplusplus
extern "C"
{
#endif

__global__ void gpu_sort(float *input, int *output, int* step) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = __float2int_rd(input[index] / *step);
}

int cuda_sort(int number_of_elements, float *a, int step)
{
  const int NUM_BUCKETS = 6;
  float *d_in;
  int *d_out;
  int *out = (int *) malloc(sizeof(float) * number_of_elements);
  int *d_step;
  vector<float> buckets[NUM_BUCKETS];
  cudaMalloc(&d_in, sizeof(float) * number_of_elements);
  cudaMalloc(&d_out, sizeof(int) * number_of_elements);
  cudaMalloc(&d_step, sizeof(int) * 1);

  cudaMemcpy(d_in, a, sizeof(float) * number_of_elements, cudaMemcpyHostToDevice);
  cudaMemcpy(d_step, &step, sizeof(int) * 1, cudaMemcpyHostToDevice);

  gpu_sort<<<number_of_elements/THREADS, THREADS>>>(d_in, d_out, d_step);
  cudaMemcpy(out, d_out, sizeof(int) * number_of_elements, cudaMemcpyDeviceToHost);
  for (int i = 0; i < number_of_elements; i++) {
    buckets[out[i]].push_back(a[i]);
  }
  for (int i = 0; i < NUM_BUCKETS; i++) {
    thrust::device_vector<float> d_vec = buckets[i];
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), buckets[i].begin());
  } 
  int index = 0;
  for (int i = 0; i < NUM_BUCKETS; i++)
  {
      for (vector<float>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++)
      {
          a[index] = *it;
          index++;
      }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_step);
  free(out);
  return 0;
}

#ifdef __cplusplus
}
#endif
