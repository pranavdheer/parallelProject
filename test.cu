#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include "lib.h"


template<bool ZIPPED>
__global__ void CalculateNodePointers(int n, int m, int* edges, int* nodes) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i <= m; i += step) {
    int prev = i > 0 ? edges[ZIPPED ? (2 * (i - 1) + 1) : (m + i - 1)] : -1;
    int next = i < m ? edges[ZIPPED ? (2 * i + 1) : (m + i)] : n;
    for (int j = prev + 1; j <= next; ++j)
      nodes[j] = i;
  }
}
uint64_t SumResults(int size, uint64_t* results) {
  thrust::device_ptr<uint64_t> ptr(results);
  return thrust::reduce(ptr, ptr + size);
}

__global__ void CalculateFlags(int m, int* edges, int* nodes, bool* flags) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < m; i += step) {
    int a = edges[2 * i];
    int b = edges[2 * i + 1];
    int deg_a = nodes[a + 1] - nodes[a];
    int deg_b = nodes[b + 1] - nodes[b];
    flags[i] = (deg_a < deg_b) || (deg_a == deg_b && a < b);
  }
}

void RemoveMarkedEdges(int m, int* edges, bool* flags) {
  thrust::device_ptr<uint64_t> ptr((uint64_t*)edges);
  thrust::device_ptr<bool> ptr_flags(flags);
  thrust::remove_if(ptr, ptr + m, ptr_flags, thrust::identity<bool>());
}

__global__ void CalculateTriangles(
  int m, const int* __restrict__ edges, const int* __restrict__ nodes,
  uint64_t* results, int deviceCount = 1, int deviceIdx = 0) {
int from =
  gridDim.x * blockDim.x * deviceIdx +
  blockDim.x * blockIdx.x +
  threadIdx.x;
int step = deviceCount * gridDim.x * blockDim.x;
uint64_t count = 0;

for (int i = from; i < m; i += step) {
  int u = edges[i], v = edges[m + i];

  int u_it = nodes[u], u_end = nodes[u + 1];
  int v_it = nodes[v], v_end = nodes[v + 1];

  int a = edges[u_it], b = edges[v_it];

  while (u_it < u_end && v_it < v_end) {
    
    int d = a - b;
    if (d <= 0)
      a = edges[++u_it];
    if (d >= 0)
      b = edges[++v_it];
    if (d == 0)
      ++count;
  }
}

results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}


void SortEdges(int m, int* edges) {
    thrust::device_ptr<uint64_t> ptr((uint64_t*)edges);
    thrust::sort(ptr, ptr + m);
  }

int NumVerticesGPU(int m, int* edges) {
    thrust::device_ptr<int> ptr(edges);
    return 1 + thrust::reduce(ptr, ptr + 2 * m, 0, thrust::maximum<int>());
  }

  __global__ void UnzipEdges(int m, int* edges, int* unzipped_edges) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = from; i < m; i += step) {
      unzipped_edges[i] = edges[2 * i];
      unzipped_edges[m + i] = edges[2 * i + 1];
    }
  }

 

void MultiGpuForward(const Edges& edges){

    int m = edges.size(), n;


    int* dev_edges;
    int* dev_nodes;

    int *h_data = (int *)malloc(m * 2 * sizeof(int));

    cudaMalloc(&dev_edges, m * 2 * sizeof(int));

    cudaMemcpyAsync(
        dev_edges, edges.data(), m * 2 * sizeof(int),
        cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    n = NumVerticesGPU(m, dev_edges);
    printf("number of nodes=%d\n",n);

    cudaMemcpy(h_data, dev_edges, m * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("raw input=\n");
    for(int i=0;i<2*m;i++)
       printf("%d ",h_data[i]);

    printf("\n");    

    SortEdges(m, dev_edges);

    cudaDeviceSynchronize();

    printf("sort input=\n");
    cudaMemcpy(h_data, dev_edges, m * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<2*m;i++)
      printf("%d ",h_data[i]);
    printf("\n");  

    int *h_nodes = (int *)malloc((n+1) * sizeof(int));
    cudaMalloc(&dev_nodes, (n + 1) * sizeof(int));


    CalculateNodePointers<true><<<5, 80>>>(
        n, m, dev_edges, dev_nodes);
    
    cudaDeviceSynchronize();

    
    cudaMemcpy(h_data, dev_nodes,(n+1)* sizeof(int), cudaMemcpyDeviceToHost);  
    printf("node array= (%d)\n",n);

    for(int i=0;i<n;i++)
      printf("%d ",h_data[i]);

    printf("\n");
    bool* dev_flags;
    cudaMalloc(&dev_flags, m * sizeof(bool));
    CalculateFlags<<<5,80>>>(
      m, dev_edges, dev_nodes, dev_flags);
    RemoveMarkedEdges(m, dev_edges, dev_flags);

    cudaMemcpy(h_data, dev_edges, m * 2 * sizeof(int), cudaMemcpyDeviceToHost);  
    printf("removed edges= \n");
    for(int i=0;i<2*m;i++)
      printf("%d ",h_data[i]);
    
    printf("\n");  

    cudaFree(dev_flags);
    cudaDeviceSynchronize();
    m /= 2;

    printf("new value of m= %d \n",m);
    UnzipEdges<<<5, 80>>>(m, dev_edges, dev_edges + 2 * m);

    cudaDeviceSynchronize();

    printf("unzipping edges= \n");
    cudaMemcpy(h_data, dev_edges,2*2*m* sizeof(int), cudaMemcpyDeviceToHost);  
    for(int i=0;i<2*2*m;i++)
    printf("%d ",h_data[i]);
    printf("\n");

    cudaMemcpyAsync(dev_edges, dev_edges + 2 * m, 2 * m * sizeof(int),cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize(); 

    printf("mem cpy async \n");
    cudaMemcpy(h_data, dev_edges,2*2*m* sizeof(int), cudaMemcpyDeviceToHost);  
    for(int i=0;i<2*2*m;i++)
    printf("%d ",h_data[i]);
    printf("\n");

    CalculateNodePointers<false><<<5, 80>>>(
      n, m, dev_edges, dev_nodes);

      printf("node pointer for unzipped array \n");
      cudaMemcpy(h_data, dev_nodes,(n+1)* sizeof(int), cudaMemcpyDeviceToHost);  
      for(int i=0;i<=n;i++)
      printf("%d ",h_data[i]);  

      cudaDeviceSynchronize();

      uint64_t* dev_results;
      cudaMalloc(&dev_results,
        5*80 * sizeof(uint64_t));
        
        CalculateTriangles<<<5,80>>>(
          m, dev_edges, dev_nodes, dev_results);

          cudaDeviceSynchronize(); 

          uint64_t result = 0;
          result = SumResults(5*80, dev_results);
         cudaFree(dev_results);     

         printf("\n\n%d\n",result);
}