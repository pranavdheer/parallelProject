// System includes
#include <stdio.h>
#include <assert.h>
#include<iostream>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include "lib.h"

using namespace std;

#define threadsPerBlock 1024


// ptr =  cuda device pointer
void debug(int *ptr,int size, string msg){

    cout<<msg<<endl;

    int* deb = (int*)malloc(size * sizeof(int));

    cudaMemcpy(deb,ptr, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<size; i++)
      cout<<deb[i]<<" ";

    cout<<"\n";

    free(deb);

}



__global__ void nodeArray(int* dev_edges, int *dev_nodes,int size, int n){

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int start = 0;
    int end = 0;
     
    int edgeIndex = id * 2;
    
    // early stopping condition or
    // outofbound condition
    if(edgeIndex == n-1 || edgeIndex + 2 >= size)
      return;

    if(dev_edges[edgeIndex] != dev_edges[edgeIndex + 2]){

        start = dev_edges[edgeIndex];
        end   = dev_edges[edgeIndex + 2];
    }

   for(int i = start+1 ; i <= end ; i++ ){
       dev_nodes[i] = edgeIndex;
   }

}

__device__ void warp_reduce_max(int smem[1024])
{
    smem[threadIdx.x] = smem[threadIdx.x+512] > smem[threadIdx.x] ? 
                        smem[threadIdx.x+512] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+256] > smem[threadIdx.x] ? 
						smem[threadIdx.x+256] : smem[threadIdx.x]; __syncthreads();

    smem[threadIdx.x] = smem[threadIdx.x+128] > smem[threadIdx.x] ? 
						smem[threadIdx.x+128] : smem[threadIdx.x]; __syncthreads();

    smem[threadIdx.x] = smem[threadIdx.x+64] > smem[threadIdx.x] ? 
						smem[threadIdx.x+64] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ? 
						smem[threadIdx.x+8] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; __syncthreads();

	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; __syncthreads();

}
__global__ void find_max_final(int* in, int* out, int n, int remaining, int num_blocks)
{
	__shared__ float smem_max[1024];

	int idx = threadIdx.x + remaining;

	int max = -inf;
	int val;

	// tail part
	int iter = 0;
	for(int i = 1; iter + idx < n; i++)
	{
		val = in[tid + iter];
		max = val > max ? val : max;
        iter = i * threadsPerBlock;
    }

	iter = 0;
	for(int i = 0; (iter + threadIdx.x) < num_blocks; i++)
	{
		val = out[threadIdx.x + iter];
		max = val > max ? val : max;
		iter = i * threadsPerBlock;
	}

	smem_max[threadIdx.x] = max;
    __syncthreads();

	if(threadIdx.x < 512)
		warp_reduce_max(smem_max);

	if(threadIdx.x == 0)
		out[blockIdx.x] = smem_max[threadIdx.x]; 
}

__global__ void find_max(int* in, int* out, int elements_per_block)
{

	__shared__ float smem_max[1024];

	int idx = threadIdx.x + blockIdx.x * elements_per_block;

	int max = -inf;

	int val;

	int elements_per_thread = elements_per_block / threadsPerBlock; 
	
    #pragma unroll
    for(int i = 0; i < elements_per_thread; i++)
    {
        val = in[idx + i * threadsPerBlock];
        max = val > max ? val : max;

    }

	smem_max[threadIdx.x] = max;
	__syncthreads();

	if(threadIdx.x < 512)
		warp_reduce_max(smem_max);
	

	if(threadIdx.x == 0)
		out[blockIdx.x] = smem_max[threadIdx.x]; 
	

}

void calculateNumVertices(int* d_in, int* d_out, int num_elements)
{

	int elements_per_block = 4; // needs to be set (random right now) ( = m * 2 / number of blocks)
		
	int num_blocks = num_elements / elements_per_block; // redundant
	int tail = num_elements - num_blocks * elements_per_block;
	int remaining = num_elements - tail;

	find_max<<<num_blocks, threadsPerBlock>>>(d_in, d_out, elements_per_block); 
    cudaDeviceSynchronize();

	find_max_final<<<1, threadsPerBlock>>>(d_in, d_out, num_elements, remaining, num_blocks);
	
}

void parallelForward(const Edges& edges){

    int m = edges.size();
    int size = 2*m;
    int* dev_edges;
    int* dev_nodes;
    int numberOfBlocks;
    int *num_vertices;
    // TODO-: sort the edges
    
    // transfer data to GPU
    cudaMalloc(&dev_edges, size * sizeof(int));

    cudaMemcpy(dev_edges, edges.data(), m * 2 * sizeof(int),
    cudaMemcpyHostToDevice);

    cudaMalloc(&numv_array, size * sizeof(int));
    calculateNumVertices(dev_edges, numv_array, size);
    cudaMemcpy(num_vertices, numv_array, sizeof(int), cudaMemcpyDeviceToHost); // add 1 to answer(num_vertices) for code after this
    // Hardcoding the node value 
    int n = 4;
     
    // allocate space for the node array
    cudaMalloc(&dev_nodes, (n + 1) * sizeof(int));


    numberOfBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,size,n);
    cudaDeviceSynchronize();
    
    debug(dev_nodes,n+1,"print node array");

}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.iteriProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Shared memory per block:   %d bytes\n", deviceProps.sharedMemPerBlock);
    }
    printf("---------------------------------------------------------\n");
}