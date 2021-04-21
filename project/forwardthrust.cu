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






void parallelForward(const Edges& edges){

    int m = edges.size();
    int size = 2*m;
    int* dev_edges;
    int* dev_nodes;
    int numberOfBlocks;

    // TODO-: sort the edges
    
    // transfer data to GPU
    cudaMalloc(&dev_edges, size * sizeof(int));

    cudaMemcpy(dev_edges, edges.data(), m * 2 * sizeof(int),
    cudaMemcpyHostToDevice);

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
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Shared memory per block:   %d bytes\n", deviceProps.sharedMemPerBlock);
    }
    printf("---------------------------------------------------------\n");
}