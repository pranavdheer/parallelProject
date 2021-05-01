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
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include "lib.h"

using namespace std;

#define threadsPerBlock 1024
#define numberOfBlocks 1
#define FILTER -1

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
/*
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    
    int start = 0;
    int end = 0;
     
    for(int id = idx; id < size / 2; id += step)
    {
        int edgeIndex = id * 2;

        //use to calculate the degree of the last node
        if(edgeIndex == 0)
        dev_nodes[n] = size / 2;
        
        int x = dev_edges[edgeIndex];
        // early stopping condition or
        // outofbound condition
        if(x == n-1 || (edgeIndex + 2) >= size)
        return;

        int y = dev_edges[edgeIndex + 2];
        if(x != y){

            start = x;
            end   = y;
        }

        for(int i = start+1 ; i <= end ; i++ ){
            dev_nodes[i] = (edgeIndex + 2) / 2; //always divisble by two
        }

    }
    */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int start = 0;
    int end = 0;
     
    for(int id = idx; ((id * 2) + 2) < size ; id += step){

        int edgeIndex = (id * 2) + 1;        
        //use to calculate the degree of the last node
        if(edgeIndex == 1){
            // dev_nodes[n] = size >> 1;
            dev_nodes[0] = 0;
        }    
    
        int x = dev_edges[edgeIndex];
        // early stopping condition or
        // outofbound condition
         
        int y = dev_edges[edgeIndex + 2];
        
        if(x != y){

            start = x;
            end   = y;
        }

        else if (x == y && edgeIndex + 2 == size-1){

            start = x;
            end = n;
            edgeIndex += 2; 
            // printf("condition = %d %d\n",start,end);
        }

        // dealing with missing nodes
        for(int i = start+1 ; i <= end ; i++ ){
            dev_nodes[i] = (edgeIndex  + 2) >> 1; //always divisble by two
        }

    }

}

__global__ void filter(int* dev_edges,int* dev_nodes,int numberOfEdges){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int id;
    for(int iter = idx; iter < numberOfEdges; iter += step)
    {
        // access every second element
        id = iter * 2; 

        // outofbound access
        //if(id >= 2*numberOfEdges)
        //return;

        int source = dev_edges[id];
        int destination   = dev_edges[id + 1];

        int sourceDegree = dev_nodes[source+1] - dev_nodes[source];
        int destinationDegree = dev_nodes[destination+1] - dev_nodes[destination]; 


        if(destinationDegree < sourceDegree || (destinationDegree == sourceDegree && destination < source)){
            dev_edges[id] = FILTER;
            dev_edges[id + 1] = FILTER;
        }    
    }     

}

__global__ void trianglecounting(int* dev_edges,int* dev_nodes, uint64_t* result, int numberOfEdges){
/*
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;    
    int id;
    uint count = 0;
    for(int iter = idx; iter < numberOfEdges / 2; iter += step)
    {
        id = iter * 2;
        // TODO: need to decide how many edges thread will be responsible for
        //if(id >= numberOfEdges)
        //    return;

        int s = dev_edges[id];
        int e = dev_edges[id + 1];

        int s_start = dev_nodes[s];
        int s_end = dev_nodes[s + 1];
        int e_start = dev_nodes[e];
        int e_end = dev_nodes[e + 1];
        
        int s_next,e_next;
        while(s_start < s_end && e_start < e_end)
        {
            s_next = dev_edges[(s_start << 1) + 1];
            e_next = dev_edges[(e_start << 1) + 1];
            int difference = s_next - e_next;
            if(difference == 0)
                count++;
            if(difference <= 0)
                s_start += 1;
            if(difference >= 0)
                e_start += 1;
        }


    }
    result[idx] = count;
*/
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int count  = 0;
        int id = 0;
    for(int iter = idx; iter<numberOfEdges / 2; iter = iter+step){

        id = iter * 2;
        int s = dev_edges[id];
        int e = dev_edges[id + 1];

        int s_start = dev_nodes[s];
        int s_end = dev_nodes[s + 1];
        int e_start = dev_nodes[e];
        int e_end = dev_nodes[e + 1];

        int s_next,e_next;
        while(s_start < s_end && e_start < e_end)
        {
            s_next = dev_edges[(s_start << 1)];
            e_next = dev_edges[(e_start << 1)];
            int difference = s_next - e_next;
            if(difference == 0)
                count++;
            if(difference <= 0)
                s_start += 1;
            if(difference >= 0)
                e_start += 1;
        }

    result[idx] = count;

    }
}
void SortEdges(int m, int* edges) {
    thrust::device_ptr<uint64_t> ptr((uint64_t*)edges);
    thrust::sort(ptr, ptr + m);
  }
void remove(int* dev_edges,int numberOfEdges){

    thrust::device_ptr<int> ptr((int*)dev_edges);
    thrust::remove(ptr, ptr + 2*numberOfEdges , -1);

}
int NumVerticesGPU(int m, int* edges) {
    thrust::device_ptr<int> ptr(edges);
    return 1 + thrust::reduce(ptr, ptr + 2 * m, 0, thrust::maximum<int>());
  }
void parallelForward(const Edges& edges){

    int numberOfEdges = edges.size();
    int* dev_edges;
    int* dev_nodes;
    uint64_t *result;
    //int numberOfBlocks;
    int numberOfNodes;

    // TODO-: sort the edges
    
    // transfer data to GPU
    cudaMalloc(&dev_edges, 2 * numberOfEdges * sizeof(int));
    cudaMalloc(&result, numberOfBlocks * threadsPerBlock * sizeof(uint64_t));
    cudaMemcpy(dev_edges, edges.data(), numberOfEdges * 2 * sizeof(int),
    cudaMemcpyHostToDevice);

    // Hardcoding the node value 
    numberOfNodes = NumVerticesGPU(numberOfEdges, dev_edges);
    // numberOfNodes = 4;
    SortEdges(numberOfEdges, dev_edges);
    // allocate space for the node array
    cudaMalloc(&dev_nodes, (numberOfNodes + 1) * sizeof(int));

    // reuse the same node-array for everything to save space
    //numberOfBlocks = (numberOfEdges + threadsPerBlock - 1) / threadsPerBlock;
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2,numberOfNodes);
    cudaDeviceSynchronize();

    printf("number of edges = %d\n", numberOfEdges);
    // compute the degree of the nodes
    //numberOfBlocks = (numberOfEdges + threadsPerBlock - 1) / threadsPerBlock;
    filter<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges);
    cudaDeviceSynchronize();

    //remove the filtered edges
    remove(dev_edges,numberOfEdges);
    printf("hello\n");
    //get the node array once again
    //note = new size of the edge array is now numberOfEdges
    //numberOfBlocks = (numberOfEdges/2 + threadsPerBlock - 1) / threadsPerBlock;
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges,numberOfNodes);
    cudaDeviceSynchronize(); 
    // note = the actual index of the element in edge array is 2*nodeArray[i]

    trianglecounting<<<numberOfBlocks,threadsPerBlock>>>(dev_edges, dev_nodes, result, numberOfEdges);
    cudaDeviceSynchronize();


    //calculate the number of triangles
    thrust::device_ptr<uint64_t> ptr(result);
    uint64_t numberoftriangles =  thrust::reduce(ptr, ptr + (numberOfBlocks * threadsPerBlock));

    //debug(result,numberOfNodes,"triangle array");

    printf("number of triangles = %lld\n",numberoftriangles);
    cudaFree(result);
    cudaFree(dev_edges);
    cudaFree(dev_nodes);

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
