// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>

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
#define numberOfBlocks 400
#define FILTER -2

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

void compare(int *ptr1 , int *ptr2 ,int size){

    int* deb1 = (int*)malloc(size * sizeof(int));
    int* deb2 = (int*)malloc(size * sizeof(int));

    cudaMemcpy(deb1,ptr1, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(deb2,ptr2, size * sizeof(int), cudaMemcpyDeviceToHost);


    for(int i=0 ;i<size;i++)
        if(deb1[i] != deb2[i])
          cout<<i<<" "<<deb1[i]<<" "<<deb2[i]<<endl;

    free(deb1);
    free(deb2);

}

__global__ void nodeArray(int* dev_edges, int *dev_nodes,int size, int n){


    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int x,y;

    // bug-: node id that were not present, were not getting updated to zero 
    // eg-: 0 and 1 is not present they should have nodeattay index as 0 [resolved]

    if (idx == 0){

        x = dev_edges[1];
        for(int i=0 ;i<=x;i++)
            dev_nodes[i] = 0;

    }
    
    for( int id = idx; id < size/2; id += step){
        
        int edgeIndex = (id * 2) + 1;
        
        x = dev_edges[edgeIndex];
        
        if(id == size/2 - 1)
          y = n;
        
        else  
          y = dev_edges[edgeIndex + 2];
        
        // dealing with missing nodes
        for(int i = x+1 ; i <= y ; i++ ){  
            dev_nodes[i] = id + 1; //always divisble by two
        }

    }
}    

__global__ void filter(int* dev_edges,int* dev_nodes,int size){

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    for(id = id*2; id < size ; id += step){

        int source = dev_edges[id];
        int destination   = dev_edges[id+1];

        int sourceDegree = dev_nodes[source+1] - dev_nodes[source];
        int destinationDegree = dev_nodes[destination+1] - dev_nodes[destination]; 


        if(destinationDegree < sourceDegree || (destinationDegree == sourceDegree && destination < source)){
            dev_edges[id] = FILTER;
            dev_edges[id + 1] = FILTER;
        }         
    }
}    

__global__ void trianglecounting(int* dev_edges,int* dev_nodes, int* result, int numberOfEdges){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;    
    int count = 0;
    int id = 0;
    
    for(int iter = idx; iter < numberOfEdges/2; iter = iter + step)
    {
      
        id = iter *2;
        int s = dev_edges[id];
        int e = dev_edges[id + 1];

        int s_start = dev_nodes[s];
        int s_end = dev_nodes[s + 1];
        int e_start = dev_nodes[e];
        int e_end = dev_nodes[e + 1];
        int s_next,e_next;

        // printf("id = %d,s = %d, s_start = %d \n",id,s,s_start);
        while(s_start < s_end && e_start < e_end)
        {
            s_next = dev_edges[(s_start << 1)];
            e_next = dev_edges[(e_start << 1)];
            int difference = s_next - e_next;
            // printf("I am here %d\n",difference);
            if(difference == 0)
                count++;
            if(difference <= 0)
                s_start += 1;
            if(difference >= 0)
                e_start += 1;
        }
    }
    result[idx] = count;

}    

void remove(int* dev_edges,int numberOfEdges){

    thrust::device_ptr<int> ptr((int*)dev_edges);
    thrust::remove(ptr, ptr + 2*numberOfEdges , FILTER);

}

void sort(int* dev_edges,int numberOfEdges){

    // sort the edges
    thrust::device_ptr<uint64_t> ptr((uint64_t*)dev_edges);
    thrust::sort(ptr, ptr + numberOfEdges);
}

void parallelForward(const Edges& edges){

    double startKernelTime = CycleTimer::currentSeconds();

    int numberOfEdges = edges.size();
    int* dev_edges;
    int* dev_nodes;
    int* result;
    int numberOfNodes;

    cudaEvent_t startNodeArray1, stopNodeArray1, startNodeArray2, stopNodeArray2,startFilter, 
    stopFilter, startTriCount, stopTriCount;

    // timer code
    cudaEventCreate(&startNodeArray1);
    cudaEventCreate(&stopNodeArray1);

    cudaEventCreate(&startNodeArray2);
    cudaEventCreate(&stopNodeArray2);

    cudaEventCreate(&startFilter);
    cudaEventCreate(&stopFilter);

    cudaEventCreate(&startTriCount);
    cudaEventCreate(&stopTriCount);
     

    // transfer data to GPU
    cudaMalloc(&dev_edges, 2 * numberOfEdges * sizeof(int));
    cudaMalloc(&result, numberOfBlocks * threadsPerBlock * sizeof(int));


    cudaMemcpy(dev_edges, edges.data(), numberOfEdges * 2 * sizeof(int),
    cudaMemcpyHostToDevice);

    sort(dev_edges,numberOfEdges);


    // Hardcoding the node value 
    numberOfNodes = 1696415;
    // numberOfNodes = 6;

    // allocate space for the node array
    cudaMalloc(&dev_nodes, (numberOfNodes + 1) * sizeof(int));

    cudaEventRecord(startNodeArray1);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2,numberOfNodes);
    cudaEventRecord(stopNodeArray1);

    cudaDeviceSynchronize();

     
    // compute the degree of the nodes
    cudaEventRecord(startFilter);
    filter<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2);
    cudaEventRecord(stopFilter);

    cudaDeviceSynchronize();

    //remove the filtered edges
    remove(dev_edges,numberOfEdges);


    //get the node array once again
    //note = new size of the edge array is now numberOfEdges

    cudaEventRecord(startNodeArray2);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges,numberOfNodes);
    cudaEventRecord(stopNodeArray2);

    cudaDeviceSynchronize(); 

    cudaEventRecord(startTriCount);
    trianglecounting<<<numberOfBlocks,threadsPerBlock>>>(dev_edges, dev_nodes, result, numberOfEdges);
    cudaEventRecord(stopTriCount);

    cudaDeviceSynchronize();

    //calculate the number of triangles
    thrust::device_ptr<int> ptr(result);
    int numberoftriangles =  thrust::reduce(ptr, ptr + (numberOfBlocks * threadsPerBlock));

    // debug(result,numberOfNodes,"triangle array");

    printf("number of triangles = %d\n",numberoftriangles);

    cudaFree(dev_edges);
    cudaFree(dev_nodes);
    cudaFree(result);

    float m1 = 0;
    cudaEventElapsedTime(&m1, startNodeArray1, stopNodeArray1);
    printf("CUDA Elapsed Time for Node Array 1 = %f ms\n", m1);

    float m2 = 0;
    cudaEventElapsedTime(&m2, startFilter, stopFilter);
    printf("CUDA Elapsed Time for Filter = %f ms\n", m2);

    float m3 = 0;
    cudaEventElapsedTime(&m3, startNodeArray2, stopNodeArray2);
    printf("CUDA Elapsed Time for Node Array 2 = %f ms\n", m3);

    float m4 = 0;
    cudaEventElapsedTime(&m4, startTriCount, stopTriCount);
    printf("CUDA Elapsed Time for Triangle Counting = %f ms\n", m4);

    cudaEventDestroy(startNodeArray1);
    cudaEventDestroy(stopNodeArray1);
    cudaEventDestroy(startNodeArray2);
    cudaEventDestroy(stopNodeArray2);
    cudaEventDestroy(startFilter);
    cudaEventDestroy(stopFilter);
    cudaEventDestroy(startTriCount);
    cudaEventDestroy(stopTriCount);
    
    
    double endKernelTime = CycleTimer::currentSeconds();
    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);


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
