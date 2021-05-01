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
#define numberOfBlocks 46
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


    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int start = 0;
    int end = 0;
     
    for(int edgeIndex = idx*2 + 1; (edgeIndex + 2) < size ; edgeIndex += step){

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

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;


    for( id = id * 2; id<numberOfEdges ; id = id+step){
    
        int count = 0;
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

    result[id >> 1] = count;

    }

}    

void remove(int* dev_edges,int numberOfEdges){

    thrust::device_ptr<int> ptr((int*)dev_edges);
    thrust::remove(ptr, ptr + 2*numberOfEdges , -1);

}

void sort(int* dev_edges,int numberOfEdges){

    // sort the edges
    thrust::device_ptr<uint64_t> ptr((uint64_t*)dev_edges);
    thrust::sort(ptr, ptr + numberOfEdges);
}

void parallelForward(const Edges& edges){

    int numberOfEdges = edges.size();
    int* dev_edges;
    int* dev_nodes;
    int* result;
    int numberOfNodes;

    
    // transfer data to GPU
    cudaMalloc(&dev_edges, 2 * numberOfEdges * sizeof(int));
    cudaMalloc(&result, numberOfEdges * sizeof(int));
    
    cudaMemcpy(dev_edges, edges.data(), numberOfEdges * 2 * sizeof(int),
    cudaMemcpyHostToDevice);


    // debug(dev_edges,2*numberOfEdges,"orig data");
    sort(dev_edges,numberOfEdges);
    debug(dev_edges,2*numberOfEdges,"sort data");

    // Hardcoding the node value 
    // numberOfNodes = 1696415;
    numberOfNodes = 4;
     
    // allocate space for the node array
    cudaMalloc(&dev_nodes, (numberOfNodes + 1) * sizeof(int));

    // reuse the same node-array for everything to save space
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2,numberOfNodes);
    cudaDeviceSynchronize();

    debug(dev_nodes,numberOfNodes+1,"node array");
     
    // compute the degree of the nodes
    filter<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2);
    cudaDeviceSynchronize();

    
    //remove the filtered edges
    remove(dev_edges,numberOfEdges);

    debug(dev_edges,numberOfEdges,"filtred array");

    //get the node array once again
    //note = new size of the edge array is now numberOfEdges
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges,numberOfNodes);
    cudaDeviceSynchronize(); 

    debug(dev_nodes,numberOfNodes+1,"node array");
    
    // note = the actual index of the element in edge array is 2*nodeArray[i]
    cout<<"start triangle counting"<<endl;
    trianglecounting<<<numberOfBlocks,threadsPerBlock>>>(dev_edges, dev_nodes, result, numberOfEdges);
    cudaDeviceSynchronize();

    //calculate the number of triangles
    thrust::device_ptr<int> ptr(result);
    int numberoftriangles =  thrust::reduce(ptr, ptr + numberOfNodes);

    // debug(result,numberOfNodes,"triangle array");

    printf("number of triangles = %d\n",numberoftriangles);

    cudaFree(dev_edges);
    cudaFree(dev_nodes);
    cudaFree(result);


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
