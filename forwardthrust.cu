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
#define inf 0x7f800000 

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

__global__ void nodeArray(int* dev_edges, int *dev_nodes,int size, int n, int flag ){


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

__global__ void filter(int* dev_edges,int* dev_nodes,int numberOfEdges){

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;

    for(id = id; id < numberOfEdges ; id += step){

        int2 sd_pair = ((int2*)dev_edges)[id];

        int sourceDegree = dev_nodes[(sd_pair.x)+1] - dev_nodes[sd_pair.x];
        int destinationDegree = dev_nodes[(sd_pair.y) + 1] - dev_nodes[sd_pair.y]; 

        if(destinationDegree < sourceDegree || (destinationDegree == sourceDegree && sd_pair.y < sd_pair.x)){
            ((int2*)dev_edges)[id] =  make_int2(FILTER, FILTER);

        }         
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

__global__ void find_max(int* in, int* out, int elements_per_block)
{

	__shared__ int smem_max[1024];


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

__global__ void find_max_final(int* in, int* out, int n, int remaining, int num_blocks)
{
	__shared__ int smem_max[1024];

	int idx = threadIdx.x + remaining;

	int max = -inf;
	int val;

	// tail part
	int iter = 0;
	for(int i = 1; iter + idx < n; i++)
	{
		val = in[idx + iter];
		max = val > max ? val : max;
        iter = i * threadsPerBlock;
    }

	iter = 0;
	for(int i = 1; (iter + threadIdx.x) < num_blocks; i++)
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


void calculateNumVertices(int* d_in, int* d_out, int num_elements)
{

	//int elements_per_block = ; // needs to be set (random right now) ( = m * 2 / number of blocks)
		
	int num_blocks = numberOfBlocks;//46;//num_elements / elements_per_block; // redundant
    int elements_per_block = num_elements / num_blocks;
	int tail = num_elements - num_blocks * elements_per_block;
	int remaining = num_elements - tail;

	find_max<<<num_blocks, threadsPerBlock>>>(d_in, d_out, elements_per_block); 
    cudaDeviceSynchronize();

	find_max_final<<<1, threadsPerBlock>>>(d_in, d_out, num_elements, remaining, num_blocks);

	
}


__global__ void trianglecounting(const int* __restrict__ dev_edges,const int* __restrict__ dev_nodes, int* result, int numberOfEdges){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int count  = 0;
 
    __shared__ int shared[12288];

    for (int i=threadIdx.x;i<12288;i=i+1024) {
        shared[i] =  dev_nodes[i];
    }
    __syncthreads();
    
    for(int iter = idx; iter<numberOfEdges / 2; iter = iter+step){


        int2 se_pair = ((int2*)dev_edges)[iter];
        int s_start,s_end,e_start,e_end;

        if(se_pair.x +1 < 12288){
           s_start = shared[se_pair.x];
           s_end = shared[se_pair.x + 1];
        }   
        else {  
            s_start = dev_nodes[se_pair.x];
            s_end = dev_nodes[se_pair.x + 1];
        }    

        if(se_pair.y + 1 < 12288){
            e_start = shared[se_pair.y];
            e_end = shared[se_pair.y + 1];
        } 
        else {

            e_start = dev_nodes[se_pair.y];
            e_end = dev_nodes[se_pair.y + 1];

        }
        
        // int s_start = dev_nodes[se_pair.x];
        // int s_end = dev_nodes[se_pair.x + 1];
        // int e_start = dev_nodes[se_pair.y];
        // int e_end = dev_nodes[se_pair.y + 1];
        
        int2 s_next,e_next;
        s_next = ((int2*)dev_edges)[s_start];
        e_next = ((int2*)dev_edges)[e_start];

        while(s_start < s_end && e_start < e_end)
        {
            // need to run and check for speed, vector accesses might have increased execution time
            int a = s_next.x;
            int b = e_next.x;

            if(a < b) {
                s_start+=1;
                s_next = ((int2*)dev_edges)[s_start];
            }
            else if(a > b) {
                e_start+=1;
                e_next = ((int2*)dev_edges)[e_start];
            }
            else {
                count++;
                s_start+=1;
                s_next = ((int2*)dev_edges)[s_start];
                e_start+=1;
                e_next = ((int2*)dev_edges)[e_start];
            }   
        }
    }
    result[idx] = count;
}

void remove(int* dev_edges,int numberOfEdges) {

    thrust::device_ptr<int> ptr((int*)dev_edges);
    thrust::remove(ptr, ptr + 2*numberOfEdges , FILTER);

}

// __global__ void remove_filter(int* dev_edges ,int* d_out  ,int numberOfEdges) {

//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int step = gridDim.x * blockDim.x;

//     for(int iter = 2*idx; iter<numberOfEdges; iter = iter+step) {

//         if ( dev_edges[idx] == FILTER ) {

//             d_out[idx] = 0;
//             d_out[idx + 1] = 0;
//         }

//         else {
//             d_out[idx] = 1;
//             d_out[idx + 1] = 1;
//         }
//     }
// } 

// void remove_filter_wrapper () {

//     remove_filter(dev_edges,d_out,numberOfEdges);
//     cudaDeviceSynchronize();

// }

void sort(int* dev_edges,int numberOfEdges){

    // sort the edges
    thrust::device_ptr<uint64_t> ptr((uint64_t*)dev_edges);
    thrust::sort(ptr, ptr + numberOfEdges);
}

int NumVerticesGPU(int m, int* edges) {
    thrust::device_ptr<int> ptr(edges);
    return 1 + thrust::reduce(ptr, ptr + 2 * m, 0, thrust::maximum<int>());
}

__device__ void warp_reduce_sum(int smem[1024])
{
    smem[threadIdx.x] = smem[threadIdx.x+512] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+256] + smem[threadIdx.x]; __syncthreads();
    smem[threadIdx.x] = smem[threadIdx.x+128] + smem[threadIdx.x]; __syncthreads();
    smem[threadIdx.x] = smem[threadIdx.x+64] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+32] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+16] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+8] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+4] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+2] + smem[threadIdx.x]; __syncthreads();
	smem[threadIdx.x] = smem[threadIdx.x+1] + smem[threadIdx.x]; __syncthreads();
}

__global__ void find_sum_final(int* in, int* out, int n, int remaining, int num_blocks)
{
	__shared__ int smem_sum[1024];

	int idx = threadIdx.x + remaining;

	int sum = 0;

	// tail part
	int iter = 0;
	for(int i = 1; iter + idx < n; i++)
	{
		sum += in[idx + iter];
        iter = i * threadsPerBlock;
    }
	iter = 0;
	for(int i = 1; (iter + threadIdx.x) < num_blocks; i++)
	{
		sum += out[threadIdx.x + iter];
		iter = i * threadsPerBlock;
	}
	smem_sum[threadIdx.x] = sum;
    __syncthreads();

	if(threadIdx.x < 512)
		warp_reduce_sum(smem_sum);

	if(threadIdx.x == 0)
		out[blockIdx.x] = smem_sum[threadIdx.x]; 
}

__global__ void find_sum(int* in, int* out, int elements_per_block)
{

	__shared__ int smem_sum[1024];

	int idx = threadIdx.x + blockIdx.x * elements_per_block;
	int sum = 0;
	int elements_per_thread = elements_per_block / threadsPerBlock; 
	
    #pragma unroll
    for(int i = 0; i < elements_per_thread; i++)
        sum += in[idx + i * threadsPerBlock];

	smem_sum[threadIdx.x] = sum;
	__syncthreads();

	if(threadIdx.x < 512)
		warp_reduce_sum(smem_sum);

	if(threadIdx.x == 0) 
		out[blockIdx.x] = smem_sum[threadIdx.x]; 
}

void calculateSum(int* d_in, int* d_out, int num_elements)
{
		
	int num_blocks = numberOfBlocks;
    int elements_per_block = num_elements/num_blocks;
	int tail = num_elements - num_blocks * elements_per_block;
	int remaining = num_elements - tail;

	find_sum<<<num_blocks, threadsPerBlock>>>(d_in, d_out, elements_per_block); 
    cudaDeviceSynchronize();

	find_sum_final<<<1, threadsPerBlock>>>(d_in, d_out, num_elements, remaining, num_blocks);
}

void parallelForward(const Edges& edges){


    int numberOfEdges = edges.size();
    int* dev_edges;
    int* dev_nodes;
    int* result;
    int* d_out;
    int numberOfNodes;
    int* out = (int*)malloc(sizeof(int));

    cudaEvent_t startNodeArray1, stopNodeArray1, startNodeArray2, stopNodeArray2,startFilter, 
    stopFilter, startTriCount, stopTriCount,startNumvertices,stopNumvertices, startSumTri, stopSumTri, startRemove,stopRemove,startSort,stopSort;

    // timer code
    cudaEventCreate(&startNodeArray1);
    cudaEventCreate(&stopNodeArray1);

    cudaEventCreate(&startNodeArray2);
    cudaEventCreate(&stopNodeArray2);

    cudaEventCreate(&startFilter);
    cudaEventCreate(&stopFilter);

    cudaEventCreate(&startTriCount);
    cudaEventCreate(&stopTriCount);

    cudaEventCreate(&startNumvertices);
    cudaEventCreate(&stopNumvertices);

    cudaEventCreate(&startSumTri);
    cudaEventCreate(&stopSumTri);

    cudaEventCreate(&startRemove);
    cudaEventCreate(&stopRemove);

    cudaEventCreate(&startSort);
    cudaEventCreate(&stopSort);

    // transfer data to GPU
    cudaMalloc(&dev_edges, 2 * numberOfEdges * sizeof(int));
    cudaMalloc(&result, numberOfBlocks * threadsPerBlock * sizeof(int));
    cudaMalloc(&d_out, 2 * numberOfEdges * sizeof(int));

    double startKernelTime = CycleTimer::currentSeconds();

    cudaMemcpy(dev_edges, edges.data(), numberOfEdges * 2 * sizeof(int),
    cudaMemcpyHostToDevice);

    cudaEventRecord(startSort);
    sort(dev_edges,numberOfEdges);
    cudaEventRecord(stopSort);

    // Hardcoding the node value 
    cudaEventRecord(startNumvertices);
    calculateNumVertices(dev_edges, d_out, numberOfEdges * 2);
    cudaEventRecord(stopNumvertices);

    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    numberOfNodes = 1 + (*out);
 
    // allocate space for the node array
    cudaMalloc(&dev_nodes, (numberOfNodes + 1) * sizeof(int));

    cudaEventRecord(startNodeArray1);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2,numberOfNodes,0);
    cudaEventRecord(stopNodeArray1);

    cudaDeviceSynchronize();

    // compute the degree of the nodes
    cudaEventRecord(startFilter);
    filter<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges);
    cudaEventRecord(stopFilter);

    cudaDeviceSynchronize();

    //remove the filtered edges
    cudaEventRecord(startRemove);
    remove(dev_edges,numberOfEdges);
    cudaEventRecord(stopRemove);

    //get the node array once again
    //note = new size of the edge array is now numberOfEdges

    cudaEventRecord(startNodeArray2);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges,numberOfNodes,1);
    cudaEventRecord(stopNodeArray2);

    // debug(dev_edges,numberOfEdges,"node array result");
    cudaDeviceSynchronize(); 

    cudaEventRecord(startTriCount);
    trianglecounting<<<numberOfBlocks,threadsPerBlock>>>(dev_edges, dev_nodes, result, numberOfEdges);
    cudaEventRecord(stopTriCount);

    cudaDeviceSynchronize();

    //calculate the number of triangles
    cudaEventRecord(startSumTri);
    calculateSum(result, d_out, numberOfBlocks * threadsPerBlock);
    cudaEventRecord(stopSumTri);

    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    int numberoftriangles = (*out);

    printf("number of triangles = %d\n",numberoftriangles);

    double endKernelTime = CycleTimer::currentSeconds();
    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);

    cudaFree(dev_edges);
    cudaFree(dev_nodes);
    cudaFree(result);
    cudaFree(d_out);
    free(out);

    float m1 = 0;
    cudaEventElapsedTime(&m1, startNumvertices, stopNumvertices);
    printf("CUDA Elapsed Time for num of vertices = %f ms\n", m1);

    float m2 = 0;
    cudaEventElapsedTime(&m2, startNodeArray1, stopNodeArray1);
    printf("CUDA Elapsed Time for nodeArray filter 1 = %f ms\n", m2);


    float m3 = 0;
    cudaEventElapsedTime(&m3, startFilter, stopFilter);
    printf("CUDA Elapsed Time for edge filter = %f ms\n", m3);

    float m4 = 0;
    cudaEventElapsedTime(&m4, startRemove, stopRemove);
    printf("CUDA Elapsed Time for edge Remove %f ms\n", m4);

    float m5 = 0;
    cudaEventElapsedTime(&m5, startNodeArray2, stopNodeArray2);
    printf("CUDA Elapsed Time for nodeArray filter 2 = %f ms\n", m5);

    float m6 = 0;
    cudaEventElapsedTime(&m6, startTriCount, stopTriCount);
    printf("CUDA Elapsed Time for calculating Triangles %f ms\n", m6);

    float m7 = 0;
    cudaEventElapsedTime(&m7, startSumTri, stopSumTri);
    printf("CUDA Elapsed Time for Summing the number of Triangles %f ms\n", m7);

    float m8 = 0;
    cudaEventElapsedTime(&m8, startSort, stopSort);
    printf("CUDA Elapsed Time for Sorting %f ms\n", m8);

    cudaEventDestroy(startNodeArray1);
    cudaEventDestroy(stopNodeArray1);

    cudaEventDestroy(startNodeArray2);
    cudaEventDestroy(stopNodeArray2);

    cudaEventDestroy(startFilter);
    cudaEventDestroy(stopFilter);

    cudaEventDestroy(startTriCount);
    cudaEventDestroy(stopTriCount);

    cudaEventDestroy(startNumvertices);
    cudaEventDestroy(stopNumvertices);

    cudaEventDestroy(startSumTri);
    cudaEventDestroy(stopSumTri);

    cudaEventDestroy(startRemove);
    cudaEventDestroy(stopRemove);

    cudaEventDestroy(startSort);
    cudaEventDestroy(stopSort);

    // printCudaInfo();
    

}

