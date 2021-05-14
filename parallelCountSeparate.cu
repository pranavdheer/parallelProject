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
#define numberOfBlocks 46
#define blocks_x_threads (threadsPerBlock * numberOfBlocks)
#define FILTER -1
#define inf 0x7f800000 
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
		
	int num_blocks = 46;
    int elements_per_block = num_elements/num_blocks;
	int tail = num_elements - num_blocks * elements_per_block;
	int remaining = num_elements - tail;

	find_sum<<<num_blocks, threadsPerBlock>>>(d_in, d_out, elements_per_block); 
    cudaDeviceSynchronize();

	find_sum_final<<<1, threadsPerBlock>>>(d_in, d_out, num_elements, remaining, num_blocks);
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

void calculateNumVertices(int* d_in, int* d_out, int num_elements)
{
		
	int num_blocks = numberOfBlocks;
    int elements_per_block = num_elements / num_blocks;
	int tail = num_elements - num_blocks * elements_per_block;
	int remaining = num_elements - tail;
    //printf("helo\n");
	find_max<<<num_blocks, threadsPerBlock>>>(d_in, d_out, elements_per_block); 
    cudaDeviceSynchronize();

	find_max_final<<<1, threadsPerBlock>>>(d_in, d_out, num_elements, remaining, num_blocks);
    //printf("dout %d \n", *d_out );
	
}

__global__ void nodeArray(const int* __restrict__ dev_edges, int *dev_nodes,int size, int n){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blocks_x_threads;
    int x,y;



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

__global__ void filter(int* dev_edges,const int* __restrict__ dev_nodes,int numberOfEdges){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blocks_x_threads;
    //int id;
    for(int iter = idx; iter < numberOfEdges; iter += step)
    {



        int2 sd_pair = ((int2*)dev_edges)[iter];
        // can't use int2 because int2 pointer moves in multiples of 2
        int sourceDegree = dev_nodes[(sd_pair.x) + 1] - dev_nodes[sd_pair.x];
        int destinationDegree = dev_nodes[(sd_pair.y) + 1] - dev_nodes[sd_pair.y];

        if(destinationDegree < sourceDegree || (destinationDegree == sourceDegree && sd_pair.y < sd_pair.x)){

            ((int2*)dev_edges)[iter] =  make_int2(FILTER, FILTER);
        }    
    }     

}
__global__ void separate(int* dev_edges, int*d_out, int numberOfEdges) {
    int from = blockDim.x * blockIdx.x + threadIdx.x;
    int step = blocks_x_threads;
    for (int i = from; i < numberOfEdges; i += step) {
        int2 sd_pair = ((int2*)dev_edges)[i];  
        d_out[i] = sd_pair.x;
        d_out[numberOfEdges + i] = sd_pair.y;

    }
}
__global__ void trianglecounting(const int* __restrict__ dev_edges,const int* __restrict__ dev_nodes, int* result, int numberOfEdges){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;    
    int count = 0;
    int id = 0;
    
    for(int iter = idx; iter < numberOfEdges; iter = iter + step)
    {
        
        id = iter;
        int s = dev_edges[id];
        int e = dev_edges[numberOfEdges + id];

        int s_start = dev_nodes[s];
        int s_end = dev_nodes[s + 1];
        int e_start = dev_nodes[e];
        int e_end = dev_nodes[e + 1];
        int s_next,e_next;
        s_next = dev_edges[(s_start)];
        e_next = dev_edges[(e_start)];


        while(s_start < s_end && e_start < e_end)
        {

            int difference = s_next - e_next;


            if(difference < 0) { 
                s_start += 1;
                s_next = dev_edges[(s_start)];
            }
            if(difference > 0) {
                e_start += 1;
                e_next = dev_edges[(e_start)];
            }
            if(difference == 0) {
                count++;
                s_start += 1;
                s_next = dev_edges[(s_start)];
                e_start += 1;
                e_next = dev_edges[(e_start)];
            }
        }
    }
    result[idx] = count;
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
    int* d_out;
    int *result;
    //int numberOfBlocks;
    int numberOfNodes;
    int* out = (int*)malloc(sizeof(int));


    cudaEvent_t startNodeArray1, stopNodeArray1, startNodeArray2, stopNodeArray2,startFilter, stopFilter, startTriCount, stopTriCount, startNumVertices, stopNumVertices, startSumTri, stopSumTri, startSeparate, stopSeparate;

    cudaEventCreate(&startNodeArray1);
    cudaEventCreate(&stopNodeArray1);
    cudaEventCreate(&startNodeArray2);
    cudaEventCreate(&stopNodeArray2);
    cudaEventCreate(&startFilter);
    cudaEventCreate(&stopFilter);
    cudaEventCreate(&startTriCount);
    cudaEventCreate(&stopTriCount);
    cudaEventCreate(&startNumVertices);
    cudaEventCreate(&stopNumVertices);
    cudaEventCreate(&startSumTri);
    cudaEventCreate(&stopSumTri);
    cudaEventCreate(&startSeparate);
    cudaEventCreate(&stopSeparate);

    double startKernelTime = CycleTimer::currentSeconds();
    // transfer data to GPU
    cudaMalloc(&dev_edges, 2 * numberOfEdges * sizeof(int));
    cudaMalloc(&result, numberOfBlocks * threadsPerBlock * sizeof(uint64_t));
    cudaMemcpy(dev_edges, edges.data(), numberOfEdges * 2 * sizeof(int),
    cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, 2 * numberOfEdges * sizeof(int));




    // Hardcoding the node value 
    cudaEventRecord(startNumVertices);
    calculateNumVertices(dev_edges, d_out, numberOfEdges * 2);
    cudaEventRecord(stopNumVertices);
    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    numberOfNodes = 1 + (*out);
    printf("number of nodes = %d\n", numberOfNodes);

    SortEdges(numberOfEdges, dev_edges);
    // allocate space for the node array
    //debug(dev_edges,numberOfEdges*2,"sorted edges");

    cudaMalloc(&dev_nodes, (numberOfNodes + 1) * sizeof(int));


    cudaEventRecord(startNodeArray1);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges*2,numberOfNodes);
    cudaEventRecord(stopNodeArray1);
    cudaDeviceSynchronize();
    //debug(dev_nodes,numberOfNodes,"node aray 1");

    printf("number of edges = %d\n", numberOfEdges);
    // compute the degree of the nodes

    cudaEventRecord(startFilter);
    filter<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges);
    cudaEventRecord(stopFilter);
    cudaDeviceSynchronize();

    //remove the filtered edges
    remove(dev_edges,numberOfEdges);
    //debug(dev_edges,numberOfEdges,"filtered edges");

    cudaEventRecord(startNodeArray2);
    nodeArray<<<numberOfBlocks,threadsPerBlock>>>(dev_edges,dev_nodes,numberOfEdges,numberOfNodes);
    cudaEventRecord(stopNodeArray2);
    cudaDeviceSynchronize(); 
    //debug(dev_nodes,numberOfNodes,"node aray 2");
    cudaEventRecord(startSeparate);
    separate<<<numberOfBlocks,threadsPerBlock>>>(dev_edges, d_out, numberOfEdges/2);
    cudaEventRecord(stopSeparate);
    //debug(d_out,numberOfEdges,"separarted array");

    cudaEventRecord(startTriCount);
    trianglecounting<<<numberOfBlocks,threadsPerBlock>>>(d_out, dev_nodes, result, numberOfEdges/2);
    cudaEventRecord(stopTriCount);
    cudaDeviceSynchronize();

    cudaEventRecord(startSumTri);
    calculateSum(result, dev_edges, numberOfBlocks * threadsPerBlock);
    cudaEventRecord(stopSumTri);



    //calculate the number of triangles

    cudaMemcpy(out, dev_edges, sizeof(int), cudaMemcpyDeviceToHost);
    int numberoftriangles = (*out);


    cudaFree(result);
    cudaFree(dev_edges);
    cudaFree(dev_nodes);
    cudaFree(d_out);

    
    double endKernelTime = CycleTimer::currentSeconds();
    //debug(result,numberOfNodes,"triangle array");
    free(out);
    printf("number of triangles = %lld\n",numberoftriangles);

    float m1 = 0;
    cudaEventElapsedTime(&m1, startNodeArray1, stopNodeArray1);
    printf("CUDA Elapsed Time for Node Array 1 %f ms\n", m1);

    float m2 = 0;
    cudaEventElapsedTime(&m2, startFilter, stopFilter);
    printf("CUDA Elapsed Time for Filter %f ms\n", m2);

    float m5 = 0;
    cudaEventElapsedTime(&m5, startNumVertices, stopNumVertices);
    printf("CUDA Elapsed Time for Num Vertices %f ms\n", m5);

    float m3 = 0;
    cudaEventElapsedTime(&m3, startNodeArray2, stopNodeArray2);
    printf("CUDA Elapsed Time for Node Array 2 %f ms\n", m3);

    float m7 = 0;
    cudaEventElapsedTime(&m7, startSeparate, stopSeparate);
    printf("CUDA Elapsed Time for separate %f ms\n", m7);

    float m4 = 0;
    cudaEventElapsedTime(&m4, startTriCount, stopTriCount);
    printf("CUDA Elapsed Time for Triangle Counting %f ms\n", m4);

    float m6 = 0;
    cudaEventElapsedTime(&m6, startSumTri, stopSumTri);
    printf("CUDA Elapsed Time for Sum Triangles %f ms\n", m6);

    cudaEventDestroy(startNodeArray1);
    cudaEventDestroy(stopNodeArray1);
    cudaEventDestroy(startNodeArray2);
    cudaEventDestroy(stopNodeArray2);
    cudaEventDestroy(startFilter);
    cudaEventDestroy(stopFilter);
    cudaEventDestroy(startTriCount);
    cudaEventDestroy(stopTriCount);
    cudaEventDestroy(startNumVertices);
    cudaEventDestroy(stopNumVertices);
    cudaEventDestroy(startSumTri);
    cudaEventDestroy(stopSumTri);
    cudaEventDestroy(startSeparate);
    cudaEventDestroy(stopSeparate);


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
