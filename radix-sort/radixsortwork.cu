#include<iostream>
#include<stdio.h>
#include<stdint.h>
#include "CycleTimer.h"
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
using namespace std;
#define threadsPerBlock 1024
//int sortBlocks;


__global__ void predicate(uint64_t *myarray , uint64_t *predicatearray , int d_numberOfElements,uint64_t bit,int bitset)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < d_numberOfElements)
            predicatearray[index] = bitset ? ((myarray[index] & bit) ? 1 : 0) : ((!(myarray[index] & bit)) ? 1 : 0);
}

__global__ void scatter(uint64_t *myarray , uint64_t *scanarray , uint64_t *predicatearray,uint64_t * scatterarray ,int d_numberOfElements,int offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < d_numberOfElements)
        if(predicatearray[index])
            scatterarray[scanarray[index] - 1 + offset] = myarray[index];
}
__global__ void scanDevice(uint64_t *myarray , int numberOfElements, uint64_t *temp,int moveIndex)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index > numberOfElements)
        return;
    temp[index] = myarray[index];
    if(index - moveIndex >= 0)
        temp[index] = temp[index] +myarray[index - moveIndex];
}
uint64_t* scanHost(uint64_t *scanarray,int numberOfElements, int sortBlocks)
{
    uint64_t *temp;
    uint64_t *temp1;
    cudaMalloc(&temp1,sizeof(uint64_t)*numberOfElements);
    cudaMalloc(&temp,sizeof(uint64_t)*numberOfElements);
    cudaMemcpy(temp1,scanarray,sizeof(uint64_t)*numberOfElements,cudaMemcpyDeviceToDevice);
    int j,k=0;
    for(j=1;j<numberOfElements;j= j*2,k++)
    {
        if(k%2 == 0)
        {
            scanDevice<<<sortBlocks,threadsPerBlock>>>(temp1,numberOfElements,temp, j);
            cudaDeviceSynchronize();
        }
        else
        {
            scanDevice<<<sortBlocks,threadsPerBlock>>>(temp,numberOfElements,temp1, j);
            cudaDeviceSynchronize();
        }
    } 
    cudaDeviceSynchronize();
    if(k%2 == 0)
        return temp1;
    else
        return temp;
}

uint64_t *partition(uint64_t *myarray,int numberOfElements,uint64_t bit, int sortBlocks)
{   
    int offset;
    uint64_t *predicatearray;
    cudaMalloc((void**)&predicatearray,sizeof(uint64_t)*numberOfElements);
    predicate<<<sortBlocks,threadsPerBlock>>>(myarray,predicatearray,numberOfElements,bit,0);
    uint64_t *scanarray;
    scanarray = scanHost(predicatearray,numberOfElements, sortBlocks);
    uint64_t *scatterarray;
    cudaMalloc((void**)&scatterarray,sizeof(uint64_t)*numberOfElements);

    scatter<<<sortBlocks,threadsPerBlock>>>(myarray,scanarray,predicatearray,scatterarray, numberOfElements,0);
    cudaMemcpy(&offset,scanarray+numberOfElements-1,sizeof(uint64_t),cudaMemcpyDeviceToHost);
    predicate<<<sortBlocks,threadsPerBlock>>>(myarray,predicatearray,numberOfElements,bit,1);
    scanarray = scanHost(predicatearray,numberOfElements, sortBlocks);
    scatter<<<sortBlocks,threadsPerBlock>>>(myarray,scanarray,predicatearray,scatterarray, numberOfElements,offset);
    return scatterarray;
}
int offset;
uint64_t *split(uint64_t *myarray,int numberOfElements,uint64_t bit,int bitset, int sortBlocks)
{   
    uint64_t *predicatearray;
    cudaMalloc((void**)&predicatearray,sizeof(uint64_t)*numberOfElements);
    predicate<<<sortBlocks,threadsPerBlock>>>(myarray,predicatearray,numberOfElements,bit,bitset);
    uint64_t *scanarray;
    scanarray = scanHost(predicatearray,numberOfElements, sortBlocks);
    uint64_t *scatterarray;
    cudaMemcpy(&offset,scanarray+numberOfElements-1,sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaMalloc((void**)&scatterarray,sizeof(uint64_t)*offset);
    scatter<<<sortBlocks,threadsPerBlock>>>(myarray,scanarray,predicatearray,scatterarray, numberOfElements,0);
    return scatterarray;
}
uint64_t *SortEdges(uint64_t *myarray , int numberOfElements, int sortBlocks)
{
    uint64_t bit;
    uint64_t *minusarray = split(myarray,numberOfElements,1LU<<63,1, sortBlocks);
    for(int i=0;i<sizeof(uint64_t)*8;i++)
    {
        bit = 1LU<<i;
        minusarray = partition(minusarray,offset,bit, sortBlocks);
    }
    uint64_t *plusarray = split(myarray,numberOfElements,1LU<<63,0, sortBlocks);
    for(int i=0;i<sizeof(uint64_t)*8;i++)
    {
        bit = 1LU<<i;
        plusarray = partition(plusarray,offset,bit, sortBlocks);
    }
    
    cudaMemcpy(myarray,minusarray,sizeof(uint64_t)*(numberOfElements-offset),cudaMemcpyDeviceToDevice);
    cudaMemcpy(myarray+(numberOfElements-offset),plusarray,sizeof(uint64_t)*offset,cudaMemcpyDeviceToDevice);
    return myarray;
}
void SortEdgesThrust(int m, uint64_t* edges) {
    thrust::device_ptr<uint64_t> ptr(edges);
    thrust::sort(ptr, ptr + m);
    }
int main()
{
    cout<<"enter the number of elements \n";
    int numberOfElements;
    cin>>numberOfElements;
    uint64_t *h_array  = new uint64_t[numberOfElements];
    for(int i=0;i<numberOfElements;i++)
    {
        h_array[i] = rand()&1;
    }
    h_array[1024] = 34;
	h_array[333] = 55;
	h_array[223] = 42; 
    h_array[5124] = 6;
    h_array[6305] = 2;  
    int sortBlocks = numberOfElements / threadsPerBlock;
    if(numberOfElements % threadsPerBlock != 0)
        sortBlocks += 1;
    cout << sortBlocks << "\n";
    uint64_t *myarray;
    cudaMalloc((void**)&myarray ,sizeof(uint64_t)*numberOfElements);
    cudaMemcpy(myarray,h_array,sizeof(uint64_t)*numberOfElements,cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaEventRecord(start);

    //cudaEventRecord(stop);
    double startKernel = CycleTimer::currentSeconds();
    SortEdgesThrust(numberOfElements, myarray);
    double endKernel = CycleTimer::currentSeconds();
    double startKernelTime = CycleTimer::currentSeconds();
    myarray = SortEdges(myarray, numberOfElements, sortBlocks);
    double endKernelTime = CycleTimer::currentSeconds();
    float m1 = 0;
    cudaEventElapsedTime(&m1, start, stop);
    printf("CUDA Elapsed Time %f ms\n", m1);
    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);
    double kernel = endKernel - startKernel;
    printf("Kernel: %.3f ms\n", 1000.f * kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_array,myarray,sizeof(uint64_t)*numberOfElements,cudaMemcpyDeviceToHost);

}