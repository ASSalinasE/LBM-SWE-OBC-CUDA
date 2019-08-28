#ifndef SETUP_CUH
#define SETUP_CUH

#include "../../include/structs.h"

#if IN == 3
	__global__ void auxArraysKernel(int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, unsigned char*);
#elif IN == 4
	__global__ void auxArraysKernel(int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, unsigned char*, unsigned char*);
#endif
__global__ void hKernel(int, int, const prec* __restrict__, const prec* __restrict__, prec*);

#endif
