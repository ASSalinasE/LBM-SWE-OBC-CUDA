#include <iostream>
#include <stdio.h>
#include "../include/structs.h"

#if IN == 3
__global__ void auxArraysKernel(int Lx, int Ly,
	const int* __restrict__ ex, const int* __restrict__ ey,
	const int* __restrict__ node_types, unsigned char* Arr_tri) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int size = Lx * Ly;
	if (i < size) {
		int y = (int)i / Lx;
		int x = i - y * Lx;
		int xi, yi, ind, indj, indk;
		int a;
		for (a = 0; a<9; a++) {
			Arr_tri[i + a * size] = 0;
		}
		if (node_types[i] == 2) {
			Arr_tri[i] = 1;
			if (y == 0) {
				if (x == 0) {
					Arr_tri[i + 3 * size] = 1;
					Arr_tri[i + 4 * size] = 1;
					Arr_tri[i + 7 * size] = 1;
				}
				else if (x == Lx - 1) {
					Arr_tri[i + size] = 1;
					Arr_tri[i + 4 * size] = 1;
					Arr_tri[i + 8 * size] = 1;
				}
				else {
					Arr_tri[i + 1 * size] = 1;
					Arr_tri[i + 3 * size] = 1;
					Arr_tri[i + 4 * size] = 1;
					Arr_tri[i + 7 * size] = 1;
					Arr_tri[i + 8 * size] = 1;
				}
			}
			else if (y == Ly - 1) {
				if (x == 0) {
					Arr_tri[i + 2 * size] = 1;
					Arr_tri[i + 3 * size] = 1;
					Arr_tri[i + 6 * size] = 1;
				}
				else if (x == Lx - 1) {
					Arr_tri[i + size] = 1;
					Arr_tri[i + 2 * size] = 1;
					Arr_tri[i + 5 * size] = 1;
				}
				else {
					Arr_tri[i + size] = 1;
					Arr_tri[i + 2 * size] = 1;
					Arr_tri[i + 3 * size] = 1;
					Arr_tri[i + 5 * size] = 1;
					Arr_tri[i + 6 * size] = 1;
				}
			}
			else {
				if (x == 0) {
					Arr_tri[i + 2 * size] = 1;
					Arr_tri[i + 3 * size] = 1;
					Arr_tri[i + 4 * size] = 1;
					Arr_tri[i + 6 * size] = 1;
					Arr_tri[i + 7 * size] = 1;
				}
				else if (x == Lx - 1) {
					Arr_tri[i + size] = 1;
					Arr_tri[i + 2 * size] = 1;
					Arr_tri[i + 4 * size] = 1;
					Arr_tri[i + 5 * size] = 1;
					Arr_tri[i + 8 * size] = 1;
				}
				else {
					for (a = 1; a<9; a++) {
						Arr_tri[i + a * size] = 1;
					}
				}
			}
		}
		else if (node_types[i] == 1) {
			Arr_tri[i] = 1;
			if (y == 0) {
				Arr_tri[i + size] = 1;
				Arr_tri[i + 4 * size] = 1;
				Arr_tri[i + 8 * size] = 1;
				Arr_tri[i + 3 * size] = 2;
				Arr_tri[i + 6 * size] = 2;
				Arr_tri[i + 7 * size] = 2;
			}
			else if (y == Ly - 1) {
				Arr_tri[i + size] = 1;
				Arr_tri[i + 2 * size] = 1;
				Arr_tri[i + 5 * size] = 1;
				Arr_tri[i + 3 * size] = 2;
				Arr_tri[i + 6 * size] = 2;
				Arr_tri[i + 7 * size] = 2;
			}
			else {
				for (a = 1; a<9; a++) {
					yi = y - ey[a];
					xi = x - ex[a];
					ind = yi * Lx + xi;
					if (node_types[ind] != 0) 
						Arr_tri[i + a * size] = 1;
					else 
						Arr_tri[i + a * size] = 2;
					if (a > 4) {
						if (node_types[ind] == 1) {
							indj = y * Lx + xi;
							indk = yi * Lx + x;
							if (node_types[indj] == 0 || node_types[indk] == 0)
								Arr_tri[i + a * size] = 2; 
						}
					}
				}
			} 
		}
	}
}
#elif IN == 4
__global__ void auxArraysKernel(int Lx, int Ly,
	const int* __restrict__ ex, const int* __restrict__ ey,
	const int* __restrict__ node_types,
	unsigned char* SC_bin, unsigned char* BB_bin) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int size = Lx * Ly;
	if (i < size) {
		int y = (int)i / Lx;
		int x = i - y * Lx;
		int xi, yi, ind, indj, indk, a;
		int valueSC = 0, valueBB = 0;
		if (node_types[i] == 2) {
			if (y == 0) {
				if (x == 0) 
					valueSC += 4 + 8 + 64;
				else if (x == Lx - 1)
					valueSC += 1 + 8 + 128;
				else 
					valueSC += 1 + 4 + 8 + 64 + 128;
			}
			else if (y == Ly - 1) {
				if (x == 0) 
					valueSC += 2 + 4 + 32;
				else if (x == Lx - 1) 
					valueSC += 1 + 2 + 16;
				else 
					valueSC += 1 + 2 + 4 + 16 + 32;
			}
			else {
				if (x == 0)
					valueSC += 2 + 4 + 8 + 32 + 64;
				else if (x == Lx - 1) 
					valueSC += 1 + 2 + 8 + 16 + 128;
				else  
					valueSC = 255;
			}
		}
		else if (node_types[i] == 1) {
			if (y == 0) {
				valueSC += 1 + 8 + 128;
				valueBB += 4 + 32 + 64;
			}
			else if (y == Ly - 1) {
				valueSC += 1 + 2 + 16;
				valueBB += 4 + 32 + 64;
			}
			else {
				for (a = 1; a<9; a++) {
					yi = y - ey[a];
					xi = x - ex[a];
					ind = yi * Lx + xi;
					if (node_types[ind] != 0) 
						valueSC += (1 << (a-1));
					else 
						valueBB += (1 << (a-1));
					if (a > 4) {
						if (node_types[ind] == 1) {
							indj = y * Lx + xi;
							indk = yi * Lx + x;
							if (node_types[indj] == 0 || node_types[indk] == 0) {
								valueSC -= (1 << (a-1));
								valueBB += (1 << (a-1));
							}
						}
					}
				}
			}
		}
		SC_bin[i] = (unsigned char) valueSC;
		BB_bin[i] = (unsigned char) valueBB;
	}
} 
#endif


__global__ void hKernel(int Lx, int Ly, const prec* __restrict__ w,
	const prec* __restrict__ b, prec* h) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < Lx*Ly) {
		h[i] = w[i] - b[i];
	}
}
