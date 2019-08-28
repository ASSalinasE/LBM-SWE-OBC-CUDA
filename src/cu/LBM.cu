#include "include/setup.cuh"
#include "../cpp/include/files.h"
#include "../include/structs.h"
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <fstream>
#include <time.h>
 
__global__ void wKernel(int Lx, int Ly, const prec* __restrict__ h,
	const prec* __restrict__ b, prec* w) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < Lx*Ly) {
		w[i] = h[i] + b[i];
	}
}

#if IN == 1
__global__ void LBMpull(int Lx, int Ly, prec g, prec e, prec tau,
	const prec* __restrict__ b, const prec* __restrict__ f1, 
	prec* f2, prec* h) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int size = Lx * Ly, j;
	prec ftemp[9], feq[9];
	prec uxlocal, uylocal;
	prec hlocal[9], blocal[9];
	prec gh, usq, ux3, uy3, uxuy5, uxuy6;
	prec fact1 = 1 / (9 * e*e);
	prec fact2 = fact1 * 0.25;
	prec factS = fact1 * 1.5;
	unsigned char trilocal[8]; 
	int nt[9]; 
	if (i < size) {
		hlocal[0] = h[i];
		if (hlocal[0] > 0){
			int y = (int)i / Lx;
			int x = i - y * Lx;

			blocal[0] = b[i];
			blocal[1] = (             x != 0   ) ? b[i      - 1] : 0;
			blocal[2] = (y != 0                ) ? b[i - Lx    ] : 0;
			blocal[3] = (             x != Lx-1) ? b[i      + 1] : 0;
			blocal[4] = (y != Ly-1             ) ? b[i + Lx    ] : 0;
			blocal[5] = (y != 0    && x != 0   ) ? b[i - Lx - 1] : 0;
			blocal[6] = (y != 0    && x != Lx-1) ? b[i - Lx + 1] : 0;
			blocal[7] = (y != Ly-1 && x != Lx-1) ? b[i + Lx + 1] : 0; 
			blocal[8] = (y != Ly-1 && x != 0   ) ? b[i + Lx - 1] : 0;

			hlocal[1] = (             x != 0   ) ? h[i      - 1] : 0;
			hlocal[2] = (y != 0                ) ? h[i - Lx    ] : 0;
			hlocal[3] = (             x != Lx-1) ? h[i      + 1] : 0;
			hlocal[4] = (y != Ly-1             ) ? h[i + Lx    ] : 0;
			hlocal[5] = (y != 0    && x != 0   ) ? h[i - Lx - 1] : 0;
			hlocal[6] = (y != 0    && x != Lx-1) ? h[i - Lx + 1] : 0;
			hlocal[7] = (y != Ly-1 && x != Lx-1) ? h[i + Lx + 1] : 0;
			hlocal[8] = (y != Ly-1 && x != 0   ) ? h[i + Lx - 1] : 0;

			ftemp[1] = (             x != 0   ) ? f1[i      - 1 +     size] : 0;
			ftemp[2] = (y != 0                ) ? f1[i - Lx     + 2 * size] : 0;
			ftemp[3] = (             x != Lx-1) ? f1[i      + 1 + 3 * size] : 0;
			ftemp[4] = (y != Ly-1             ) ? f1[i + Lx     + 4 * size] : 0;
			ftemp[5] = (y != 0    && x != 0   ) ? f1[i - Lx - 1 + 5 * size] : 0;
			ftemp[6] = (y != 0    && x != Lx-1) ? f1[i - Lx + 1 + 6 * size] : 0;
			ftemp[7] = (y != Ly-1 && x != Lx-1) ? f1[i + Lx + 1 + 7 * size] : 0;
			ftemp[8] = (y != Ly-1 && x != 0   ) ? f1[i + Lx - 1 + 8 * size] : 0;

			for (int a = 0; a < 9; a++){
				nt[a] = 0;
				if (hlocal[a] > 0) nt[a] = 2;
			}

			for (int a = 1; a < 9; a++){
				if (!((y == 0 && (a == 2 || a == 5 || a == 6)) ||
					  (y == Ly-1 && (a == 4 || a == 7 || a == 8)) ||
					  (x == 0 && (a == 1 || a == 5 || a == 8)) ||
					  (x == Lx-1 && (a == 3 || a == 6 || a == 7))))
					if (nt[a] == 0) nt[0] = 1;
			}

			for (int a = 0; a<8; a++) trilocal[a] = 0;
			if (nt[0] == 2) {
				if (y == 0) {
					if (x == 0) {
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[6] = 1; 
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[3] = 1;
						trilocal[7] = 1;
					}
					else {
						trilocal[0] = 1;
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[6] = 1;
						trilocal[7] = 1;
					}
				}
				else if (y == Ly - 1) {
					if (x == 0) {
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[5] = 1;
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[4] = 1;
					}
					else {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[4] = 1;
						trilocal[5] = 1;
					}
				}
				else {
					if (x == 0) {
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[5] = 1;
						trilocal[6] = 1;
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[3] = 1;
						trilocal[4] = 1;
						trilocal[7] = 1;
					}
					else {
						for (int a = 0; a<8; a++) {
							trilocal[a] = 1;
						}
					}
				}
			}
			else if (nt[0] == 1) {
				if (y == 0) {
					trilocal[0] = 1;
					trilocal[3] = 1;
					trilocal[7] = 1;
					trilocal[2] = 2;
					trilocal[5] = 2;
					trilocal[6] = 2;
				}
				else if (y == Ly - 1) {
					trilocal[0] = 1;
					trilocal[1] = 1;
					trilocal[4] = 1;
					trilocal[2] = 2;
					trilocal[5] = 2;
					trilocal[6] = 2;
				}
				else {
					for (int a = 1; a<9; a++){
						if (nt[a] != 0)
							trilocal[a-1] = 1;
						else
							trilocal[a-1] = 2;
					}
					if (nt[5] == 0 || (nt[5] != 0 && (nt[1] == 0 || nt[2] == 0))) trilocal[4] = 2;
					if (nt[6] == 0 || (nt[6] != 0 && (nt[2] == 0 || nt[3] == 0))) trilocal[5] = 2;
					if (nt[7] == 0 || (nt[7] != 0 && (nt[3] == 0 || nt[4] == 0))) trilocal[6] = 2;
					if (nt[8] == 0 || (nt[8] != 0 && (nt[4] == 0 || nt[1] == 0))) trilocal[7] = 2;
				} 
			}

			ftemp[0] = f1[i]; 
			#if BN == 1
				if(trilocal[0] == 1) ftemp[1] = ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS; else ftemp[1] = f1[i +     size];
				if(trilocal[1] == 1) ftemp[2] = ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS; else ftemp[2] = f1[i + 2 * size];
				if(trilocal[2] == 1) ftemp[3] = ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS; else ftemp[3] = f1[i + 3 * size];
				if(trilocal[3] == 1) ftemp[4] = ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS; else ftemp[4] = f1[i + 4 * size];
				if(trilocal[4] == 1) ftemp[5] = ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25; else ftemp[5] = f1[i + 5 * size];
				if(trilocal[5] == 1) ftemp[6] = ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25; else ftemp[6] = f1[i + 6 * size];
				if(trilocal[6] == 1) ftemp[7] = ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25; else ftemp[7] = f1[i + 7 * size];
				if(trilocal[7] == 1) ftemp[8] = ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25; else ftemp[8] = f1[i + 8 * size];

				if(trilocal[0] == 2) ftemp[1] = ftemp[3];
				if(trilocal[1] == 2) ftemp[2] = ftemp[4];
				if(trilocal[2] == 2) ftemp[3] = ftemp[1];
				if(trilocal[3] == 2) ftemp[4] = ftemp[2];
				if(trilocal[4] == 2) ftemp[5] = ftemp[7];
				if(trilocal[5] == 2) ftemp[6] = ftemp[8];
				if(trilocal[6] == 2) ftemp[7] = ftemp[5];
				if(trilocal[7] == 2) ftemp[8] = ftemp[6];
			#elif BN == 2
				ftemp[1] = (trilocal[0] == 1) ? (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) : f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) ? (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) : f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) ? (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) : f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) ? (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) : f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) ? (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) : f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) ? (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) : f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) ? (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) : f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) ? (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) : f1[i + 8 * size];

				ftemp[1] = (trilocal[0] == 2) ? ftemp[3] : ftemp[1];
				ftemp[2] = (trilocal[1] == 2) ? ftemp[4] : ftemp[2];
				ftemp[3] = (trilocal[2] == 2) ? ftemp[1] : ftemp[3];
				ftemp[4] = (trilocal[3] == 2) ? ftemp[2] : ftemp[4];
				ftemp[5] = (trilocal[4] == 2) ? ftemp[7] : ftemp[5];
				ftemp[6] = (trilocal[5] == 2) ? ftemp[8] : ftemp[6];
				ftemp[7] = (trilocal[6] == 2) ? ftemp[5] : ftemp[7];
				ftemp[8] = (trilocal[7] == 2) ? ftemp[6] : ftemp[8];
			#else
				ftemp[1] = (trilocal[0] == 1) * (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) + (trilocal[0] != 1) * f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) * (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) + (trilocal[1] != 1) * f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) * (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) + (trilocal[2] != 1) * f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) * (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) + (trilocal[3] != 1) * f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) * (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) + (trilocal[4] != 1) * f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) * (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) + (trilocal[5] != 1) * f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) * (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) + (trilocal[6] != 1) * f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) * (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) + (trilocal[7] != 1) * f1[i + 8 * size];

				ftemp[1] += (trilocal[0] == 2) * (ftemp[3] - ftemp[1]);
				ftemp[2] += (trilocal[1] == 2) * (ftemp[4] - ftemp[2]);
				ftemp[3] += (trilocal[2] == 2) * (ftemp[1] - ftemp[3]);
				ftemp[4] += (trilocal[3] == 2) * (ftemp[2] - ftemp[4]);
				ftemp[5] += (trilocal[4] == 2) * (ftemp[7] - ftemp[5]);
				ftemp[6] += (trilocal[5] == 2) * (ftemp[8] - ftemp[6]);
				ftemp[7] += (trilocal[6] == 2) * (ftemp[5] - ftemp[7]);
				ftemp[8] += (trilocal[7] == 2) * (ftemp[6] - ftemp[8]);
			#endif

			hlocal[0] = ftemp[0] + (ftemp[1] + ftemp[2] + ftemp[3] + ftemp[4]) + (ftemp[5] + ftemp[6] + ftemp[7] + ftemp[8]);
			uxlocal = e * ((ftemp[1] - ftemp[3]) + (ftemp[5] - ftemp[6] - ftemp[7] + ftemp[8])) / hlocal[0];
			uylocal = e * ((ftemp[2] - ftemp[4]) + (ftemp[5] + ftemp[6] - ftemp[7] - ftemp[8])) / hlocal[0];

			h[i] = hlocal[0];

			gh = 1.5 * g * hlocal[0];
			usq = 1.5 * (uxlocal * uxlocal + uylocal * uylocal);
			ux3 = 3.0 * e * uxlocal;
			uy3 = 3.0 * e * uylocal;
			uxuy5 = ux3 + uy3;
			uxuy6 = uy3 - ux3;

			feq[0] = hlocal[0] - fact1 * hlocal[0] * (5.0 * gh + 4.0 * usq);
			feq[1] = fact1 * hlocal[0] * (gh + ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[2] = fact1 * hlocal[0] * (gh + uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[3] = fact1 * hlocal[0] * (gh - ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[4] = fact1 * hlocal[0] * (gh - uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[5] = fact2 * hlocal[0] * (gh + uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[6] = fact2 * hlocal[0] * (gh + uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			feq[7] = fact2 * hlocal[0] * (gh - uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[8] = fact2 * hlocal[0] * (gh - uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			for (j = 0; j < 9; j++)
				f2[i + j * size] = ftemp[j] - (ftemp[j] - feq[j]) / tau;
		}
	}
} 
#elif IN == 2
__global__ void LBMpull(int Lx, int Ly, prec g, prec e, prec tau,
	const prec* __restrict__ b, const int* __restrict__ node_types,
	const prec* __restrict__ f1, prec* f2, prec* h) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int size = Lx * Ly, j;
	prec ftemp[9], feq[9];
	prec uxlocal, uylocal;
	prec hlocal[9], blocal[9];
	prec gh, usq, ux3, uy3, uxuy5, uxuy6;
	prec fact1 = 1 / (9 * e*e);
	prec fact2 = fact1 * 0.25;
	prec factS = fact1 * 1.5;
	unsigned char trilocal[8]; 
	int nt[9]; 
	if (i < size) {
		nt[0] = node_types[i];
		if (nt[0] != 0){
			int y = (int)i / Lx;
			int x = i - y * Lx;

			blocal[0] = b[i];
			blocal[1] = (             x != 0   ) ? b[i      - 1] : 0;
			blocal[2] = (y != 0                ) ? b[i - Lx    ] : 0;
			blocal[3] = (             x != Lx-1) ? b[i      + 1] : 0;
			blocal[4] = (y != Ly-1             ) ? b[i + Lx    ] : 0;
			blocal[5] = (y != 0    && x != 0   ) ? b[i - Lx - 1] : 0;
			blocal[6] = (y != 0    && x != Lx-1) ? b[i - Lx + 1] : 0;
			blocal[7] = (y != Ly-1 && x != Lx-1) ? b[i + Lx + 1] : 0;
			blocal[8] = (y != Ly-1 && x != 0   ) ? b[i + Lx - 1] : 0;

			hlocal[0] = h[i];
			hlocal[1] = (             x != 0   ) ? h[i      - 1] : 0;
			hlocal[2] = (y != 0                ) ? h[i - Lx    ] : 0;
			hlocal[3] = (             x != Lx-1) ? h[i      + 1] : 0;
			hlocal[4] = (y != Ly-1             ) ? h[i + Lx    ] : 0;
			hlocal[5] = (y != 0    && x != 0   ) ? h[i - Lx - 1] : 0;
			hlocal[6] = (y != 0    && x != Lx-1) ? h[i - Lx + 1] : 0;
			hlocal[7] = (y != Ly-1 && x != Lx-1) ? h[i + Lx + 1] : 0;
			hlocal[8] = (y != Ly-1 && x != 0   ) ? h[i + Lx - 1] : 0;

			ftemp[1] = (             x != 0   ) ? f1[i      - 1 +     size] : 0;
			ftemp[2] = (y != 0                ) ? f1[i - Lx     + 2 * size] : 0;
			ftemp[3] = (             x != Lx-1) ? f1[i      + 1 + 3 * size] : 0;
			ftemp[4] = (y != Ly-1             ) ? f1[i + Lx     + 4 * size] : 0;
			ftemp[5] = (y != 0    && x != 0   ) ? f1[i - Lx - 1 + 5 * size] : 0;
			ftemp[6] = (y != 0    && x != Lx-1) ? f1[i - Lx + 1 + 6 * size] : 0;
			ftemp[7] = (y != Ly-1 && x != Lx-1) ? f1[i + Lx + 1 + 7 * size] : 0;
			ftemp[8] = (y != Ly-1 && x != 0   ) ? f1[i + Lx - 1 + 8 * size] : 0;

			for (int a = 0; a<8; a++) trilocal[a] = 0;
			if (nt[0] == 2) {
				if (y == 0) {
					if (x == 0) {
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[6] = 1;
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[3] = 1;
						trilocal[7] = 1;
					}
					else {
						trilocal[0] = 1;
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[6] = 1;
						trilocal[7] = 1;
					}
				}
				else if (y == Ly - 1) {
					if (x == 0) {
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[5] = 1;
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[4] = 1;
					}
					else {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[4] = 1;
						trilocal[5] = 1;
					}
				}
				else {
					if (x == 0) {
						trilocal[1] = 1;
						trilocal[2] = 1;
						trilocal[3] = 1;
						trilocal[5] = 1;
						trilocal[6] = 1;
					}
					else if (x == Lx - 1) {
						trilocal[0] = 1;
						trilocal[1] = 1;
						trilocal[3] = 1;
						trilocal[4] = 1;
						trilocal[7] = 1;
					}
					else {
						for (int a = 0; a<8; a++) {
							trilocal[a] = 1;
						}
					}
				}
			}
			else if (nt[0] == 1) {
				if (y == 0) {
					trilocal[0] = 1;
					trilocal[3] = 1;
					trilocal[7] = 1;
					trilocal[2] = 2;
					trilocal[5] = 2;
					trilocal[6] = 2;
				}
				else if (y == Ly - 1) {
					trilocal[0] = 1;
					trilocal[1] = 1;
					trilocal[4] = 1;
					trilocal[2] = 2;
					trilocal[5] = 2;
					trilocal[6] = 2;
				}
				else {
					nt[1] = node_types[i - 1 ];
					nt[2] = node_types[i - Lx];
					nt[3] = node_types[i + 1 ];
					nt[4] = node_types[i + Lx];
					nt[5] = node_types[i - Lx - 1];
					nt[6] = node_types[i - Lx + 1];
					nt[7] = node_types[i + Lx + 1];
					nt[8] = node_types[i + Lx - 1];
					for (int a = 1; a<9; a++){
						if (nt[a] != 0)
							trilocal[a-1] = 1;
						else
							trilocal[a-1] = 2;
					}
					if (nt[5] == 0 || (nt[5] == 1 && (nt[1] == 0 || nt[2] == 0))) trilocal[4] = 2;
					if (nt[6] == 0 || (nt[6] == 1 && (nt[2] == 0 || nt[3] == 0))) trilocal[5] = 2;
					if (nt[7] == 0 || (nt[7] == 1 && (nt[3] == 0 || nt[4] == 0))) trilocal[6] = 2;
					if (nt[8] == 0 || (nt[8] == 1 && (nt[4] == 0 || nt[1] == 0))) trilocal[7] = 2;
				} 
			}

			ftemp[0] = f1[i]; 
			#if BN == 1
				if(trilocal[0] == 1) ftemp[1] = ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS; else ftemp[1] = f1[i +     size];
				if(trilocal[1] == 1) ftemp[2] = ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS; else ftemp[2] = f1[i + 2 * size];
				if(trilocal[2] == 1) ftemp[3] = ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS; else ftemp[3] = f1[i + 3 * size];
				if(trilocal[3] == 1) ftemp[4] = ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS; else ftemp[4] = f1[i + 4 * size];
				if(trilocal[4] == 1) ftemp[5] = ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25; else ftemp[5] = f1[i + 5 * size];
				if(trilocal[5] == 1) ftemp[6] = ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25; else ftemp[6] = f1[i + 6 * size];
				if(trilocal[6] == 1) ftemp[7] = ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25; else ftemp[7] = f1[i + 7 * size];
				if(trilocal[7] == 1) ftemp[8] = ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25; else ftemp[8] = f1[i + 8 * size];

				if(trilocal[0] == 2) ftemp[1] = ftemp[3];
				if(trilocal[1] == 2) ftemp[2] = ftemp[4];
				if(trilocal[2] == 2) ftemp[3] = ftemp[1];
				if(trilocal[3] == 2) ftemp[4] = ftemp[2];
				if(trilocal[4] == 2) ftemp[5] = ftemp[7];
				if(trilocal[5] == 2) ftemp[6] = ftemp[8];
				if(trilocal[6] == 2) ftemp[7] = ftemp[5];
				if(trilocal[7] == 2) ftemp[8] = ftemp[6];
			#elif BN == 2
				ftemp[1] = (trilocal[0] == 1) ? (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) : f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) ? (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) : f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) ? (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) : f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) ? (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) : f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) ? (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) : f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) ? (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) : f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) ? (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) : f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) ? (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) : f1[i + 8 * size];

				ftemp[1] = (trilocal[0] == 2) ? ftemp[3] : ftemp[1];
				ftemp[2] = (trilocal[1] == 2) ? ftemp[4] : ftemp[2];
				ftemp[3] = (trilocal[2] == 2) ? ftemp[1] : ftemp[3];
				ftemp[4] = (trilocal[3] == 2) ? ftemp[2] : ftemp[4];
				ftemp[5] = (trilocal[4] == 2) ? ftemp[7] : ftemp[5];
				ftemp[6] = (trilocal[5] == 2) ? ftemp[8] : ftemp[6];
				ftemp[7] = (trilocal[6] == 2) ? ftemp[5] : ftemp[7];
				ftemp[8] = (trilocal[7] == 2) ? ftemp[6] : ftemp[8];
			#else
				ftemp[1] = (trilocal[0] == 1) * (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) + (trilocal[0] != 1) * f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) * (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) + (trilocal[1] != 1) * f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) * (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) + (trilocal[2] != 1) * f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) * (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) + (trilocal[3] != 1) * f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) * (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) + (trilocal[4] != 1) * f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) * (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) + (trilocal[5] != 1) * f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) * (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) + (trilocal[6] != 1) * f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) * (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) + (trilocal[7] != 1) * f1[i + 8 * size];

				ftemp[1] += (trilocal[0] == 2) * (ftemp[3] - ftemp[1]);
				ftemp[2] += (trilocal[1] == 2) * (ftemp[4] - ftemp[2]);
				ftemp[3] += (trilocal[2] == 2) * (ftemp[1] - ftemp[3]);
				ftemp[4] += (trilocal[3] == 2) * (ftemp[2] - ftemp[4]);
				ftemp[5] += (trilocal[4] == 2) * (ftemp[7] - ftemp[5]);
				ftemp[6] += (trilocal[5] == 2) * (ftemp[8] - ftemp[6]);
				ftemp[7] += (trilocal[6] == 2) * (ftemp[5] - ftemp[7]);
				ftemp[8] += (trilocal[7] == 2) * (ftemp[6] - ftemp[8]);
			#endif

			hlocal[0] = ftemp[0] + (ftemp[1] + ftemp[2] + ftemp[3] + ftemp[4]) + (ftemp[5] + ftemp[6] + ftemp[7] + ftemp[8]);
			uxlocal = e * ((ftemp[1] - ftemp[3]) + (ftemp[5] - ftemp[6] - ftemp[7] + ftemp[8])) / hlocal[0];
			uylocal = e * ((ftemp[2] - ftemp[4]) + (ftemp[5] + ftemp[6] - ftemp[7] - ftemp[8])) / hlocal[0];

			h[i] = hlocal[0];

			gh = 1.5 * g * hlocal[0];
			usq = 1.5 * (uxlocal * uxlocal + uylocal * uylocal);
			ux3 = 3.0 * e * uxlocal;
			uy3 = 3.0 * e * uylocal;
			uxuy5 = ux3 + uy3;
			uxuy6 = uy3 - ux3;

			feq[0] = hlocal[0] - fact1 * hlocal[0] * (5.0 * gh + 4.0 * usq);
			feq[1] = fact1 * hlocal[0] * (gh + ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[2] = fact1 * hlocal[0] * (gh + uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[3] = fact1 * hlocal[0] * (gh - ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[4] = fact1 * hlocal[0] * (gh - uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[5] = fact2 * hlocal[0] * (gh + uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[6] = fact2 * hlocal[0] * (gh + uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			feq[7] = fact2 * hlocal[0] * (gh - uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[8] = fact2 * hlocal[0] * (gh - uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			for (j = 0; j < 9; j++)
				f2[i + j * size] = ftemp[j] - (ftemp[j] - feq[j]) / tau;
		}
	}
} 
#elif IN == 3
__global__ void LBMpull(int Lx, int Ly, prec g, prec e, prec tau,
	const prec* __restrict__ b, const unsigned char* __restrict__ Arr_tri, 
	const prec* __restrict__ f1, prec* f2, prec* h) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int size = Lx * Ly, j;
	prec ftemp[9], feq[9];
	prec uxlocal, uylocal;
	prec hlocal[9], blocal[9];
	prec gh, usq, ux3, uy3, uxuy5, uxuy6;
	prec fact1 = 1 / (9 * e*e);
	prec fact2 = fact1 * 0.25;
	prec factS = fact1 * 1.5;
	unsigned char trilocal[8]; 
	if (i < size) {			
		int check = 0;
		for (j = 1; j < 9; j++)
			trilocal[j-1] = Arr_tri[i + j * size];
		for (j = 0; j < 8; j++)
			check += trilocal[j];
		if (check != 0){
			int y = (int)i / Lx;
			int x = i - y * Lx;
			blocal[0] = b[i];
			blocal[1] = (             x != 0   ) ? b[i      - 1] : 0;
			blocal[2] = (y != 0                ) ? b[i - Lx    ] : 0;
			blocal[3] = (             x != Lx-1) ? b[i      + 1] : 0;
			blocal[4] = (y != Ly-1             ) ? b[i + Lx    ] : 0;
			blocal[5] = (y != 0    && x != 0   ) ? b[i - Lx - 1] : 0;
			blocal[6] = (y != 0    && x != Lx-1) ? b[i - Lx + 1] : 0;
			blocal[7] = (y != Ly-1 && x != Lx-1) ? b[i + Lx + 1] : 0; 
			blocal[8] = (y != Ly-1 && x != 0   ) ? b[i + Lx - 1] : 0;

			hlocal[0] = h[i];
			hlocal[1] = (             x != 0   ) ? h[i      - 1] : 0;
			hlocal[2] = (y != 0                ) ? h[i - Lx    ] : 0;
			hlocal[3] = (             x != Lx-1) ? h[i      + 1] : 0;
			hlocal[4] = (y != Ly-1             ) ? h[i + Lx    ] : 0;
			hlocal[5] = (y != 0    && x != 0   ) ? h[i - Lx - 1] : 0;
			hlocal[6] = (y != 0    && x != Lx-1) ? h[i - Lx + 1] : 0;
			hlocal[7] = (y != Ly-1 && x != Lx-1) ? h[i + Lx + 1] : 0;
			hlocal[8] = (y != Ly-1 && x != 0   ) ? h[i + Lx - 1] : 0;

			ftemp[1] = (             x != 0   ) ? f1[i      - 1 +     size] : 0;
			ftemp[2] = (y != 0                ) ? f1[i - Lx     + 2 * size] : 0;
			ftemp[3] = (             x != Lx-1) ? f1[i      + 1 + 3 * size] : 0;
			ftemp[4] = (y != Ly-1             ) ? f1[i + Lx     + 4 * size] : 0;
			ftemp[5] = (y != 0    && x != 0   ) ? f1[i - Lx - 1 + 5 * size] : 0;
			ftemp[6] = (y != 0    && x != Lx-1) ? f1[i - Lx + 1 + 6 * size] : 0;
			ftemp[7] = (y != Ly-1 && x != Lx-1) ? f1[i + Lx + 1 + 7 * size] : 0;
			ftemp[8] = (y != Ly-1 && x != 0   ) ? f1[i + Lx - 1 + 8 * size] : 0;

			ftemp[0] = f1[i]; 
			#if BN == 1
				if(trilocal[0] == 1) ftemp[1] = ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS; else ftemp[1] = f1[i +     size];
				if(trilocal[1] == 1) ftemp[2] = ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS; else ftemp[2] = f1[i + 2 * size];
				if(trilocal[2] == 1) ftemp[3] = ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS; else ftemp[3] = f1[i + 3 * size];
				if(trilocal[3] == 1) ftemp[4] = ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS; else ftemp[4] = f1[i + 4 * size];
				if(trilocal[4] == 1) ftemp[5] = ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25; else ftemp[5] = f1[i + 5 * size];
				if(trilocal[5] == 1) ftemp[6] = ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25; else ftemp[6] = f1[i + 6 * size];
				if(trilocal[6] == 1) ftemp[7] = ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25; else ftemp[7] = f1[i + 7 * size];
				if(trilocal[7] == 1) ftemp[8] = ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25; else ftemp[8] = f1[i + 8 * size];

				if(trilocal[0] == 2) ftemp[1] = ftemp[3];
				if(trilocal[1] == 2) ftemp[2] = ftemp[4];
				if(trilocal[2] == 2) ftemp[3] = ftemp[1];
				if(trilocal[3] == 2) ftemp[4] = ftemp[2];
				if(trilocal[4] == 2) ftemp[5] = ftemp[7];
				if(trilocal[5] == 2) ftemp[6] = ftemp[8];
				if(trilocal[6] == 2) ftemp[7] = ftemp[5];
				if(trilocal[7] == 2) ftemp[8] = ftemp[6];
			#elif BN == 2
				ftemp[1] = (trilocal[0] == 1) ? (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) : f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) ? (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) : f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) ? (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) : f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) ? (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) : f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) ? (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) : f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) ? (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) : f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) ? (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) : f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) ? (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) : f1[i + 8 * size];

				ftemp[1] = (trilocal[0] == 2) ? ftemp[3] : ftemp[1];
				ftemp[2] = (trilocal[1] == 2) ? ftemp[4] : ftemp[2];
				ftemp[3] = (trilocal[2] == 2) ? ftemp[1] : ftemp[3];
				ftemp[4] = (trilocal[3] == 2) ? ftemp[2] : ftemp[4];
				ftemp[5] = (trilocal[4] == 2) ? ftemp[7] : ftemp[5];
				ftemp[6] = (trilocal[5] == 2) ? ftemp[8] : ftemp[6];
				ftemp[7] = (trilocal[6] == 2) ? ftemp[5] : ftemp[7];
				ftemp[8] = (trilocal[7] == 2) ? ftemp[6] : ftemp[8];
			#else
				ftemp[1] = (trilocal[0] == 1) * (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) + (trilocal[0] != 1) * f1[i +     size];
				ftemp[2] = (trilocal[1] == 1) * (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) + (trilocal[1] != 1) * f1[i + 2 * size];
				ftemp[3] = (trilocal[2] == 1) * (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) + (trilocal[2] != 1) * f1[i + 3 * size];
				ftemp[4] = (trilocal[3] == 1) * (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) + (trilocal[3] != 1) * f1[i + 4 * size];
				ftemp[5] = (trilocal[4] == 1) * (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) + (trilocal[4] != 1) * f1[i + 5 * size];
				ftemp[6] = (trilocal[5] == 1) * (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) + (trilocal[5] != 1) * f1[i + 6 * size];
				ftemp[7] = (trilocal[6] == 1) * (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) + (trilocal[6] != 1) * f1[i + 7 * size];
				ftemp[8] = (trilocal[7] == 1) * (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) + (trilocal[7] != 1) * f1[i + 8 * size];

				ftemp[1] += (trilocal[0] == 2) * (ftemp[3] - ftemp[1]);
				ftemp[2] += (trilocal[1] == 2) * (ftemp[4] - ftemp[2]);
				ftemp[3] += (trilocal[2] == 2) * (ftemp[1] - ftemp[3]);
				ftemp[4] += (trilocal[3] == 2) * (ftemp[2] - ftemp[4]);
				ftemp[5] += (trilocal[4] == 2) * (ftemp[7] - ftemp[5]);
				ftemp[6] += (trilocal[5] == 2) * (ftemp[8] - ftemp[6]);
				ftemp[7] += (trilocal[6] == 2) * (ftemp[5] - ftemp[7]);
				ftemp[8] += (trilocal[7] == 2) * (ftemp[6] - ftemp[8]);
			#endif

			hlocal[0] = ftemp[0] + (ftemp[1] + ftemp[2] + ftemp[3] + ftemp[4]) + (ftemp[5] + ftemp[6] + ftemp[7] + ftemp[8]);
			uxlocal = e * ((ftemp[1] - ftemp[3]) + (ftemp[5] - ftemp[6] - ftemp[7] + ftemp[8])) / hlocal[0];
			uylocal = e * ((ftemp[2] - ftemp[4]) + (ftemp[5] + ftemp[6] - ftemp[7] - ftemp[8])) / hlocal[0];

			h[i] = hlocal[0];

			gh = 1.5 * g * hlocal[0];
			usq = 1.5 * (uxlocal * uxlocal + uylocal * uylocal);
			ux3 = 3.0 * e * uxlocal;
			uy3 = 3.0 * e * uylocal;
			uxuy5 = ux3 + uy3;
			uxuy6 = uy3 - ux3;

			feq[0] = hlocal[0] - fact1 * hlocal[0] * (5.0 * gh + 4.0 * usq);
			feq[1] = fact1 * hlocal[0] * (gh + ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[2] = fact1 * hlocal[0] * (gh + uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[3] = fact1 * hlocal[0] * (gh - ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[4] = fact1 * hlocal[0] * (gh - uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[5] = fact2 * hlocal[0] * (gh + uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[6] = fact2 * hlocal[0] * (gh + uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			feq[7] = fact2 * hlocal[0] * (gh - uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[8] = fact2 * hlocal[0] * (gh - uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			for (j = 0; j < 9; j++)
				f2[i + j * size] = ftemp[j] - (ftemp[j] - feq[j]) / tau;
		}
	}
} 
#else
__global__ void LBMpull(int Lx, int Ly, prec g, prec e, prec tau,
	const prec* __restrict__ b, const unsigned char* __restrict__ SC_bin, 
	const unsigned char* __restrict__ BB_bin, const prec* __restrict__ f1, 
	prec* f2, prec* h) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;			
	int size = Lx * Ly, j;
	prec ftemp[9], feq[9];
	prec uxlocal, uylocal;
	prec hlocal[9], blocal[9];
	prec gh, usq, ux3, uy3, uxuy5, uxuy6;
	prec fact1 = 1 / (9 * e*e);
	prec fact2 = fact1 * 0.25;
	prec factS = fact1 * 1.5;
	unsigned char SC,BB; 
	if (i < size) {
		SC = SC_bin[i];
		BB = BB_bin[i];
		if(SC + BB != 0){
			int y = (int)i / Lx;
			int x = i - y * Lx;
			blocal[0] = b[i];
			blocal[1] = (             x != 0   ) ? b[i      - 1] : 0;
			blocal[2] = (y != 0                ) ? b[i - Lx    ] : 0;
			blocal[3] = (             x != Lx-1) ? b[i      + 1] : 0;
			blocal[4] = (y != Ly-1             ) ? b[i + Lx    ] : 0;
			blocal[5] = (y != 0    && x != 0   ) ? b[i - Lx - 1] : 0;
			blocal[6] = (y != 0    && x != Lx-1) ? b[i - Lx + 1] : 0;
			blocal[7] = (y != Ly-1 && x != Lx-1) ? b[i + Lx + 1] : 0; 
			blocal[8] = (y != Ly-1 && x != 0   ) ? b[i + Lx - 1] : 0;

			hlocal[0] = h[i];
			hlocal[1] = (             x != 0   ) ? h[i      - 1] : 0;
			hlocal[2] = (y != 0                ) ? h[i - Lx    ] : 0;
			hlocal[3] = (             x != Lx-1) ? h[i      + 1] : 0; 
			hlocal[4] = (y != Ly-1             ) ? h[i + Lx    ] : 0;
			hlocal[5] = (y != 0    && x != 0   ) ? h[i - Lx - 1] : 0;
			hlocal[6] = (y != 0    && x != Lx-1) ? h[i - Lx + 1] : 0;
			hlocal[7] = (y != Ly-1 && x != Lx-1) ? h[i + Lx + 1] : 0;
			hlocal[8] = (y != Ly-1 && x != 0   ) ? h[i + Lx - 1] : 0;

			ftemp[1] = (             x != 0   ) ? f1[i      - 1 +     size] : 0;
			ftemp[2] = (y != 0                ) ? f1[i - Lx     + 2 * size] : 0;
			ftemp[3] = (             x != Lx-1) ? f1[i      + 1 + 3 * size] : 0;
			ftemp[4] = (y != Ly-1             ) ? f1[i + Lx     + 4 * size] : 0;
			ftemp[5] = (y != 0    && x != 0   ) ? f1[i - Lx - 1 + 5 * size] : 0;
			ftemp[6] = (y != 0    && x != Lx-1) ? f1[i - Lx + 1 + 6 * size] : 0;
			ftemp[7] = (y != Ly-1 && x != Lx-1) ? f1[i + Lx + 1 + 7 * size] : 0;
			ftemp[8] = (y != Ly-1 && x != 0   ) ? f1[i + Lx - 1 + 8 * size] : 0;

			ftemp[0] = f1[i]; 
			#if BN == 1
				if((SC>>0) & 1) ftemp[1] = ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS; else ftemp[1] = f1[i +     size];
				if((SC>>1) & 1) ftemp[2] = ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS; else ftemp[2] = f1[i + 2 * size];
				if((SC>>2) & 1) ftemp[3] = ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS; else ftemp[3] = f1[i + 3 * size];
				if((SC>>3) & 1) ftemp[4] = ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS; else ftemp[4] = f1[i + 4 * size];
				if((SC>>4) & 1) ftemp[5] = ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25; else ftemp[5] = f1[i + 5 * size];
				if((SC>>5) & 1) ftemp[6] = ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25; else ftemp[6] = f1[i + 6 * size];
				if((SC>>6) & 1) ftemp[7] = ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25; else ftemp[7] = f1[i + 7 * size];
				if((SC>>7) & 1) ftemp[8] = ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25; else ftemp[8] = f1[i + 8 * size];

				if((BB>>(0)) & 1) ftemp[1] = ftemp[3];
				if((BB>>(1)) & 1) ftemp[2] = ftemp[4];
				if((BB>>(2)) & 1) ftemp[3] = ftemp[1];
				if((BB>>(3)) & 1) ftemp[4] = ftemp[2];
				if((BB>>(4)) & 1) ftemp[5] = ftemp[7];
				if((BB>>(5)) & 1) ftemp[6] = ftemp[8];
				if((BB>>(6)) & 1) ftemp[7] = ftemp[5];
				if((BB>>(7)) & 1) ftemp[8] = ftemp[6];
			#elif BN == 2
				ftemp[1] = ((SC>>0) & 1) ? (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS) : f1[i +     size];
				ftemp[2] = ((SC>>1) & 1) ? (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS) : f1[i + 2 * size];
				ftemp[3] = ((SC>>2) & 1) ? (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS) : f1[i + 3 * size];
				ftemp[4] = ((SC>>3) & 1) ? (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS) : f1[i + 4 * size];
				ftemp[5] = ((SC>>4) & 1) ? (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) : f1[i + 5 * size];
				ftemp[6] = ((SC>>5) & 1) ? (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) : f1[i + 6 * size];
				ftemp[7] = ((SC>>6) & 1) ? (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) : f1[i + 7 * size];
				ftemp[8] = ((SC>>7) & 1) ? (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) : f1[i + 8 * size];

				ftemp[1] = ((BB>>(0)) & 1) ? ftemp[3] : ftemp[1];
				ftemp[2] = ((BB>>(1)) & 1) ? ftemp[4] : ftemp[2];
				ftemp[3] = ((BB>>(2)) & 1) ? ftemp[1] : ftemp[3];
				ftemp[4] = ((BB>>(3)) & 1) ? ftemp[2] : ftemp[4];
				ftemp[5] = ((BB>>(4)) & 1) ? ftemp[7] : ftemp[5];
				ftemp[6] = ((BB>>(5)) & 1) ? ftemp[8] : ftemp[6];
				ftemp[7] = ((BB>>(6)) & 1) ? ftemp[5] : ftemp[7];
				ftemp[8] = ((BB>>(7)) & 1) ? ftemp[6] : ftemp[8]; 
			#else
				//int x = i%Lx, y = i/Lx;
				ftemp[1] = ((SC>>0) & 1) * (ftemp[1] - g * (hlocal[0] + hlocal[1]) * (blocal[0] - blocal[1]) * factS       ) + !((SC>>0) & 1) * f1[i +     size];
				ftemp[2] = ((SC>>1) & 1) * (ftemp[2] - g * (hlocal[0] + hlocal[2]) * (blocal[0] - blocal[2]) * factS       ) + !((SC>>1) & 1) * f1[i + 2 * size];
				ftemp[3] = ((SC>>2) & 1) * (ftemp[3] - g * (hlocal[0] + hlocal[3]) * (blocal[0] - blocal[3]) * factS       ) + !((SC>>2) & 1) * f1[i + 3 * size];
				ftemp[4] = ((SC>>3) & 1) * (ftemp[4] - g * (hlocal[0] + hlocal[4]) * (blocal[0] - blocal[4]) * factS       ) + !((SC>>3) & 1) * f1[i + 4 * size];
				ftemp[5] = ((SC>>4) & 1) * (ftemp[5] - g * (hlocal[0] + hlocal[5]) * (blocal[0] - blocal[5]) * factS * 0.25) + !((SC>>4) & 1) * f1[i + 5 * size];
				ftemp[6] = ((SC>>5) & 1) * (ftemp[6] - g * (hlocal[0] + hlocal[6]) * (blocal[0] - blocal[6]) * factS * 0.25) + !((SC>>5) & 1) * f1[i + 6 * size];
				ftemp[7] = ((SC>>6) & 1) * (ftemp[7] - g * (hlocal[0] + hlocal[7]) * (blocal[0] - blocal[7]) * factS * 0.25) + !((SC>>6) & 1) * f1[i + 7 * size];
				ftemp[8] = ((SC>>7) & 1) * (ftemp[8] - g * (hlocal[0] + hlocal[8]) * (blocal[0] - blocal[8]) * factS * 0.25) + !((SC>>7) & 1) * f1[i + 8 * size];

				ftemp[1] += ((BB>>(0)) & 1) * (ftemp[3] - ftemp[1]);
				ftemp[2] += ((BB>>(1)) & 1) * (ftemp[4] - ftemp[2]);
				ftemp[3] += ((BB>>(2)) & 1) * (ftemp[1] - ftemp[3]);
				ftemp[4] += ((BB>>(3)) & 1) * (ftemp[2] - ftemp[4]); 
				ftemp[5] += ((BB>>(4)) & 1) * (ftemp[7] - ftemp[5]);
				ftemp[6] += ((BB>>(5)) & 1) * (ftemp[8] - ftemp[6]);
				ftemp[7] += ((BB>>(6)) & 1) * (ftemp[5] - ftemp[7]);
				ftemp[8] += ((BB>>(7)) & 1) * (ftemp[6] - ftemp[8]);
			#endif

			hlocal[0] = ftemp[0] + (ftemp[1] + ftemp[2] + ftemp[3] + ftemp[4]) + (ftemp[5] + ftemp[6] + ftemp[7] + ftemp[8]);
			uxlocal = e * ((ftemp[1] - ftemp[3]) + (ftemp[5] - ftemp[6] - ftemp[7] + ftemp[8])) / hlocal[0];
			uylocal = e * ((ftemp[2] - ftemp[4]) + (ftemp[5] + ftemp[6] - ftemp[7] - ftemp[8])) / hlocal[0];

			h[i] = hlocal[0];

			gh = 1.5 * g * hlocal[0];
			usq = 1.5 * (uxlocal * uxlocal + uylocal * uylocal);
			ux3 = 3.0 * e * uxlocal;
			uy3 = 3.0 * e * uylocal;
			uxuy5 = ux3 + uy3;
			uxuy6 = uy3 - ux3;

			feq[0] = hlocal[0] - fact1 * hlocal[0] * (5.0 * gh + 4.0 * usq);
			feq[1] = fact1 * hlocal[0] * (gh + ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[2] = fact1 * hlocal[0] * (gh + uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[3] = fact1 * hlocal[0] * (gh - ux3 + 0.5 * ux3*ux3 * 9 * fact1 - usq);
			feq[4] = fact1 * hlocal[0] * (gh - uy3 + 0.5 * uy3*uy3 * 9 * fact1 - usq);
			feq[5] = fact2 * hlocal[0] * (gh + uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[6] = fact2 * hlocal[0] * (gh + uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			feq[7] = fact2 * hlocal[0] * (gh - uxuy5 + 0.5 * uxuy5*uxuy5 * 9 * fact1 - usq);
			feq[8] = fact2 * hlocal[0] * (gh - uxuy6 + 0.5 * uxuy6*uxuy6 * 9 * fact1 - usq);
			for (j = 0; j < 9; j++)
				f2[i + j * size] = ftemp[j] - (ftemp[j] - feq[j]) / tau;
		}
	} 
} 
#endif

__global__ void feqKernel(int Lx, int Ly, prec g, prec e,
	const prec* __restrict__ h, prec* f) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < Lx*Ly) {
		prec hi = h[i];
		prec gh1 = g * hi * hi / (6.0 * e * e);
		prec gh2 = gh1 / 4;
		f[i] = hi - 5.0 * gh1;
		f[i + (    Lx*Ly)] = gh1;
		f[i + (2 * Lx*Ly)] = gh1;
		f[i + (3 * Lx*Ly)] = gh1;
		f[i + (4 * Lx*Ly)] = gh1;
		f[i + (5 * Lx*Ly)] = gh2;
		f[i + (6 * Lx*Ly)] = gh2;
		f[i + (7 * Lx*Ly)] = gh2;
		f[i + (8 * Lx*Ly)] = gh2;
	}
}

__global__ void TSkernel(prec* TSdata, const prec* __restrict__ w,
	const int* __restrict__ TSind, int t, int deltaTS, int NTS, int TTS) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < NTS) {
		int n = t / deltaTS;
		TSdata[i*TTS + n] = w[TSind[i]];
	}
}

 
void LBMTimeStep(mainDStruct devi, cudaStruct devEx, int t, int deltaTS, cudaEvent_t ct1, cudaEvent_t ct2, prec *msecs) {
	float dt;

	if (t % 2 == 0){
		cudaEventRecord(ct1);
		#if IN == 1
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.f1, devEx.f2, devEx.h);
		#elif IN == 2
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devi.node_types, devEx.f1, devEx.f2, devEx.h);
		#elif IN == 3
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.Arr_tri, devEx.f1, devEx.f2, devEx.h);
		#else
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.SC_bin, devEx.BB_bin, devEx.f1, devEx.f2, devEx.h);
		#endif
	}
	else{
		cudaEventRecord(ct1);
		#if IN == 1
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.f2, devEx.f1, devEx.h);
		#elif IN == 2
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devi.node_types, devEx.f2, devEx.f1, devEx.h);
		#elif IN == 3
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.Arr_tri, devEx.f2, devEx.f1, devEx.h);
		#else
			LBMpull << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.tau,
			devi.b, devEx.SC_bin, devEx.BB_bin, devEx.f2, devEx.f1, devEx.h);
		#endif
	}

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	*msecs += dt;

	if (t%deltaTS == 0) {
		wKernel << <devi.Ngrid, devi.Nblocks >> >(devi.Lx, devi.Ly, devEx.h, devi.b, devi.w);
		TSkernel << <devi.NTS, 1 >> > (devi.TSdata, devi.w, devi.TSind, t, deltaTS, devi.NTS, devi.TTS);
	}
}

void setup(mainDStruct devi, cudaStruct devEx, int deltaTS) {
	#if IN == 3
		auxArraysKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.ex, devEx.ey, devi.node_types,
		devEx.Arr_tri);
	#elif IN == 4
		auxArraysKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.ex, devEx.ey, devi.node_types,
		devEx.SC_bin, devEx.BB_bin);
	#endif
	hKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devi.w, devi.b, devEx.h);

	feqKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.g, devEx.e, devEx.h, devEx.f1);

	TSkernel << <devi.NTS, 1 >> > (devi.TSdata, devi.w, devi.TSind, 0, deltaTS, devi.NTS, devi.TTS);
}

void copyAndWriteResultData(mainHStruct host, mainDStruct devi, cudaStruct devEx, int t, std::string outputdir) {

	wKernel << <devi.Ngrid, devi.Nblocks >> > (devi.Lx, devi.Ly, devEx.h, devi.b, devi.w);

	cudaMemcpy(host.w, devi.w, devi.Lx*devi.Ly * sizeof(prec), cudaMemcpyDeviceToHost);

	writeOutput(devi.Lx*devi.Ly, t, host.w, outputdir);
}

void copyAndWriteTSData(mainHStruct host, mainDStruct devi, int deltaTS, prec Dt, std::string outputdir) {

	cudaMemcpy(host.TSdata, devi.TSdata, devi.TTS*devi.NTS * sizeof(prec), cudaMemcpyDeviceToHost);

	writeTS(devi.TTS, devi.NTS, deltaTS, Dt, host.TSdata, outputdir);
}

void LBM(mainHStruct host, mainDStruct devi, cudaStruct devEx, int* time_array, prec Dt, std::string outputdir) {
	cudaFuncSetCacheConfig(LBMpull, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(feqKernel, cudaFuncCachePreferL1);

	int tMax = time_array[0];
	int deltaOutput = time_array[1];
	int deltaTS = time_array[2];
	int t = 0;
	cudaEvent_t ct1, ct2;
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	prec msecs = 0;
	setup(devi, devEx, deltaTS);
	std::cout << std::fixed << std::setprecision(1);
	while (t <= tMax) {
		LBMTimeStep(devi, devEx, t, deltaTS, ct1, ct2, &msecs);
		t++;
		if (deltaOutput != 0 && t%deltaOutput == 0) {
			std::cout << "\rTime step: " << t << " (" << 100.0*t / tMax << "%)";
			copyAndWriteResultData(host, devi, devEx, t, outputdir);
		}
	}
	copyAndWriteResultData(host, devi, devEx, t, outputdir);
	copyAndWriteTSData(host, devi, deltaTS, Dt, outputdir);
	std::cout << std::endl << "Tiempo total: " << msecs << "[ms]" << std::endl;
	std::cout << std::endl << "Tiempo promedio por iteracion: " << msecs / tMax << "[ms]" << std::endl;
}

