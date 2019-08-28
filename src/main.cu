#include <string>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "include/structs.h"
#include "cpp/include/files.h"
#include "cu/include/LBM.cuh"
#include <time.h>
#include <sys/types.h> 
#include <sys/stat.h>

#if defined(_WIN32)  
#include <direct.h> 
#endif

void freemem(mainHStruct host, mainDStruct devi, cudaStruct devEx) {
	/*delete[] host.b;
	delete[] host.w;
	delete[] host.ux;
	delete[] host.uy;
	delete[] host.node_types;
	*/
	cudaFree(devi.b);
	cudaFree(devi.w);
	cudaFree(devi.node_types);
	cudaFree(devi.TSind);
	cudaFree(devi.TSdata);

	cudaFree(devEx.ex);
	cudaFree(devEx.ey);
	cudaFree(devEx.h);
	cudaFree(devEx.f1);
	cudaFree(devEx.f2);
	#if IN == 3
		cudaFree(devEx.Arr_tri);
	#elif IN == 4
		cudaFree(devEx.SC_bin);
		cudaFree(devEx.BB_bin);
	#endif
}

void getTSIndex(int* TSind, prec* TSx, prec* TSy, prec x0, prec y0,
	int* bb, int Lx, int Ly, prec Dx, int NTS) {
	int k, min_i, min_x, min_y;
	std::cout << "TS nodes located in dry zones:" << std::endl;
	std::cout << std::fixed << std::setprecision(2);
	for (k = 0; k < NTS; k++) {
		std::cout << "Node at (x = " << TSx[k] << ", y = " << TSy[k] << "). ";
		min_x = (int)(TSx[k] - x0) / Dx;
		min_y = (int)(TSy[k] - y0) / Dx;
		min_i = min_x + min_y * Lx;
		while(bb[min_i] == 0){
			std::cout << "Nearest node type: " << bb[min_i] << std::endl;
			min_i = min_i - 1;
		}
		std::cout << "Using the nearest submerged node at(x = " << x0 + Dx*min_x << ", y = " << y0 + Dx*min_y << ")." << std::endl;
		TSind[k] = min_i;
	}
}

int dirExists(const char *path) {
	struct stat info;

	if (stat(path, &info) != 0)
		return 0;
	else if (info.st_mode & S_IFDIR)
		return 1;
	else
		return 0;
}

int main(int argc, char* argv[]) {
	if (argv == NULL || argc < 2) {
		std::cout << "Please specify arguments!" << std::endl;
		exit(EXIT_FAILURE);
	}
	int time_array[3], Lx, Ly, NTS, Nblocks;
	prec Dx, x0, y0, tau, g, Dt;

	std::string scenario;
	std::string test;
	std::string dir;

	readConf(dir, scenario, test, time_array, &tau, &g, &Dt, &Nblocks, argv[1]);

	test = scenario + "_" + test;
	std::string outputdir = dir + "Outputs/outputs_";
	std::string inputdir = dir + "../Inputs/";

	outputdir.append(test);
	std::string outputdir_temp = outputdir;
	int c = 1;
	std::string cstr = "";
	while (dirExists(outputdir_temp.c_str())) {
		cstr = static_cast<std::ostringstream*>(&(std::ostringstream() << c))->str();
		outputdir_temp = outputdir + "_" + cstr;
		c++;
	}
	outputdir = outputdir_temp;
	#if defined(_WIN32)  
		_mkdir(outputdir.c_str());
	#else
		mkdir(outputdir.c_str(), 0733);
	#endif
	std::cout << "Created directory: " << outputdir << std::endl;

	mainHStruct host;
	mainDStruct devi;
	cudaStruct devEx;

	readInput(&host.b, &host.w, &host.node_types, test, inputdir, &Lx, &Ly, &Dx, &x0, &y0);

	prec *TSx, *TSy;
	readTSloc(&TSx, &TSy, &NTS, scenario, inputdir);
	int TTS = int(ceil((prec)time_array[0] / (prec)time_array[2]));
	host.TSdata = new prec[TTS * NTS];
	host.TSind = new int[NTS];
	getTSIndex(host.TSind, TSx, TSy, x0, y0, host.node_types, Lx, Ly, Dx, NTS);

	uint num_bytes_d = Lx * Ly * sizeof(prec);
	uint num_bytes_i = Lx * Ly * sizeof(int);
	int Ngrid = int(ceil((prec)Lx * (prec)Ly / (prec)Nblocks));
	int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
	int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
	prec e = Dx / Dt;

	writeConf(Lx, Ly, tau, Dx, Dt, outputdir);
	writeOutput(Lx*Ly, 0, host.w, outputdir);

	devi.Lx = Lx;
	devi.Ly = Ly;
	devi.NTS = NTS;
	devi.TTS = TTS;
	devi.Nblocks = Nblocks;
	devi.Ngrid = Ngrid;

	cudaMalloc((void**)&devi.w, num_bytes_d); 
	cudaMalloc((void**)&devi.b, num_bytes_d);
	cudaMalloc((void**)&devi.node_types, num_bytes_i);
	cudaMalloc((void**)&devi.TSdata, TTS * NTS * sizeof(prec));
	cudaMalloc((void**)&devi.TSind, NTS * sizeof(int));

	cudaMemcpy(devi.b, host.b, num_bytes_d, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.w, host.w, num_bytes_d, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.node_types, host.node_types, num_bytes_i, cudaMemcpyHostToDevice);
	cudaMemcpy(devi.TSind, host.TSind, NTS * sizeof(int), cudaMemcpyHostToDevice);

	devEx.tau = tau;
	devEx.g = g;
	devEx.e = e;
	cudaMalloc((void**)&devEx.ex, 9 * sizeof(int));
	cudaMalloc((void**)&devEx.ey, 9 * sizeof(int));
	cudaMalloc((void**)&devEx.h, num_bytes_d);
	cudaMalloc((void**)&devEx.f1, 9 * num_bytes_d);
	cudaMalloc((void**)&devEx.f2, 9 * num_bytes_d);
	#if IN == 3
		cudaMalloc((void**)&devEx.Arr_tri, 9 * Lx * Ly * sizeof(unsigned char));
	#elif IN == 4
		cudaMalloc((void**)&devEx.SC_bin, Lx * Ly * sizeof(unsigned char));
		cudaMalloc((void**)&devEx.BB_bin, Lx * Ly * sizeof(unsigned char));
	#endif

	cudaMemcpy(devEx.ex, ex, 9 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devEx.ey, ey, 9 * sizeof(int), cudaMemcpyHostToDevice);
	clock_t t1, t2; 
	std::cout << "\nStart\n";
	t1 = clock();
	LBM(host, devi, devEx, time_array, Dt, outputdir);
	t2 = clock();

	std::cout << std::endl << "Tiempo total: " << 1000.0 * (prec)(t2 - t1) / CLOCKS_PER_SEC << "[ms]" << std::endl;

	freemem(host, devi, devEx);
	return 0;
} 

