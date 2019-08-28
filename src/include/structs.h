#ifndef STRUCTS_H
#define STRUCTS_H

#define FILE_BATCH_SIZE 1000000
#ifndef PREC  
#define PREC 64
#endif
#ifndef IN
#define IN 1
#endif
#ifndef BN
#define BN 1
#endif
#if PREC==64
	typedef double prec;
#else
	typedef float prec;
#endif

typedef struct mainHStruct {
	int* node_types;
	prec* b;
	prec* w;
	int* TSind;
	prec* TSdata;
} mainHStruct;

typedef struct mainDStruct {
	int Lx;
	int Ly;
	int NTS;
	int TTS;
	int Nblocks;
	int Ngrid;
	int* node_types;
	prec* b;
	prec* w; 
	int* TSind;
	prec* TSdata;
} mainDStruct;

typedef struct cudaStruct {
	prec tau;
	prec g;
	prec e;
	int* ex;
	int* ey;
	#if IN == 3
		unsigned char* Arr_tri;
	#elif IN == 4
		unsigned char* SC_bin;
		unsigned char* BB_bin;
	#endif
	prec* h;
	prec* f1;
	prec* f2;
} cudaStruct;

#endif
