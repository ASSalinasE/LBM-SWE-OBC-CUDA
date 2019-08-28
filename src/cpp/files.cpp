#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "include/files.h"
#include "../include/structs.h"

void readConf(std::string& dir, std::string& scenario,
	std::string& test, int *timearray, prec *tau,
	prec *g, prec *Dt, int *Nblocks, std::string file) {
	std::ifstream myfile;
	myfile.open(file.c_str(), std::ios::in);
	if (!myfile.is_open()) {
		std::cout << "Config file doesn't exist." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Reading configuration from " << file << std::endl;
	std::string skip;
	myfile >> skip >> skip >> dir;
	myfile >> skip >> skip >> scenario;
	myfile >> skip >> skip >> test;
	myfile >> skip >> skip >> timearray[0];
	myfile >> skip >> skip >> timearray[1];
	myfile >> skip >> skip >> timearray[2];
	myfile >> skip >> skip >> *tau;
	myfile >> skip >> skip >> *g;
	myfile >> skip >> skip >> *Dt;
	myfile >> skip >> skip >> *Nblocks;
	myfile.close();
}

prec stod(char* word, int len) {
	prec val = 0, ord = 1E-16;
	int dig;
	for (int i = len-1; i >= 0; i--) {
		if (word[i] != '.') { 
			dig = word[i] - '0';
			val += dig * ord;
			ord *= 10;
		}
	}
	return val;
}

void readInput(prec** b, prec** w,
	int** node_types, std::string test, std::string inputdir,
	int *Lx, int *Ly, prec *Dx, prec* x0, prec* y0) {
	FILE *fp;
	std::string fullfile = inputdir + test + ".txt"; 
	if ((fp = fopen(fullfile.c_str(), "r")) == NULL) {
		std::cout << "Input file doesn't exist." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Reading input from " << fullfile << std::endl;
	#if PREC==64
		fscanf(fp, "%d %d %lf %lf %lf\n", Lx, Ly, Dx, x0, y0);
	#else
		fscanf(fp, "%d %d %f %f %f\n", Lx, Ly, Dx, x0, y0);
	#endif

	prec* bl = new prec[(*Lx)*(*Ly)];
	prec* wl = new prec[(*Lx)*(*Ly)];
	int* node_typesl = new int[(*Lx)*(*Ly)];
	int wc = 0, bc = 0, len = 0, buflen;
	prec val;
	char buffer[FILE_BATCH_SIZE], word[50];
	buffer[FILE_BATCH_SIZE - 1] = '\0';
	while (wc < (*Lx)*(*Ly) || wc < (*Lx)*(*Ly)) {
		buflen = fread(buffer, 1, FILE_BATCH_SIZE - 1, fp);
		for (int i = 0; i <= buflen; i++) {
			if (buffer[i] == ' ' || buffer[i] == '\n' || buffer[i] == '\r') {
				word[len] = '\0';
				if (len == 1)
					node_typesl[wc - 1] = word[0] - '0';
				else if (len > 1) {
					val = stod(word,len);
					if (wc == bc) {
						bl[bc] = val;
						bc++;
					}
					else {
						wl[wc] = val;
						wc++;
					}
				}
				len = 0;
			}
			else {
				word[len] = buffer[i];
				len++;
			}
		}
		if (buflen != FILE_BATCH_SIZE - 1)
			break;
		if (buffer[buflen-1] != ' ' && buffer[buflen-1] != '\n' && buffer[buflen-1] != '\r')
			fseek(fp, 1-len, SEEK_CUR);
		len = 0;
	}
	fclose(fp);
	*w = wl;
	*b = bl;
	*node_types = node_typesl;
}

void readTSloc(prec** x, prec** y, int* NTS, std::string scenario, std::string inputdir) {
	std::ifstream myfile;
	std::string fullfile = inputdir + "TS/" + scenario + ".txt";
	myfile.open(fullfile.c_str(), std::ios::in);
	if (!myfile.is_open()) {
		std::cout << "Time series locations file doesn't exist." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Reading time series location from " << fullfile << std::endl;
	myfile >> *NTS;

	prec* xl = new prec[(*NTS)];
	prec* yl = new prec[(*NTS)];

	for (int i = 0; i < (*NTS); i++) {
		myfile >> xl[i] >> yl[i];
	}
	myfile.close();
	*x = xl;
	*y = yl;
}

void writeOutput(int L, int t, prec* w, std::string outputdir) {
	FILE *fp;
	std::ostringstream numero; 
	numero << std::setw(5) << std::setfill('0') << std::right << (t);
	std::string fullfile = outputdir + "/output_" + numero.str() + ".dat";
	if ((fp = fopen(fullfile.c_str(), "wb")) == NULL) {
		std::cout << "Can't create output file." << std::endl;
		exit(EXIT_FAILURE);
	}
	fwrite(&w[0], sizeof(prec), L, fp);
	fclose(fp);
}

void writeConf(int Lx, int Ly, prec tau, prec Dx, prec Dt,
	std::string outputdir) {
	std::ofstream myfile;
	std::string fullfile = outputdir + "/config.txt";
	myfile.open(fullfile.c_str(), std::ios_base::out);
	myfile.precision(16);
	myfile.setf(std::ios::scientific);
	myfile << Lx << " " << Ly << " " << tau << " " << Dx << " " << Dt << std::endl;
	myfile.close();
}

void writeTS(int TTS, int NTS, int deltaTS, prec Dt, prec* TSdata, std::string outputdir) {
	std::ofstream myfile;
	std::string fullfile = outputdir + "/TSoutput.txt";
	myfile.open(fullfile.c_str(), std::ios_base::out);
	myfile.precision(16);
	myfile.setf(std::ios::scientific);
	int i, j;
	for (j = 0; j < TTS; j++) {
		myfile << j * deltaTS*Dt << " ";
	}
	myfile << std::endl;
	for (i = 0; i < NTS; i++) {
		myfile << TSdata[i*TTS];
		for (j = 1; j < TTS; j++) {
			myfile << " " << TSdata[i*TTS + j];
		}
		myfile << std::endl;
	}
	myfile.close();
}
