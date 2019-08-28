#ifndef FILES_HH
#define FILES_HH

#include "../../include/structs.h"
#include <string>

void readConf(std::string&, std::string&, std::string&, int*,
	prec*, prec*, prec*, int*, std::string);

void readInput(prec**, prec**, int**, std::string, std::string, 
	int*, int*, prec*, prec*, prec*);

void readTSloc(prec**, prec**, int*, std::string, std::string);

void writeOutput(int, int, prec*, std::string);

void writeConf(int, int, prec, prec, prec, std::string);

void writeTS(int, int, int, prec, prec*, std::string);

#endif