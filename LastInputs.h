#ifndef LASTINPUTS
#define LASTINPUTS

#include "repast_hpc/Point.h"
#include <stdio.h>

typedef struct LastInputs LastInputs;

struct LastInputs{

	int sumTeam1Contacts;
	int sumTeam2Contacts;
	int Occupant;
	int willPandwillT0;
	int willPandwillT1;
	int qOccupant;

	//make constuctors for the struct
	LastInputs();
	LastInputs(int _sumTeam1Contacts, int _sumTeam2Contacts,  int _Occupant, int _willPandwillT0, int _willPandwillT1, int _qOccupant);

	 /* For archive packaging */
	    template<class Archive>
	    void serialize(Archive &ar, const unsigned int version){
	        ar & sumTeam1Contacts;
	        ar & sumTeam2Contacts;
	        ar & Occupant;
	        ar & willPandwillT0;
	        ar & willPandwillT1;
	        ar & qOccupant;

	    }

	    LastInputs& operator=(const LastInputs& li);
	    bool operator==(const LastInputs& li);
	    bool operator!=(const LastInputs& li);



};

#endif
