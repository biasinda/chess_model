//chessModel.h header file for chessModel


#ifndef CHESS_MODEL
#define CHESS_MODEL

#include "repast_hpc/Schedule.h"
#include "repast_hpc/Properties.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedSpaces.h"
#include "repast_hpc/Spaces.h"
#include "repast_hpc/Point.h"
#include "repast_hpc/AgentRequest.h"
#include "repast_hpc/Grid.h"
#include "repast_hpc/Context.h"

//display includes
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <boost/mpi.hpp>  //include boost library

#include "algorithm"
//HEADER

#include "Box.h"
#include "PayONPos.h"
#include "movePackage.h"


//#include "chessbox.h"
//#include "chessboard.h"

#include <iostream>
#include <fstream>


/* Agent Package Provider */
class BoxPackageProvider {
	
private:
    repast::SharedContext<Box>* boxs;
	
public:
	
    BoxPackageProvider(repast::SharedContext<Box>* boxPtr);
	
    void providePackage(Box * box, std::vector<BoxPackage>& out);
	
    void provideContent(repast::AgentRequest req, std::vector<BoxPackage>& out);
	
};

/* Agent Package Receiver */
class BoxPackageReceiver {
	
private:
    repast::SharedContext<Box>* boxs;
	
public:
	
    BoxPackageReceiver(repast::SharedContext<Box>* boxPtr);
	
    Box * createAgent(BoxPackage package);
	
    void updateAgent(BoxPackage package);
	
};




class chessHPCModel{


	int rank;
	int originX;
	int originY;
	int stopAt;
	int countOfBoxes;

	int nProcInX;
	int nProcInY;

	//int that keeps track of which team has move
	int teamToMove;

	int king1isChecked;
	int king2isChecked;

	//learning rate alpha
	double alpha;

	//discounting gamma
	double gamma;


	repast::Properties* props;

	//context, used between processes, changed to a pointer so dont have to instantiate
	repast::SharedContext<Box>* context;
	//repast::SharedContext<InterProcessAgent>* iProcContext;


	//making the grid for the chessboard

	repast::SharedSpaces<Box>::SharedStrictDiscreteSpace* grid; //using typedef SharedDiscreteSpace<T, StrictBorders, SimpleAdder<T> >

	//tried to make grid on only one process...
	//repast::Spaces<Box>::SingleStrictDiscreteSpace* grid;

	BoxPackageProvider* provider;
	BoxPackageReceiver* receiver;

	//InterProcessAgentPackageProvider* IPAprovider;
	//InterProcessAgentPackageReceiver* IPAreceiver;

	//char** board;




public:



	chessHPCModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm);
	~chessHPCModel();
	void init(int gamesPlayed);
	void playGame();
	void playMove();
	void initSchedule(repast::ScheduleRunner& runner);
	void recordResults();
	void synchAgents();

	void addBoxesToChessBoard();

	void normalizeWeights(int occupant);

	void setupBoard();
	//method that returns the max payoff and position for a piece, returns one move (the one with the highest payoff) and its payoff as a PayPos struct
	std::vector<PayONPos> getAllMovesPayoffs(Box* box);

		//used in getMaxPayoffand Position,
		//method that returns what pieces I will protect and what pieces I will threaten in next move
		//will return in pos0 the pieces it will protect in the next turn and in pos1 the pieces it will threaten
		std::vector<int> getWillThreatandProt(int team, repast::Point<int> move,int qOccupant, repast::Point<int> currentLocation);

		PayONPos getRandomMove();


		//method to select move to take for a team based on payoff
		std::vector<PayONPos> getMoves();

		PayONPos selectMove();

     //method to actually make move given a PayONPos
     void makeMove(PayONPos bestMove);
     void undoLastMove(PayONPos lastMove);

     //ssetupContacts method, used too initialize contacts in beginning
     //(only used once)
     void setupContacts();

     //setBox method to initalize board
     void setBox(repast::Point<int> pointToSet, int occupantToSet, int teamToSet);

     //update the contacts
     void updateOldContacts(PayONPos bestMove);
     void updateNewContacts(PayONPos bestMove);

     //method to getLegalMoves for any box, will return a movePackage struct
     movePackage getLegalMoves(Box* box);

     	 //method that returns all pawn moves, depends on team because pawn moves depend on team
     	 movePackage getPawnMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getPawnProtects(int team, repast::Point<int> currentLocation);

     	 movePackage getKnightMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getKnightProtects(int team, repast::Point<int> currentLocation);

     	 movePackage getBishopMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getBishopProtects(int team, repast::Point<int> currentLocation);

     	 movePackage getRookMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getRookProtects(int team, repast::Point<int> currentLocation);

     	 movePackage getQueenMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getQueenProtects(int team, repast::Point<int> currentLocation);

     	 movePackage getKingMoves(int team, repast::Point<int> currentLocation);
     	 movePackage getKingProtects(int team, repast::Point<int> currentLocation);


     //method that checks if king is checked.
     //If yes, can he be saved? if yes move him, if no end game and declare winner
     void victoryCondition();

     	 int victoryCheck();

     //method to share agents across processes
     void requestIPAAgents();
     void requestAllAgents();

     void cancelAgentRequests();

     //method to display board
     void displayBoard0();

     void displayBoard1();

     void setBoard(int occupant,int team,repast::Point<int> position);


     //method to check if the move i am trying to make is on my process, if it isnt
     //request that agent from the other process.
     void moveRequest(repast::Point<int> move);

     void synchStates();

     void readPawnWeights();
     void readKnightWeights();
     void readBishopWeights();
     void readRookWeights();
     void readQueenWeights();
     void readKingWeights();



     void writeRandomWeights(int team);
		 void writeRandomPawnWeights(int team);
		 void writeRandomKnightWeights(int team);
		 void writeRandomBishopWeights(int team);
		 void writeRandomRookWeights(int team);
		 void writeRandomQueenWeights(int team);
		 void writeRandomKingWeights(int team);


     void writeLastWeights(int team);
		void writeLastPawnWeights(int team);
		void writeLastBishopWeights(int team);
		void writeLastKnightWeights(int team);
		void writeLastRookWeights(int team);
		void writeLastQueenWeights(int team);
		void writeLastKingWeights(int team);

     //method to perform the reinforcement learning
     //ie update the weights of the payoff function

	 void learning();
     void reinforcementLearning(PayONPos lastBestMove, PayONPos bestMove);
     void finalReinforcementLearning();

     std::vector<Box*> pawnEnd(int team);
     void promotePawn(Box* pawnEndBox);




     void plotWeights();
     	 void plotKnight1Weights();
     	 void plotKnight2Weights();

     	 void plotRook1Weights();
     	 void plotRook2Weights();

     	void plotPawn1Weights();
     	void plotPawn2Weights();

     void clearAllTxtFiles();

     void displayEat(int oldOccupant);

};//end chessHPCModel






#endif
