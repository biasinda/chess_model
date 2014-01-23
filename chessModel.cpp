//chessModel.cpp for multiAgentChess program

//the model file, constructor, destructor, init etc.. see header file chessModel.h for all methods

#include <stdio.h>
#include <boost/mpi.hpp>  //include boost library
#include "repast_hpc/RepastProcess.h"  //interprocess communication and encapsulates process
#include "repast_hpc/Schedule.h"
#include <vector>
#include <algorithm>
#include "repast_hpc/AgentId.h"
#include "repast_hpc/AgentImporterExporter.h"

#include "repast_hpc/SharedContext.h"

//HEADERS
#include "chessModel.h" //include header file for model class

//plotting
//#include "mgl2/mgl.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Box package provider
BoxPackageProvider::BoxPackageProvider(repast::SharedContext<Box>* boxPtr): boxs(boxPtr){ }

void BoxPackageProvider::providePackage(Box * box, std::vector<BoxPackage>& out){
    repast::AgentId id = box->getId();
    BoxPackage package(id.id(), id.startingRank(), id.agentType(), id.currentRank(), box->getOccupant(), box->getTeam1Contacts(), box->getTeam2Contacts(), box->getTeam(),box->bestMove, box->lastInputs );
    out.push_back(package);
}

void BoxPackageProvider::provideContent(repast::AgentRequest req, std::vector<BoxPackage>& out){
    std::vector<repast::AgentId> ids = req.requestedAgents();
    for(size_t i = 0; i < ids.size(); i++){
        providePackage(boxs->getAgent(ids[i]), out);
    }

}

//Box package receiver
BoxPackageReceiver::BoxPackageReceiver(repast::SharedContext<Box>* boxPtr): boxs(boxPtr){ }

Box * BoxPackageReceiver::createAgent(BoxPackage package){
    repast::AgentId id(package.id, package.rank, package.type);
    id.currentRank(package.currentRank);
    return new Box(id, package.Occupant, package.team1Contacts, package.team2Contacts, package.team, package.bestMove, package.lastInputs);
}

void BoxPackageReceiver::updateAgent(BoxPackage package){
    repast::AgentId id(package.id, package.rank, package.type);
    Box * box = boxs->getAgent(id);
    box->set(package.currentRank, package.Occupant, package.team1Contacts, package.team2Contacts, package.team, package.bestMove, package.lastInputs);
}


//vector that will contain weights of payoff function for each agent

int n=5;//number of weights


 //team1  weights
  std::vector<double> w_pawn1(n);
  std::vector<double> w_knight1(n);
  std::vector<double> w_bishop1(n);
  std::vector<double> w_rook1(n);
  std::vector<double> w_queen1(n);
  std::vector<double> w_king1(n);


  //team2 weights
   std::vector<double> w_pawn2(n);
   std::vector<double> w_knight2(n);
   std::vector<double> w_bishop2(n);
   std::vector<double> w_rook2(n);
   std::vector<double> w_queen2(n);
   std::vector<double> w_king2(n);


   //vector of pieces still on the board
   std::vector<int> team1pieces;
   std::vector<int> team2pieces;

 //time series to store weights in
   /*
   std::vector<double> w_pawn(n);
   std::vector<double> w_knight(n);
   std::vector<double> w_bishop(n);
   std::vector<double> w_rook(n);
   std::vector<double> w_queen(n);
   std::vector<double> w_king(n);


 std::vector<std::vector<double> > pawn_ts;
 std::vector<std::vector<double> > knight_ts;
 std::vector<std::vector<double> > bishop_ts;
 std::vector<std::vector<double> > rook_ts;
 std::vector<std::vector<double> > queen_ts;
 std::vector<std::vector<double> > king_ts;
 */

char** board;

double cut;//for random move (entropy)

int checkIfRand;

int moveCount;

PayONPos lastBestMove1;
PayONPos lastBestMove2;

PayONPos finalMove;
std::vector<PayONPos> allMoves;

std::vector<PayONPos> allFinalMovesTeam1;
std::vector<PayONPos> allFinalMovesTeam2;

int kingIsChecked;

int teamThatWon;
int teamThatLost;

int useRandomWeights;
int useTeam2RandomWeights;
int displayBoard;
int displayMoves;

int team1ReinforcementLearning;
int team2ReinforcementLearning;

int normalize;

//array to contain agent poinsters in order to dele in the end, avoid memory leak
std::vector<Box*> boxAgentVector;

//chess model constructor
chessHPCModel::chessHPCModel(std::string propsFile, int argc, char** argv , boost::mpi::communicator* comm){


	//get the RepastProcess instance in order to get the rank
	//repast::RepastProcess* rp = repast::RepastProcess::instance();
	//rank = rp->rank();

	//passing the communicator (comm) to the props file, this way one process will read props and pass to the others
	//argc and argv are used to pass properties through cmdline


	props = new repast::Properties(propsFile, argc, argv, comm);

	// // //Log4CL::instance()->get_logger("root").log(INFO, "retrieving properties...");

	stopAt = repast::strToInt(props->getProperty( "stop.at" ) );
	countOfBoxes = repast::strToInt(props->getProperty( "count.of.boxes" ) );

	alpha=repast::strToDouble(props->getProperty( "learning.rate" ) );
	gamma=repast::strToDouble(props->getProperty( "discounting.rate" ) );

	//cut=repast::strToDouble(props->getProperty( "entropy" ) );
	cut=repast::strToDouble(props->getProperty( "entropy" ) );;

	nProcInX = repast::strToInt(props->getProperty( "processors.in.x" ) );
	nProcInY = repast::strToInt(props->getProperty( "processors.in.y" ) );

	useRandomWeights = repast::strToInt(props->getProperty( "random.weights" ) );
	useTeam2RandomWeights = repast::strToInt(props->getProperty( "team2.random.weights" ) );
	displayBoard = repast::strToInt(props->getProperty( "display.board" ) );
	displayMoves = repast::strToInt(props->getProperty( "display.moves" ) );

	team1ReinforcementLearning=repast::strToInt(props->getProperty( "team1.reinforcement.learning" ) );
	team2ReinforcementLearning=repast::strToInt(props->getProperty( "team2.reinforcement.learning" ) );

	normalize=repast::strToInt(props->getProperty( "normalize" ) );

	// // //Log4CL::instance()->get_logger("root").log(INFO, "got properties");

	context = new repast::SharedContext<Box>(comm);

	//iProcContext= new repast::SharedContext<InterProcessAgent>(comm);
	//make file to write out to

	// // //Log4CL::instance()->get_logger("root").log(INFO, "made provider");

	provider = new BoxPackageProvider( context );
	receiver = new BoxPackageReceiver( context );

	moveCount=0;
	kingIsChecked=0;


	board= new char*[8];

	for(int v=0;v<8;v++){
		board[v]=new char[4];
	}

	// // //Log4CL::instance()->get_logger("root").log(INFO, "made board");


						if(useRandomWeights==1){
							writeRandomWeights(1);
							writeRandomWeights(2);

						}
						else if(useTeam2RandomWeights==1){
							writeRandomWeights(2);
						}

						//read weights from text files...
						readPawnWeights();
						readBishopWeights();
						readKnightWeights();
						readRookWeights();
						readQueenWeights();
						readKingWeights();

						//normalize all the weights
						normalizeWeights(PAWNP);
						normalizeWeights(KNIGHT);
						normalizeWeights(BISHOP);
						normalizeWeights(ROOK);
						normalizeWeights(QUEEN);
						normalizeWeights(KING);

	king1isChecked=0;
	king2isChecked=0;

	teamThatWon=0;
	teamThatLost=0;

	//initialize team vectors
	int ps[16]={ 1,1,1,1,1,1,1,1,2,2,3,3,5,5,10,20};
	team1pieces.assign(&ps[0], &ps[0]+16);
	team2pieces.assign(&ps[0], &ps[0]+16);

	//clear chessgame.txt file
	std::ofstream ofs;
	ofs.open ("chessgame.txt", std::ofstream::out | std::ofstream::trunc);
	ofs.close();

	std::stringstream m;


	rank = repast::RepastProcess::instance()->rank();

	//create a grid on which to play chess

		//process dimensions, am running chess game on one processor
		std::vector<int>pDim(2);
		pDim[0]=nProcInX;
		pDim[1]=nProcInY;

		m.str("");
		m << "pDim[0]=" << pDim[0] << " , pDim[1]=" << pDim[1] ;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );

		grid = new repast::SharedSpaces<Box>::SharedStrictDiscreteSpace("chessboard", repast::GridDimensions( repast::Point<double>(8,8) ) , pDim , 1, comm );

		//grid = new repast::Spaces<Box>::SingleStrictDiscreteSpace("chessboard", repast::GridDimensions( repast::Point<double>(8,8) ) );

		context->addProjection( grid );
		// // //Log4CL::instance()->get_logger("root").log(INFO,"chessboard added to context ...");

		//add agents to context and chessboard
		originX =  grid->dimensions().origin().getX();
		originY =  grid->dimensions().origin().getY();

		//showing origins
		std::stringstream m2;
		m2 << "originX=" << originX << ", originY="<< originY;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m2.str() );

		// // //Log4CL::instance()->get_logger("root").log(INFO,"creating boxes and adding to chessboard...");

		//adding boxes to chessboard
		chessHPCModel::addBoxesToChessBoard();

		// // //Log4CL::instance()->get_logger("root").log(INFO,"finished adding boxes");


		//set team that is starting game:
			teamToMove=1;


			// // //Log4CL::instance()->get_logger("root").log(INFO,"FINISHED INIT()");



// // //Log4CL::instance()->get_logger("root").log(INFO, "finished creating model");

}//end chess model class


void chessHPCModel::writeRandomPawnWeights(int team){


	std::ofstream myfile;
	if(team==1){
	myfile.open("weights/pawn_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
	if(team==2){

	myfile.open("weights/pawn_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();

	}
}


void chessHPCModel::writeRandomKnightWeights(int team){
	std::ofstream myfile;

	if(team==1){
	myfile.open("weights/knight_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
	if(team==2){

	myfile.open("weights/knight_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();

	}
}



void chessHPCModel::writeRandomBishopWeights(int team){
	std::ofstream myfile;

	if(team==1){
	myfile.open("weights/bishop_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
	if(team==2){


	myfile.open("weights/bishop_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();

	}
}

void chessHPCModel::writeRandomRookWeights(int team){
	std::ofstream myfile;
	if(team==1){
	myfile.open("weights/rook_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}

	if(team==2){
	myfile.open("weights/rook_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
}

void chessHPCModel::writeRandomQueenWeights(int team){
	std::ofstream myfile;

	if(team==1){
	myfile.open("weights/queen_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
	if(team==2){
	myfile.open("weights/queen_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}
}

void chessHPCModel::writeRandomKingWeights(int team){
	std::ofstream myfile;

	if(team==1){


	myfile.open("weights/king_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();

	}

	if(team==2){
	myfile.open("weights/king_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<repast::Random::instance()->nextDouble() <<"\n";
	}

	myfile.close();
	}

}

void chessHPCModel::writeRandomWeights(int team){

	writeRandomPawnWeights(team);
	writeRandomKnightWeights(team);
	writeRandomBishopWeights(team);
	writeRandomRookWeights(team);
	writeRandomQueenWeights(team);
	writeRandomKingWeights(team);



}


void chessHPCModel::readPawnWeights(){
	// // //Log4CL::instance()->get_logger("root").log(INFO, "read pawn weights ");

	double q;
	size_t r=0;

	std::ifstream readPawn("weights/pawn_weights1.txt");
	while(readPawn>>q){

		w_pawn1.at(r)=q;
		std::cout<<"\n pawn1: "<<q<<"\n";
		r++;
	}

	readPawn.close();

	double qq;
	size_t rr=0;


	std::ifstream readPawn1("weights/pawn_weights2.txt");
	while(readPawn1>>qq){

		w_pawn2.at(rr)=qq;
		std::cout<<"\n pawn2: "<<qq<<"\n";
		rr++;
	}

	readPawn1.close();
}

void chessHPCModel::readBishopWeights(){
	double q;
	size_t r=0;

	std::ifstream readPawn("weights/bishop_weights1.txt");
	while(readPawn>>q){

		w_bishop1.at(r)=q;
		r++;
	}
	readPawn.close();
	r=0;

	std::ifstream readPawn1("weights/bishop_weights2.txt");
	while(readPawn1>>q){

		w_bishop2.at(r)=q;
		r++;
	}
	readPawn1.close();
}

void chessHPCModel::readKnightWeights(){
	double q;
	size_t r=0;

	std::ifstream readPawn("weights/knight_weights1.txt");
	while(readPawn>>q){

		w_knight1.at(r)=q;
		r++;
	}

	readPawn.close();
	r=0;
	std::ifstream readPawn1("weights/knight_weights2.txt");
	while(readPawn1>>q){

		w_knight2.at(r)=q;
		r++;
	}

	readPawn1.close();
}

void chessHPCModel::readRookWeights(){
	double q;
	size_t r=0;

	std::ifstream readPawn("weights/rook_weights1.txt");
	while(readPawn>>q){

		w_rook1.at(r)=q;

		r++;
	}

	readPawn.close();
	r=0;

	std::ifstream readPawn1("weights/rook_weights2.txt");
	while(readPawn1>>q){

		w_rook2.at(r)=q;

		r++;
	}

	readPawn1.close();
}

void chessHPCModel::readQueenWeights(){
	double q;
	size_t r=0;

	std::ifstream readPawn("weights/queen_weights1.txt");
	while(readPawn>>q){

		w_queen1.at(r)=q;
		r++;
	}

	readPawn.close();
	r=0;

	std::ifstream readPawn1("weights/queen_weights2.txt");
	while(readPawn1>>q){

		w_queen2.at(r)=q;
		r++;
	}

	readPawn1.close();
}

void chessHPCModel::readKingWeights(){
	double q;
	size_t r=0;

	std::ifstream readPawn("weights/king_weights1.txt");
	while(readPawn>>q){

		w_king1.at(r)=q;
		r++;
	}

	readPawn.close();
	r=0;

	std::ifstream readPawn1("weights/king_weights2.txt");
	while(readPawn1>>q){

		w_king2.at(r)=q;
		r++;
	}

	readPawn1.close();
}


//destructor and delete props
chessHPCModel::~chessHPCModel(){
		delete props;
		delete provider;
		delete receiver;
		delete context;
		//delete grid;
}

//init method to intialize game (setup agents, grid and game)
void chessHPCModel::init(int gamesPlayed){
	// // //Log4CL::instance()->get_logger("root").log(INFO,"running init()...");

	std::stringstream m;
    repast::ScheduleRunner& runner =repast::RepastProcess::instance()->getScheduleRunner();
	m<<"Game"<<gamesPlayed<<"- current tick: " <<runner.currentTick();
	// // //Log4CL::instance()->get_logger("root").log(INFO, m.str());



}//end init()


//synch method
void chessHPCModel::synchAgents() {

	// // //Log4CL::instance()->get_logger("root").log(INFO, "Sync...");

	repast::RepastProcess::instance()->synchronizeAgentStates<BoxPackage, BoxPackageProvider, BoxPackageReceiver>(*provider, *receiver,  REQUEST_AGENTS_ALL ); //synchronizeAgentStates
	// // //Log4CL::instance()->get_logger("root").log(INFO, " syncAgentSTATES done");

	repast::RepastProcess::EXCHANGE_PATTERN exchangePattern = repast::RepastProcess::POLL;


	repast::RepastProcess::instance()->synchronizeAgentStatus<Box, BoxPackage, BoxPackageProvider, BoxPackageReceiver , BoxPackageReceiver> ( *context, *provider, *receiver, *receiver, exchangePattern );
	// // //Log4CL::instance()->get_logger("root").log(INFO, " syncAgentSTATUS done");


	//use synPI to synchronize info about shared projection
	//repast::RepastProcess::instance()->synchronizeProjectionInfo<Box, BoxPackage , BoxPackageProvider, BoxPackageReceiver, BoxPackageReceiver >(*context, *provider, *receiver, *receiver, exchangePattern, false );
	//// // //Log4CL::instance()->get_logger("root").log(INFO, " syncPROJECTION done");

}//end synchAgents()


//initSchedule method, used to set the schedule
void chessHPCModel::initSchedule(repast::ScheduleRunner& runner){


	runner.scheduleEvent(1, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::setupBoard) ) );

	runner.scheduleEvent(2, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::setupContacts) ) );

	//runner.scheduleEvent(3, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::synchStates) ) );

	runner.scheduleEvent(3,5, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::displayBoard0) ) );

	runner.scheduleEvent(4,5, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::displayBoard1) ) );


	runner.scheduleEvent(5,5, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::playGame) ) );

	runner.scheduleEvent(6,5, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::victoryCondition) ) );

	runner.scheduleEvent(7,5, repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::learning) ) );


	runner.scheduleEndEvent(repast::Schedule::FunctorPtr(new repast::MethodFunctor<chessHPCModel> (this, &chessHPCModel::plotWeights) ) );

	//schedule the simulation to stop
	runner.scheduleStop(stopAt);

}


std::vector<PayONPos> chessHPCModel::getAllMovesPayoffs(Box* box){
	//method that returns all possible payoffs and positions for a piece
	//returns vector of moves for each piece (ie all its legal moves and the payoffs associated with that move)


	//make a vector of structs paypos
	std::vector<PayONPos> payposVect;
	PayONPos payONposMax;

	PayONPos temp;

	movePackage allLegalMoves;
	std::vector<int> payoffVector;

	std::vector<int> currentLocationVector;



	//get agent id
	repast::AgentId id = box->getId();

	//get agent occupant of box I am moving FROM
	int occupant = box->getOccupant();

	//get team
	int team= box->getTeam();

	//get location of box in the grid using getLocation
	grid->getLocation(id,currentLocationVector );



	//convert to repast::Point...
	repast::Point<int> currentLocation( currentLocationVector[0], currentLocationVector[1] );

	double w[n];



				switch(occupant){

				case PAWNP:

					allLegalMoves=getPawnMoves(team, currentLocation);

					//want to get the weights from the global variable
					//in order to use the weights that are being updated by my learning funnction

					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_pawn1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_pawn2.at(i);

						}

					}


					break;

				case KNIGHT:


					allLegalMoves=getKnightMoves(team, currentLocation);
					//weights for willPT
					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_knight1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_knight2.at(i);

						}

					}

					break;

				case BISHOP:


					allLegalMoves=getBishopMoves(team, currentLocation);
					//weights for willPT
					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_bishop1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_bishop2.at(i);

						}

					}
					break;

				case ROOK:


					allLegalMoves=getRookMoves(team, currentLocation);
					//weights for willPT
					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_rook1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_rook2.at(i);

						}

					}
					break;

				case QUEEN:


					allLegalMoves=getQueenMoves(team, currentLocation);
					//weights for willPT
					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_queen1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_queen2.at(i);

						}

					}
					break;

				case KING:


					allLegalMoves=getKingMoves(team, currentLocation);
					//weights for willPT
					if(team==1){

						for(int i=0;i<n;i++){
							w[i]=w_king1.at(i);

						}

					}else if(team==2){

						for(int i=0;i<n;i++){
							w[i]=w_king2.at(i);

						}

					}
					break;

				}//end switch




			//go through all legal moves and get Payoff
			for (std::size_t k = 0; k < allLegalMoves.legalMoves.size(); k++) {
				//get legal move
				repast::Point<int> move = allLegalMoves.legalMoves[k];

					std::vector<int> willPandwillT;

					//for each legal move get pieces will protect and threaten next turn
					 willPandwillT=chessHPCModel::getWillThreatandProt(team, move, occupant, currentLocation);


				//get object at that position, object at the position I can move to
				Box* boxToCheck = grid->getObjectAt(move);

				//getPayoff for that object
				//add payoff and move to vector of payoffs

				//need to add willThreaten and Protect to payoff- need to figure out weights


				temp.payoff=boxToCheck->getPayoff(team, occupant, w, willPandwillT); //+ w[0]*willPandwillT[0] + w[1]*willPandwillT[1];

				temp.oldPosition=currentLocation;
				temp.newPosition=move;
				temp.team=team;
				temp.occupant=occupant;

				temp.otherTeam=boxToCheck->getTeam();

				temp.occupantThatWasEaten=boxToCheck->getOccupant();

				//also pass the last inputs used to determine the move to the PayONPos, used later in the reinforcement learning

				temp.lastInputs=boxToCheck->lastInputs;

				payposVect.push_back( temp );

			} //end for(allLegalMoves)


		//returns all moves for piece in question
		return payposVect;


}//end getAllMovesPayoffs()

std::vector<int> chessHPCModel::getWillThreatandProt(int team,repast::Point<int> move, int qOccupant, repast::Point<int> currentLocation){

	//get all legal moves from this move and get pieces i will protect/threaten
	//used in getMaxPayoffand Position,
	//method that returns what pieces I will protect and what pieces I will threaten in next move
	//will return in pos0 the sum of the pieces it will protect in the next turn and in pos1 the sum of the pieces it will threaten

	std::vector<int> out;

			movePackage nextMoves;
			//will need a switch case...


					switch(qOccupant){

					case PAWNP:
					nextMoves=getPawnProtects(team, move);
					break;

					case KNIGHT:
					nextMoves=getKnightProtects(team, move);
					break;

					case ROOK:
					nextMoves=getRookProtects(team, move);
					break;

					case QUEEN:
					nextMoves=getQueenProtects(team, move);
					break;

					case KING:
					nextMoves=getKingProtects(team, move);
					break;

					}//end switch




			std::vector<int> willProtect;
			std::vector<int> willThreatn;

			int sumWillThreatn;
			int sumWillProtect;

			//now go through all these moves
			for(std::size_t j=0;j<nextMoves.legalMoves.size();j++ ){

				//only add box occupant to pieces will protect if it is not in the same position as myself
				//to avoid protecting myself
				repast::Point<int> move=nextMoves.legalMoves[j];

				if(currentLocation[0]!=move[0] && currentLocation[1]!=move[1]){

				Box* nextBox=grid->getObjectAt(nextMoves.legalMoves[j]);

					//now if where I can move next is occupied by a friend/enemy
					//add this piece to willProtect/willThreaten



					if(nextBox->getTeam()==team){
						//if same team add occupant to willProtect
						willProtect.push_back(nextBox->getOccupant());
					}else if(nextBox->getTeam()!=team && nextBox->getTeam()!=0){
						//if differnt team and not empty, add occupant to willThreaten
						willThreatn.push_back(nextBox->getOccupant());
					}

				}//end if(currentLocation...)

			}//end for(nextMoves)


if(willThreatn.size()!=0){
sumWillThreatn=std::accumulate(willThreatn.begin(),willThreatn.end(),0);
} else{
	sumWillThreatn=0;
}


if(willProtect.size()!=0){
sumWillProtect=std::accumulate(willProtect.begin(),willProtect.end(),0);
}else{

	sumWillProtect=0;
}

//now want to return the sum of pieces i will protect and threaten
out.push_back(sumWillProtect);
out.push_back(sumWillThreatn);

return out;

}//end getWilProtandThr()


//put this in init()
void chessHPCModel::addBoxesToChessBoard(){

	// // //Log4CL::instance()->get_logger("root").log(INFO,"ADDING BOXES");
	repast::AgentId id;
	std::stringstream m;
	int team;
	Box* box;
	int count=0; //count used for AgentId's


	//add observer box to  process

	repast::AgentId id1(100,rank,1);
	id1.currentRank(rank);
	Box* observer= new Box(id1);
	context->addAgent(observer);
	// // //Log4CL::instance()->get_logger("root").log(INFO,"added observer box"); //need to be on grid in order to share among processes...
	grid->moveTo(observer->getId(),repast::Point<int>(originX,originY+2*rank) );


	//add agents to both processes
	chessHPCModel::requestIPAAgents();

			//add the agent to the vector of agents
			boxAgentVector.push_back(observer);


	repast::Point<int> pointToSet(0,0);

	// // //Log4CL::instance()->get_logger("root").log(INFO,"made variables");

	//add empty boxes to all grid points
	//note: we add all agents to both processes, so both processes will have a copy of each agent
	//the position of the agent will determine localoty or non_locality for the specific process

	//Box* box;

	for(int j=0;j<8;j++){
		for(int i=0; i<countOfBoxes;i++){

			repast::AgentId id(count, rank, 0);

			id.currentRank(rank);
			box = new Box(id,0,0);


			context->addAgent(box); //adding boxes

			//put pawn on chessboard all in first row

			grid->moveTo(box->getId(),repast::Point<int>(originX+j,originY+i) );

						//add the agent to the vector of agents
						boxAgentVector.push_back(box);

			setBoard(0,0,repast::Point<int>(originX+j,originY+i));


			//std::stringstream m;
			//m << "box"<< box->getId() << " added to position: " << originX+j << "," << originY+i << "  occupant " << box->getOccupant() ;
			//// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );

			//zero (team and occupant) all the boxes, dont need already set to zero in Box(id,0,0)
			//chessHPCModel::setBox(repast::Point<int>(originX+j,originY+i),0,0);

			count=count+1;


		}//end for i

		//m.str(" ");
		//m << "finished adding "<< j << "th column"<<" , count ="<<count ;
		//// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );

	}//end for j


	// // //Log4CL::instance()->get_logger("root").log(INFO,"ADDED ALL EMPTY BOXES");


}//end addBoxes...



void chessHPCModel::setupBoard(){

	//set intial positions of agents
	// // //Log4CL::instance()->get_logger("root").log(INFO,"SETUP BOARD");
	repast::AgentId id;
	std::stringstream m;
	int team;
	Box* box;
	repast::Point<int> pointToSet(0,0);



	if (rank==0){

	//adding team1
		team=1;

		//PAWNS
			for(int j = 0; j < 8; j++){

				int i=1;

					pointToSet[0]=j;
					pointToSet[1]=i;



					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),PAWNP,1);

			}//end for j



			//END PAWNS


				//ROOKS

					pointToSet[0]=0;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),ROOK,1);

					pointToSet[0]=0+7;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),ROOK,1);


				//END ROOKS


				//KNIGHTS

					pointToSet[0]=0+1;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KNIGHT,1);

					pointToSet[0]=0+6;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KNIGHT,1);


				//END KNIGHTS


				//BISHOPS

					pointToSet[0]=0+2;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),BISHOP,1);

					pointToSet[0]=0+5;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),BISHOP,1);

				//END BISHOPS

				//KING

					pointToSet[0]=0+4;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KING,1);

				//END KING


				//QUEEN


					pointToSet[0]=0+3;
					pointToSet[1]=0;

					chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),QUEEN,1);


				//END QUEEN




		//end adding team1
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}//end rank0

	if(rank==1){


	//adding team2
						team=2;

						//PAWNS
							for(int j = 0; j < 8; j++){

								int i=1;

									pointToSet[0]=j;
									pointToSet[1]=6;


									chessHPCModel::setBox( repast::Point<int>(pointToSet[0],pointToSet[1]),PAWNP,team);

							}//end for j

						//END PAWNS


								//ROOKS

									pointToSet[0]=0;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),ROOK,team);

									pointToSet[0]=0+7;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),ROOK,team);


								//END ROOKS


								//KNIGHTS

									pointToSet[0]=0+1;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KNIGHT,team);

									pointToSet[0]=0+6;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KNIGHT,team);


								//END KNIGHTS


								//BISHOPS

									pointToSet[0]=0+2;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),BISHOP,team);

									pointToSet[0]=0+5;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),BISHOP,team);

								//END BISHOPS

								//KING

									pointToSet[0]=0+4;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),KING,team);

								//END KING


								//QUEEN


									pointToSet[0]=0+3;
									pointToSet[1]=7;

									chessHPCModel::setBox(repast::Point<int>(pointToSet[0],pointToSet[1]),QUEEN,team);


								//END QUEEN

						//end adding team2


	}//end rank2



/*
	//to test use only a few pieces
			if(rank==0){
				chessHPCModel::setBox(repast::Point<int>(0,0),KING,1);
				//chessHPCModel::setBox(repast::Point<int>(1,0),PAWNP,1);
				chessHPCModel::setBox(repast::Point<int>(1,1),QUEEN,2);
				chessHPCModel::setBox(repast::Point<int>(2,2),PAWNP,2);

				//chessHPCModel::setBox(repast::Point<int>(0,2),PAWNP,1);
				//chessHPCModel::setBox(repast::Point<int>(1,2),PAWNP,1);


				//chessHPCModel::setBox(repast::Point<int>(3,2),KNIGHT,2);
				chessHPCModel::setBox(repast::Point<int>(0,3),BISHOP,2);
				//chessHPCModel::setBox(repast::Point<int>(5,3),KING,1);


				//chessHPCModel::setBox(repast::Point<int>(7,0),PAWNP,1);
				//chessHPCModel::setBox(repast::Point<int>(6,0),PAWNP,1);

			}
			if(rank==1){
				chessHPCModel::setBox(repast::Point<int>(0,7),PAWNP,2);
				chessHPCModel::setBox(repast::Point<int>(1,7),PAWNP,2);
				chessHPCModel::setBox(repast::Point<int>(2,7),KNIGHT,2);
				chessHPCModel::setBox(repast::Point<int>(7,7),KING,2);



				//chessHPCModel::setBox(repast::Point<int>(5,5),PAWNP,1);
				//chessHPCModel::setBox(repast::Point<int>(6,6),QUEEN,1);



			}

*/



		//now request all agents in order to be able to move all over the board
	requestAllAgents();
	setupContacts();
	// // //Log4CL::instance()->get_logger("root").log(INFO,"REQUESTED ALL AGENTS");

		//after moving and updating contacts need to synchronize



	// // //Log4CL::instance()->get_logger("root").log(INFO,"DONE SETTING UP BOARD");



}//end setupBoard()

void chessHPCModel::setBoard(int occupant, int team, repast::Point<int> point){

	char piece;

	if(team==0){
		piece='_';
	}

	else if(team==1){

	switch(occupant){
	case 0:
		piece='_';
		break;
	case PAWNP:
		piece='p';
		break;
	case KNIGHT:
		piece='h';
		break;
	case ROOK:
		piece='r';
		break;
	case BISHOP:
		piece='b';
		break;
	case QUEEN:
		piece='q';
		break;
	case KING:
		piece='k';
		break;

	}

	}else if(team==2){
		switch(occupant){
		case 0:
			piece='_';
			break;
		case PAWNP:
			piece='P';
			break;
		case KNIGHT:
			piece='H';
			break;
		case ROOK:
			piece='R';
			break;
		case BISHOP:
			piece='B';
			break;
		case QUEEN:
			piece='Q';
			break;
		case KING:
			piece='K';
			break;

		}

	}

board[point[0]][point[1]]=piece;



}



void chessHPCModel::playGame(){

	// // //Log4CL::instance()->get_logger("root").log(INFO, "PLAYING GAME");


	synchStates();

	std::stringstream m;
	m << "						TEAM" << teamToMove << "'S TURN";
	// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );

	int qTeam=teamToMove;

	allMoves.clear(); //clear the vector of all possibel moves
	allMoves=chessHPCModel::getMoves(); //get all possible moves and store them in the global variable allMoves



if(rank==0 && displayMoves==1){
			//to test display all the moves we have for team 1
			m.str("");
			m << "\n\n\n\n						team" << teamToMove << " has "<<allMoves.size()<< "moves.";
			Log4CL::instance()->get_logger("root").log( INFO,m.str() );

			//display the payoff for each move
			for(int i=0;i<allMoves.size();i++){
				m.str("");
		m << i << ". piece" << allMoves[i].occupant << " in position ("
				<< allMoves[i].oldPosition[0] << ","
				<< allMoves[i].oldPosition[1] << ") to ("
				<< allMoves[i].newPosition[0]<<","<< allMoves[i].newPosition[1]<< ") has payoff = "
				<< allMoves[i].payoff << "  and SumTeam1cs="
				<< allMoves[i].lastInputs.sumTeam1Contacts<< "  and SumTeam2cs="
				<< allMoves[i].lastInputs.sumTeam2Contacts << " and has willPandwillT0 ="<< allMoves[i].lastInputs.willPandwillT0
				<< " and has willPandwillT1 ="<< allMoves[i].lastInputs.willPandwillT1;
		Log4CL::instance()->get_logger("root").log( INFO,m.str() );

			}

}//rank==0


	PayONPos moveSelectMove;

		//select best move
		moveSelectMove=chessHPCModel::selectMove(); //among allMoves select the best move, ie the one with highest payoff

		/*
					m.str("");
					m << "				selectMove returns move with payoff" << moveSelectMove.payoff;
					// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );
		 */


								//perform move
								chessHPCModel::makeMove(moveSelectMove);




		// // //Log4CL::instance()->get_logger("root").log(INFO, "PLAYING GAME: finished ");
}


std::vector<PayONPos> chessHPCModel::getMoves(){


	//go through all boxes, if the box is on our team add its moves to all possible moves, synchronize between processors
	//finally return all possible moves for our team

	// // //Log4CL::instance()->get_logger("root").log(INFO,"SELECTING MOVE");


	//vector to put agents
	std::vector<Box*> boxesToCheck;

	//use select agents to get all agents on both processes since I already shared all agents on all processes
	//dont use LOCAL, but get all the boxes and therefore all the moves
	context->selectAgents(boxesToCheck,0,true);


	std::stringstream m;
	//m<<"getMoves(): selectAgents() returned a size "<<boxesToCheck.size()<<" vector";
	//// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );

	//go through local agents and get the PayONPos for each move
	//add each move to allMoves
	std::vector<Box*>::iterator it=boxesToCheck.begin();
	int c=0;

	while(it !=boxesToCheck.end()){


			//make Point I want to check
			repast::Point<int> checkPoint(0, 0);

			//get Object
			Box* boxToCheck = *it;

			std::vector<int> currentLoc;
			grid->getLocation(boxToCheck,currentLoc);
			//get its team and occupant
			int team= boxToCheck->getTeam();
			int occupant= boxToCheck->getOccupant();

			/*
			m.str("");
			m<<"got agent at location: "<<currentLoc[0]<<","<<currentLoc[1]<<" occupant is: "<< occupant <<" , count ="<<c;
			// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );
			*/


			if (team==teamToMove){ //the box belongs to my team (its qteams turn...)

				//for each piece get a vector of moves
				std::vector<PayONPos> moves=chessHPCModel::getAllMovesPayoffs(boxToCheck);

				//add each move from that vector to allMoves (all the possible moves of the team)
				for(int i=0;i<moves.size();i++){
				allMoves.push_back( moves[i] );
				}

			}

			it++;
			c++;

	}//end while(it...)

	//SYNCHRONIZE
	//before returning vector, sort by payoffs, so last move will be one with highest payoff. PROBELM: if two moves have the same payoff the two processors may sort them differently...
	std::sort( allMoves.begin(), allMoves.end(), PayONPosFunctor() );

				std::vector<Box*> ipaBoxs;
			//to make sure both processors have the same ordering:

				//get the ipa agents
				 context->selectAgents(repast::SharedContext<Box>::LOCAL,ipaBoxs,1,false);

					//on rank 0 i set bestMove for both agents to allMoves, now they both have the allMove vector from process 0.
					ipaBoxs[0]->bestMove=allMoves;



			//now I synchronize across processes...
			repast::RepastProcess::instance()->synchronizeAgentStates<BoxPackage, BoxPackageProvider, BoxPackageReceiver>(*provider, *receiver ); //synchronizeAgentStates

			if(rank==1){
			//now reset the allMove vector on both processes...
			std::vector<Box*> ipaLBoxs; context->selectAgents(repast::SharedContext<Box>::NON_LOCAL,ipaLBoxs,1,false);



			//from this local agent retrieve the bestMove vector
			allMoves=ipaLBoxs[0]->bestMove;
			}



return allMoves;


}//end getMoves()

PayONPos chessHPCModel::selectMove(){

	//after going through all boxes, get the max payonpos and return it
	std::stringstream m;

	if(allMoves.size()!=0){
	//since the elements were sorted by payoff in getMoves we can simply use the last element in the vector
		finalMove=allMoves[allMoves.size()-1];
		//finalMove= *std::max_element( allMoves.begin(), allMoves.end(), PayONPosFunctor() );

		//now also get a random move for exploration, uses allMoves vector
		PayONPos randomMove=getRandomMove();


		//now want to select between the randomMove and the bestMove

				//get a random number btwn 0 and 1
				checkIfRand=0;
				double e=rand()/double(RAND_MAX);
									m.str("");
									m<<" random number is "<<e;
									//// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );

				//cut is entropy condition
				if(e>cut){
					// // //Log4CL::instance()->get_logger("root").log(INFO,"picked random move");

					//update the final move to the random move if we selected that one
					finalMove=randomMove;
					checkIfRand=1;

				}

							//try displaying the last inputs
							m<<" select move: position"<<finalMove.newPosition[0]<<"'"<<finalMove.newPosition[1] <<"  finalMove.lastInputs.sumt1c= "<<finalMove.lastInputs.sumTeam1Contacts<< " and lastInputs.qOccupant ="<< finalMove.lastInputs.qOccupant;
							// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );

	}else if(allMoves.size()==0){

		m.str("");
		m<<" SelectMove: team "<< teamToMove <<" has no moves! ";
		// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );

		//declare a draw
		std::ofstream myfile;


			//make a victory check: will check if the kings whose turn it is is checked
			kingIsChecked=victoryCheck();

			//king checked->check mate
			if(kingIsChecked==1){
				//set the teamThatWon variable
				if(teamToMove==1){
					teamThatWon=2;
				}else if(teamToMove==2){
					teamThatWon=1;
				}
				teamThatLost=teamToMove;

				if(rank==0){
					myfile.open("chessgameStats.txt", std::ios::app);
					myfile<< "Team "<< teamThatWon << " won in " <<moveCount<< " moves, entropy: "<<cut <<". \n";
					myfile.close();

					myfile.open("chessgameLength.txt", std::ios::app);
					myfile<<moveCount<<"\n";
					myfile.close();

					myfile.open("chessgame.txt", std::ios::app);
					myfile<< "Team "<< teamThatWon << " won in " <<moveCount<< " moves. \n";
					myfile.close();
				}

				//king not checked->draw
			}else if(kingIsChecked==0){

				if(rank==0){
					myfile.open("chessgameStats.txt", std::ios::app);
					myfile<< "Draw in " <<moveCount<< " moves, entropy: "<<cut <<" - team "<< teamToMove<<" has no more moves. \n";
					myfile.close();

					myfile.open("chessgameLength.txt", std::ios::app);
					myfile<<moveCount<<"\n";
					myfile.close();

					myfile.open("chessgame.txt", std::ios::app);
					myfile<< "Draw in " <<moveCount<< " moves, entropy: "<<cut <<" - team "<< teamToMove<<" has no more moves. \n";
					myfile.close();
				}

				//set draw
				teamThatWon=3;
				teamThatLost=3;

			}

			//stop simulation, and
			repast::ScheduleRunner& runner =repast::RepastProcess::instance()->getScheduleRunner();
			runner.stop();



	}



	// // //Log4CL::instance()->get_logger("root").log(INFO,"FINISHED SELECTING MOVE");


	return finalMove;



}//end selectMove

PayONPos chessHPCModel::getRandomMove(){

	int ranY=rand() %8;
	PayONPos randomMove;

	int totalMoves=allMoves.size();
	int ranPos=rand() %allMoves.size();

	randomMove=allMoves[ranPos];


	return randomMove;

}//end getRandomMove

void chessHPCModel::makeMove(PayONPos bestMove){



	//given a PayONPos, moves the piece from the old to the new position

	finalMove=bestMove;

	// // //Log4CL::instance()->get_logger("root").log(INFO,"MAKING MOVE");
	//put in do smoething method
	//method that will make move, remove occupant from old position and set occupant of new position
	repast::Point<int> oldPosition = bestMove.oldPosition;
	repast::Point<int> newPosition = bestMove.newPosition;

	int team=bestMove.team;
	int occupant= bestMove.occupant;


	//get boxes at new and old position
	Box* boxNewPosition=grid->getObjectAt(newPosition);
	Box* boxOldPosition=grid->getObjectAt(oldPosition);





	if(team==1 || team==2){

			//make move


						boxNewPosition->setOccupant(occupant);
						boxNewPosition->setTeam(team); //set team in new position to team of new occupant
						//set new position
						setBoard(occupant,team, newPosition);



						boxOldPosition->setOccupant(0); // set occupant at old position to 0, box is now empty after move
						boxOldPosition->setTeam(0); //set team in old position to 0 (empty)

						//once moved update board representation
						//set old position
						setBoard(0,0,oldPosition);
	}//end if team
/*
						//only do reinforcement learning if the move was not random:
						if(checkIfRand==0){


							//to make sure the right team is learning need to skip a step

							if(moveCount>1){
								//only if the move count is greater than 1, otherwise have no best move to use...

								//do reinforcement learning on last best move
								if(bestMove.team==1){
									reinforcementLearning(lastBestMove1, bestMove);
								}
								else if(bestMove.team==2){
									reinforcementLearning(lastBestMove2, bestMove);
								}

							}


							if(bestMove.team==1){
								//store last best move
								lastBestMove1.occupant=bestMove.occupant;
								lastBestMove1.team=bestMove.team;
								lastBestMove1.oldPosition=bestMove.oldPosition;
								lastBestMove1.newPosition=bestMove.newPosition;
								lastBestMove1.payoff=bestMove.payoff;
								lastBestMove1.lastInputs=bestMove.lastInputs;

							}else if(bestMove.team==2){
								lastBestMove2.occupant=bestMove.occupant;
								lastBestMove2.team=bestMove.team;
								lastBestMove2.oldPosition=bestMove.oldPosition;
								lastBestMove2.newPosition=bestMove.newPosition;
								lastBestMove2.payoff=bestMove.payoff;
								lastBestMove2.lastInputs=bestMove.lastInputs;


							}

						}//end if checkifrand

*/

				setupContacts();


			/*
			m1.str("");
			m1 << "finalMove.sumT1cs= "<< bestMove.lastInputs.sumTeam1Contacts;
			// // //Log4CL::instance()->get_logger("root").log(INFO, m1.str());
*/

				std::ofstream myfile;
				myfile.open("chessgame.txt",std::ios::app );
					if(checkIfRand==1){
						myfile<<"random move: ";
					}

	myfile << "trying move!! move" << moveCount << ": piece " << occupant
			<< " of team" << team << " from (" << oldPosition[0] << ","
			<< oldPosition[1] << ") to (" << newPosition[0] << ","
			<< newPosition[1] << "): payoff is: "<< bestMove.payoff<< "\n";
	myfile.close();



}

void chessHPCModel::undoLastMove(PayONPos lastMove){
	//undoes the last move
	//used in while king is checked loop
	// // //Log4CL::instance()->get_logger("root").log(INFO,"undoing last move");

	repast::Point<int> oldPosition=lastMove.oldPosition;
	repast::Point<int> newPosition=lastMove.newPosition;

	int occupantThatMoved=lastMove.occupant;
	int occupantThatWasEaten=lastMove.occupantThatWasEaten;

	int teamThatMoved=lastMove.team;

	int otherTeam=lastMove.otherTeam;

	//to undo the move

	Box* boxAtOldPosition=grid->getObjectAt(oldPosition);
	Box* boxAtNewPosition=grid->getObjectAt(newPosition);

	boxAtNewPosition->setOccupant(occupantThatWasEaten);
	boxAtNewPosition->setTeam(otherTeam);

	boxAtOldPosition->setOccupant(occupantThatMoved);
	boxAtOldPosition->setTeam(teamThatMoved);

	//update the board also:
							//set new position
							setBoard(occupantThatWasEaten,otherTeam, newPosition);

							//set old position
							setBoard(occupantThatMoved,teamThatMoved, oldPosition);


							setupContacts();
							//after undoing move reset the board contacts

}

void chessHPCModel::normalizeWeights(int occupant){


	if(normalize==1){
	//normalize all the weights
	double wSum1;
	double wSum2;

	std::vector<double> absW1;
	std::vector<double> absW2;
	std::stringstream m;

	switch (occupant) {

	case PAWNP:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_pawn1[i] ) );
			absW2.push_back( std::abs( w_pawn2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_pawn1.size(); i++) {

			w_pawn1[i] = w_pawn1[i] / wSum1;
			w_pawn2[i] = w_pawn2[i] / wSum2;

		}

		wSum1 = std::accumulate(w_pawn1.begin(), w_pawn1.end(), 0.0);
		wSum2 = std::accumulate(w_pawn2.begin(), w_pawn2.end(), 0.0);

		break;

	case KNIGHT:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_knight1[i] ) );
			absW2.push_back( std::abs( w_knight2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_knight1.size(); i++) {
			w_knight1[i] = w_knight1[i] / wSum1;
			w_knight2[i] = w_knight2[i] / wSum2;
		}
		wSum1 = std::accumulate(w_knight1.begin(), w_knight1.end(), 0.0);
		wSum2 = std::accumulate(w_knight2.begin(), w_knight2.end(), 0.0);
		break;

	case BISHOP:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_bishop1[i] ) );
			absW2.push_back( std::abs( w_bishop2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_bishop1.size(); i++) {
			w_bishop1[i] = w_bishop1[i] / wSum1;
			w_bishop2[i] = w_bishop2[i] / wSum2;
		}

		wSum1 = std::accumulate(w_bishop1.begin(), w_bishop1.end(), 0.0);
		wSum2 = std::accumulate(w_bishop2.begin(), w_bishop2.end(), 0.0);
		break;

	case ROOK:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_rook1[i] ) );
			absW2.push_back( std::abs( w_rook2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_rook1.size(); i++) {
			w_rook1[i] = w_rook1[i] / wSum1;
			w_rook2[i] = w_rook2[i] / wSum2;
		}

		wSum1 = std::accumulate(w_rook1.begin(), w_rook1.end(), 0.0);
		wSum2 = std::accumulate(w_rook2.begin(), w_rook2.end(), 0.0);
		break;

	case QUEEN:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_queen1[i] ) );
			absW2.push_back( std::abs( w_queen2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_queen1.size(); i++) {
			w_queen1[i] = w_queen1[i] / wSum1;
			w_queen2[i] = w_queen2[i] / wSum2;
		}

		wSum1 = std::accumulate(w_queen1.begin(), w_queen1.end(), 0.0);
		wSum2 = std::accumulate(w_queen2.begin(), w_queen2.end(), 0.0);
		break;

	case KING:

		//weights can be negative so need absolute value to get actual distance between them
		for(int i=0;i<w_pawn1.size();i++){
			absW1.push_back( std::abs( w_king1[i] ) );
			absW2.push_back( std::abs( w_king2[i] ) );

		}


		//find sum of weights
		wSum1 = std::accumulate(absW1.begin(), absW1.end(), 0.0);
		wSum2 = std::accumulate(absW2.begin(), absW2.end(), 0.0);

		for (int i = 0; i < w_king1.size(); i++) {
			w_king1[i] = w_king1[i] / wSum1;
			w_king2[i] = w_king2[i] / wSum2;
		}

		wSum1 = std::accumulate(w_king1.begin(), w_king1.end(), 0.0);
		wSum2 = std::accumulate(w_king2.begin(), w_king2.end(), 0.0);

		break;

	} //end switch

	/*
				if(rank==0){

					if(wSum1>1.5 || wSum2>1.5 ){
						std::cout<<"warning!"<< occupant<<" normalizeweights: wsum1= "<<wSum1<< " , wsum2= "<<wSum2<<"\n";

						std::ofstream file;
						file.open("chessgameStats.txt", std::ios::app);
						file<<"warning!"<< occupant<<" normalizeweights: wsum1= "<<wSum1<< " , wsum2= "<<wSum2<<"\n";
						file.close();
					}

				}
*/

	}//end if nirmalize==1
}


movePackage chessHPCModel::getPawnMoves(int team, repast::Point<int> currentLocation){

	movePackage allLegalMoves;
	allLegalMoves.team=team;
	allLegalMoves.occupant=PAWNP; //set occupant to pawn
	repast::Point<int> checkPoint(0,0);
	Box* boxToCheck;

	//PAWN TEAM 1

			if (team==1 ){ // pawn moves depend on team they are on


				//MOVE FORWARD

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0];
					checkPoint[1]=currentLocation[1]+1;

					//need to check if the position i want to check is on my processor, if it is not need to request that position from the other processor...



					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


					//get Occupant of box to check
					int occupantToCheck = boxToCheck->getOccupant();


						if (occupantToCheck==0){// if box is empty add location to moves

							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

						}
						else if(boxToCheck->getTeam()==1){// same team, do not add box to possible moves since cannot move to it, break loop
							//do not update k
						}
						else if (boxToCheck->getTeam()==2){// different team, dont add box to possible moves since cannot move to it (since pawn)

						}
						//note do not need to do anything if went off grid, note could improve this code, do not need to check team

					}//end if(boxtocheck)



				//MOVE FORWARD RIGHT

				//make position and Point I want to check

				checkPoint[0]=currentLocation[0]+1;
				checkPoint[1]=currentLocation[1]+1;
				//get Object at Position I want to check
				boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

				//check if we went off the grid
				if (boxToCheck!=0){
					//get Occupant of box to check
					int occupantToCheck = boxToCheck->getOccupant();


						if (occupantToCheck==0 || boxToCheck->getTeam()==1 ){// if box is empty do not add box to possible move since this is a pawn and
											//cannot move to empty boxes on sides

						}

						else if (boxToCheck->getTeam()==2){// different team,  add box to possible moves and break loop since cannot move further
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


						}

					}//end if(boxtocheck)



				//MOVE FORWARD LEFT

				//make position and Point I want to check

				checkPoint[0]=currentLocation[0]-1;
				checkPoint[1]=currentLocation[1]+1;
				//get Object at Position I want to check
				boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

				if (boxToCheck!=0){
				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();


					if (occupantToCheck==0 || boxToCheck->getTeam()==1){// if box is empty/friendly do not add box to possible move since this
																		// is a pawn and cannot move to empty boxes on sides

					}

					else if (boxToCheck->getTeam()==2){// different team,  add box to possible moves and break loop since cannot move further
						allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


					}

				}//end if(boxtocheck)

			}//END PAWN TEAM 1

			//PAWN TEAM 2

							if (team==2 ){ // pawn moves depend on team they are on, other pieces no


								//MOVE DOWN

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]-1;

									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


									//get Occupant of box to check
									int occupantToCheck = boxToCheck->getOccupant();


										if (occupantToCheck==0){// if box is empty add location to possible moves
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
										}
										else if(boxToCheck->getTeam()==2){// same team, do not add box to possible moves since cannot move to it, break loop

										}
										else if (boxToCheck->getTeam()==1){// different team, dont add box to possible moves since cannot move to it (since pawn)

										}
										//note do not need to do anything if went off grid, note could improve this code, do not need to check team

									}//end if(boxtocheck)



								//MOVE DOWN RIGHT

								//make position and Point I want to check
								checkPoint[0]=currentLocation[0]+1;
								checkPoint[1]=currentLocation[1]-1;

								//get Object at Position I want to check
								boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								//check if we went off the grid
								if (boxToCheck!=0){
									//get Occupant of box to check
									int occupantToCheck = boxToCheck->getOccupant();


										if (occupantToCheck==0 || boxToCheck->getTeam()==2 ){// if box is empty do not add box to possible move since this is a pawn and
															//cannot move to empty boxes on sides

										}

										else if (boxToCheck->getTeam()==1){// different team,  add box to possible moves and break loop since cannot move further
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}

									}//end if(boxtocheck)



								//MOVE DOWN LEFT

								//make position and Point I want to check

								checkPoint[0]=currentLocation[0]-1;
								checkPoint[1]=currentLocation[1]-1;

								//get Object at Position I want to check
								boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if (boxToCheck!=0){
								//get Occupant of box to check
								int occupantToCheck = boxToCheck->getOccupant();


									if (occupantToCheck==0 || boxToCheck->getTeam()==2){// if box is empty/friendly do not add box to possible move since this
																						// is a pawn and cannot move to empty boxes on sides

									}

									else if (boxToCheck->getTeam()==1){// different team,  add box to possible moves and break loop since cannot move further
										allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

									}

								}//end if(boxtocheck)

							}//END PAWN TEAM 2


							return allLegalMoves;


}//end getPawnMoves
movePackage chessHPCModel::getPawnProtects(int team, repast::Point<int> currentLocation){
	//returns moves that pawn threatens and protects, not necessarily moves the pawn can make

	movePackage allLegalMoves;
	allLegalMoves.team=team;
	allLegalMoves.occupant=PAWNP; //set occupant to pawn
	repast::Point<int> checkPoint(0,0);
	Box* boxToCheck;

	//PAWN TEAM 1

			if (team==1 ){ // pawn moves depend on team they are on




				//MOVE FORWARD RIGHT

				//make position and Point I want to check

				checkPoint[0]=currentLocation[0]+1;
				checkPoint[1]=currentLocation[1]+1;
				//get Object at Position I want to check
				boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

				//check if we went off the grid
				if (boxToCheck!=0){
					//get Occupant of box to check
					int occupantToCheck = boxToCheck->getOccupant();


						if (occupantToCheck==0 || boxToCheck->getTeam()==1 ){// if box is empty add box to protects since this is a pawn and
											//can protect empty boxes on sides
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

						}

						else if (boxToCheck->getTeam()==2){// different team,  add box to possible moves and break loop since cannot move further
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

						}

					}//end if(boxtocheck)



				//MOVE FORWARD LEFT

				//make position and Point I want to check

				checkPoint[0]=currentLocation[0]-1;
				checkPoint[1]=currentLocation[1]+1;
				//get Object at Position I want to check
				boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

				if (boxToCheck!=0){
				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();


					if (occupantToCheck==0 || boxToCheck->getTeam()==1){
						// if box is empty add box to protects since this is a pawn and
						//can protect empty boxes on sides
						allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					}

					else if (boxToCheck->getTeam()==2){// different team,  add box to possible moves and break loop since cannot move further
						allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


					}

				}//end if(boxtocheck)

			}//END PAWN TEAM 1

			//PAWN TEAM 2

							if (team==2 ){ // pawn moves depend on team they are on, other pieces no


								//MOVE DOWN RIGHT

								//make position and Point I want to check
								checkPoint[0]=currentLocation[0]+1;
								checkPoint[1]=currentLocation[1]-1;

								//get Object at Position I want to check
								boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								//check if we went off the grid
								if (boxToCheck!=0){
									//get Occupant of box to check
									int occupantToCheck = boxToCheck->getOccupant();


										if (occupantToCheck==0 || boxToCheck->getTeam()==2 ){

											// if box is empty add box to protects since this is a pawn and
											//can protect empty boxes on sides
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

										}

										else if (boxToCheck->getTeam()==1){// different team,  add box to possible moves and break loop since cannot move further
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}

									}//end if(boxtocheck)



								//MOVE DOWN LEFT

								//make position and Point I want to check

								checkPoint[0]=currentLocation[0]-1;
								checkPoint[1]=currentLocation[1]-1;

								//get Object at Position I want to check
								boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if (boxToCheck!=0){
								//get Occupant of box to check
								int occupantToCheck = boxToCheck->getOccupant();


									if (occupantToCheck==0 || boxToCheck->getTeam()==2){

										// if box is empty add box to protects since this is a pawn and
										//can protect empty boxes on sides
										allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
									}

									else if (boxToCheck->getTeam()==1){// different team,  add box to possible moves and break loop since cannot move further
										allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

									}

								}//end if(boxtocheck)

							}//END PAWN TEAM 2


							return allLegalMoves;


}//end getPawnMoves


movePackage chessHPCModel::getKnightMoves(int team, repast::Point<int> currentLocation){


	movePackage allLegalMoves;
		allLegalMoves.team=team;
		allLegalMoves.occupant=KNIGHT;
		repast::Point<int> checkPoint(0,0);
		Box* boxToCheck;


			//FORWARD RIGHT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+1;
			checkPoint[1]=currentLocation[1]+2;

			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//
					//

				}

			}//end if(boxtocheck)


			//FORWARD LEFT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-1;
			checkPoint[1]=currentLocation[1]+2;

			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//
					//

				}


			}//end if(boxtocheck)



			//LEFT FORWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-2;
			checkPoint[1]=currentLocation[1]+1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if ( boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//
					//

				}


			}//end if(boxtocheck)

			//LEFT BACKWARDS

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-2;
			checkPoint[1]=currentLocation[1]-1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


				}


			}//end if(boxtocheck)


			//BACKWARDS LEFT

			//make position and Point I want to check
			checkPoint[0]=currentLocation[0]-1;
			checkPoint[1]=currentLocation[1]-2;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//
					//

				}


			}//end if(boxtocheck)

			//BACKWARDS RIGHT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+1;
			checkPoint[1]=currentLocation[1]-2;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

				}


			}//end if(boxtocheck)


			//RIGHT FORWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+2;
			checkPoint[1]=currentLocation[1]+1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

				}


			}//end if(boxtocheck)

			//RIGHT BACKWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+2;
			checkPoint[1]=currentLocation[1]-1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


				//get Occupant of box to check
				int occupantToCheck = boxToCheck->getOccupant();

				if (boxToCheck->getTeam()==team ){// same team- dont add move
				}

				else if (occupantToCheck==0 || boxToCheck->getTeam()!=team){// different team or empty-add move
					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//
					//

				}

			}//end if(boxtocheck)


return allLegalMoves;

}//end getKnightMoves
movePackage chessHPCModel::getKnightProtects(int team, repast::Point<int> currentLocation){


	movePackage allLegalMoves;
		allLegalMoves.team=team;
		allLegalMoves.occupant=KNIGHT;
		repast::Point<int> checkPoint(0,0);
		Box* boxToCheck;


			//FORWARD RIGHT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+1;
			checkPoint[1]=currentLocation[1]+2;

			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){

					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

			}//end if(boxtocheck)


			//FORWARD LEFT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-1;
			checkPoint[1]=currentLocation[1]+2;

			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){

					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


			}//end if(boxtocheck)



			//LEFT FORWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-2;
			checkPoint[1]=currentLocation[1]+1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


			}//end if(boxtocheck)

			//LEFT BACKWARDS

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]-2;
			checkPoint[1]=currentLocation[1]-1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){

					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

			}//end if(boxtocheck)


			//BACKWARDS LEFT

			//make position and Point I want to check
			checkPoint[0]=currentLocation[0]-1;
			checkPoint[1]=currentLocation[1]-2;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


			}//end if(boxtocheck)

			//BACKWARDS RIGHT

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+1;
			checkPoint[1]=currentLocation[1]-2;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){

					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

			}//end if(boxtocheck)


			//RIGHT FORWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+2;
			checkPoint[1]=currentLocation[1]+1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){

					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

			}//end if(boxtocheck)

			//RIGHT BACKWARD

			//make position and Point I want to check

			checkPoint[0]=currentLocation[0]+2;
			checkPoint[1]=currentLocation[1]-1;
			//get Object at Position I want to check
			boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

			//check if we went off the grid
			if (boxToCheck!=0){


					allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


			}//end if(boxtocheck)


return allLegalMoves;

}//end getKnightMoves


movePackage chessHPCModel::getBishopMoves(int team, repast::Point<int> currentLocation){

			movePackage allLegalMoves;
			allLegalMoves.team=team;
			allLegalMoves.occupant=BISHOP;
			repast::Point<int> checkPoint(0,0);
			Box* boxToCheck;

						//RIGHT FORWARD

								for(int i=1; i<9; i++){


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else {//if hit edge break
										break;
									}


								}//end for loop



						//LEFT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


						//RIGHT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


						//LEFT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


return allLegalMoves;
}//end getBishopMoves
movePackage chessHPCModel::getBishopProtects(int team, repast::Point<int> currentLocation){

			movePackage allLegalMoves;
			allLegalMoves.team=team;
			allLegalMoves.occupant=BISHOP;
			repast::Point<int> checkPoint(0,0);
			Box* boxToCheck;

						//RIGHT FORWARD

								for(int i=1; i<9; i++){


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team-  add move but break since cannot protect further pieces
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and break since cannot protect further
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));



										}

									}//end if(boxtocheck)


									else {//if hit edge break
										break;
									}


								}//end for loop



						//LEFT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


						//RIGHT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


						//LEFT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop


return allLegalMoves;
}//end getBishopMoves


movePackage chessHPCModel::getRookMoves(int team, repast::Point<int> currentLocation){
	movePackage allLegalMoves;
				allLegalMoves.team=team;
				allLegalMoves.occupant=ROOK;
				repast::Point<int> checkPoint(0,0);
				Box* boxToCheck;

		//RIGHT

				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0]+i;
					checkPoint[1]=currentLocation[1];
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							// now should end the for loop, since i know cant continue forwards
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

		//LEFT

				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0]-i;
					checkPoint[1]=currentLocation[1];
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							// now should end the for loop, since i know cant continue forwards
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

		//FORWARD


				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0];
					checkPoint[1]=currentLocation[1]+i;
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							// now should end the for loop, since i know cant continue forwards
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

	//BACKWARDS
				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0];
					checkPoint[1]=currentLocation[1]-i;
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							// now should end the for loop, since i know cant continue forwards
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop


	return allLegalMoves;

}//end getRookMoves
movePackage chessHPCModel::getRookProtects(int team, repast::Point<int> currentLocation){
	movePackage allLegalMoves;
				allLegalMoves.team=team;
				allLegalMoves.occupant=ROOK;
				repast::Point<int> checkPoint(0,0);
				Box* boxToCheck;

		//RIGHT

				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0]+i;
					checkPoint[1]=currentLocation[1];
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

		//LEFT

				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0]-i;
					checkPoint[1]=currentLocation[1];
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

		//FORWARD


				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0];
					checkPoint[1]=currentLocation[1]+i;
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop

	//BACKWARDS
				for(int i=1; i<9; i++){

					//make position and Point I want to check

					checkPoint[0]=currentLocation[0];
					checkPoint[1]=currentLocation[1]-i;
					//get Object at Position I want to check
					boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

					//check if we went off the grid
					if (boxToCheck!=0){


						//get Occupant of box to check
						int occupantToCheck = boxToCheck->getOccupant();

						if (boxToCheck->getTeam()==team ){// same team- dont add move
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
							break;
						}

						else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//
							break;

						}else if (occupantToCheck==0 ){// empty-add move and keep looking
							allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

							//
							//

						}

					}//end if(boxtocheck)


					else{//if hit edge break
						break;
					}


				}//end for loop


	return allLegalMoves;

}//end getRookMoves


movePackage chessHPCModel::getQueenMoves(int team, repast::Point<int> currentLocation){


	movePackage allLegalMoves;
				allLegalMoves.team=team;
				allLegalMoves.occupant=QUEEN;
				repast::Point<int> checkPoint(0,0);
				Box* boxToCheck;

						//RIGHT

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop



						//LEFT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

						//FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

						//BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//RIGHT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//LEFT

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//RIGHT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)

									else{//if hit edge break
										break;
									}





								}//end for loop

					//LEFT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)

									else{//if hit edge break
										break;
									}





								}//end for loop


				return allLegalMoves;
}//end getQueenMoves
movePackage chessHPCModel::getQueenProtects(int team, repast::Point<int> currentLocation){

	//// // //Log4CL::instance()->get_logger("root").log(INFO,"getQueenMoves");
	movePackage allLegalMoves;
				allLegalMoves.team=team;
				allLegalMoves.occupant=QUEEN;
				repast::Point<int> checkPoint(0,0);
				Box* boxToCheck;

						//RIGHT

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team-  add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop



						//LEFT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

						//FORWARD
								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]+i;



									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));


									//check if we went off the grid
									if (boxToCheck!=0){

													/*
													std::stringstream m1;
													m1<< "checking point:"<<checkPoint[0]<<","<<checkPoint[1]<<" team="<<boxToCheck->getTeam();
													// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );
													*/


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											//// // //Log4CL::instance()->get_logger("root").log(INFO,"GETQUEEN MOVE FORWARD DIFFERNT TEAM AND NOT EMPTY");
											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											//// // //Log4CL::instance()->get_logger("root").log(INFO,"GETQUEEN MOVE FORWARD DIFFERNT TEAM AND EMPTY");
											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

						//BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//RIGHT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//LEFT

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)


									else{//if hit edge break
										break;
									}


								}//end for loop

					//RIGHT BACKWARDS

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]+i;
									checkPoint[1]=currentLocation[1]-i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)

									else{//if hit edge break
										break;
									}





								}//end for loop

					//LEFT FORWARD

								for(int i=1; i<9; i++){

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-i;
									checkPoint[1]=currentLocation[1]+i;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									//check if we went off the grid
									if (boxToCheck!=0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));
											break;
										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//
											break;

										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//
											//

										}

									}//end if(boxtocheck)

									else{//if hit edge break
										break;
									}





								}//end for loop


				return allLegalMoves;
}//end getQueenMoves


movePackage chessHPCModel::getKingMoves(int team, repast::Point<int> currentLocation){

					movePackage allLegalMoves;
					allLegalMoves.team=team;
					allLegalMoves.occupant=KING;

					std::vector<int> kingThreateners;
					repast::Point<int> checkPoint(0,0);
					Box* boxToCheck;


							//RIGHT

											//make position and Point I want to check

											checkPoint[0]=currentLocation[0]+1;
											checkPoint[1]=currentLocation[1];
											//get Object at Position I want to check
											boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

											//check if we went off the grid
										if(boxToCheck!=0){

											//get the pieces that threaten this move, depends on team of king
											if (team==1){
											kingThreateners= boxToCheck->getTeam2Contacts();
											}
											else if (team==2){
											kingThreateners= boxToCheck->getTeam1Contacts();
											}


											 //check that the box is unthreatened
											if (kingThreateners.size()==0){


												//get Occupant of box to check
												int occupantToCheck = boxToCheck->getOccupant();


												if (boxToCheck->getTeam()==team ){// same team- dont add move
													// now should end the for loop, since i know cant continue forwards

												}

												else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));




												}else if (occupantToCheck==0 ){// empty-add move and keep looking
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

													//
													//

												}

											}//end if(king.thretners.size)

											}//end if(boxtocheck)

											else{//if hit edge break

											}





							//RIGHT FORWARD


											//make position and Point I want to check

											checkPoint[0]=currentLocation[0]+1;
											checkPoint[1]=currentLocation[1]+1;
											//get Object at Position I want to check
											boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

										if(boxToCheck!=0){
											//get the pieces that threaten this move, depends on team of king
											if (team==1){
											kingThreateners= boxToCheck->getTeam2Contacts();
											}
											else if (team==2){
												kingThreateners= boxToCheck->getTeam1Contacts();
											}

											//check if we went off the grid
											if (kingThreateners.size()==0){


												//get Occupant of box to check
												int occupantToCheck = boxToCheck->getOccupant();

												if (boxToCheck->getTeam()==team ){// same team- dont add move

												}

												else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


												}else if (occupantToCheck==0 ){// empty-add move and keep looking
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

													//
													//

												}

											}//end if(kingthreateners.size)

											}//end if(boxtocheck)

											else{//if hit edge break

											}




							//RIGHT BACK


										//make position and Point I want to check

										checkPoint[0]=currentLocation[0]+1;
										checkPoint[1]=currentLocation[1]-1;

										//get Object at Position I want to check
										boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));


										if(boxToCheck!=0){

											//get the pieces that threaten this move, depends on team of king
											if (team==1){
												kingThreateners= boxToCheck->getTeam2Contacts();
											}
											else if (team==2){
												kingThreateners= boxToCheck->getTeam1Contacts();
											}


											//check if we went off the grid
											if (kingThreateners.size()==0){


												//get Occupant of box to check
												int occupantToCheck = boxToCheck->getOccupant();

												if (boxToCheck->getTeam()==team ){// same team- dont add move
													// now should end the for loop, since i know cant continue forwards


												}

												else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

													//
													//


												}else if (occupantToCheck==0 ){// empty-add move and keep looking
													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

													//
													//

												}

											}//end if(threateners.size()=0)

										}//end if(boxtocheck)

										else {//if hit edge break

										}

							//BACK


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]-1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

							if(boxToCheck!=0){

										//get the pieces that threaten this move, depends on team of king
										if (team == 1) {
											kingThreateners = boxToCheck->getTeam2Contacts();
										} else if (team == 2) {
											kingThreateners = boxToCheck->getTeam1Contacts();
										}

									//check if we went off the grid
									if (kingThreateners.size()==0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards

										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

										}

									}//end if(king.threater.size)

									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//FORWARD

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]+1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

							if(boxToCheck!=0){

										//get the pieces that threaten this move, depends on team of king
										if (team == 1) {
											kingThreateners = boxToCheck->getTeam2Contacts();
										} else if (team == 2) {
											kingThreateners = boxToCheck->getTeam1Contacts();
										}
									//check if we went off the grid
									if (kingThreateners.size()==0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards

										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));



										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}

									}//end if(kingthreateners.size)

									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT FORWARD

							//make position and Point I want to check

							checkPoint[0]=currentLocation[0]-1;
							checkPoint[1]=currentLocation[1]+1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if(boxToCheck!=0){

									//get the pieces that threaten this move, depends on team of king
									if (team == 1) {
										kingThreateners = boxToCheck->getTeam2Contacts();
									} else if (team == 2) {
										kingThreateners = boxToCheck->getTeam1Contacts();
									}
									//check if we went off the grid
									if (kingThreateners.size()==0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards

										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));

										}

									}//end if(kingthreateners.size)

									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-1;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if(boxToCheck!=0){

									//get the pieces that threaten this move, depends on team of king
									if (team == 1) {
										kingThreateners = boxToCheck->getTeam2Contacts();
									} else if (team == 2) {
										kingThreateners = boxToCheck->getTeam1Contacts();
									}

									//check if we went off the grid
									if (kingThreateners.size()==0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards

										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));



										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));



										}

									}//end if(kingthreateners.size)

									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT BACKWARDS



									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-1;
									checkPoint[1]=currentLocation[1]-1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

									if(boxToCheck!=0){

									if (team == 1) {
										kingThreateners = boxToCheck->getTeam2Contacts();
									} else if (team == 2) {
										kingThreateners = boxToCheck->getTeam1Contacts();
									}

									//check if we went off the grid
									if (kingThreateners.size()==0){


										//get Occupant of box to check
										int occupantToCheck = boxToCheck->getOccupant();

										if (boxToCheck->getTeam()==team ){// same team- dont add move
											// now should end the for loop, since i know cant continue forwards

										}

										else if ( boxToCheck->getTeam()!=team && occupantToCheck!=0 ){// different team-add move and end loop since cannot continue
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));



										}else if (occupantToCheck==0 ){// empty-add move and keep looking
											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}

									}//end if(kingthreatener.size)

									}//end if(boxtocheck)

									else{//if hit edge break

									}


	return allLegalMoves;
}//end getKingMoves
movePackage chessHPCModel::getKingProtects(int team, repast::Point<int> currentLocation){

					movePackage allLegalMoves;
					allLegalMoves.team=team;
					allLegalMoves.occupant=KING;

					std::vector<int> kingThreateners;
					repast::Point<int> checkPoint(0,0);
					Box* boxToCheck;


							//RIGHT

											//make position and Point I want to check

											checkPoint[0]=currentLocation[0]+1;
											checkPoint[1]=currentLocation[1];
											//get Object at Position I want to check
											boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

										if(boxToCheck!=0){


													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


											}//end if(boxtocheck)

											else{//if hit edge break

											}





							//RIGHT FORWARD


											//make position and Point I want to check

											checkPoint[0]=currentLocation[0]+1;
											checkPoint[1]=currentLocation[1]+1;
											//get Object at Position I want to check
											boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

										if(boxToCheck!=0){

													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


											}//end if(boxtocheck)

											else{//if hit edge break

											}




							//RIGHT BACK


										//make position and Point I want to check

										checkPoint[0]=currentLocation[0]+1;
										checkPoint[1]=currentLocation[1]-1;

										//get Object at Position I want to check
										boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));


										if(boxToCheck!=0){


													allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


										}//end if(boxtocheck)

										else {//if hit edge break

										}

							//BACK


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]-1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

							if(boxToCheck!=0){


											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//FORWARD

									//make position and Point I want to check

									checkPoint[0]=currentLocation[0];
									checkPoint[1]=currentLocation[1]+1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

							if(boxToCheck!=0){


											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT FORWARD

							//make position and Point I want to check

							checkPoint[0]=currentLocation[0]-1;
							checkPoint[1]=currentLocation[1]+1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if(boxToCheck!=0){


											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT


									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-1;
									checkPoint[1]=currentLocation[1];
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

								if(boxToCheck!=0){


											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


									}//end if(boxtocheck)

									else{//if hit edge break

									}

							//LEFT BACKWARDS



									//make position and Point I want to check

									checkPoint[0]=currentLocation[0]-1;
									checkPoint[1]=currentLocation[1]-1;
									//get Object at Position I want to check
									boxToCheck = grid->getObjectAt(repast::Point<int>(checkPoint[0],checkPoint[1]));

							if(boxToCheck!=0){


											allLegalMoves.legalMoves.push_back(repast::Point<int>(checkPoint[0],checkPoint[1]));


									}//end if(boxtocheck)

									else{//if hit edge break

									}

	return allLegalMoves;
}//end getKingMoves


void chessHPCModel::setupContacts(){
	//go through all boxes and initialize the contacts for each box
	// sets up the team contacts fo each box in order to use them for moves, payoff and other
	// // //Log4CL::instance()->get_logger("root").log(INFO,"running setupContacts()...");


	std::vector<PayONPos> bestMoves;


	std::vector<Box*> boxesToCheck;
	//get all local agents
	context->selectAgents(boxesToCheck,0,false); //get all agents of type 0

	/*
		std::stringstream m1;
		m1 << "setupContacts, boxesToCheck.size()="<< boxesToCheck.size()<<" , first selected agent:"<<boxesToCheck[0]->getId();
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );
*/

	std::vector<Box*>::iterator itt=boxesToCheck.begin();

	//!!!IMPORTANT all boxes need to be cleared before can (re)setup contacts!
	//first go through all boxes and clear the contacts
	while(itt !=boxesToCheck.end()){

		Box* boxToCheck = *itt;
		boxToCheck->clearAllContacts();

		itt++;
	}



	//reset iterator
	std::vector<Box*>::iterator it=boxesToCheck.begin();



	//go through all local boxes
	while(it !=boxesToCheck.end()){

			std::vector<int> currentLocation;


			//get Object at Position I want to check
			Box* boxToCheck = *it;



			//get objects location
			grid->getLocation(boxToCheck, currentLocation);

			//get its team
			int team= boxToCheck->getTeam();
			int occupant= boxToCheck->getOccupant();

			//make Point I want to check
			repast::Point<int> checkPoint(currentLocation[0], currentLocation[1]);

			std::stringstream m;
			movePackage allLegalMoves;

			//get its possible moves
			switch (occupant) {
			case 0:
				//if the querying box is empty
				//do nothing (??)
				break;
			case PAWNP:
				allLegalMoves = getPawnProtects(team, checkPoint);

				break;
			case KNIGHT:
				allLegalMoves = getKnightProtects(team, checkPoint);

				break;
			case BISHOP:
				allLegalMoves = getBishopProtects(team, checkPoint);

				break;
			case ROOK:
				allLegalMoves = getRookProtects(team, checkPoint);

				break;
			case QUEEN:
				allLegalMoves = getQueenProtects(team, checkPoint);

				break;
			case KING:
				allLegalMoves = getKingProtects(team, checkPoint);

				break;
			}//end switch



			//go through all moves and add piece to contacts
			for(std::size_t k=0;k<allLegalMoves.legalMoves.size();k++){

				repast::Point<int> move=allLegalMoves.legalMoves[k];


				//movePackage.legalMoves are repast::Point<int> 's so ok

				//get object at that location
				Box* box=grid->getObjectAt(move);


				if(team==1){
					//if the team that is checking is from team 1, add the contact to team1Contacts
					box->addTeam1Contact(occupant);
				}else if (team==2){//else add the contact to team2Contacts
					box->addTeam2Contact(occupant);
				}

			}//end for(allLegalMoves)

			it++;

	}//end while(it..)




	// // //Log4CL::instance()->get_logger("root").log(INFO,"finished setupContacts");


/*
  std::stringstream m;
	//test a box:
	Box* testBox=grid->getObjectAt(repast::Point<int> (3,originY+2));
	// // //Log4CL::instance()->get_logger("root").log(INFO,"gotBox");


	std::vector<int> testBoxteam1Contacts=testBox->getTeam1Contacts();
	// // //Log4CL::instance()->get_logger("root").log(INFO,"gotteam1Contacts");

	m.str("");
	m<<"occupant "<<testBox->getOccupant()<<" at position"<< 3 <<","<< originY+2 <<" , has team1Contacts of size "<< testBoxteam1Contacts.size()<<"\n 1."<<testBoxteam1Contacts[0]<<" 2."<<testBoxteam1Contacts[1]<<" 3."<<testBoxteam1Contacts[2];
	// // //Log4CL::instance()->get_logger("root").log( INFO,m.str() );
*/

if(moveCount==0){

	// // //Log4CL::instance()->get_logger("root").log(INFO, "STARTING GAME!!!!!!!!!!!!!!!!!!!!!!!!!");
	std::cout<<"\n \n \n";

}

}//end setupContacts()

void chessHPCModel::setBox(repast::Point<int> pointToSet, int occupantToSet, int teamToSet){

	//method to set the board in board representation (visualization)
	//// // //Log4CL::instance()->get_logger("root").log(INFO,"running setBox()");
	Box* boxToSet=grid->getObjectAt(pointToSet);


	boxToSet->setTeam(teamToSet);

	boxToSet->setOccupant(occupantToSet);

	setBoard(occupantToSet,teamToSet,pointToSet);

}


//iterator issue(changed to size_t)
void chessHPCModel::victoryCondition(){



	//checks if the king is checked, if he is checkd force the his team to either move him,
	//save him by eating his checker, or save him by moving someone between him and the matter
	//if none are available, declares check mate and ends agme

	// // //Log4CL::instance()->get_logger("root").log(INFO, "checking victory condition");
	std::vector<int> kingThreateners;
	movePackage kingPossibleMoves;
	std::vector<PayONPos> kingPayoffVector;

	PayONPos temp;

	std::vector<Box*> boxesToCheck;
	//get all local agents
	context->selectAgents(boxesToCheck,0,false); //disregard error, eclipse issue

	std::vector<Box*>::iterator it=boxesToCheck.begin();

	//go through all local boxes and find king (of team 1 and team2)
	while(it !=boxesToCheck.end()){


		std::vector<int> currentLocation;

		//get object, team and occupant
		Box* boxToCheck=*it;
		int team= boxToCheck->getTeam();
		int occupant= boxToCheck->getOccupant();

		//get object location
		grid->getLocation(boxToCheck,currentLocation);

		//make point object from ,location
		repast::Point<int> checkPoint(currentLocation[0],currentLocation[1]);

		//if the occupant is the king and it is the king on the team about to move
					if (occupant==KING && team==teamToMove){

						//check if he is threatened
						if(team==1){
							kingThreateners= boxToCheck->getTeam2Contacts();

						}else if(team==2){
							kingThreateners= boxToCheck->getTeam1Contacts();
						}



						//king is not checked, means we found a move that works
						if(kingThreateners.size()==0){
						// // //Log4CL::instance()->get_logger("root").log(INFO, " \nKING NOT checked, accept move");

									kingIsChecked=0;

									std::ofstream myfile;
									myfile.open("chessgame.txt",std::ios::app );
									myfile<< "KING of team "<<teamToMove<<" NOT checked, accept move \n";
									myfile.close();


									break; //end the while loop

						}

						//if kingThreateners is nonempty->check
						if(kingThreateners.size()!=0){

							kingIsChecked=1;

							// // //Log4CL::instance()->get_logger("root").log(INFO, " king is checked");

							std::stringstream m;
							m << " king of team "<< team <<" is threatened by" <<kingThreateners.size() << " pieces, 1st threatener: "<<kingThreateners[0];
							// // //Log4CL::instance()->get_logger("root").log(INFO, m.str());

							std::ofstream myfile;
							myfile.open("chessgame.txt", std::ios::app);
							myfile<< "king "<< team<< " is checked!\n";
							myfile.close();

							while(kingIsChecked==1){


								undoLastMove(finalMove);
								//erase the move from the possible moves, since it is not a possible move, as it leaves the king checked
								std::vector<PayONPos>::iterator it=find(allMoves.begin(), allMoves.end(), finalMove);
								allMoves.erase(it);

								if(allMoves.size()==0){
									//have no more moves to make and am checked
									//so we have a check mate

										//set the teamThatWon variable
										if(teamToMove==1){
											teamThatWon=2;
										}else if(teamToMove==2){
											teamThatWon=1;
										}
										teamThatLost=teamToMove;

									std::ofstream myfile;
									myfile.open("chessgame.txt", std::ios::app);
									myfile<< "Check Mate!  "<< teamToMove<< " loses!\n";
									myfile.close();


									if(rank==0){
									myfile.open("chessgameStats.txt", std::ios::app);
									myfile<< "Team "<< teamThatWon << " won in " <<moveCount<< " moves, entropy: "<<cut <<". \n";
									myfile.close();

									myfile.open("chessgameLength.txt", std::ios::app);
									myfile<<moveCount<<"\n";
									myfile.close();

									}

									m.str("");
									m<<"CHECK MATE! team "<<teamToMove<< " loses";
									// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );

									kingIsChecked=0;
									//stop the simulation
									repast::ScheduleRunner& runner =repast::RepastProcess::instance()->getScheduleRunner();
									runner.stop();



								}else{


									//now that i removed move that doesnt work, select the next best move and so on, until king is no longer checked
									finalMove=chessHPCModel::selectMove();
									chessHPCModel::makeMove(finalMove);
									kingIsChecked=victoryCheck();
								}


							}//end while(kingischecked=1)


						}//end if( kingThreateners.size()!=0 )


					}//end if(occupant==KING)


		it++;

	}//end while(it!=...)



	//now can update team that is moving and movecount

			//display occupants of box moved to and moved away from
			repast::Point<int> posMovedTo=finalMove.newPosition;
			repast::Point<int> posMovedFrom=finalMove.oldPosition;

			//now get the boxes at those locations
			Box* boxMovedFrom=grid->getObjectAt(posMovedFrom);
			Box* boxMovedTo=grid->getObjectAt(posMovedTo);

			int occupantOldPos=boxMovedFrom->getOccupant();
			int occupantNewPos=boxMovedTo->getOccupant();


			//set team to move and add move to vector of final moves
			if (teamToMove == 1) {

				allFinalMovesTeam1.push_back(finalMove);
				teamToMove = 2;

			} else if (teamToMove == 2) {

				allFinalMovesTeam2.push_back(finalMove);
				teamToMove = 1;

			}

			//update the move count
			moveCount = moveCount + 1;

			//update piece count
			// // //Log4CL::instance()->get_logger("root").log(INFO, " updating piece count ");
				int occupantThatWasEaten=finalMove.occupantThatWasEaten;
				int otherTeam=finalMove.otherTeam;
/*
std::stringstream m;
m<<"team1pieces size "<<team1pieces.size()<< " , team2pieces size "<< team2pieces.size();
// // //Log4CL::instance()->get_logger("root").log(INFO,m.str() );
*/


				if(occupantThatWasEaten!=0 && otherTeam==1){
					//remove occupant from team vector
					std::vector<int>::iterator it=std::find(team1pieces.begin(), team1pieces.end(), occupantThatWasEaten);
					//// // //Log4CL::instance()->get_logger("root").log(INFO, " found piece ");

					team1pieces.erase(it);

				}

				if(occupantThatWasEaten!=0 && otherTeam==2){
					//remove occupant from team vector
					std::vector<int>::iterator it=std::find(team2pieces.begin(), team2pieces.end(), occupantThatWasEaten);
					//// // //Log4CL::instance()->get_logger("root").log(INFO, " found piece ");
					team2pieces.erase(it);

				}



				//CHECK STILL ENOUGH PIECES

				//should also check the number of pieces left on the board
				if( (team1pieces.size()<=2 && team2pieces.size()<=2) || (team1pieces.size()==3 && team2pieces.size()==1) || (team1pieces.size()==1 && team2pieces.size()==3) ){

					//sum the pieces in order to determine draw combinations:
					int sum1pieces= std::accumulate(team1pieces.begin(), team1pieces.end(),0);
					int sum2pieces= std::accumulate(team2pieces.begin(), team2pieces.end(),0);

					//there can be no pawns:
					std::vector<int>::iterator it1=std::find(team1pieces.begin(), team1pieces.end(), PAWNP);
					std::vector<int>::iterator it2=std::find(team2pieces.begin(), team2pieces.end(), PAWNP);

					if( it1==team1pieces.end() && it2==team2pieces.end() ){//did not find any pawns


						if ((sum1pieces == 20 && sum2pieces == 20)// two kings
								|| (sum1pieces == 23 && sum2pieces == 20)//king bishop vs king
								|| (sum1pieces == 20 && sum2pieces == 23)
								|| (sum1pieces == 23 && sum2pieces == 23)
								|| (sum1pieces == 22 && sum2pieces == 20)//king knight vs king
								|| (sum1pieces == 20 && sum2pieces == 22)
								|| (sum1pieces == 22 && sum2pieces == 22)
								|| (sum1pieces == 24 && sum2pieces == 20)// king knight knight vs king
								|| (sum1pieces == 20 && sum2pieces == 24)
								|| (sum1pieces == 24 && sum2pieces == 24))
						{

							//declare draw
							std::ofstream myfile;
							if(rank==0){

								myfile.open("chessgameStats.txt", std::ios::app);
								myfile<< "Draw in " <<moveCount<< " moves, entropy: "<<cut <<" -not enough pieces. \n";
								myfile.close();

								myfile.open("chessgameLength.txt", std::ios::app);
								myfile<<moveCount<<"\n";
								myfile.close();

								myfile.open("chessgame.txt", std::ios::app);
								myfile<< "Draw in " <<moveCount<< " moves, entropy: "<<cut <<" -not enough pieces. \n";
								myfile.close();

							}


							//leave the weights
							//if we get a draw we store both sets of weights
							teamThatWon=3;
							teamThatLost=3;



							//stop simulation, and
							repast::ScheduleRunner& runner =repast::RepastProcess::instance()->getScheduleRunner();
							runner.stop();

						}
					}
				}




}//end victoryCondition()

int chessHPCModel::victoryCheck(){

	// // //Log4CL::instance()->get_logger("root").log(INFO, " victorycheck");
	//method returns 1 if king is checked, 0 if he is not
			std::vector<int> kingThreateners;
			movePackage kingPossibleMoves;
			std::vector<PayONPos> kingPayoffVector;

			PayONPos temp;

			std::vector<Box*> boxesToCheck;
			//get all local agents
			context->selectAgents(boxesToCheck,0,false); //disregard error, eclipse issue

			std::vector<Box*>::iterator it=boxesToCheck.begin();

			//go through all local boxes and find king (of team 1 and team2)
			while(it !=boxesToCheck.end()){

				std::vector<int> currentLocation;

				//get object, team and occupant
				Box* boxToCheck=*it;
				int team= boxToCheck->getTeam();
				int occupant= boxToCheck->getOccupant();

				//get object location
				grid->getLocation(boxToCheck,currentLocation);

				//make point object from ,location
				repast::Point<int> checkPoint(currentLocation[0],currentLocation[1]);

				//if the occupant is the king and it is the king on the team about to move
				if (occupant==KING && team==teamToMove){

					//check if he is threatened
					if(team==1){
						kingThreateners= boxToCheck->getTeam2Contacts();

					}else if(team==2){
						kingThreateners= boxToCheck->getTeam1Contacts();
					}

					if(kingThreateners.size()==0){


						kingIsChecked=0;




					}//end if(kingThr==0)
					else if(kingThreateners.size()!=0){
						kingIsChecked=1;

					}

					break;

				}//end if(occ=king...)

				it++;

			}//end while(it...)
			return kingIsChecked;

}

void chessHPCModel::requestAllAgents(){

	// // //Log4CL::instance()->get_logger("root").log(INFO,"running requestAllAgents()");
	//to share the ipa agent across processes

	//make an agentRequest object
	repast::AgentRequest req(rank);

	//make vector to put in agent to share
	std::vector<Box*> agentsToShare;

	//get the local agents to be shared
	context->selectAgents(repast::SharedContext<Box>::LOCAL,agentsToShare,0,false);

	int newRank;

	if(rank==0){
		newRank=1;
	}else if(rank==1){
		newRank=0;
	}

	std::stringstream m1;
	m1 << "before- process"<< rank << ": #of agents=" << context->size()<< "  agetnsToShare.size="<<agentsToShare.size() ;
	// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

	for(size_t j = 0; j < agentsToShare.size(); j++){

		//change agent id into id of process to share to
		repast::AgentId local=agentsToShare[j]->getId();

		repast::AgentId nonlocal(local.id(),newRank,0);

		nonlocal.currentRank(newRank);

		req.addRequest(nonlocal);


	}

					std::vector<repast::AgentId> reqId=req.requestedAgents();
					m1.str("");
					m1 << "1st requested id:"<< reqId[0] ;
					// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

	repast::RepastProcess::instance()->requestAgents<Box, BoxPackage,
			BoxPackageProvider, BoxPackageReceiver, BoxPackageReceiver>(
			*context, req, *provider, *receiver, *receiver);

	m1.str("");
	m1 << "after- process"<< rank << ": #of agents=" << context->size() ;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

}//end request()

void chessHPCModel::requestIPAAgents(){

	// // //Log4CL::instance()->get_logger("root").log(INFO,"running requestIPAAgents()");
	//to share the ipa agent across processes

	//make an agentRequest object
	repast::AgentRequest req(rank);

	//make vector to put in agent to share
	std::vector<Box*> agentsToShare;

	//get the local agents to be shared
	context->selectAgents(repast::SharedContext<Box>::LOCAL,agentsToShare,1,false);

	int newRank;

	if(rank==0){
		newRank=1;
	}else if(rank==1){
		newRank=0;
	}

	std::stringstream m1;
	m1 << "before- process"<< rank << ": #of agents=" << context->size()<< "  agetnsToShare.size="<<agentsToShare.size() ;
	// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

	for(size_t j = 0; j < agentsToShare.size(); j++){

		//change agent id into id of process to share to
		repast::AgentId local=agentsToShare[j]->getId();

		repast::AgentId nonlocal(local.id(),newRank,1);

		nonlocal.currentRank(newRank);

		req.addRequest(nonlocal);


	}

					std::vector<repast::AgentId> reqId=req.requestedAgents();
					m1.str("");
					m1 << "1st requested id:"<< reqId[0] ;
					// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

	repast::RepastProcess::instance()->requestAgents<Box, BoxPackage,
			BoxPackageProvider, BoxPackageReceiver, BoxPackageReceiver>(
			*context, req, *provider, *receiver, *receiver);

	m1.str("");
	m1 << "after- process"<< rank << ": #of agents=" << context->size() ;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

		// // //Log4CL::instance()->get_logger("root").log(INFO,"finished requestIPAAgents()");

}//end request()

void chessHPCModel::cancelAgentRequests(){



	repast::AgentRequest req(rank);

	std::vector<Box*> agentsToCancel;
	std::vector<Box*> agentsToRemove;


	context->selectAgents(repast::SharedContext<Box>::NON_LOCAL, agentsToCancel, false);
	context->selectAgents(agentsToRemove, false);



	std::stringstream m1;
	m1 << "agentsToCancel.size()= "<< agentsToCancel.size();
	// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );


	m1.str("");
	m1 << "before cancelRequests: process"<< rank << ": #of agents=" << context->size() ;
	// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

/*
	std::vector<Box*>::iterator it=agentsToCancel.begin();

	while(it !=agentsToCancel.end()){
		req.addCancellation((*it)->getId() );

		it++;
	}


	repast::RepastProcess::instance()->requestAgents<Box, BoxPackage,
			BoxPackageProvider, BoxPackageReceiver, BoxPackageReceiver>(
			*context, req, *provider, *receiver, *receiver);

*/

	//remove all agents
	std::vector<Box*>::iterator itt=agentsToRemove.begin();

	while(itt!=agentsToRemove.end()){
		context->removeAgent( (*itt)->getId() );
		itt++;
	}

			m1.str("");
			m1 << "after cancelRequests: process"<< rank << ": #of agents=" << context->size() ;
			// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );






}


void chessHPCModel::displayBoard0(){

	if(displayBoard==1){


	if (rank==0){


		//try writing somethig to a file
		std::ofstream myfile;
		myfile.open("chessgame.txt", std::ios::app);


		std::cout<<"\n";
		std::cout<<"move#"<<moveCount<<"\n";
		std::cout<<"\n";
		std::cout<<"team"<<teamToMove<<"'s turn\n";
		std::cout<<"\n";

				myfile<<"\n";
				myfile<<"move#"<<moveCount<<"\n";
				myfile<<"\n";
				myfile<<"team"<<teamToMove<<"'s turn\n";
				myfile<<"\n";




		for(int x=0;x<8;x++){
			std::cout<<x;
				myfile<<x;


		}

		std::cout<<"\n";
				myfile<<"\n";


		for(int x=0;x<8;x++){
					std::cout<<"_";
						myfile<<"_";

				}

		std::cout<<"\n";
			myfile<<"\n";

		for(int y=originY;y<originY+4;y++){

			for(int x=0;x<8;x++){

				std::cout<<board[x][y];
						myfile<<board[x][y];


			}

			std::cout<<"|"<<originY+y<<"\n";
					myfile<<"|"<<originY+y<<"\n";
		}


		myfile.close();


	}//end if(rank0)

	}//end diplayBoard==1

}//end displayBoard1

void chessHPCModel::displayBoard1(){

	if(displayBoard==1){

	if(rank==1){

				std::ofstream myfile;
				myfile.open("chessgame.txt", std::ios::app);



		for(int y=originY;y<originY+4;y++){

			for(int x=0;x<8;x++){

				std::cout<<board[x][y];
					myfile<<board[x][y];
			}

			std::cout<<"|"<<y<<"\n";
				myfile<<"|"<<y<<"\n";
		}

		std::cout<<"\n";
		std::cout<<"\n";
			myfile<<"\n\n";


		myfile.close();
	}//end if(rank)

	}//end diplayBoard==1



}//end displayBoard1

void chessHPCModel::synchStates(){


	//now SYNCHRONIZE the values across processes
	repast::RepastProcess::instance()->synchronizeAgentStates<BoxPackage, BoxPackageProvider, BoxPackageReceiver>(*provider, *receiver ); //synchronizeAgentStates
	// // //Log4CL::instance()->get_logger("root").log(INFO, " syncAgentSTATES done");


}

std::vector<Box*> chessHPCModel::pawnEnd(int team){

	std::vector<Box*> pawnEndBoxes;

	if(team==1){
		// get all the boxes in the last row, which is for team 1 case row (0-7,7)
		for(int i=0;i<8;i++){

			//get the Box at that location:
			Box* boxToCheck=grid->getObjectAt(repast::Point<int>(i,7));
			int occupant=boxToCheck->getOccupant();
			int team=boxToCheck->getTeam();
			if(occupant==PAWNP && team==1){
				//add the box to the output vector
				pawnEndBoxes.push_back(boxToCheck);

			}
		}

	}//end if team =1

	if(team==2){
		// get all the boxes in the last row, which is for team 2 case row (0-7,0)
		for(int i=0;i<8;i++){

			//get the Box at that location:
			Box* boxToCheck=grid->getObjectAt(repast::Point<int>(i,0));
			int occupant=boxToCheck->getOccupant();
			int team=boxToCheck->getTeam();
			if(occupant==PAWNP && team==2){
				//add the box to the output vector
				pawnEndBoxes.push_back(boxToCheck);

			}
		}

	}//end if team =2

	return pawnEndBoxes;

}

void chessHPCModel::promotePawn(Box* pawnEndBox){

	//make a vector of possibel new occupants
	int newOccupant;

	//90% of the time pick queen
	double e=rand()/double(RAND_MAX);
		if(e>.9){
			newOccupant=KNIGHT;
		}else if(e<.9){
			newOccupant=QUEEN;
		}

	//promote the pawn
	pawnEndBox->setOccupant(newOccupant);

	//change the board
	std::vector<int> loc;
	grid->getLocation(pawnEndBox,loc);

	repast::Point<int> pos(loc[0],loc[1]);
	setBoard(newOccupant, pawnEndBox->team, pos);

	//update team piece vector:
	if(pawnEndBox->team==1){

		std::vector<int>::iterator it=std::find(team1pieces.begin(), team1pieces.end(), 1);
		// // //Log4CL::instance()->get_logger("root").log(INFO, " found piece ");
		//remove pawn
		team1pieces.erase(it);

		//add new piece
		team1pieces.push_back(newOccupant);


	}
	//update team piece vector:
	if(pawnEndBox->team==2){

		std::vector<int>::iterator it=std::find(team2pieces.begin(), team2pieces.end(), 1);
		//remove pawn
		team2pieces.erase(it);

		//add new piece
		team2pieces.push_back(newOccupant);


	}

	std::ofstream myfile;
	if(rank==0){
		myfile.open("chessgame.txt", std::ios::app);
		myfile<< "Promoted a pawn!\n";
		myfile.close();
	}

	setupContacts();
}

void chessHPCModel::learning(){

	//check if a pawn has reached end of board, if so change him to another piece:
	std::vector<Box*> pawnEnd1=pawnEnd(1);
	std::vector<Box*> pawnEnd2=pawnEnd(2);

	//check if the vectors are empty
	//if they are not, promote the pawns

			if(pawnEnd1.size()!=0){
				for(int i=0;i<pawnEnd1.size();i++){
					promotePawn(pawnEnd1[i]);
				}

			}

			if(pawnEnd2.size()!=0){
				for(int i=0;i<pawnEnd2.size();i++){
					promotePawn(pawnEnd2[i]);
				}

			}



	//method to be put in scheduler. performs reinforcement learning
	//and stores last two selected moves to be used in learning

	//to make sure the right team is learning need to skip a step for each team

	if(moveCount>2){
		//only if the move count is greater than 2, otherwise have no best move to use...

		//do reinforcement learning on last best move
		if( finalMove.team==1 && team1ReinforcementLearning==1 ){
			reinforcementLearning(lastBestMove1, finalMove);
		}
		else if( finalMove.team==2 && team2ReinforcementLearning==1 ){
			reinforcementLearning(lastBestMove2, finalMove);
		}

	}

	//this will be scheduled after the victory condition
	//store the finalMove in a last bestMove,according to team
	if(finalMove.team==1){
		//store last best move
		lastBestMove1.occupant=finalMove.occupant;
		lastBestMove1.team=finalMove.team;
		lastBestMove1.oldPosition=finalMove.oldPosition;
		lastBestMove1.newPosition=finalMove.newPosition;
		lastBestMove1.payoff=finalMove.payoff;
		lastBestMove1.lastInputs=finalMove.lastInputs;

	}else if(finalMove.team==2){
		lastBestMove2.occupant=finalMove.occupant;
		lastBestMove2.team=finalMove.team;
		lastBestMove2.oldPosition=finalMove.oldPosition;
		lastBestMove2.newPosition=finalMove.newPosition;
		lastBestMove2.payoff=finalMove.payoff;
		lastBestMove2.lastInputs=finalMove.lastInputs;

	}

}

void chessHPCModel::reinforcementLearning(PayONPos lastBestMove, PayONPos bestMove){


	//// // //Log4CL::instance()->get_logger("root").log(INFO," reinforcement learning");


	//update alpha
	//alpha=alpha/moveCount;

	//method that updates weights of payoff function
	//using a stochastic gradient descent method
	//w(t+1)=w(t)+a*(Y(t+1)-Y(t))*delw(Y(t))

	//for now since the payoff function is linear can easily compute delw(Y)
	//but should change later to take derivative explicitly

	//will take last selected move and piece, then let opponent move, then take new board configuration and
	//simulate another move of the same piece and use this for the update method


	//get info from lastBestMove


	//normalize all the weights








	int occupant=lastBestMove.occupant;
	repast::Point<int> oldPosition=lastBestMove.oldPosition;
	repast::Point<int> newPosition=lastBestMove.newPosition;
	int team=lastBestMove.team;
	int lastPayoff=lastBestMove.payoff;

	//now simulate a next move using the current board configuration

	//first get the current box my piece is at:
	Box* box=grid->getObjectAt(newPosition);


		//now i get the last inputs that were used to compute the payoff in the previous time step, so I get them from the previous next position, ie the postion I moved to
		LastInputs lastInputs=lastBestMove.lastInputs;

		//test the operator
		std::ofstream myfile1;
		myfile1.open("chessgame.txt", std::ios::app);
		myfile1<< "\n Reinforcement Learning position: "<<newPosition[0]<<","<<newPosition[1] <<" team"<< team <<": lastInputs.qOcc="<<lastInputs.qOccupant<<" , lastInputs.t1c = "<< lastInputs.sumTeam1Contacts;
		myfile1.close();

		/*
		std::stringstream m1;
		m1 << " Reinforcement Learning newPosition="<< newPosition[0] <<","<<newPosition[1];
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

		m1.str("");
		m1 << " Reinforcement Learning team"<< team <<": lastInputs.qOcc="<<lastInputs.qOccupant<<" , lastInputs.t1c = "<< lastInputs.sumTeam1Contacts<< " and box.lastInputs.t1c = "<< box->lastInputs.sumTeam1Contacts;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );

		 */

	int nextPayoff=gamma*bestMove.payoff;

	int nextTeam=bestMove.team;

	/*				//SIMULATING MOVE INSTEAD OF USING NEXT ACTUAL MOVE
					//now simulate a next move using the current board configuration

					//first get the current box my piece is at:
					Box* box=grid->getObjectAt(newPosition); //should be occupant that just moved

					//then from this box get all its moves
					std::vector<PayONPos> simulatedMoves=chessHPCModel::getAllMovesPayoffs(box);

					//from these moves find move with max payoff
					PayONPos bestSimulatedMove =*std::max_element(simulatedMoves.begin(), simulatedMoves.end(), PayONPosFunctor() );

					//if several moves have same simulated moves, may have different bestSimulated move on different processors, so need to share seleceted move across processes
					std::vector<Box*> ipaBoxs;

					//to make sure both processors have the same ordering:
						//get the ipa agents


						 context->selectAgents(repast::SharedContext<Box>::LOCAL,ipaBoxs,1,false);

						//on rank 0 i set bestSimulatedMove for both agents to allMoves, now they both have the allMove vector from process 0.

						ipaBoxs[0]->bestMove[0]=bestSimulatedMove;



				//now I synchronize across processes...
				repast::RepastProcess::instance()->synchronizeAgentStates<BoxPackage, BoxPackageProvider, BoxPackageReceiver>(*provider, *receiver ); //synchronizeAgentStates

				if(rank==1){
				//now reset the bestSimulatedMove on process 1
				std::vector<Box*> ipaLBoxs; context->selectAgents(repast::SharedContext<Box>::NON_LOCAL,ipaLBoxs,1,false);



				//from this nonlocal agent retrieve the bestMove vector
				bestSimulatedMove=ipaLBoxs[0]->bestMove[0];
				}

				//now the simulated move selected will be the same on both processes.

				//now i only need the payoff from this move to update my value function
				int nextPayoff=bestSimulatedMove.payoff;

	*/


	//now preform update using stoch grad desc
	//check which piece needs his weights updated:
	std::vector<double> column;
	std::ofstream myfile;
/*
		m1.str("");
		m1 << "reinforcement learning- occupant"<< occupant << " , oldTeam " << team<< " nextTeam "<< nextTeam<<" payoff diff="<<nextPayoff-lastPayoff;
		// // //Log4CL::instance()->get_logger("root").log( INFO,m1.str() );
*/

				myfile1.open("chessgame.txt", std::ios::app);
				myfile1<<"\n Reinforcement Learning, occupant"<< occupant <<" , team "<< team<<" , alpha="<<alpha<<", payoff difference="<<nextPayoff-lastPayoff<<" , nextPayoff= "<<nextPayoff<<" lastPayoff="<<lastPayoff<<"\n";
				myfile1.close();




	switch (occupant)
	{
	case PAWNP:
		//// // //Log4CL::instance()->get_logger("root").log(INFO," pawn learning");
		//store all the weights in order to see their evolution


		//store all current weights
		//see getPayoff in Box.cpp for details of derivative with respect to weights...

		//store weights in ts(timeseries variable)
		if(team==1){



		w_pawn1.at(0)=w_pawn1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_pawn1.at(1)=w_pawn1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_pawn1.at(2)=w_pawn1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_pawn1.at(3)=w_pawn1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_pawn1.at(4)=w_pawn1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(PAWNP);

		if(rank==0){


		//write weights to txt file
		std::ofstream myfile;
		myfile.open("timeseries/pawn1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_pawn1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}


		}

		if(team==2){




		w_pawn2.at(0)=w_pawn2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_pawn2.at(1)=w_pawn2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_pawn2.at(2)=w_pawn2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_pawn2.at(3)=w_pawn2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_pawn2.at(4)=w_pawn2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(PAWNP);

		if(rank==0){


		//write weights to txt file
		std::ofstream myfile;
		myfile.open("timeseries/pawn2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_pawn2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}
		}


		break;

	case KNIGHT:
		//// // //Log4CL::instance()->get_logger("root").log(INFO," knight learning");
		//store all the weights in order to see their evolution


		//stored all current weights

		if(team==1){



		w_knight1.at(0)=w_knight1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_knight1.at(1)=w_knight1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_knight1.at(2)=w_knight1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_knight1.at(3)=w_knight1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_knight1.at(4)=w_knight1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(KNIGHT);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/knight1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_knight1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}
		}

		if(team==2){


		w_knight2.at(0)=w_knight2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_knight2.at(1)=w_knight2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_knight2.at(2)=w_knight2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_knight2.at(3)=w_knight2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_knight2.at(4)=w_knight2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(KNIGHT);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/knight2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_knight2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}
		}


		break;
	case BISHOP:
		//// // //Log4CL::instance()->get_logger("root").log(INFO," bishop learning");
		//store all the weights in order to see their evolution



		//stored all current weights
		if(team==1){



		w_bishop1.at(0)=w_bishop1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_bishop1.at(1)=w_bishop1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_bishop1.at(2)=w_bishop1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_bishop1.at(3)=w_bishop1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_bishop1.at(4)=w_bishop1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(BISHOP);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/bishop1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_bishop1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}

		if(team==2){


		w_bishop2.at(0)=w_bishop2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_bishop2.at(1)=w_bishop2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_bishop2.at(2)=w_bishop2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_bishop2.at(3)=w_bishop2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_bishop2.at(4)=w_bishop2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(BISHOP);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/bishop2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_bishop2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}


		break;
	case ROOK:
		//// // //Log4CL::instance()->get_logger("root").log(INFO," rook learning");

		if(team==1){



		w_rook1.at(0)=w_rook1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_rook1.at(1)=w_rook1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_rook1.at(2)=w_rook1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_rook1.at(3)=w_rook1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_rook1.at(4)=w_rook1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(ROOK);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/rook1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_rook1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}

		if(team==2){




		w_rook2.at(0)=w_rook2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_rook2.at(1)=w_rook2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_rook2.at(2)=w_rook2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_rook2.at(3)=w_rook2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_rook2.at(4)=w_rook2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(ROOK);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/rook2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_rook2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}


		break;
	case QUEEN:

		//// // //Log4CL::instance()->get_logger("root").log(INFO," queen learning");


		if(team==1){




		w_queen1.at(0)=w_queen1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_queen1.at(1)=w_queen1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_queen1.at(2)=w_queen1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_queen1.at(3)=w_queen1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_queen1.at(4)=w_queen1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(QUEEN);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/queen1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_queen1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}

		if(team==2){


		w_queen2.at(0)=w_queen2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_queen2.at(1)=w_queen2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_queen2.at(2)=w_queen2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_queen2.at(3)=w_queen2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_queen2.at(4)=w_queen2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(QUEEN);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/queen2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_queen2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}


		break;
	case KING:
		//// // //Log4CL::instance()->get_logger("root").log(INFO," king learning");

		if(team==1){




		w_king1.at(0)=w_king1.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_king1.at(1)=w_king1.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_king1.at(2)=w_king1.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_king1.at(3)=w_king1.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_king1.at(4)=w_king1.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(KING);

		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/king1_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_king1.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}

		if(team==2){



		w_king2.at(0)=w_king2.at(0)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT0;
		w_king2.at(1)=w_king2.at(1)+alpha*(nextPayoff-lastPayoff)*lastInputs.willPandwillT1;
		w_king2.at(2)=w_king2.at(2)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam1Contacts;
		w_king2.at(3)=w_king2.at(3)+alpha*(nextPayoff-lastPayoff)*lastInputs.sumTeam2Contacts;
		w_king2.at(4)=w_king2.at(4)+alpha*(nextPayoff-lastPayoff)*lastInputs.Occupant;

		normalizeWeights(KING);


		if(rank==0){



		std::ofstream myfile;
		myfile.open("timeseries/king2_wts.txt", std::ios::app);

		//write the weights to a file
			for(int i=0;i<n;i++){
				myfile<<w_king2.at(i)<<" ";

			}
				myfile<<"\n";

		myfile.close();

		}

		}


		break;
	}






}

void chessHPCModel::finalReinforcementLearning() {
	// // //Log4CL::instance()->get_logger("root").log(INFO,	"final reinforcement learning");

	//normalize all the weights
			normalizeWeights(PAWNP);
			normalizeWeights(KNIGHT);
			normalizeWeights(BISHOP);
			normalizeWeights(ROOK);
			normalizeWeights(QUEEN);
			normalizeWeights(KING);

			PayONPos wReward;
			wReward.payoff=10;
			wReward.team=teamThatWon;

			PayONPos lReward;
			lReward.payoff=-10;
			lReward.team=teamThatLost;

			PayONPos tReward1;
			tReward1.payoff=0;
			tReward1.team=1;

			PayONPos tReward2;
			tReward2.payoff=0;
			tReward2.team=2;

			// // //Log4CL::instance()->get_logger("root").log(INFO," made rewards");

			if(allFinalMovesTeam1.size()!=0 && allFinalMovesTeam2.size()!=0){

	//perform the final learning using the +1/-1 as reward for team that won/lost, 0 if they tie, applies learning to all moves in the game.
			if(teamThatWon==1){

					for(int i=0;i<allFinalMovesTeam1.size();i++){
						reinforcementLearning(allFinalMovesTeam1[i],wReward);
					}

					for(int i=0;i<allFinalMovesTeam2.size();i++){
						reinforcementLearning(allFinalMovesTeam2[i],lReward);
					}

			}else if (teamThatWon==2){

				// // //Log4CL::instance()->get_logger("root").log(INFO," team 2 won ");

					for(int i=0;i<allFinalMovesTeam2.size();i++){
						reinforcementLearning(allFinalMovesTeam2[i],wReward);
					}

					for(int i=0;i<allFinalMovesTeam1.size();i++){
						reinforcementLearning(allFinalMovesTeam1[i],lReward);
					}


			}
/*			else if (teamThatWon==3){

					for(int i=0;i<allFinalMovesTeam1.size();i++){
						reinforcementLearning(allFinalMovesTeam1[i],tReward1);
					}

					for(int i=0;i<allFinalMovesTeam2.size();i++){
						reinforcementLearning(allFinalMovesTeam2[i],tReward2);
					}

			}
*/

			}//end if allTeamMoves emtpy...


			if(rank==0){
			std::ofstream myfile;
			myfile.open("chessgame.txt", std::ios::app);
			myfile << "finalRL: teamThatLost = " << teamThatLost << " , teamThatWon = "
					<< teamThatWon << ".\n";
			myfile.close();

			myfile.open("chessgameWinners", std::ios::app);
			myfile << teamThatWon<< "\n";
			myfile.close();
			}
}


void chessHPCModel::plotWeights(){

	// // //Log4CL::instance()->get_logger("root").log(INFO," plotting weights");
	//plotKnight1Weights();
	//plotRook1Weights();
	//plotPawn1Weights();

	//plotKnight2Weights();
	//plotRook2Weights();
	//plotPawn2Weights();

	//do last reinforcement learning
	if(moveCount>2){
		//only if the move count is greater than 2, otherwise have no best move to use...

		//do reinforcement learning on last best move
		if(finalMove.team==1 && team1ReinforcementLearning==1){
			reinforcementLearning(lastBestMove1, finalMove);
		}
		else if(finalMove.team==2 && team2ReinforcementLearning==1){
			reinforcementLearning(lastBestMove2, finalMove);
		}

	}


	//write weights to text files
	finalReinforcementLearning();
	// // //Log4CL::instance()->get_logger("root").log(INFO," finished final RL");
	//write final weights to txt file:

	if(rank==0){
		writeLastWeights(teamThatWon);
		writeLastWeights(teamThatLost);
	}


}


void chessHPCModel::writeLastWeights(int team){

	// // //Log4CL::instance()->get_logger("root").log(INFO," writing last weights");
		writeLastPawnWeights(team);
		writeLastBishopWeights(team);
		writeLastKnightWeights(team);
		writeLastRookWeights(team);
		writeLastQueenWeights(team);
		writeLastKingWeights(team);

		std::ofstream myfile;
		myfile.open("chessgame.txt", std::ios::app);
		myfile<< "Writing final weights to txt file. \n";
		myfile.close();

		//clearAllTxtFiles();



}
void chessHPCModel::writeLastPawnWeights(int team){


	std::ofstream myfile;

	if(team==1 || team==3){


	myfile.open("weights/pawn_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_pawn1[i] <<"\n";
	}

	myfile.close();

	}//end team 1

	if(team==2 || team==3){

	myfile.open("weights/pawn_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_pawn2[i] <<"\n";
	}

	myfile.close();

	}//end team2

}

void chessHPCModel::writeLastKnightWeights(int team){
	std::ofstream myfile;

	if(team==1 || team==3){
	myfile.open("weights/knight_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_knight1[i] <<"\n";
	}

	myfile.close();
	}//end team1

	if(team==2 || team==3){

	myfile.open("weights/knight_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_knight2[i] <<"\n";
	}

	myfile.close();

	}//end team2
}

void chessHPCModel::writeLastBishopWeights(int team){
	std::ofstream myfile;

	if(team==1 || team==3){
	myfile.open("weights/bishop_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_bishop1[i] <<"\n";
	}

	myfile.close();
	}//end team1

	if(team==2 || team==3){
	myfile.open("weights/bishop_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_bishop2[i] <<"\n";
	}

	myfile.close();

	}//end team2
}

void chessHPCModel::writeLastRookWeights(int team){
	std::ofstream myfile;

	if(team==1 || team==3){
	myfile.open("weights/rook_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_rook1[i] <<"\n";
	}

	myfile.close();
	}

	if(team==2 || team==3){

	myfile.open("weights/rook_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_rook2[i] <<"\n";
	}

	myfile.close();

	}
}

void chessHPCModel::writeLastQueenWeights(int team){
	std::ofstream myfile;

	if(team==1 || team==3){
	myfile.open("weights/queen_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_queen1[i] <<"\n";
	}

	myfile.close();

	}

	if(team==2 || team==3){
	myfile.open("weights/queen_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_queen2[i] <<"\n";
	}

	myfile.close();
	}
}

void chessHPCModel::writeLastKingWeights(int team){
	std::ofstream myfile;

	if(team==1 || team==3){
	myfile.open("weights/king_weights1.txt");

	for(int i=0;i<n;i++){
		myfile<<w_king1[i] <<"\n";
	}

	myfile.close();
	}

	if(team==2 || team==3){

	myfile.open("weights/king_weights2.txt");

	for(int i=0;i<n;i++){
		myfile<<w_king2[i] <<"\n";
	}

	myfile.close();

	}

}


/*
void chessHPCModel::plotKnight1Weights(){


	  mglGraph gr;
	  //gr.FPlot("sin(pi*x)");
	  //gr.WriteFrame("pawnTimeSeries.png");

	  //try plotting the pawn weight time series from pawn_ts

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;


	  double k0[knight1_ts.size()];  //this gives size of row
	  double k1[knight1_ts.size()];
	  double k2[knight1_ts.size()];
	  double k3[knight1_ts.size()];
	  double k4[knight1_ts.size()];

	  double x[knight1_ts.size()];


	  for(int i=0;i<knight1_ts.size();i++){
		  k0[i]=knight1_ts[i][0];
		  k1[i]=knight1_ts[i][1];
		  k2[i]=knight1_ts[i][2];
		  k3[i]=knight1_ts[i][3];
		  k4[i]=knight1_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,knight1_ts.size() );
	  y1.Set( k1,knight1_ts.size()  );
	  y2.Set( k2,knight1_ts.size()  );
	  y3.Set( k3,knight1_ts.size()  );
	  y4.Set( k4,knight1_ts.size()  );
	  xx.Set( x,knight1_ts.size()   );

	 // y.Set( k4,knight1_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,knight1_ts.size(),0,1);
	  gr.SetRange('x',0,knight1_ts.size());

	  gr.SetRange('y',0,1);
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-protectors","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-threateners","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=knight1_ts.size()/2;

	  gr.Legend(0,0);


	  gr.Title("Knight1 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/knight1WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/knight1WeightsTimeSeries.png","",false);
}
void chessHPCModel::plotKnight2Weights(){


	  mglGraph gr;
	  //gr.FPlot("sin(pi*x)");
	  //gr.WriteFrame("pawnTimeSeries.png");

	  //try plotting the pawn weight time series from pawn_ts

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;


	  double k0[knight2_ts.size()];  //this gives size of row
	  double k1[knight2_ts.size()];
	  double k2[knight2_ts.size()];
	  double k3[knight2_ts.size()];
	  double k4[knight2_ts.size()];

	  double x[knight2_ts.size()];


	  for(int i=0;i<knight2_ts.size();i++){
		  k0[i]=knight2_ts[i][0];
		  k1[i]=knight2_ts[i][1];
		  k2[i]=knight2_ts[i][2];
		  k3[i]=knight2_ts[i][3];
		  k4[i]=knight2_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,knight2_ts.size() );
	  y1.Set( k1,knight2_ts.size()  );
	  y2.Set( k2,knight2_ts.size()  );
	  y3.Set( k3,knight2_ts.size()  );
	  y4.Set( k4,knight2_ts.size()  );
	  xx.Set( x,knight2_ts.size()   );

	 // y.Set( k4,knight2_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,knight2_ts.size(),0,1);
	  gr.SetRange('x',0,knight2_ts.size());

	  gr.SetRange('y',0,1);
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-threateners","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-protectors","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=knight2_ts.size()/2;

	  gr.Legend(0,0);


	  gr.Title("Knight2 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/knight1WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/knight2WeightsTimeSeries.png","",false);
}


void chessHPCModel::plotPawn1Weights(){


	  mglGraph gr;
	  //gr.FPlot("sin(pi*x)");
	  //gr.WriteFrame("pawnTimeSeries.png");

	  //try plotting the pawn weight time series from pawn_ts

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;


	  double k0[pawn1_ts.size()];  //this gives size of row
	  double k1[pawn1_ts.size()];
	  double k2[pawn1_ts.size()];
	  double k3[pawn1_ts.size()];
	  double k4[pawn1_ts.size()];

	  double x[pawn1_ts.size()];


	  for(int i=0;i<pawn1_ts.size();i++){
		  k0[i]=pawn1_ts[i][0];
		  k1[i]=pawn1_ts[i][1];
		  k2[i]=pawn1_ts[i][2];
		  k3[i]=pawn1_ts[i][3];
		  k4[i]=pawn1_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,pawn1_ts.size() );
	  y1.Set( k1,pawn1_ts.size()  );
	  y2.Set( k2,pawn1_ts.size()  );
	  y3.Set( k3,pawn1_ts.size()  );
	  y4.Set( k4,pawn1_ts.size()  );
	  xx.Set( x,pawn1_ts.size()   );

	 // y.Set( k4,pawn1_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,pawn1_ts.size(),0,1);
	  gr.SetRange('x',0,pawn1_ts.size());

	  gr.SetRange('y',0,1);
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-protectors","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-threateners","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=pawn1_ts.size()/2;

	  gr.Legend(0,0);


	  gr.Title("Pawn1 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/pawn1WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/pawn1WeightsTimeSeries.png","",false);
}
void chessHPCModel::plotPawn2Weights(){


	  mglGraph gr;
	  //gr.FPlot("sin(pi*x)");
	  //gr.WriteFrame("pawnTimeSeries.png");

	  //try plotting the pawn weight time series from pawn_ts

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;


	  double k0[pawn2_ts.size()];  //this gives size of row
	  double k1[pawn2_ts.size()];
	  double k2[pawn2_ts.size()];
	  double k3[pawn2_ts.size()];
	  double k4[pawn2_ts.size()];

	  double x[pawn2_ts.size()];


	  for(int i=0;i<pawn2_ts.size();i++){
		  k0[i]=pawn2_ts[i][0];
		  k1[i]=pawn2_ts[i][1];
		  k2[i]=pawn2_ts[i][2];
		  k3[i]=pawn2_ts[i][3];
		  k4[i]=pawn2_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,pawn2_ts.size() );
	  y1.Set( k1,pawn2_ts.size()  );
	  y2.Set( k2,pawn2_ts.size()  );
	  y3.Set( k3,pawn2_ts.size()  );
	  y4.Set( k4,pawn2_ts.size()  );
	  xx.Set( x,pawn2_ts.size()   );

	 // y.Set( k4,pawn2_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,pawn2_ts.size(),0,1);
	  gr.SetRange('x',0,pawn2_ts.size());

	  gr.SetRange('y',0,1);
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-threateners","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-protectors","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=pawn2_ts.size()/2;

	  gr.Legend(0,.8);


	  gr.Title("Pawn2 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/pawn2WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/pawn2WeightsTimeSeries.png","",false);
}


void chessHPCModel::plotRook1Weights(){


	  mglGraph gr;
	  //gr.FPlot("sin(pi*x)");
	  //gr.WriteFrame("pawnTimeSeries.png");

	  //try plotting the pawn weight time series from pawn_ts

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;


	  double k0[rook1_ts.size()];  //this gives size of row
	  double k1[rook1_ts.size()];
	  double k2[rook1_ts.size()];
	  double k3[rook1_ts.size()];
	  double k4[rook1_ts.size()];

	  double x[rook1_ts.size()];


	  for(int i=0;i<rook1_ts.size();i++){
		  k0[i]=rook1_ts[i][0];
		  k1[i]=rook1_ts[i][1];
		  k2[i]=rook1_ts[i][2];
		  k3[i]=rook1_ts[i][3];
		  k4[i]=rook1_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,rook1_ts.size() );
	  y1.Set( k1,rook1_ts.size()  );
	  y2.Set( k2,rook1_ts.size()  );
	  y3.Set( k3,rook1_ts.size()  );
	  y4.Set( k4,rook1_ts.size()  );
	  xx.Set( x,rook1_ts.size()   );

	 // y.Set( k4,rook1_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,rook1_ts.size(),0,1);
	  gr.SetRange('x',0,rook1_ts.size());

	  gr.SetRange('y',0,1);
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-protectors","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-threateners","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=rook1_ts.size()/2;

	  gr.Legend(0,0);


	  gr.Title("Rook1 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/rook1WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/rook1WeightsTimeSeries.png","",false);
}
void chessHPCModel::plotRook2Weights(){

	mglGraph gr;

	  mglData y0;
	  mglData y1;
	  mglData y2;
	  mglData y3;
	  mglData y4;
	  mglData xx;



	  double k0[rook2_ts.size()];  //this gives size of row
	  double k1[rook2_ts.size()];
	  double k2[rook2_ts.size()];
	  double k3[rook2_ts.size()];
	  double k4[rook2_ts.size()];

	  double x[rook2_ts.size()];

	  //find maximum
	    double max = 0;

	    for(int i = 0; i < rook2_ts.size(); i++){
	        for(int j = 0; j< n; j++){
	            if(max < rook2_ts[i][j])
	            {
	                max = rook2_ts[i][j];

	            }
	        }
	    }

	  for(int i=0;i<rook2_ts.size();i++){
		  k0[i]=rook2_ts[i][0];
		  k1[i]=rook2_ts[i][1];
		  k2[i]=rook2_ts[i][2];
		  k3[i]=rook2_ts[i][3];
		  k4[i]=rook2_ts[i][4];

		  //vector for axis
		  x[i]=i;

	  }

	  //now plot in mathgl
	  y0.Set( k0,rook2_ts.size() );
	  y1.Set( k1,rook2_ts.size()  );
	  y2.Set( k2,rook2_ts.size()  );
	  y3.Set( k3,rook2_ts.size()  );
	  y4.Set( k4,rook2_ts.size()  );
	  xx.Set( x,rook2_ts.size()   );

	 // y.Set( k4,rook2_ts.size()  );

	  gr.SetFontSize(3.0);
	  gr.Alpha(true);

	  gr.SetOrigin(0,0);
	 // gr.SetRanges(0,rook2_ts.size(),0,1);
	  gr.SetRange('x',0,rook2_ts.size());

	  int max1=max;
	  double ymax=std::max(max1,1);

	  gr.SetRange('y',0, ymax );
	  gr.Axis();

	  gr.Plot(xx,y0,"b");
	  gr.AddLegend("weight-willProtect","b");

	  gr.Plot(xx,y1,"g");
	  gr.AddLegend("weight-willThreaten","g");

	  gr.Plot(xx,y2,"r");
	  gr.AddLegend("weight-threateners","r");

	  gr.Plot(xx,y3,"c");
	  gr.AddLegend("weight-protectors","c");

	  gr.Plot(xx,y4,"m");
	  gr.AddLegend("weight-occupant","m");

	  gr.Label('x',"move#");
	  gr.Label('y',"weight");

	  int mid=rook2_ts.size()/2;

	  gr.Legend(0,0);


	  gr.Title("Rook2 Weights TimeSeries","");
	  //gr.WriteFrame("timeseries/rook2WeightsTimeSeries.png");

	  gr.WritePNG("timeseries/rook2WeightsTimeSeries.png","",false);

}
*/

