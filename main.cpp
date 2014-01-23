//main.cpp for multiAgentChess program

//the main method... starts the game

#include <stdio.h>
#include <boost/mpi.hpp>  //include boost library
#include "repast_hpc/RepastProcess.h"  //interprocess communication and encapsulates process

//HEADERS
#include "chessModel.h" //include header file for model class



//learning parameter alpha should decrease as more games are played

//keep track of number of games played
int gamesPlayed;

//keep track of number of victories for each team;
int games1Won;
int games2Won;

//change the entropy level as the games progress
double entropyV[7];


int main(int argc, char** argv){




	//the repast process requires a configuration file
	// for logging and other settings, we pass this as an argument
	std::string configFile = argv[1];

	std::string propsFile  = argv[2]; // The name of the properties file is Arg 2


						//boost environment variable
						boost::mpi::environment env(argc, argv);

						//boost communicator, used to retrieve info about and
						//communicating between processes
						boost::mpi::communicator world;

						//must be called before rp is used, passes
						//the config file (configure logging)
						repast::RepastProcess::init(configFile);


						Log4CL::instance()->get_logger("root").log(INFO, "creating model...");


						chessHPCModel* model = new chessHPCModel(propsFile, argc, argv, &world);

						//make a schedule runner variable, manages schedule
						repast::ScheduleRunner& runner = repast::RepastProcess::instance()->getScheduleRunner();


						//run the init schedule method (adds agents)
						model->init(gamesPlayed);

						Log4CL::instance()->get_logger("root").log(INFO, "running initSchedule()...");
						model->initSchedule(runner);

						//run the schedule runner
						runner.run();

						//once finished delete model pointer (used 'new', stack memory etc...)
						delete model;
						//notify the rp that simulation is completed (dont know why this is here)

						repast::RepastProcess::instance()->done();







return 0;

}//end main






