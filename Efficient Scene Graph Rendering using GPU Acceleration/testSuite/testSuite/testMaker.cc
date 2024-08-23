#include "testMaker.h"


void setOutputFile (const char* outputFileName) {
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("output file could not be created" ) ;
	}
}

void generateMesh (const int &frameSizeX, const int &frameSizeY, const int &globalPositionX, const int &globalPositionY, const int &opacity) {

	fprintf (outputFile, "%d %d\n", frameSizeX, frameSizeY) ;
	fprintf (outputFile, "%d %d\n", globalPositionX, globalPositionY) ;
	fprintf (outputFile, "%d\n", opacity) ;
	for (int row = 0 ; row < frameSizeX; row++) {
		for (int col = 0; col < frameSizeY; col++) {
			 fprintf (outputFile, "%d ", rand () % GRAY_SCALE) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

void generateEdges (const int &numNodes, const int &numEdges)  {

	assert (numNodes == numEdges+1) ;
	std::vector<std::vector <int> > graph (numNodes) ;
	int outDegreeOfSource = 1 + rand () % numEdges ;
	int edgeNum = 0 ;
	std::unordered_set <int> taken ;
	taken.insert (0) ;
	for ( ; edgeNum < outDegreeOfSource; edgeNum++) {
		int u = 0 ; int v ;
		do { v = rand () % numNodes ;} while (taken.find (v) != taken.end ()) ;
		taken.insert (v) ;
		graph[u].push_back (v) ;
	}
	for ( ; edgeNum < numEdges; edgeNum++) {
		int u, v;
		do { u = rand () % numNodes ;} while (taken.find (u) == taken.end ()) ;
		do { v = rand () % numNodes ;} while (taken.find (v) != taken.end ()) ;
		taken.insert (u) ;
		taken.insert (v) ;
		graph[u].push_back (v) ;
	}
	for (int u = 0 ; u < numNodes ; u++) {
		for (int v = 0 ; v < graph[u].size () ; v++) {
			fprintf (outputFile, "%d %d\n", u, graph[u][v]) ;	
		}
	}
}

void generateTranslations (const int &numTranslations, const int &numNodes) {
	
	int nodeNum, command, amount ;
	for (int translationNo = 0 ; translationNo < numTranslations; translationNo++) {
		
		nodeNum = rand () % numNodes ;
		int command = rand () % NUM_COMMANDS ;

		switch (command) {
			case __ROTATE__ : 
				amount = rand () % 360 ;
				break ;
			case __UP__ :
				amount = rand () % MAX_MOVE_Y ;
				break ;
			case __DOWN__ :
				amount = rand () % MAX_MOVE_Y ;
				break ;
			case __LEFT__ :
				amount = rand () % MAX_MOVE_X ;
				break ;
			case __RIGHT__ :
				amount = rand () % MAX_MOVE_X ;
				break ;
			default :
				amount = 0 ;
				break ;
		}
		fprintf (outputFile, "%d %d %d\n", nodeNum, command, amount) ;
	}
}
