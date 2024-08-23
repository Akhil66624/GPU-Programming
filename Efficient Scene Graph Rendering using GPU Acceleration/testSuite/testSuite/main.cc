#include "testMaker.h"
#include <algorithm>

FILE* outputFile = NULL ;
int main (int argc, char** argv) {
	
	const char* outputFileName = argv[1] ;
	srand (atoi (argv[2]) ) ;
	/*int upperLimitNodes = atoi(argv[2]) ;
	int upperLimitEdges = atoi(argv[3]) ;
	int upperLimitTranslations = atoi(argv[4]) ;
	int bigFrameX = atoi(argv[5]) ; 
	int bigFrameY = atoi(argv[6]) ;
	int maxOpac = atoi(argv[7]) ;
	srand (atoi (argv[8]) ) ;
	int upperLimitSmallFrameX = atoi (argv[9]) ;
	int upperLimitSmallFrameY = atoi (argv[10]) ;
	*/
	int upperLimitNodes = 10000000;
	int upperLimitEdges = 9999999;
	int upperLimitTranslations = 10000000;
	int bigFrameX =  100000; 
	int bigFrameY =  100000;
	int maxOpac = 300000000;
	int upperLimitSmallFrameX = 100;
	int upperLimitSmallFrameY = 100;
	int upperLimitOnSum = 100000000 ;


	int numNodes = std::max (3, rand () % upperLimitNodes) ;
	int numEdges = numNodes-1 ;
	int numTranslations = std::max (2, rand () % upperLimitTranslations) ;
	int acceptedRandomnessForMeshSize = (upperLimitOnSum - (2 * upperLimitNodes)) ;
	int upperLimitFrameSize = upperLimitOnSum/numNodes ;
	if (upperLimitFrameSize < 10000) {
		upperLimitSmallFrameX = rand () % std::min(100, upperLimitFrameSize + 1) ;
		upperLimitSmallFrameY = upperLimitFrameSize/upperLimitSmallFrameX ;
	}
	setOutputFile (outputFileName) ;

	fprintf (outputFile, "%d\n", numNodes) ;
	bigFrameX = rand () % bigFrameX + 1 ;
	bigFrameY = rand () % bigFrameY + 1 ;
	fprintf (outputFile, "%d %d\n", bigFrameX+1, bigFrameY+1) ;
	
	int frameSizeX, frameSizeY, globalPositionX, globalPositionY, opacity ;
	std::unordered_set <int> takenOpacities ;
	for (int i=0; i<numNodes; i++) {
		frameSizeX = rand () % upperLimitSmallFrameX;
		frameSizeX = std::max (2, frameSizeX) ;
		frameSizeY = rand () % upperLimitSmallFrameY;
		frameSizeY = std::max (2, frameSizeY) ;
		acceptedRandomnessForMeshSize -= (frameSizeX*frameSizeY) ;
		assert (acceptedRandomnessForMeshSize + 2 * numNodes >= 0) ;
		globalPositionX = rand () % bigFrameX ;
		globalPositionY = rand () % bigFrameY ;
		do {opacity = rand () % maxOpac ;} while (takenOpacities.find (opacity) != takenOpacities.end ()) ; 
		takenOpacities.insert (opacity) ;
		generateMesh (frameSizeX, frameSizeY, globalPositionX, globalPositionY, opacity) ;
	}	
	
	fprintf (outputFile, "%d\n", numEdges) ;
	generateEdges (numNodes, numEdges) ;

	fprintf (outputFile, "%d\n", numTranslations) ;
	generateTranslations (numTranslations, numNodes) ;

}
