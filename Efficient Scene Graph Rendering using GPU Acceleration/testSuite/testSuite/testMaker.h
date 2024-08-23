/*
 * to Make test Cases for SceneGraph
 * test case pattern :
 * 1. The meshes.
 * 2. The edges.
 * 3. The translations.
 * */

#ifndef __TEST_MAKER__
#define __TEST_MAKER__

#define GRAY_SCALE 255 
#define NUM_COMMANDS 4 
#define MAX_MOVE_X 100
#define MAX_MOVE_Y 100
enum Command {__UP__, __DOWN__, __LEFT__, __RIGHT__, __ROTATE__} ;

#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <unordered_set>

extern FILE* outputFile ;

void setOutputFile (const char* outputFileName) ;

void generateMesh (const int &frameSizeX, const int &frameSizeY, const int &globalPositionX, const int &globalPositionY, const int &opacity) ;

void generateEdges (const int &numNodes, const int &numEdges) ;

void generateTranslations (const int &numTranslations, const int &numNodes) ;

#endif
