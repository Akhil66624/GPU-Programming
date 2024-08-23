/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


__device__ int directChildren(int *dCsr, int *dOffset, int translating_vertex, int *direktions, int x, int y, int V = 0) {
	if (translating_vertex == 0) {
		for (int i = 0; i < V; i++) {
			atomicAdd(direktions + 2 * i, x);
			atomicAdd(direktions + 2 * i + 1, y);
		}
		return -1;
	}
	int start = dOffset[translating_vertex];
	int end = dOffset[translating_vertex + 1];

	int cur_child;
	atomicAdd(direktions + 2 * translating_vertex, x);
	atomicAdd(direktions + 2 * translating_vertex + 1, y);

	for(int i = start; i < end; i++) {
		cur_child = dCsr[i];
		directChildren(dCsr, dOffset, cur_child, direktions, x, y);
	}
	return 1;
}

__global__ void dirFinder(int *dCsr, int *dOffset, int *dTranslations, int *direktions, int numTranslations, int V, int E) {
	long long id = threadIdx.x + blockDim.x * blockIdx.x;
	int translation_num = id;

	if (translation_num < numTranslations) {
		int translating_vertex = dTranslations[translation_num];
		int command = dTranslations[numTranslations + translation_num];
		int amt = dTranslations[numTranslations * 2 + translation_num];
		int x = command/2;
		int y = command%2;
		int X = amt * (1 - x * x) * (2 * y - 1);
		int Y = amt * x * x * (2 * y - 1);
		directChildren(dCsr, dOffset, translating_vertex, direktions, X, Y, V);
	}
}

__global__ void setOpacity(int *dCanvas, int *direktions, int *dOpacity, int **dMesh, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int V, int y_max, int x_max) {
	int x = threadIdx.x;
	int vertex_id = blockIdx.x;
	int hor_start = dGlobalCoordinatesY[vertex_id] + direktions[2 * vertex_id + 1];
	int hor_end = dGlobalCoordinatesY[vertex_id] + direktions[2 * vertex_id + 1] + dFrameSizeY[vertex_id];

	int ver_start = dGlobalCoordinatesX[vertex_id] + direktions[2 * vertex_id];

	if (x < dFrameSizeX[vertex_id]){
		for(int i = hor_start; i < hor_end; i++) {
			if (ver_start + x < x_max and ver_start + x >= 0 && i >= 0 && i < y_max) {
				atomicMin(dCanvas + (y_max * (ver_start + x) + i), -dOpacity[vertex_id]);
			}
		}
	}
}

__global__ void renderCanvas(int *dCanvas, int *direktions, int *dOpacity, int **dMesh, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int V, int y_max, int x_max) {
	int x = threadIdx.x;
	int vertex_id = blockIdx.x;
	int hor_start = dGlobalCoordinatesY[vertex_id] + direktions[2 * vertex_id + 1];
	int hor_end = dGlobalCoordinatesY[vertex_id] + direktions[2 * vertex_id + 1] + dFrameSizeY[vertex_id];

	int ver_start = dGlobalCoordinatesX[vertex_id] + direktions[2 * vertex_id];

	if (x < dFrameSizeX[vertex_id]){
		for(int i = hor_start; i < hor_end; i++) {
			if (ver_start + x < x_max and ver_start + x >= 0 && i >= 0 && i < y_max) {
				if (dCanvas[(y_max * (ver_start + x) + i)] == -dOpacity[vertex_id]) {
					dCanvas[(y_max * (ver_start + x) + i)] = dMesh[vertex_id][x * dFrameSizeY[vertex_id] + i - hor_start];
				}
			}
		}
	}
}

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

	int *direktions;
	int *dCanvas;

	int *dOffset;
	int *dCsr;
	int *dOpacity ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **dMesh ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *dGlobalCoordinatesX ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *dGlobalCoordinatesY ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *dFrameSizeX ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *dFrameSizeY ;

	int *dTranslations;
	cudaMalloc(&dTranslations, numTranslations * 3 * sizeof(int));

	cudaMalloc(&direktions, 2 * V * sizeof(int));
	cudaMalloc(&dCanvas, frameSizeX * frameSizeY * sizeof(int));
	cudaMalloc(&dOffset, (V + 1) * sizeof(int));
	cudaMalloc(&dCsr, E * sizeof(int));

	int* translationsArr;
	translationsArr = (int*) malloc (sizeof (int) * 3 * numTranslations);

	for (int i = 0; i < numTranslations; i++) {
		translationsArr[i] = translations[i][0];
		translationsArr[i + numTranslations] = translations[i][1];
		translationsArr[i + 2*numTranslations] = translations[i][2];
	}

	cudaMemcpy(dTranslations, translationsArr, 3 * numTranslations * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dOffset, hOffset, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dCsr, hCsr, E * sizeof(int), cudaMemcpyHostToDevice);

	// 1st Kernel Launch //
	dirFinder<<<ceil(float(numTranslations) / 1024), 1024>>>(dCsr, dOffset, dTranslations, direktions, numTranslations, V, E);

	// Clear up after 1st kernel
	cudaFree(dTranslations);
	free(translationsArr);

	cudaMalloc(&dOpacity, V * sizeof(int));

	// Setting up the meshes
	cudaMalloc(&dMesh, V * sizeof(int*));
	int **dMeshesAdresses;
	dMeshesAdresses = (int**) malloc (sizeof (int*) * V);
	for (int i = 0; i < V; i++) {
		cudaMalloc(&(dMeshesAdresses[i]), hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int));
	}

	// Code runs correctly upto here
	cudaMalloc(&dGlobalCoordinatesX, V * sizeof(int));
	cudaMalloc(&dGlobalCoordinatesY, V * sizeof(int));
	cudaMalloc(&dFrameSizeX, V * sizeof(int));
	cudaMalloc(&dFrameSizeY, V * sizeof(int));
	
	cudaMemcpy(dOpacity, hOpacity, V * sizeof(int), cudaMemcpyHostToDevice);
	for (int i = 0; i < V; i++) {
		cudaMemcpy(dMeshesAdresses[i], hMesh[i], hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int), cudaMemcpyHostToDevice);
	}
	for (int i = 0; i < V; i++) {
		cudaMemcpy(dMesh, dMeshesAdresses, V * sizeof(int*), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dFrameSizeX, hFrameSizeX, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dFrameSizeY, hFrameSizeY, V * sizeof(int), cudaMemcpyHostToDevice);

	// 2nd Kernel Launch //
	setOpacity<<<V, 100>>>(dCanvas, direktions, dOpacity, dMesh, dGlobalCoordinatesX, dGlobalCoordinatesY, dFrameSizeX, dFrameSizeY, V, frameSizeX, frameSizeY);

	// 3rd Kernel Launch //
	renderCanvas<<<V, 100>>>(dCanvas, direktions, dOpacity, dMesh, dGlobalCoordinatesX, dGlobalCoordinatesY, dFrameSizeX, dFrameSizeY, V, frameSizeX, frameSizeY);

	cudaMemcpy(hFinalPng, dCanvas, frameSizeX * frameSizeY * sizeof(int), cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}