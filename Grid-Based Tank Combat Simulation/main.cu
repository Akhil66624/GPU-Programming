#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/count.h>
using namespace std;

//*******************************************

// Write down the kernels here


__global__ void simulateGame(int tank, int T, int * coordinatesX, int * coordinatesY, int * deviceHealth, int* deviceScore, int *ptrHealth)
{

    __shared__ int distanceMin;
    distanceMin=INT_MAX;

    int code=blockIdx.x;

    if(deviceHealth[code]<=0 || (tank%T)==0)
    {
        return ;
    }

    int srcX = coordinatesX[code], srcY = coordinatesY[code];
    int destX = coordinatesX[(code+tank)%T], destY = coordinatesY[(code+tank)%T];

    int tankItr=threadIdx.x;
    long long int slopeA=destY-srcY, slopeB=destX-srcX;

    int p2X=coordinatesX[tankItr], p2Y=coordinatesY[tankItr];
    long long int temp1=(p2Y - srcY), temp2=(p2X - srcX);

    long int distance=-1;
    if(tankItr != code && deviceHealth[tankItr] > 0 && (temp1 * slopeB==slopeA * temp2) && ((destX - srcX >= 0) ^ (p2X - srcX < 0)) && ((destY - srcY >= 0) ^ (p2Y - srcY < 0)) )
    {
        distance=abs(p2X-srcX)+abs(p2Y-srcY);
        atomicMin(&distanceMin, distance);
    }

    __syncthreads();

    if(distanceMin!=INT_MAX && distanceMin==distance )
    {
        deviceScore[code]++;
        atomicSub(&ptrHealth[tankItr], 1);
    }
}

__global__ void initializeHealth(int H, int * deviceHealth, int * deviceScore){
    int id = threadIdx.x;

    deviceScore[id] = 0;
    deviceHealth[id] = H;
}

//***********************************************

int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
        fscanf( inputfilepointer, "%d", &xcoord[i] );
        fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int * deviceScore;
    int * deviceHealth;
    int * coordinatesX;
    int * coordinateY;

    cudaMalloc(& deviceHealth, sizeof(int)*T);
    cudaMalloc(&deviceScore, sizeof(int)*T);

    cudaMalloc(&coordinatesX, sizeof(int)*T);
    cudaMemcpy(coordinatesX, xcoord, sizeof(int)*T, cudaMemcpyHostToDevice);

    cudaMalloc(&coordinateY, sizeof(int)*T);
    cudaMemcpy(coordinateY, ycoord, sizeof(int)*T, cudaMemcpyHostToDevice);

    initializeHealth<<<1, T>>>(H, deviceHealth, deviceScore);
    cudaDeviceSynchronize();

    int counterTanks=T;

    thrust::device_vector<int> healthVector(T, H);
    int * ptrHealth=thrust::raw_pointer_cast(healthVector.data());
    int tank=1;

    while(counterTanks>1)
    {
        simulateGame<<<T, T>>>(tank, T, coordinatesX, coordinateY, deviceHealth, deviceScore, ptrHealth);
        cudaMemcpy(deviceHealth, ptrHealth, sizeof(int)*T, cudaMemcpyDeviceToDevice);

        counterTanks = thrust::count_if(healthVector.begin(), healthVector.end(), thrust::placeholders::_1 > 0);
        tank++;
    }
    cudaMemcpy(score, deviceScore, sizeof(int)*T, cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
