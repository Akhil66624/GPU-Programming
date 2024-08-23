g++ -g -std=c++17 -c testMaker.cc -o testMaker.o
g++ -g -std=c++17 -c main.cc -o main.o
g++ -g -std=c++17 testMaker.o main.o -o testCaseGenerator.o
