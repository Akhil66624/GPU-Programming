# Makefile for compiling SceneNode.cc, Renderer.cc, and main.cc with g++

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17

# Source files
SOURCES = testMaker.cc main.cc

# Object files
OBJECTS = $(SOURCES:.cc=.o)

# Header files
HEADERS = testMaker.h

# Target executable
TARGET = testCaseGenerator.o

# Default target
all: $(TARGET)

# Linking object files to create the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

# Compiling source files
%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)


