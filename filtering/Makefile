CXX=g++
CXXFLAGS=-std=c++11 -O3 -fno-stack-protector

all: convert-from-snap-main.e

convert-from-snap-main.e: convert-from-snap-main.cpp graph.cpp graph.h
	$(CXX) $(CXXFLAGS) $^ -o $@


clean:
	rm -f *.o *.e
