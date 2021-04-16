#include<iostream>
#include<vector>
#include<stdlib.h>
#include "lib.h"

using namespace std;

typedef std::vector< std::pair<int, int> > Edges;

void MultiGpuForward(const Edges&);

int main(int argc, char** argv){

Edges edges;

edges.push_back(std::make_pair(0,1));
edges.push_back(std::make_pair(0,2));
edges.push_back(std::make_pair(0,3));
edges.push_back(std::make_pair(0,4));

edges.push_back(std::make_pair(1,4));
edges.push_back(std::make_pair(1,2));
edges.push_back(std::make_pair(1,0));

edges.push_back(std::make_pair(2,1));
edges.push_back(std::make_pair(2,3));
edges.push_back(std::make_pair(2,0));

edges.push_back(std::make_pair(3,2));
edges.push_back(std::make_pair(3,4));
edges.push_back(std::make_pair(3,0));

edges.push_back(std::make_pair(4,1));
edges.push_back(std::make_pair(4,3));
edges.push_back(std::make_pair(4,0));



MultiGpuForward(edges);

}