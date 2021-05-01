#include<iostream>
#include<vector>
#include<stdlib.h>
#include <fstream>
#include <string.h>
#include <unordered_map>
#include <algorithm>
#include "lib.h"

using namespace std;

typedef std::vector< std::pair<int, int> > Edges;
void printCudaInfo();
void parallelForward(const Edges&);
Edges edges;

void customTest(){


edges.push_back(std::make_pair(0,1));
edges.push_back(std::make_pair(0,2));
edges.push_back(std::make_pair(0,3));


edges.push_back(std::make_pair(1,0));
edges.push_back(std::make_pair(1,2));

edges.push_back(std::make_pair(2,0));
edges.push_back(std::make_pair(2,1));
edges.push_back(std::make_pair(2,3));

edges.push_back(std::make_pair(3,0));
edges.push_back(std::make_pair(3,2));
parallelForward(edges);


}

void MakeUndirected(Edges* edges) {
const size_t n = edges->size();
for (uint i = 0; i < n; ++i) {
    pair<int, int> edge = (*edges)[i];
    swap(edge.first, edge.second);
    edges->push_back(edge);
}  
}
void RemoveDuplicateEdges(Edges* edges) {
std::sort(edges->begin(), edges->end());
edges->erase(unique(edges->begin(), edges->end()), edges->end());
}
void RemoveSelfLoops(Edges* edges) {
for (uint i = 0; i < edges->size(); ++i) {
    if ((*edges)[i].first == (*edges)[i].second) {
    edges->at(i) = edges->back();
    edges->pop_back();
    --i;
    }
}
}
inline void NormalizeEdges(Edges* edges) {
    MakeUndirected(edges);
    RemoveDuplicateEdges(edges);
    RemoveSelfLoops(edges);
    //PermuteEdges(edges);
    //PermuteVertices(edges);
}
void WriteEdgesToFile(const char* filename) {
  ofstream out(filename, ios::binary);
  int m = edges.size();
  out.write((char*)&m, sizeof(int));
  out.write((char*)edges.data(), 2 * m * sizeof(int));
}

void readFromFile(char* filename, char* filewrite){
  std::unordered_map<int, int> nodes;
  int next_node = 0;
    ifstream in(filename);
    string buf;
    int node_a, node_b;

    while (getline(in, buf)){
        if (buf.empty() || buf[0] == '#')
            continue;
        sscanf(buf.c_str(), "%d %d", &node_a, &node_b);
        if (!nodes.count(node_a))
            nodes[node_a] = next_node++;
        if (!nodes.count(node_b))
            nodes[node_b] = next_node++;
        edges.push_back(std::make_pair(nodes[node_a], nodes[node_b]));
        //edges.push_back(std::make_pair(node_a,node_b));

    }
    printf("read file\n");
    NormalizeEdges(&edges);
    printf("normalized\n");
    WriteEdgesToFile(filewrite);
    printf("written to file\n");
}


int main(int argc, char** argv){


    if(argc <= 1){
        cout<<"Invalid command"<<endl;
        exit(1);
    }

    char* test = argv[1];
    char* file = argv[2]; 
    char* file2 = argv[3];
    
    // enter custom dataset
    if( !strcmp(test,"custom")){
        cout<<"Custom dataset result"<<endl;
        customTest();
    }
    // print cuda information
    else if(!strcmp(test,"pCuda")){
        printCudaInfo();
    }
    // read from file
    else if(!strcmp(test,"file")) {
        if(argc <= 3 ){
            cout<<"Enter file name "<<endl;
            exit(1);
        }       
        readFromFile(file, file2);

        if(edges.size() > 0 )
            parallelForward(edges);    
    }
    // invalid arguments
    else{
        cout<<"Invalid command"<<endl;
        exit(1);

    }


}