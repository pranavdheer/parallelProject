#include<iostream>
#include<vector>
#include<stdlib.h>
#include <fstream>
#include <string.h>
#include "lib.h"

using namespace std;

typedef std::vector< std::pair<int, int> > Edges;
void printCudaInfo();
void parallelForward(const Edges&);
Edges edges;

void customTest(){


edges.push_back(std::make_pair(0,2));
edges.push_back(std::make_pair(0,1));
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

void readFromFile(char* filename){

  ifstream in(filename, ios::binary);
  int m;
  in.read((char*)&m, sizeof(int));
  edges.resize(m);
  in.read((char*)edges.data(), 2 * m * sizeof(int));
  cout<<"done reading the file"<<"\n"<<"number of edges = "<<edges.size()<<endl;
}


int main(int argc, char** argv){


    if(argc <= 1){
        cout<<"Invalid command"<<endl;
        exit(1);
    }

    char* test = argv[1];
    char* file = argv[2]; 
     
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
       if(argc <= 2 ){
            cout<<"Enter file name "<<endl;
            exit(1);
       }       
       readFromFile(file);

       if(edges.size() > 0 )
            parallelForward(edges);    
    }
    // invalid arguments
    else{
        cout<<"Invalid command"<<endl;
        exit(1);

    }


}