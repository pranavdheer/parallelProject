#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include "lib.h"
#include <vector>
#include <utility>
using namespace std;

int NumVertices(const Edges& edges) {
  int num_vertices = 0;
  for (const pair<int, int>& edge : edges)
    num_vertices = max(num_vertices, 1 + max(edge.first, edge.second));
  return num_vertices;
}

void MakeUndirected(Edges* edges) {
  const size_t n = edges->size();
  for (int i = 0; i < n; ++i) {
    pair<int, int> edge = (*edges)[i];
    swap(edge.first, edge.second);
    edges->push_back(edge);
  }  
}
void RemoveDuplicateEdges(Edges* edges) {
  sort(edges->begin(), edges->end());
  edges->erase(unique(edges->begin(), edges->end()), edges->end());
}
void RemoveSelfLoops(Edges* edges) {
  for (int i = 0; i < edges->size(); ++i) {
    if ((*edges)[i].first == (*edges)[i].second) {
      edges->at(i) = edges->back();
      edges->pop_back();
      --i;
    }
  }
}
void PermuteEdges(Edges* edges) {
  random_shuffle(edges->begin(), edges->end());
}

void PermuteVertices(Edges* edges) {
  vector<int> p(NumVertices(*edges));
  for (int i = 0; i < p.size(); ++i)
    p[i] = i;
  random_shuffle(p.begin(), p.end());
  for (pair<int, int>& edge : *edges) {
    edge.first = p[edge.first];
    edge.second = p[edge.second];
  }
}


inline void NormalizeEdges(Edges* edges) {
  MakeUndirected(edges);
  RemoveDuplicateEdges(edges);
  RemoveSelfLoops(edges);
  PermuteEdges(edges);
  PermuteVertices(edges);
}


void WriteEdgesToFile(const Edges& edges, const char* filename) {
  ofstream out(filename, ios::binary);
  int m = edges.size();
  out.write((char*)&m, sizeof(int));
  out.write((char*)edges.data(), 2 * m * sizeof(int));
}

Edges ReadEdgesFromSNAPFile(const char* filename) {
  Edges edges;
  unordered_map<int, int> nodes;
  int next_node = 0;
  ifstream in(filename);
  assert(in.is_open());
  string buf;
  while (getline(in, buf)) {
    if (buf.empty() || buf[0] == '#')
      continue;  // ignore empty lines and comments
    int node_a, node_b;
    sscanf(buf.c_str(), "%d %d", &node_a, &node_b);
    if (!nodes.count(node_a))
      nodes[node_a] = next_node++;
    if (!nodes.count(node_b))
      nodes[node_b] = next_node++;
    edges.push_back(make_pair(nodes[node_a], nodes[node_b]));
  }
  return edges;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " IN OUT" << endl;
    exit(1);
  }

  Edges edges = ReadEdgesFromSNAPFile(argv[1]);
  NormalizeEdges(&edges);
  WriteEdgesToFile(edges, argv[2]);
}