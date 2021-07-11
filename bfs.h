#ifndef BFS_H
#define BFS_H

using namespace std;

class Graph {
public:
    vector<int> adj;                                        //Adjacency list
    vector<int> edge;                                       //Edge offset
    vector<int> edgeSize;
    int numVertices = 0;
    int numEdges = 0;
};

void readGraph(Graph &g);
void runbfs(int startVertex, Graph &g, vector<int> &distance);
void cleanup();

#endif
