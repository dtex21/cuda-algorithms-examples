//For more comments go to bfs.h
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include "bfs.h"

using namespace std;

int *d_adj, *d_edge, *d_edgeSize, *d_distance;

void readGraph(Graph &g) {
    int n, m, u, v;
    vector<int> vetrices;
    
    cout << "Give number of vertices:" << endl;;
    cin >> n;
    cout << "Give number of edges:" << endl;
    cin >> m;
    
    vector<vector<int> > adj(n);
    cout << "Give paths between vetrices:" << endl;         //From A to B to C, etc.
    for (int i = 0; i < m; i++) {
            cout << "Path " << i << endl;                   //Path indexing to make my life easier
            cin >> u >> v;
            adj[u].push_back(v);
            vetrices.push_back(u);
            vetrices.push_back(v);
    }

    for (int i = 0; i < n; i++) {                           //Calculate offset
        g.edge.push_back(g.adj.size());                    
        g.edgeSize.push_back(adj[i].size());
        for (auto e: adj[i]) {
            g.adj.push_back(e);
        }
    }

    g.numVertices = n;
    g.numEdges = g.adj.size();
    sort(vetrices.begin(), vetrices.end());                 
    vetrices.erase(unique(vetrices.begin(), vetrices.end()), vetrices.end());
    
    cout << "List of vertices: ";
    for (auto v : vetrices)
        cout << v << " ";
    cout << endl;
}
  //                               //
 //////////// CUDA KERNEL //////////
//                               //

__global__ void bfs(int N, int level, int *d_adj, int *d_edge, int *d_edgeSize, int *d_distance, bool *explored) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    *explored = false;

    if (idx < N && d_distance[idx] == level) {
        for (int i = d_edge[idx]; i < d_edge[idx] + d_edgeSize[idx]; i++) {
            int v = d_adj[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                *explored = true;
            }
        }
    }
}

void runbfs(int start, Graph &g, vector<int> &distance) {
    bool *explored;
    size_t Vsize = g.numVertices * sizeof(int);
    size_t Esize = g.numEdges * sizeof(int);

    distance[start] = 0;

    cudaMallocHost((void **) &explored, sizeof(int));
    cudaMalloc(&d_adj, Esize);
    cudaMalloc(&d_edge, Vsize);
    cudaMalloc(&d_edgeSize, Vsize);
    cudaMalloc(&d_distance, Vsize);
    cudaMemcpy(d_adj, g.adj.data(), Esize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge, g.edge.data(), Vsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSize, g.edgeSize.data(), Vsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance.data(), Vsize, cudaMemcpyHostToDevice);

    *explored = true;
    int level = 0;

    while (*explored) {
        *explored = false;
        bfs <<< 64, 64, Vsize >>>(g.numVertices, level, d_adj, d_edge, d_edgeSize, d_distance, explored);
        cudaDeviceSynchronize();
        level++;
    }
    cudaMemcpy(distance.data(), d_distance, Vsize, cudaMemcpyDeviceToHost);
}
void cleanup() {
    cudaFree(d_adj);
    cudaFree(d_edge);
    cudaFree(d_edgeSize);
    cudaFree(d_distance);
}
  //                      //
 ////////// MAIN ////////// 
//                      //

int main() {
    Graph g;
    readGraph(g);
        
    vector<int> distance(g.numVertices, numeric_limits<int>::max());
    int start;
    cout << "Give starting vertex: " << endl;
    cin >> start;
    
    runbfs(start, g, distance);

    for (auto d : distance)
        cout << "Distance from " << start << " is " << d << endl;
    cleanup();
    return 0;
}

