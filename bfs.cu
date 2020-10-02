// Για επιπλέον σχόλια: => bfs.h
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
    cout << "Give paths between vetrices:" << endl;         // Δηλαδή από την κορυφή A στην κορυφή B, έπειτα B -> C, κτλ
    for (int i = 0; i < m; i++) {
            cout << "Path " << i << endl;                   // Εδώ απλώς δίνω έναν αριθμό στις διαδρομές για να είναι πιο εύκολη η εισαγωγή των στοιχείων, δεν σημαίνει κάτι
            cin >> u >> v;
            adj[u].push_back(v);
            vetrices.push_back(u);                          // Σε αυτή και στην επόμενη γραμμή,
            vetrices.push_back(v);                          // βάζω όλες τις κορυφές σε ένα vector (διπλές και τριπλές, όπως εισάγονται απ' τον χρήστη) για να τις τυπώσω αργότερα
    }

    for (int i = 0; i < n; i++) {                           // Υπολογισμός της μετατόπισης και του μεγέθους της
        g.edge.push_back(g.adj.size());                    
        g.edgeSize.push_back(adj[i].size());
        for (auto e: adj[i]) {
            g.adj.push_back(e);
        }
    }

    g.numVertices = n;
    g.numEdges = g.adj.size();
    // Ταξινόμιση και διαγραφή των διπλών και τριπλών κορυφών. Έπειτα τύπωμα
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

    if (idx < N && d_distance[idx] == level) {              // Αν ο δείκτης (idx) είναι μικρότερος από το σύνολο των κορυφών και η απόσταση από τον γονέα είναι ίση με το επίπεδο που είμαστε στο γράφημα
        for (int i = d_edge[idx]; i < d_edge[idx] + d_edgeSize[idx]; i++) {
            int v = d_adj[i];                               // Για κάθε μετατόπιση συγκεκριμένου μεγεθους, θέτω ως v κάθε στοιχείο της adjacency list
            if (level + 1 < d_distance[v]) {                // Και αν κάθε επόμενο επίπεδο είναι μικρότερο απ' την απόσταση κάθε μιας από τις κορυφές
                d_distance[v] = level + 1;                  // Θέτω την απόσταση ως το επόμενο επίπεδο
                *explored = true;                           // Και η κορυφή έχει εξερευνηθεί
            }
        }
    }
}

void runbfs(int start, Graph &g, vector<int> &distance) {
    bool *explored;
    size_t Vsize = g.numVertices * sizeof(int);
    size_t Esize = g.numEdges * sizeof(int);

    distance[start] = 0;
    // Διανέμω μνήμη για τον CPU
    cudaMallocHost((void **) &explored, sizeof(int));
    // Και για την GPU
    cudaMalloc(&d_adj, Esize);
    cudaMalloc(&d_edge, Vsize);
    cudaMalloc(&d_edgeSize, Vsize);
    cudaMalloc(&d_distance, Vsize);
    // Αντιγραφή των τιμών στην συσκευή
    cudaMemcpy(d_adj, g.adj.data(), Esize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge, g.edge.data(), Vsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeSize, g.edgeSize.data(), Vsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance.data(), Vsize, cudaMemcpyHostToDevice);

    // Καλούμε το kernel μέχρι για κάθε επίπεδο μέχρι το τέλος του γραφήματος
    *explored = true;
    int level = 0;
    while (*explored) {
        *explored = false;
        bfs <<< 64, 64, Vsize >>>(g.numVertices, level, d_adj, d_edge, d_edgeSize, d_distance, explored);
        cudaDeviceSynchronize();
        level++;
    }
    // Αντιγράφουμε τις τιμές των αποστάσεων στον host
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
    Graph g;                                                // Δημιουργία και
    readGraph(g);                                           // Ανάγνωση του νέου γραφήματος
        
    vector<int> distance(g.numVertices, numeric_limits<int>::max());
    int start;
    cout << "Give starting vertex: " << endl;
    cin >> start;
    
    runbfs(start, g, distance);
    // Τύπωμα της απόστασης
    for (auto d : distance)
        cout << "Distance from " << start << " is " << d << endl;
    cleanup();
    return 0;
}

