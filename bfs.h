#ifndef BFS_H
#define BFS_H

using namespace std;

// Μια κλάση με τις ιδιότητες του γραφήματος
class Graph {
public:
    vector<int> adj;                                        // Η adjacency list
    vector<int> edge;                                       // Η μετατόπιση των ακμών
    vector<int> edgeSize;                                   // Το μέγεθος της
    int numVertices = 0;                                    // Ο αριθμός των κορυφών
    int numEdges = 0;                                       // Ο αριθμός των ακμών
};

void readGraph(Graph &g);                                   // Δημιουργεί και διαβάζει ένα γράφημα
void runbfs(int startVertex, Graph &g, vector<int> &distance);  // Καλεί το kernel
void cleanup();                                             // Καθαρίζει την μνήμη

#endif
