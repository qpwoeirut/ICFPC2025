/*
Takes in a route and the resulting labels and outputs a graph
Route: 18n doors
Labels: 18n + 1 2-bit labels

Graph: n nodes with 6 edges each

Uses backtracking to search the possible ways to merge the labels into a graph.
*/

#include <iostream>
#include <fstream>

using namespace std;

const int MN = 30;
const int K = 18;

int route[MN * K];
int labels[MN * K + 1];

int N;
int id[MN * K + 1];

int edge_id[32][6];
int edge_label[32][6];

void recurse(int idx) {
    // cerr << idx << ": ";
    // for (int i=0; i<N*K+1; ++i) {
    //     cerr << id[i] << " ";
    // }
    // cerr << endl;

    if (idx == N * K + 1) {
        cout << "Found solution" << endl;
        for (int i=0; i<N*K+1; ++i) {
            cout << id[i] << " ";
        }
        cout << endl;
        for (int i=0; i<32; ++i) {
            for (int j=0; j<6; ++j) {
                cout << edge_id[i][j] << " ";
            }
            cout << " ";
        }
        cout << endl;
        return;
    }

    for (int i=0; i*4<N; ++i) {
        bool ok = true;
        id[idx] = labels[idx] << 3 | i;
        
        int old_edge_id = -1;
        if (idx > 0) {
            const int prev_id = id[idx - 1];
            old_edge_id = edge_id[prev_id][route[idx - 1]];
            ok &= old_edge_id == -1 || old_edge_id == id[idx];
            edge_id[prev_id][route[idx - 1]] = id[idx];
        }

        int old_edge_label = -1;
        if (idx < N * K) {
            old_edge_label = edge_label[id[idx]][route[idx]];
            ok &= old_edge_label == -1 || old_edge_label == labels[idx + 1];
            edge_label[id[idx]][route[idx]] = labels[idx + 1];
        }

        if (ok) recurse(idx + 1);

        if (idx > 0) {
            edge_id[id[idx - 1]][route[idx - 1]] = old_edge_id;
        }
        if (idx < N * K) {
            edge_label[id[idx]][route[idx]] = old_edge_label;
        }
    }
    id[idx] = -1;
}

int main() {
    ifstream fin("route.txt");

    fin >> N;

    cout << "Running solver with N = " << N << endl;

    for (int i=0; i<N*K; ++i) {
        fin >> route[i];
    }
    for (int i=0; i<N*K+1; ++i) {
        fin >> labels[i];
        id[i] = -1;
    }

    for (int i=0; i<32; ++i) {
        for (int j=0; j<6; ++j) {
            edge_id[i][j] = -1;
            edge_label[i][j] = -1;
        }
    }

    recurse(0);

//    ofstream fout("graph.txt");
}