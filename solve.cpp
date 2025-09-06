/*
Takes in a route and the resulting labels and outputs a graph
Route: 18n doors
Labels: 18n + 1 2-bit labels

Graph: n nodes with 6 edges each

Uses backtracking to search the possible ways to merge the labels into a graph.
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>

using namespace std;

const int MN = 30;
const int K = 18;

int route[MN * K];
int labels[MN * K + 1];

int N;
int id[MN * K + 1];

int edge_id[32][6];
int edge_label[32][6];

multiset<int> inbound[32];

bool edges_match(int node_id) {
    map<int, int> outbound_ct;
    for (int j=0; j<6; ++j) ++outbound_ct[edge_id[node_id][j]];
    int matched = 0;
    int outbound_total = 0;
    for (auto [k, v]: outbound_ct) {
        if (k != -1) {
            matched += min((int)(inbound[node_id].count(k)), v);
            outbound_total += v;
        }
    }
    return inbound[node_id].size() + outbound_total - matched <= 6;
}

map<int, int> used_ids;
void recurse(int idx) {
    if (idx == N * K + 1) {
        cout << "Found solution" << endl;
        for (int i=0; i<N*K+1; ++i) {
            cout << id[i] << " ";
        }
        cout << endl;
        for (int i=0; i<N+3; ++i) {
            for (int j=0; j<6; ++j) {
                cout << edge_id[i][j] << " ";
            }
            cout << " ";
        }
        cout << endl;
        return;
    }

    for (int i=0; i*4<N; ++i) {
        id[idx] = i << 2 | labels[idx];
        ++used_ids[id[idx]];
        bool ok = used_ids.size() <= N;
        
        int old_edge_id = -1;
        if (idx > 0) {
            const int prev_id = id[idx - 1];
            old_edge_id = edge_id[prev_id][route[idx - 1]];
            ok &= old_edge_id == -1 || old_edge_id == id[idx];
            edge_id[prev_id][route[idx - 1]] = id[idx];
            if (old_edge_id == -1) {
                inbound[id[idx]].insert(prev_id);
                ok &= edges_match(id[idx]);
            }
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
            if (old_edge_id == -1) {
                inbound[id[idx]].erase(inbound[id[idx]].find(id[idx - 1]));
            }
        }
        if (idx < N * K) {
            edge_label[id[idx]][route[idx]] = old_edge_label;
        }
        if (--used_ids[id[idx]] == 0) {
            used_ids.erase(id[idx]);
            // If multiple unused IDs can be assigned, only recurse on the lowest.
            break;
        }
    }
    id[idx] = -1;
}

int main() {
    ifstream fin("route.txt");
    fin.exceptions(ios_base::badbit | ios_base::failbit);

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