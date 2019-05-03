#include<bits/stdc++.h>
#include<cstring>
#include <fstream>
#include "utility.h"
using namespace std;

double** new_2d_mat(int row, int col) {
    double** x = new double*[row];
    for(int i = 0; i < row; ++i) {
        x[i] = new double[col];
        memset(x[i], 0, sizeof(double)*col);
    }
    return x;
}
void free_matrix(double**x, int row) {
    for(int i = 0; i < row; ++i) {
        delete[] x[i];
    }
    delete[] x;
}
double* double_array(int col) {
    double* x = new double[col];
    memset(x, 0, sizeof(double)*col);
    return x;
}
void free_vector(double* x){
    delete []x;
}

void csv_read(const char* in_string, double*x) {
    const char* pos = in_string;
    int i = 0;
    for(; pos - 1 != NULL && pos[0] != '\0' && pos[0] != '#';)
    {
        float value = atof(pos);
        x[i++] = value;
        pos = strchr(pos, ',') + 1;
    }
}

void load_feature_matrix(const char* train_feature, double**x) {
    int cur_row = 0;
    ifstream ifs(train_feature);
    string line;
    while(getline(ifs, line)) {
        csv_read(line.c_str(), x[cur_row]);
        cur_row++;
    }
}
void load_target(const char* train_target, double* y) {
    int cur_row = 0;
    ifstream ifs(train_target);//target 0 or 1
    int temp;
    while(ifs >> temp) {
        cout<<temp<<" ";
        y[cur_row++] = temp;
    }
}
