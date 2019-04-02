using namespace std;
#include<bits/stdc++.h>
#include <string>
extern double** dmatrix(int row, int col);
extern void  free_matrix(double**x, int row);
extern double* dvector(int col);
extern void free_vector(double* x);
extern void csv_load_feature(const char* train_feature, double**x);
extern void csv_read(const char* in_string, double*x);
extern void load_target(const char* train_target, double* y);
