using namespace std;
#include<bits/stdc++.h>
#include <string>
extern double** dmatrix(int row, int col);
extern void  free_matrix(double**x, int row);
extern double* dvector(int col);
extern void free_vector(double* x);
extern bool ends_with(const string& str, const string& a);
extern void csv_load_feature(const char* train_feature, double**x);
extern void csv_read(const char* in_string, double*x);
extern void svm_read_line(const string& line, double* x, double y);
extern void svm_load(const string& filename, double** x, double* y);
extern void load_target(const char* train_target, double* y);
