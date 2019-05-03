using namespace std;
#include<bits/stdc++.h>
#include <string>

extern double* double_array(int col);
extern void free_vector(double* x);
extern double** new_2d_mat(int row, int col);
extern void  free_matrix(double**x, int row);
extern void load_feature_matrix(const char* train_feature, double**x);
extern void load_target_labels(const char* train_target, double* y);
