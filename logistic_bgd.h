#include<bits/stdc++.h>
#include<cstring>
#include <fstream>
#include "utility.h"
using namespace std;

class LogisticRegression_Batch
{
private:
    double *old_weights;
    double *new_weights;
    int no_of_features;
    double new_bias;
    double old_bias;
public:
    LogisticRegression_Batch(int);
    float get_random();
    double sigmoid(double);
    int predict(double *);
    double h(double *);
    double h(double *,double *,int,double);
    double fit(double **,int,int,double *,double,double,double,int);
    double euclidean_distance(const double *,const double *,int);
};
static double dot_product(const double *,const double *,int);
double **scale(double **,int,int);
