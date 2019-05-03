#include<bits/stdc++.h>
#include<cstring>
#include <fstream>
#include "utility.h"
using namespace std;

class LogisticRegression
{
private:
    double *old_weights;
    double *new_weights;
    int no_of_features;
    double bias;
    double bias_old;
public:
    LogisticRegression(int);
    float get_random();
    double sigmoid(double);
    int binary(double *);
    double h(double *);
    double h(double *,double *,int,double);
    void fit(double **, int, int, double *, double alpha = 0.1, double l2 = 0, double l1 = 0.0, int batch_size = 0);
    double distance(const double *,const double *,int);
};
static double dot_product(const double *,const double *,int);
double **scale(double **,int,int);
