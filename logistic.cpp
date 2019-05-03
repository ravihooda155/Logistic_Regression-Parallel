#include <bits/stdc++.h>
#include <omp.h>
#include "utility.h"
#include "logistic.h"

using namespace std;

LogisticRegression::LogisticRegression(int dimension)
{
    new_weights = new double[dimension];
    old_weights = new double[dimension];
    no_of_features = dimension;
    bias = 0.0;
    bias_old = 0.0;
}

double **scale(double **x, int m, int n)
{
    double **scale_x = new_2d_mat(m, n);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    { //feature
        double mean = 0.0;
        double var = 0.0;
        for (int j = 0; j < m; j++)
        {
            mean += x[j][i];
            var += x[j][i] * x[j][i];
        }
        mean = mean / m;
        for (int j = 0; j < m; j++)
        {
            var += x[j][i] * x[j][i] + mean * mean - 2 * mean * x[j][i];
        }
        var = var / m;
        double std = sqrt(var);
        for (int j = 0; j < m; j++)
        {
            scale_x[j][i] = (x[j][i] - mean) / var;
        }
    }
    return scale_x;
}

static double dot_product(const double *v1, const double *v2, int n)
{
    double r = 0.0;
    //#pragma omp parallel for reduction(+:r)
    for (int i = 0; i < n; ++i)
    {
        r += v1[i] * v2[i];
    }
    return r;
}

float LogisticRegression::get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}
double LogisticRegression::distance(const double *v1, const double *v2, int n)
{
    double sum = 0;
    for (int i = 0; i < n; ++i)
    {
        double minus = v1[i] - v2[i];
        double r = minus * minus;
        sum += r;
    }
    return sqrt(sum);
}
double LogisticRegression::sigmoid(double x)
{
    return exp(x) / (1.0 + exp(x));
}
int LogisticRegression::binary(double *x)
{
    return dot_product(x, new_weights, no_of_features) + bias;
}
double LogisticRegression::h(double *x)
{
    return h(x, new_weights, no_of_features, bias);
}
double LogisticRegression::h(double *x, double *weight, int n, double bias)
{
    double y = dot_product(x, weight, n) + bias;
    return sigmoid(y);
}
void LogisticRegression::fit(double **nx, int m, int n, double *y, double alpha, double l2, double l1, int batch_size)
{
    int max_iters = 1000;
    memset(old_weights, 0, sizeof(old_weights[0]) * no_of_features);
    memset(new_weights, 0, sizeof(new_weights[0]) * no_of_features);

    for (int i = 0; i < no_of_features; ++i)
        old_weights[i] = get_random();
    double **x = nx;
    double *predict = new double[m];
    double cross_entropy_loss = 0;

#pragma omp parallel
    {
        for (int iter = 0; iter <= max_iters; ++iter)
        {
            //predict
            cross_entropy_loss = 0;
#pragma omp for reduction(+ \
                          : cross_entropy_loss)
            for (int i = 0; i < m; ++i)
            {
                predict[i] = h(x[i], old_weights, no_of_features, bias_old);
                double g = 0.0;
                for (int k = 0; k < no_of_features; ++k)
                {
                    double gradient = 0.0;
                    gradient = (predict[i] - y[i]) * x[i][k];
                    new_weights[k] = old_weights[k] - (alpha)*gradient / m - (l2 / m) * old_weights[k];
                }
                //if (new_weights[k] < 11){ new_weights[k] = 0; }
                //   #pragma omp atomic update
                g = (predict[i] - y[i]);
                bias = bias_old;
                bias_old = bias - (alpha)*g / m - (l2 / m) * bias;
                //#pragma omp atomic update
                cross_entropy_loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
                std::swap(old_weights, new_weights);
            }
            //if (iter % 100 == 0)
            //    cout << "cross_entropy_loss:" << cross_entropy_loss << endl;
        }
    }
}

// ~LogisticRegression() {
//     delete[] old_weights;
//     delete[] new_weights;
// }
