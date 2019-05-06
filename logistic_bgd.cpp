#include <bits/stdc++.h>
#include <omp.h>
#include "utility.h"
# include "logistic_bgd.h"

double **scale(double **x, int m, int n)
{
    double **scale_x = new_2d_mat(m, n);
    double t = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    { 
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
#pragma omp parallel for reduction(+:r)
        for (int i = 0; i < n; ++i)
        {
            r += v1[i] * v2[i];
        }
        return r;
    }

    LogisticRegression_Batch::LogisticRegression_Batch(int dimensions)
    {
        no_of_features=dimensions;
        new_weights = new double[dimensions];
        old_weights = new double[dimensions];
        new_bias = 0.0;
        old_bias = 0.0;
    }

    double LogisticRegression_Batch::euclidean_distance(const double *v1, const double *v2, int n)
    {
        double sum = 0;
#pragma omp parallel for reduction(+ \
                                   : sum)
        for (int i = 0; i < n; ++i)
        {
            double minus = v1[i] - v2[i];
            double r = minus * minus;
            sum += r;
        }
        return sqrt(sum);
    }
    float LogisticRegression_Batch::get_random()
    {
        static std::default_random_engine e;
        static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
        return dis(e);
    }
    double LogisticRegression_Batch::sigmoid(double x)
    {
        return exp(x) / (1.0 + exp(x));
    }
    int LogisticRegression_Batch::predict(double *x)
    {
        return dot_product(x, new_weights, no_of_features) + new_bias;
    }
    double LogisticRegression_Batch::h(double *x)
    {
        return h(x, new_weights, no_of_features, new_bias);
    }
    double LogisticRegression_Batch::h(double *x, double *weight, int n, double bias)
    {
        double y = dot_product(x, weight, n) + bias;
        return sigmoid(y);
    }
    double LogisticRegression_Batch::fit(double **nx, int m, int n, double *y, double alpha, double l2, double l1 , int batch_size)
    {
        int max_iters = 1000;
        memset(old_weights, 0, sizeof(old_weights[0]) * no_of_features);
        memset(new_weights, 0, sizeof(new_weights[0]) * no_of_features);
        for (int i = 0; i < no_of_features; ++i)
            old_weights[i] = get_random();
        //double** x = scale(nx, m, n);
        double **x = nx;
        double *predict = new double[m];
        auto_ptr<double> ptr(predict);
        double last_cross_entropy_loss = 1e10;
        double cross_entropy_loss = 0;
        double g = 0.0;
        double gradient = 0.0;
#pragma omp parallel
        {
            for (int iter = 0; iter < max_iters; ++iter)
            {
                //predict
                int batches = m / batch_size;
                int k1 = 0;
                int k2 = batch_size;
                cross_entropy_loss = 0;

                for (int j = 0; j < batches; j++)
                {

                #pragma omp for reduction(+: cross_entropy_loss)
                    for (int i = k1; i < k2; ++i)
                    {
                        predict[i] = h(x[i], old_weights, no_of_features, old_bias);
                        cross_entropy_loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
                    }
                   
                    last_cross_entropy_loss = cross_entropy_loss;
                    //cout<<"loss: "<<iter<<" :"<<last_mrse<<endl;
                    std::swap(old_weights, new_weights);
                    new_bias = old_bias;
                    //update each weight

                    for (int k = 0; k < no_of_features; ++k)
                    {
                        gradient = 0.0;
#pragma omp for reduction(+: gradient)
                        for (int i = k1; i < k2; ++i)
                        {
                            //predict[i] = h(x[i], old_weights, no_of_features, old_bias);
                            gradient += (predict[i] - y[i]) * x[i][k];
                        }
                        new_weights[k] = old_weights[k] - (alpha)*gradient / m - (l2 / m) * old_weights[k];
                        //if (new_weights[k] < 11){ new_weights[k] = 0; }
                    }
                    //update bias
                    g = 0.0;
#pragma omp for reduction(+: g)
                    for (int i = k1; i < k2; ++i)
                    {
                        //predict[i] = h(x[i], old_weights, no_of_features, old_bias);
                        g += (predict[i] - y[i]);
                    }
                    old_bias = new_bias - (alpha)*g / m - (l2 / m) * new_bias;
                    k1 += batch_size;
                    k2 += batch_size;
                    // cout<<k1<<" "<<k2<<endl;
                }

#pragma omp for reduction(+: cross_entropy_loss)
                for (int i = k1; i < m; ++i)
                {
                    predict[i] = h(x[i], old_weights, no_of_features, old_bias);
                    cross_entropy_loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
                }

                last_cross_entropy_loss = cross_entropy_loss;
                //cout<<"loss: "<<iter<<" :"<<last_mrse<<endl;
                std::swap(old_weights, new_weights);
                new_bias = old_bias;
                //update each weight

                for (int k = 0; k < no_of_features; ++k)
                {
                    gradient = 0.0;
#pragma omp for reduction(+: gradient)
                    for (int i = k1; i < m; ++i)
                    {
                        //predict[i] = h(x[i], old_weights, no_of_features, old_bias);
                        gradient += (predict[i] - y[i]) * x[i][k];
                    }
                    new_weights[k] = old_weights[k] - (alpha)*gradient / m - (l2 / m) * old_weights[k];
                    //if (new_weights[k] < 11){ new_weights[k] = 0; }
                }
                //update bias
                g = 0.0;
#pragma omp for reduction(+: g)
                for (int i = k1; i < m; ++i)
                {
                    g += (predict[i] - y[i]);
                }
                old_bias = new_bias - (alpha)*g / m - (l2 / m) * new_bias;
               
            }
        }
        return euclidean_distance(new_weights, old_weights, no_of_features);
    }