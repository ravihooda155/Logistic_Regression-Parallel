#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <omp.h>
#include "utility.h"
#include "logistic.h"
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        cerr << "Usage: " << argv[0] << " <train_file> <test_file> <train_rows> <test_rows> <features> <classes>" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
    const char *train = argv[1];
    const char *test = argv[2];
    int train_row = atoi(argv[3]);
    int test_row = atoi(argv[4]);
    int col = atoi(argv[5]); 
    double **train_data = new_2d_mat(train_row, col + 1);
    double **train_x = new_2d_mat(train_row, col);
    double *train_y = double_array(train_row);
    double **scale_train_x = new_2d_mat(train_row, col);
    double **test_data = new_2d_mat(test_row, col + 1);
    double **test_x = new_2d_mat(test_row, col);
    double *test_y = double_array(test_row);
    double **scale_test_x = new_2d_mat(test_row, col);
    load_feature_matrix(train, train_data);
    load_feature_matrix(test, test_data);
    LogisticRegression model(col);
    int num_classes = atoi(argv[6]);
    vector<LogisticRegression> models(num_classes, col);
    
    double **pred_test = new_2d_mat(test_row, num_classes);
    double **pred_train = new_2d_mat(train_row, num_classes);
    double **confuse = new_2d_mat(num_classes, num_classes);
    int cnt = 0;

 double time=omp_get_wtime();

#pragma omp parallel
    {
    #pragma omp for collapse(2)
    for (int i = 0; i < train_row; i++)
    {
        for (int j = 0; j < col + 1; j++)
        {
            if (j < col)
                train_x[i][j] = train_data[i][j];
            else
                train_y[i] = int(train_data[i][j]);
        }
    }
#pragma omp for collapse(2)
    for (int i = 0; i < test_row; i++)
    {
        for (int j = 0; j < col + 1; j++)
        {
            if (j < col)
                test_x[i][j] = test_data[i][j];
            else
                test_y[i] = int(test_data[i][j]);
        }
    }
    scale_train_x = scale(train_x, train_row, col);
    scale_test_x = scale(test_x, test_row, col);
#pragma omp for
        for (int i = 0; i < num_classes; i++)
        {
            LogisticRegression model(col);
            double *y_temp = double_array(train_row);
            for (int j = 0; j < train_row; j++)
            {
                if (train_y[j] == i)
                {
                    y_temp[j] = 1;
                }
                else
                    y_temp[j] = 0;
            }
            model.fit(scale_train_x,train_row, col,y_temp,0.1,0,0,0);
            cout<<"-----------------"<<endl;
            cout << i << endl;
            models[i] = model;
        }
        //cout<<models.size()<<endl;
#pragma omp for
        for (int i = 0; i < train_row; i++)
        {
            double mx = 0;
            int cl = 0;
            int label = train_y[i];
            for (int j = 0; j < num_classes; j++)
            {
                LogisticRegression model(col);
                model = models[j];
                pred_train[i][j] = model.h(scale_train_x[i]);
                if (mx < pred_train[i][j])
                {
                    mx = pred_train[i][j];
                    cl = j;
                }
            }
            if (label == cl)  cnt++;
        }
#pragma omp for
        for (int i = 0; i < test_row; i++)
        {
            double mx = 0;
            int cl = 0;
            int label = test_y[i];
            for (int j = 0; j < num_classes; j++)
            {
                LogisticRegression model(col);
                model = models[j];
                pred_test[i][j] = model.h(scale_test_x[i]);
                if (mx < pred_test[i][j])
                {
                    mx = pred_test[i][j];
                    cl = j;
                }
            }
            confuse[label][cl]++;
        }
    }
    cout << cnt << endl;
    cout << "Confusion Matrix:" << endl;
    cout << "Labels/Predict" << endl;
    cout << " \t";
    for (int i = 0; i < num_classes; i++)
        cout << i << "\t";
    cout << endl;
    float total = 0;
    float eq = 0;
    for (int i = 0; i < num_classes; i++)
    {
        cout << i << "\t";
        for (int j = 0; j < num_classes; j++)
        {
            cout << confuse[i][j] << "\t";
            total += confuse[i][j];
            if (i == j)
                eq += confuse[i][j];
        }
        cout << endl;
    }
    cout << "Train Accuracy:" << double(double(cnt) / double(train_row)) << endl;
    cout << "Test Accuracy:" << double(eq / total) << endl;

    cout<<"Total execution time:"<<omp_get_wtime()-time<<endl;
    return 0;
}