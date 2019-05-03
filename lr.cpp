#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <omp.h>
#include "utility.h"

using namespace std;

double** scale(double**x,int m, int n){
    double** scale_x = new_2d_mat(m,n);
    double t=omp_get_wtime();
    #pragma omp parallel for
        for(int i = 0; i < n ;++i){//feature
            double mean = 0.0;
            double var = 0.0;
            for(int j = 0; j < m; j++){
                mean += x[j][i];
                var += x[j][i] * x[j][i];
            }
            mean = mean/m;
            for(int j = 0; j < m; j++){
                var += x[j][i] * x[j][i] + mean * mean - 2*mean*x[j][i];
            }
            var = var/m;
            double std=sqrt(var);
            for(int j = 0; j < m; j++){
                scale_x[j][i] = (x[j][i] - mean)/var;
            }
        }
    //cout<<omp_get_wtime()-t<<endl;
    return scale_x;
}

float get_random(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}
class LogisticRegression {
public:
    static double inner_prod(const double* v1, const double* v2, int n) {
        double r = 0.0;
        for(int i = 0; i < n; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }
    double distance(const double* v1, const double* v2, int n) {
        double sum = 0;
        for(int i = 0; i < n; ++i) {
            double minus = v1[i] - v2[i];
            double r = minus * minus;
            sum += r;
        }
        return sqrt(sum);
    }
    double sigmoid(double x) {
        return exp(x) / (1.0 + exp(x));
    }
    int binary(double* x){
        return inner_prod(x, _weight_new, _dim) + _bias;
    }
    double h(double* x) {
        return h(x, _weight_new, _dim, _bias);
    }
    double h(double* x, double* weight, int n, double bias) {
        double y =  inner_prod(x, weight, n) + bias;
        return sigmoid(y);
    }
    double fit(double**nx, int m, int n, double* y, double alpha = 0.1, double l2 = 0, double l1=0.0, int batch_size=128) {
        int max_iters = 10000;
        memset(_weight_old, 0, sizeof(_weight_old[0])*_dim);
        memset(_weight_new, 0, sizeof(_weight_new[0])*_dim);
        for (int i=0; i <_dim; ++i)
            _weight_old[i] = get_random();
        //double** x = scale(nx, m, n);
        double**x = nx;
        double* predict = new double[m];
        auto_ptr<double> ptr(predict);
        double last_cross_entropy_loss = 1e10;
        double cross_entropy_loss = 0;
        #pragma omp parallel
        {
        for(int iter = 0; iter < max_iters; ++iter) {
            //predict
            double cross_entropy_loss = 0;
            int batches = m/batch_size;
            int k1=0;
            int k2=batch_size;
            for(int j = 0; j<batches;j++){
               // #pragma omp parallel for
                for(int i = k1; i < k2; ++i) {
                    predict[i] = h(x[i], _weight_old, _dim, _bias_old);
                 //   #pragma omp atomic update
                    cross_entropy_loss += - ((y[i]*log(predict[i]) + (1-y[i])*log(1-predict[i]))/m);
                }
                // if(iter%1000==0)
                //     cout << "cross_entropy_loss:" << cross_entropy_loss << endl;
                // if (last_cross_entropy_loss - cross_entropy_loss < 0.0001){
                //     return cross_entropy_loss;
                // }
                last_cross_entropy_loss = cross_entropy_loss;
                //cout<<"loss: "<<iter<<" :"<<last_mrse<<endl;
                std::swap(_weight_old, _weight_new);
                _bias = _bias_old;
                //update each weight
               #pragma omp for collapse(2)
                for(int k = 0; k < _dim; ++k) {
                    double gradient = 0.0;
                    for(int i = k1; i < k2; ++i) {
                        gradient += (predict[i] - y[i]) * x[i][k];
                    }
                    _weight_new[k] = _weight_old[k] - (alpha) * gradient/m - (l2/m) * _weight_old[k];
                    //if (_weight_new[k] < 11){ _weight_new[k] = 0; }
                }
                //update bias
                double g = 0.0;
               #pragma omp for
                for(int i = k1; i < k2; ++i) {
                    g += (predict[i] - y[i]);
                }
                _bias_old = _bias - (alpha) * g/m - (l2/m) * _bias;
                k1 += batch_size;
                k2 += batch_size;
               // cout<<k1<<" "<<k2<<endl;
            }
           #pragma omp for
            for(int i = k1; i < m; ++i) {
                    predict[i] = h(x[i], _weight_old, _dim, _bias_old);
                    #pragma omp atomic update
                    cross_entropy_loss += - ((y[i]*log(predict[i]) + (1-y[i])*log(1-predict[i]))/m);
                }
                // if (last_cross_entropy_loss - cross_entropy_loss < 0.0001){
                //     return cross_entropy_loss;
                // }
                last_cross_entropy_loss = cross_entropy_loss;
                //cout<<"loss: "<<iter<<" :"<<last_mrse<<endl;
                std::swap(_weight_old, _weight_new);
                _bias = _bias_old;
                //update each weight
                #pragma omp for collapse(2)
                for(int k = 0; k < _dim; ++k) {
                    double gradient = 0.0;
                    for(int i = k1; i < m; ++i) {
                        gradient += (predict[i] - y[i]) * x[i][k];
                    }
                    _weight_new[k] = _weight_old[k] - (alpha) * gradient/m - (l2/m) * _weight_old[k];
                    //if (_weight_new[k] < 11){ _weight_new[k] = 0; }
                }
                //update bias
                double g = 0.0;
              #pragma omp for
                for(int i = k1; i < m; ++i) {
                    g += (predict[i] - y[i]);
                }
                _bias_old = _bias - (alpha) * g/m - (l2/m) * _bias;
            if(iter%1000==0)
                    cout << "cross_entropy_loss:" << cross_entropy_loss << endl;
        }
        }
        return distance(_weight_new, _weight_old, _dim);
    }
    // void save(std::ostream& os) {
    //     os << "\t"<<"bias:" << _bias << " "<<endl;
    //     for(int i = 0; i < _dim; ++i)
    //         os <<"\t" << i << ":" << _weight_new[i] << " "<<endl ;
    //     os << endl;
    // }
    LogisticRegression(int dim): _dim(dim) {
        _weight_new = new double[dim];
        _weight_old = new double[dim];
        _bias = 0.0;
        _bias_old = 0.0;
    }
    // ~LogisticRegression() {
    //     delete[] _weight_old;
    //     delete[] _weight_new;
    // }
private:
    double* _weight_old;
    double* _weight_new;
    int _dim;
    double _bias;
    double _bias_old;
};
int main(int argc, char* argv[]) {
    if(argc < 7) {
        cerr << "Usage: " << argv[0] << " <train_file> <test_file> <train_rows> <test_rows> <features> <classes>" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
    const char* train = argv[1];
    const char* test = argv[2];
    int train_row = atoi(argv[3]);
    int test_row = atoi(argv[4]);
    int col = atoi(argv[5]); //add bias
    double**train_data = new_2d_mat(train_row, col+1);
    double **train_x = new_2d_mat(train_row, col); 
    double* train_y = double_array(train_row);
    double **scale_train_x = new_2d_mat(train_row, col);
    double**test_data = new_2d_mat(test_row, col+1);
    double **test_x = new_2d_mat(test_row, col); 
    double* test_y = double_array(test_row);
    double **scale_test_x = new_2d_mat(test_row, col);

    load_feature_matrix(train, train_data);
    load_feature_matrix(test, test_data);
    LogisticRegression model(col);

    int num_classes=atoi(argv[6]);
    double** pred_test = new_2d_mat(test_row, num_classes);
    double** pred_train = new_2d_mat(train_row, num_classes);
    double** confuse = new_2d_mat(num_classes, num_classes);
    int cnt = 0;

    
#pragma omp parallel
{
    #pragma omp for collapse(2)
    for (int i=0;i< train_row;i++){
        for(int j=0;j<col+1;j++){
            if(j<col)
                train_x[i][j] = train_data[i][j];
            else
                train_y[i] = int(train_data[i][j]);
        }
    }
    #pragma omp for collapse(2)
    for (int i=0;i< test_row;i++){
        for(int j=0;j<col+1;j++){
            if(j<col)
                test_x[i][j] = test_data[i][j];
            else
                test_y[i] = int(test_data[i][j]);
        }
    }
    
    vector<LogisticRegression> models(num_classes, col);
    scale_train_x = scale(train_x,train_row, col);
    scale_test_x = scale(test_x,test_row, col);
    
    #pragma omp for
    for (int i = 0;i<num_classes; i++){
        LogisticRegression model(col);
        double* y_temp = double_array(train_row);
        for(int j=0;j<train_row;j++){
            if(train_y[j]==i){
                //cout<<y[j]<<" ";
                y_temp[j] = 1;
            }
            else
                y_temp[j] = 0;
        }
        model.fit(scale_train_x, train_row, col, y_temp, 0.1);
        //   for(int  j = 0; j < test_row; ++j) {
        //     pred_test[j][i] = model.h(scale_test_x[j]);
            //int label = (int)y[i];
            //confuse[label][pred]++;
        //}
        cout<<i<<endl;
        //model.save(std::cout);
       models[i] =model;
    }
    //cout<<models.size()<<endl;
    #pragma omp for
    for (int i=0;i<train_row;i++){
        double mx = 0;
        int cl = 0;
        int label = train_y[i];
        for(int j=0;j<num_classes;j++){
            LogisticRegression model(col);
            model = models[j];
            pred_train[i][j] = model.h(scale_train_x[i]);
            if(mx<pred_train[i][j]){
                mx = pred_train[i][j];
                cl = j;
                //cout<<j<<" "<<5<<endl;
            }
            //cout<<pred_test[i][j]<<" ";
        }
        //cout<<cl<<endl;
        if(label==cl)
            cnt++;
    }
    #pragma omp for
    for (int i=0;i<test_row;i++){
        double mx = 0;
        int cl = 0;
        int label = test_y[i];
        for(int j=0;j<num_classes;j++){
            LogisticRegression model(col);
            model = models[j];
            pred_test[i][j] = model.h(scale_test_x[i]);
            if(mx<pred_test[i][j]){
                mx = pred_test[i][j];
                cl = j;
                //cout<<j<<" "<<5<<endl;
            }
            //cout<<pred_test[i][j]<<" ";
        }
        //cout<<cl<<endl;
        //cout<<endl;
        //cout<<label<<" ";
        confuse[label][cl]++;
    }
}
    //cout<<cnt<<endl;
    cout<<"Confusion Matrix:"<<endl;
    cout<<"Labels/Predict"<<endl;
    cout<<" \t";
    for(int i=0;i<num_classes;i++)
        cout<<i<<"\t";
    cout<<endl;
    float total=0;
    float eq=0;
    for(int i=0;i<num_classes;i++){
        cout<<i<<"\t";
        for(int j=0;j<num_classes;j++){
            cout<<confuse[i][j]<<"\t";
            total+=confuse[i][j];
            if(i==j)
            eq+=confuse[i][j];
        }
        cout<<endl;
    }
    cout<<"Train Accuracy:"<<double(double(cnt)/double(train_row))<<endl;
    cout<<"Test Accuracy:"<<double(eq/total)<<endl;
   
    return 0;
}