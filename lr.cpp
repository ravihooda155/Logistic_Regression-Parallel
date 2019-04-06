#include <iostream>
#include <random>
#include <cmath>
#include <memory>
#include <fstream>
#include "data.h"
using namespace std;

float get_random(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}
 double** scale(double**x,int m, int n){
        double** scale_x = dmatrix(m,n);
        for(int i = 0; i < n ;++i){//feature
            double mean = 0.0;
            double var = 0.0;
            for(int j = 0; j < m; j++){
                mean += x[j][i];
                var += x[j][i] * x[j][i];
            }
            mean = mean/m;
            var = var/m - mean * mean;
            for(int j = 0; j < m; j++){
                scale_x[j][i] = (x[j][i] - mean)/var;
            }
        }
        return scale_x;
}
class LR {
public:
    static double inner_prod(const double* v1, const double* v2, int n) {
        double r = 0.0;
        for(int i = 0; i < n; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }
    double sigmoid(double x) {
        return exp(x)/ (1.0 + exp(x));
    }
    double binary(double* x){
        return sigmoid(inner_prod(x, _weight_new, _dim) + _bias);
    }
    double h(double* x) {
        return h(x, _weight_new, _dim, _bias);
    }
    double h(double* x, double* weight, int n, double bias) {
        double y =  inner_prod(x, weight, n) + bias;
        return sigmoid(y);
    }
    //y: 0,1
    void fit(double**nx, int m, int n, double* y, double alpha = 0.01, double l2 = 0.0, double l1=0.0, int itr = 5000) {
        int max_iters = itr;
        memset(_weight_old, 0, sizeof(_weight_old[0])*_dim);
        memset(_weight_new, 0, sizeof(_weight_new[0])*_dim);
        for (int i=0; i <_dim; ++i)
            _weight_old[i] = get_random();
        // for(int i=0;i<_dim;i++){
        //     cout<<_weight_old[i]<<" ";
        // }
        //cout<<endl;
        //double** x = scale(nx, m, n);
        double**x = nx;
        double* predict = new double[m];
	
        double last_cross_entropy_loss = 0;
        for(int iter = 0; iter < max_iters; ++iter) {
            //predict
            double mrse = 0;
            double cross_entropy_loss = 0;
            for(int i = 0; i < m; ++i) {
                predict[i] = h(x[i], _weight_old, _dim, _bias_old);
                //mrse += (y[i] - predict[i]) * (y[i] - predict[i]);
                cross_entropy_loss += - (y[i]*log(predict[i]) + (1-y[i])*log(1-predict[i]));
            }
        
            last_cross_entropy_loss = cross_entropy_loss;
            std::swap(_weight_old, _weight_new);
            _bias = _bias_old;
            //update each weight
            for(int k = 0; k < _dim; ++k) {
                double gradient = 0.0;
                for(int i = 0; i < m; ++i) {
                    gradient += (predict[i] - y[i]) * x[i][k];
                }
                _weight_new[k] = _weight_old[k] - (alpha/m) *( gradient + l2 * _weight_old[k]);
                //if (_weight_new[k] < 11){ _weight_new[k] = 0; }
            }
            //update bias
            double g = 0.0;
            for(int i = 0; i < m; ++i) {
                g += (predict[i] - y[i]);
            }
            _bias_old = _bias - alpha * g/m;
        }
        //return distance(_weight_new, _weight_old, _dim);
    }
    void save(std::ostream& os) {
        os << "\t"<<"bias:" << _bias << " "<<endl;
        for(int i = 0; i < _dim; ++i)
            os <<"\t" << i << ":" << _weight_new[i] << " "<<endl ;
        os << endl;
    }
    LR(int dim): _dim(dim) {
        _weight_new = new double[dim];
        _weight_old = new double[dim];
        _bias = 0.0;
        _bias_old = 0.0;
    }
private:
    double* _weight_old;
    double* _weight_new;
    int _dim;
    double _bias;
    double _bias_old;
};
int main(int argc, char* argv[]) {
    if(argc < 5) {
        cerr << "Usage: " << argv[0] << " <train_feature> <train_target> <rows> <cols> <classes> [test]" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
    const char* feature = argv[1];
    const char* target = argv[2];
    int num_classes = atoi(argv[5]);
    int row = atoi(argv[3]);
    int col = atoi(argv[4]); //add bias
    double**x = dmatrix(row, col);
    double **scale_x = dmatrix(row, col);
    double* y = dvector(row);
    //load_data(train_instance, x,y);  //if train_target\ttrain_feature are merged in one file
    csv_load_feature(feature, x);
    load_target(target, y);
    vector<LR> v;
    scale_x = scale(x,row, col);
    for (int i = 0;i<num_classes; i++){
        LR model(col);
        int n = sizeof(y)/sizeof(y[0]);
        double* y_temp = dvector(row);
        for(int j=0;j<n;j++){
            if(y[j]==i)
                y_temp[j] = 1;
            else
                y_temp[j] = 0;
        }
        model.fit(scale_x, row, col, y_temp, 0.001);
        cout<<i<<endl;
        //model.save(std::cout);
        v.push_back(model);
    }
    double** pred = dmatrix(row, num_classes);
    for(int j=0;j<num_classes;j++){
        LR model = v[j];
        //double** confuse = dmatrix(num_classes, num_classes);
        for(int i = 0; i < row; ++i) {
            pred[i][j] = model.binary(x[i]);
            //int label = (int)y[i];
            //confuse[label][pred]++;
        }
    }
    double** confuse = dmatrix(num_classes, num_classes);
    int cnt = 0;
   
    for (int i=0;i<row;i++){
        double mx = 0;
        int cl = 0;
        for(int j=0;j<num_classes;j++){
            if(mx<pred[i][j]){
                mx = pred[i][j];
                cl = j;
                //cout<<j<<" "<<5<<endl;
            }
            //cout<<pred[i][j]<<" ";
        }
        //cout<<cl<<endl;
        //cout<<endl;
        int label = (int)y[i];
        //cout<<label<<" ";
        if(label==cl)
            cnt++;
        confuse[label][cl]++;
    }
    cout<<cnt<<endl;
    cout<<"Confusion Matrix:"<<endl;
     cout<<"Labels/Predict"<<endl;
    cout<<" \t";
    for(int i=0;i<num_classes;i++)
        cout<<i<<"\t";
    cout<<endl;
    for(int i=0;i<num_classes;i++){
        cout<<i<<"\t";
        for(int j=0;j<num_classes;j++){
            cout<<confuse[i][j]<<"\t";
        }
        cout<<endl;
    }
    
    return 0;
}
