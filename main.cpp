#include <iostream>
#include <random>
#include <cmath>
#include <memory>
#include <fstream>
#include "utility.h"
using namespace std;

float get_random(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);
}
 double** scale(double**x,int m, int n){
        double** x_scale = new_2d_array(m,n);
        int i=0;
        while(i<n)
        {
            double variance = 0.0;
            double mean = 0.0;
            int j=0;
            while(j<m)
            {
                variance += x[j][i]*x[j][i];
                mean += x[j][i];
                j++;
            }
            mean = mean/m;
            variance = variance/m - mean * mean;
            for(int j = 0; j < m; j++){
                x_scale[j][i] = (x[j][i] - mean)/variance;
            }
            i++;
        }
        return x_scale;
}
class LR {
public:
    static double vec_prod(const double* vect1, const double* vect2, int n) {
        double res = 0.0;
        int i=0;
        while(i<n)
        {
            res+= vect2[i]*vect1[i];
            i++;
        }
        return res;
    }
    double sigmoid(double x) {
        return exp(x)/ (1.0 + exp(x));
    }
    double binary(double* x){
        return sigmoid(vec_prod(x, new_weights, dimensions) + _bias);
    }
   
    void fit(double**nx, int m, int n, double* y, double alpha = 0.01, double l2 = 10, double l1=0.0, int itr = 5000) {
        int max_iters = itr;
        memset(old_weights, 0, sizeof(old_weights[0])*dimensions);
        memset(new_weights, 0, sizeof(new_weights[0])*dimensions);
        for (int i=0; i <dimensions; ++i)
            old_weights[i] = get_random();
        double** x = nx;
        double* predict = new double[m];
	
        double last_cross_entropy_loss = 0;
        for(int iter = 0; iter < max_iters; ++iter) {
            //predict
            double mrse = 0;
            double cross_entropy_loss = 0;
            for(int i = 0; i < m; ++i) {
                predict[i] = sigmoid(double( vec_prod(x[i], old_weights, dimensions) + old_bias));;
                cross_entropy_loss += - (y[i]*log(predict[i]) + (1-y[i])*log(1-predict[i]));
            }
        
            last_cross_entropy_loss = cross_entropy_loss;
            std::swap(old_weights, new_weights);
            _bias = old_bias;
            //update each weight
            int k=0;
            while(k<dimensions)
             {
                double gradient = 0.0;
                int i=0;
                while(i<m) 
                {
                    gradient += x[i][k]*(predict[i] - y[i]);
                    i++;
                }
                new_weights[k] = old_weights[k] - (alpha/m) *( gradient + l2 * old_weights[k]);
                k++;
                //if (new_weights[k] < 11){ new_weights[k] = 0; }
            }
            //update bias
            double g = 0.0;
            for(int i = 0; i < m; ++i) {
                g += (predict[i] - y[i]);
            }
            old_bias = _bias - alpha * g/m;
        }
    }
    void save() {
        cout<< "\t"<<"Bias:" << _bias << " "<<endl;
        for(int i = 0; i < dimensions; i++)
        {    cout <<"\t" << i << ":" << new_weights[i];
             cout << " "<<endl ;
        }
        cout << endl;
    }
    LR(int dim): dimensions(dim) {
        new_weights = new double[dim];
        old_weights = new double[dim];
        _bias = 0.0;
        old_bias = 0.0;
    }
private:
    
    int dimensions;
    double _bias;
    double old_bias;
    double* old_weights;
    double* new_weights;
};
int main(int argc, char* argv[]) {
    if(argc < 6) {
        cout<< "Incorrect Input Format";
        exit;
    }
    
    
    const char* feature_num = argv[4];
    int num_classes = atoi(argv[3]);
    
    int col = atoi(argv[2]); 
    int row = atoi(argv[1]);
    const char* target_values = argv[5];
    double**x = new_2d_array(row, col);
    double **scale_x = new_2d_array(row, col);
    double* y = new_1d_array(row);
    //load_data(train_instance, x,y);  //if train_target\ttrain_feature are merged in one file
   
    csv_file(x,feature_num,target_values,y);
    vector<LR> v;
    scale_x = scale(x,row, col);
    for (int i = 0;i<num_classes; i++){
        LR model(col);
        int n = sizeof(y)/sizeof(y[0]);
        double* y_temp = new_1d_array(row);
        for(int j=0;j<n;j++){
            if(y[j]==i)
                y_temp[j] = 1;
            else
                y_temp[j] = 0;
        }
        model.fit(scale_x, row, col, y_temp, 0.001);
        cout<<i<<endl;
        model.save();
        v.push_back(model);
    }
    double** pred = new_2d_array(row, num_classes);
    for(int j=0;j<num_classes;j++){
        LR model = v[j];
        //double** confuse = new_2d_array(num_classes, num_classes);
        for(int i = 0; i < row; ++i) {
            pred[i][j] = model.binary(x[i]);
            //int label = (int)y[i];
            //confuse[label][pred]++;
        }
    }
    double** confuse = new_2d_array(num_classes, num_classes);
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
