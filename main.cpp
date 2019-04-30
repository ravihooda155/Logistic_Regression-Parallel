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
       double** scale_x = new_2d_array(m,n);
        for(int i = 0; i < n ;++i){//feature
            //double mean = 0.0;
            //double var = 0.0;
            double minn=(double)INT_MAX;
            double maxx=(double)INT_MIN;
            for(int j = 0; j < m; j++){
                minn=min(minn,x[j][i]);
                maxx=max(maxx,x[j][i]);
            }
            for(int j = 0; j < m; j++){
                scale_x[j][i] = (x[j][i] - minn)/(maxx-minn==0?maxx:(maxx-minn));
               // cout<<"org: "<<x[j][i]<<" scaled: "<<scale_x[j][i]<<endl;
            }
            //cout<<"-----------------"<<endl;
        }
        return scale_x;
}
class LogisticRegression {

private:
    int dimensions;
    double bias;
    double old_bias;
    double* old_weights;
    double* new_weights;
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
    double predict(double* x){
        double h_x=vec_prod(x, new_weights, dimensions); 
        return sigmoid(h_x+bias);
    }
   
    void fit(double**nx, int m, int n, double* y, double alpha = 0.1, double l2 = 0.0, double l1=0.0, int itr = 1000) {
        int max_iters = itr;
        memset(old_weights, 0, sizeof(old_weights[0])*dimensions);
        memset(new_weights, 0, sizeof(new_weights[0])*dimensions);
        for (int i=0; i <dimensions; ++i)
            old_weights[i] = get_random();
        double** x = nx;
        double* predict = new double[m];
	
        for(int iter = 0; iter < max_iters; ++iter) {
            double cross_entropy_loss = 0;
            for(int i = 0; i < m; ++i) {
                predict[i] = double( vec_prod(x[i], old_weights, dimensions) + old_bias);
                //cout<<"predict:"<<predict[i]<<endl;
                cross_entropy_loss += (y[i]-predict[i])*(y[i]-predict[i]);
            }
            cout<<"loss: "<<iter<<" :"<<cross_entropy_loss<<endl;
            //last_cross_entropy_loss = cross_entropy_loss;
            std::swap(old_weights, new_weights);
            bias = old_bias;
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
            old_bias = bias - alpha * g/m;
        }
    }
    void print() {
        cout<< "\t"<<"Bias:" << bias << " "<<endl;
        for(int i = 0; i < dimensions; i++)
        {    cout <<"\t" << i << ":" << new_weights[i];
             cout << " "<<endl ;
        }
        cout << endl;
    }
    LogisticRegression(int dim): dimensions(dim) {
        new_weights = new double[dim];
        old_weights = new double[dim];
        bias = 0.0;
        old_bias = 0.0;
    }
};
int main(int argc, char* argv[]) {
    
   
    if(argc < 6) {
        cout<< "Incorrect Input Format";
        exit(0);
    }
    
    
    const char* feature_num = argv[1];
    const char* target_values = argv[2];
    int row = atoi(argv[3]);
    int col = atoi(argv[4]);
    int num_classes = atoi(argv[5]);
    
    double**x = new_2d_array(row, col);
    double **scale_x = new_2d_array(row, col);
    double* y = new_1d_array(row);
   
    csv_file(x,feature_num,target_values,y,row);
    vector<LogisticRegression> v;
   
    scale_x = scale(x,row, col);

    for (int i = 0;i<num_classes; i++){
        LogisticRegression model(col);
        int n = sizeof(y)/sizeof(y[0]);
        double* y_temp = new_1d_array(row);
        for(int j=0;j<n;j++){
            if(y[j]==i)
                y_temp[j] = 1;
            else
                y_temp[j] = 0;
        }
        model.fit(scale_x, row, col, y_temp, 0.4);
        model.print();
        v.push_back(model);
    }
    double** pred = new_2d_array(row, num_classes);
    for(int j=0;j<num_classes;j++){
        LogisticRegression model = v[j];
        for(int i = 0; i < row; ++i) {
            pred[i][j] = model.predict(scale_x[i]);
        }
    }
    cout<<"****************************************"<<endl;

    double** confuse = new_2d_array(num_classes, num_classes);
    int cnt = 0;
   
    for(int i=0;i<row;i++){
        for(int j=0;j<num_classes;j++){
            cout<<pred[i][j]<<" ";
        }
        cout<<endl;
    }


    for (int i=0;i<row;i++){
        double mx=0;
        int cl = 0;
        for(int j=0;j<num_classes;j++){
            if(mx<pred[i][j]){
                mx = pred[i][j];
                cl = j;
            }
        }
        cout<<"class:"<<y[i]<<" prediction:"<<cl<<endl;
        int label = (int)y[i];
        if(label==cl)  cnt++;
        confuse[label][cl]++;
    }
    cout<<((double)cnt/row)<<endl;
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
