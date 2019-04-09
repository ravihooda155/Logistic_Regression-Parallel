#include "data.h"
#include<bits/stdc++.h>
#include<cstring>
#include <fstream>
using namespace std;

void csv_file(double** ptr,const char* train_feature,const char* train_target, double* labels) {
    //cout<<"csv_file entry"<<endl;
    int cur_row = 0,i=0;
    ifstream input;
    input.open(train_feature);
    string line;
    while(getline(input,line)){
        //read_data(ptr[cur_row],line);
        const char* pos=line.c_str();
        int i=0;
       //cout<<line<<endl;
       while(pos - 1!=NULL && pos[0] != '\0' && pos[0] != '#') {
        float value = atof(pos);
        ptr[cur_row][i++] = value;
        pos = strchr(pos,',')+1;
       }
       cur_row++;
    }
    double c;
    ifstream input1;
    input1.open(train_target);
    while(input1>>c){
        labels[i++]=c;
    }
}


double* new_1d_array(int n) {
    double* new_array = (double*)malloc(sizeof(double)*n);
    memset(new_array, 0, sizeof(double)*n);
    return new_array;
}
void free_1d_mem(double* ptr){
    free(ptr);
}

double** new_2d_array(int m, int n) {
    //cout<<"matrix init"<<endl;
    double** ptr=new double*[m];
    for(int i=0;i<m;i++){
        ptr[i]=new double[n];
    	memset(ptr[i],0,sizeof(double)*n);
    }
    
    return ptr;
}

void free_2d_mem(int m,double** ptr) {
    //cout<<"vector init"<<endl;
    for(int i=0;i<m; i++){
	free(ptr[i]);
    }
    free(ptr);
}




