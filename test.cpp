#include <iostream>
#include <omp.h>
using namespace std;


int main(){


    int i;

    #pragma omp parallel for
    for(i=0;i<33;i++){
        cout<<omp_get_thread_num()<<":"<<i <<endl;
    }
    return 0;
}