#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

void integrator_generate_constants(void){
    printf("Generaring constants.\n\n");
    mpf_set_default_prec(512);
    mpf_t* _h = malloc(sizeof(mpf_t)*8);
    for (int i=0;i<8;i++){
        mpf_init(_h[i]);
    }
    mpf_t* _r = malloc(sizeof(mpf_t)*28);
    for (int i=0;i<28;i++){
        mpf_init(_r[i]);
    }
    mpf_t* _c = malloc(sizeof(mpf_t)*21);
    mpf_t* _d = malloc(sizeof(mpf_t)*21);
    for (int i=0;i<21;i++){
        mpf_init(_c[i]);
        mpf_init(_d[i]);
    }
    mpf_set_str(_h[0],"0.0",10);
    mpf_set_str(_h[1],"0.0562625605369221464656521910318",10);
    mpf_set_str(_h[2],"0.180240691736892364987579942780",10);
    mpf_set_str(_h[3],"0.352624717113169637373907769648",10);
    mpf_set_str(_h[4],"0.547153626330555383001448554766",10);
    mpf_set_str(_h[5],"0.734210177215410531523210605558", 10);
    mpf_set_str(_h[6],"0.885320946839095768090359771030",10);
    mpf_set_str(_h[7],"0.977520613561287501891174488626",10);

    int l=0;
    for (int j=1;j<8;++j) {
        for(int k=0;k<j;++k) {
            // rr[l] = h[j] - h[k];
            mpf_sub(_r[l],_h[j],_h[k]);
            ++l;
        }
    }
    //c[0] = -h[1];
    mpf_neg(_c[0],_h[1]);
    //d[0] =  h[1];
    mpf_set(_d[0],_h[1]);
    l=0;
    for (int j=2;j<7;++j) {
        ++l;
        // c[l] = -h[j] * c[l-j+1];
        mpf_mul(_c[l], _h[j], _c[l-j+1]);
        mpf_neg(_c[l], _c[l]);
        //d[l] =  h[1] * d[l-j+1];
        mpf_mul(_d[l], _h[1], _d[l-j+1]);
        for(int k=2;k<j;++k) {
            ++l;
            //c[l] = c[l-j] - h[j] * c[l-j+1];
            mpf_mul(_c[l], _h[j], _c[l-j+1]);
            mpf_sub(_c[l], _c[l-j], _c[l]);
            //d[l] = d[l-j] + h[k] * d[l-j+1];
            mpf_mul(_d[l], _h[k], _d[l-j+1]);
            mpf_add(_d[l], _d[l-j], _d[l]);
        }
        ++l;
        //c[l] = c[l-j] - h[j];
        mpf_sub(_c[l], _c[l-j], _h[j]);
        //d[l] = d[l-j] + h[j]; 
        mpf_add(_d[l], _d[l-j], _h[j]);
    }

    // Output   
    printf("r = [");
    for (int i=0;i<28;i++){
         gmp_printf ("\'%.*Ff\'", 32, _r[i]);
         if (i!=27) printf(", ");
    }
    printf("]\n");
    printf("c = [");
    for (int i=0;i<21;i++){
         gmp_printf ("\'%.*Ff\'", 32, _c[i]);
         if (i!=20) printf(", ");
    }
    printf("]\n");
    printf("d = [");
    for (int i=0;i<21;i++){
         gmp_printf ("\'%.*Ff\'", 32, _d[i]);
         if (i!=20) printf(", ");
    }
    printf("]\n");
    exit(0);
}

int main() {
    integrator_generate_constants();
    return 0;
}