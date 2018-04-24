#include "rrg.h"

void tensorProdH::product(const ITensor& A, ITensor& B) const {
    B = (dim.L > dim.R ? ten.L*(A*ten.R) : (ten.L*A)*ten.R).noprime(Select);
    return;
    }

void tensorProdH::MultMv(Real* v, Real* w) {
    auto l = dim.L , r = dim.R , p = int(commonIndex(ten.L,ten.R,Link));
    Real *t = (Real *)malloc(l*r*p*sizeof(*t));
   
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,l,r*p,r,scl.R,v,l,dat.R.data(),r,0.0,t,l);
    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,l,r,p*l,scl.L,dat.L.data(),p*l,t,p*l,0.0,w,l);
    
    free(t);
    return;
    }

int tensorProdH::size() const {
    return dim.L*dim.R; 
    }
