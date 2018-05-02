#include "rrg.h"

template<class Tensor>
void tensorProdH<Tensor>::product(const Tensor& A, Tensor& B) const {
    B = (dim.L > dim.R ? ten.L*(A*ten.R) : (ten.L*A)*ten.R).noprime(Select);
    return;
    }

template<class Tensor>
void tensorProdH<Tensor>::MultMv(Real* v, Real* w) {
    auto l = dim.L , r = dim.R , p = int(commonIndex(ten.L,ten.R,Link));
    Real *t = (Real *)malloc(l*r*p*sizeof(*t));
   
    gemm_wrapper(false,false,l,r*p,r,scl.R,v,dat.R.data(),0.0,t);
    gemm_wrapper(true ,false,l,r,p*l,scl.L,dat.L.data(),t,0.0,w);
    
    free(t);
    return;
    }
template void tensorProdH<ITensor>::MultMv(Real* v , Real* w);
//template void tensorProdH<IQTensor>::MultMv(Real* v , Real* w);

template<class Tensor>
int tensorProdH<Tensor>::size() const {
    return dim.L*dim.R; 
    }
