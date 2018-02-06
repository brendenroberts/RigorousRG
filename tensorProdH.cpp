#include "rrg.h"

void tensorProdH::product(const ITensor& A, ITensor& B) const {
    ITensor ret;
    for(auto& Hpair : pairH) {
        auto cur = A*C;
        cur *= Hpair.L;
        cur *= Hpair.R;
        ret = (ret ? ret + noprime(cur)*C : noprime(cur)*C);
        }

    B = ret;
    return;
    }

void tensorProdH::MultMv(Real* v, Real* w) {
    auto l = int(findtype(pairH[0].L,Select));
    auto r = int(findtype(pairH[0].R,Select));
    Real *t = (Real *)malloc(l*r*sizeof(*t));
    vector<Real> L,R;
    int i = 0;

    for(auto& Hpair : pHvec) {
        L = Hpair.L;
        R = Hpair.R;
        
        if(L.size() == 1)
            cblas_dsymm(CblasColMajor,CblasRight,CblasUpper,l,r,pHscl[i].R,R.data(),r,v,l,(i?1.0:0.0),w,l);
        else if(R.size() == 1)
            cblas_dsymm(CblasColMajor, CblasLeft,CblasUpper,l,r,pHscl[i].L,L.data(),l,v,l,(i?1.0:0.0),w,l);
        else {
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,l,r,r,pHscl[i].R,v,l,R.data(),r,0.0,t,l);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,l,r,l,pHscl[i].L,L.data(),l,t,l,(i?1.0:0.0),w,l);
            }
        i++;
        }
    
    free(t);
    return;
    }

int tensorProdH::size() const {
    return int(findtype(C,Link)); 
    }
