#ifdef USE_ARPACK 
#include "arssym.h"
#undef F77NAME
#endif

#include "rrg.h"

#ifndef USE_ARPACK
template <class BigMatrixT , class Tensor>
vector<Real> davidsonT(BigMatrixT const& , vector<Tensor>& phi , Args const& args = Args::global());
#endif

template<class Tensor>
void tensorProdH<Tensor>::product(const Tensor& A, Tensor& B) const {
    B = (dim.L > dim.R ? ten.L*(A*ten.R) : (ten.L*A)*ten.R).noprime(Select);
    return;
    }
template void tensorProdH<ITensor>::product(const ITensor& , ITensor&) const;

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
void tensorProdH<Tensor>::diag(int s , bool doI) {
    auto nn = size();
    auto sL = findtype(ten.L,Select); sL.noprime();
    auto sR = findtype(ten.R,Select); sR.noprime();
    auto si = extIndex::gen(sL,s,0);

    fprintf(stderr,"dim H = %d... ",nn);
    if(doI || nn >= 15000) { // iterative diag: ARPACK++ (best for large problems) or ITensor
        if(nn >= 15000 && !doI) fprintf(stderr,"H too large, iterative diag\n");
        
        #ifdef USE_ARPACK
        auto tol = 1e-16;
        ARSymStdEig<Real, tensorProdH<Tensor> > tprob;
        for(int i = 0 , nconv = 0 ; nconv < s ; ++i) {
            if(i != 0) tol *= 1e1;
            tprob.DefineParameters(nn,s,this,&tensorProdH<ITensor>::MultMv,"SA",min(2*s,nn-1),tol,10000*s);
            nconv = tprob.FindEigenvectors();
            fprintf(stderr,"nconv = %d (tol %1.0e)\n",nconv,tol);
            }
        vector<Real> vdat;
        vdat.reserve(s*nn);
        auto vraw = tprob.RawEigenvectors();
        vdat.assign(vraw,vraw+s*nn);
        evc = ITensor({sL,sR,si},Dense<Real>(std::move(vdat)));   
        #else
        vector<Tensor> ret;
        for(int i = 0 ; i < s ; ++i) ret.push_back(randomTensor(sL,sR));
        davidsonT(*this,ret,{"ErrGoal",eps,"MaxIter",10000*s,"DebugLevel",-1});
        fprintf(stderr,"done\n");
        evc = Tensor(sL,sR,si);
        for(auto i : args(ret)) evc += ret[i]*setElt(si(i+1));
        #endif
    } else { // full matrix diag routine, limited to small parameters (s,D)
        Tensor S;
        diagHermitian(-ten.L*ten.R,evc,S,{"Maxm",s});
        fprintf(stderr,"done\n");
        evc *= delta(commonIndex(evc,S),si);
        }

    return;
    }
template void tensorProdH<ITensor>::diag(int , bool);
