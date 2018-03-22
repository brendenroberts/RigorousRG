#include <algorithm>
#include "rrg.h"

template<class Tensor>
Tensor applyGate(const Tensor a , const Tensor b) {
    auto ret = a;
    ret *= prime(b,Site);
    ret.mapprime(2,1,Site);
    return ret;
    }

template<class Tensor>
void applyToMPO(MPOt<Tensor>& eH , const SiteOp<Tensor>& T , int N , int first , int lr , Real err) {
    auto i = T.i;
    auto si = eH.sites().si(i);
    
    if(!first) eH.position(lr?i+1:i,{"Cutoff",1e-18});
    if(i != N) {
        Tensor U,S,V;
        U = first || i == 1 ? Tensor(si,prime(si)) : Tensor(si,prime(si),leftLinkInd(eH,i));
        svdL(first?T.A:applyGate(eH.A(i)*eH.A(i+1),T.A),U,S,V,{"Cutoff",err});
        eH.Aref(i) =   lr ? U*S : U;
        eH.Aref(i+1) = lr ? V : S*V;
        }
    else eH.Anc(i) = first?T.A:applyGate(eH.A(i),T.A);
    
    eH.leftLim(lr||i==N?i-1:i); 
    eH.rightLim(lr||i==N?i+1:i+2); 

    return;
    }

bool icomp(SiteOp<Tensor>& a , SiteOp<Tensor>& b) { return a.i < b.i; }

template<class Tensor>
void TrotterExp(MPOt<Tensor>& eH , double t , int M , vector<SiteOp<Tensor> >& terms , Real err) {
    using SO = SiteOp<Tensor>;
    const SiteSet& hs = eH.sites();
    vector<SO> eHevn , eHodd , eH2odd;
    int N = hs.N();
    Real tstep = 1.0/(t*(double)M);

    struct lo { bool operator() (const SO& a , const SO& b) { return a.i < b.i; } } lo;
    struct hi { bool operator() (const SO& a , const SO& b) { return a.i > b.i; } } hi;

    // sort them first! then push in forward, reverse order
    for(const auto& it : terms) {
        if(!it.A) continue;
        int i = it.i;
        if(i%2) {
             eHodd.push_back(SO(i,expHermitian(-tstep*it.A)));
            eH2odd.push_back(SO(i,expHermitian(-0.5*tstep*it.A)));
            }
        else eHevn.push_back(SO(i,expHermitian(-tstep*it.A)));
        } 
    
    std::sort(eHodd.begin(),eHodd.end(),hi);
    std::sort(eH2odd.begin(),eH2odd.end(),hi);
    std::sort(eHevn.begin(),eHevn.end(),lo);

    for(const auto& g : eH2odd) applyToMPO(eH,g,N,1,RIGHT,err);
    
    for(int n = 0 ; n < M ; ++n) {
        eH.Aref(1) *= 1e10/norm(eH.A(1));
        for(const auto& g : eHevn) applyToMPO(eH,g,N,0,LEFT,err);
        if(n != M-1) for(const auto& g : eHodd) applyToMPO(eH,g,N,0,RIGHT,err);
        }
    
    for(const auto& g : eH2odd) applyToMPO(eH,g,N,0,RIGHT,err);
    eH.Aref(1) *= 1e10/norm(eH.A(1));
 
    return;
    }

template<class MPOType>
void twoLocalTrotter(MPOType& eH , double t , int M , AutoMPO& ampo , Real err) {
    using Tensor = typename MPOType::TensorT;
    const auto& hs = eH.sites();
    int i , N = hs.N();
    Tensor term;
 
    vector<SiteOp<Tensor> > tns(N);
    for(const auto& ht : ampo.terms()) {
        auto st = ht.ops;
        if(st.size() == 1) { // onsite term
            i = st[0].i;
            term = ht.coef*Tensor(hs.op(st[0].op,i));
            if(i != N) term *= Tensor(hs.op("Id",i+1));
        } else { // pairing term
            i = std::min(st[0].i,st[1].i);
            term = ht.coef*Tensor(hs.op(st[0].op,st[0].i))*Tensor(hs.op(st[1].op,st[1].i));
            if(i != N && st[0].i == st[1].i) // user not very familiar with AutoMPO
                term *= Tensor(hs.op("Id",i+1)); 
            }
        tns[i-1] = SiteOp<Tensor>(i,(tns[i-1].i == 0 ? term : tns[i-1].A+term));
        }

    TrotterExp(eH,t,M,tns,err);

    return;
    }
template void twoLocalTrotter(MPO& eH , double t , int M , AutoMPO& ampo , Real err);
template void twoLocalTrotter(IQMPO& eH , double t , int M , AutoMPO& ampo , Real err);
