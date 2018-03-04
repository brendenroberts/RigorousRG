#include "rrg.h"

template<class Tensor>
Tensor applyGate(const Tensor a , const Tensor b) {
    auto ret = a;
    ret *= prime(b,Site);
    ret.mapprime(2,1,Site);
    return ret;
    }

template<class Tensor>
void applyToMPO(MPOt<Tensor>& eH , const SiteOp<Tensor>& T , int N , int first , Real err) {
    auto i = T.i;
    eH.position(i,{"Cutoff",1e-20});
    if(i != N) {
        Tensor U,S,V;
        if(first || i == 1) U = Tensor(eH.sites().si(i),eH.sites().siP(i));
        else U = Tensor(eH.sites().si(i),eH.sites().siP(i),leftLinkInd(eH,i));
        //svdL(first?T.A:applyGate(eH.A(i)*eH.A(i+1),T.A),U,S,V,{"Cutoff",err});
        //eH.Aref(i) = U;
        //eH.Aref(i+1) = S*V;
        eH.svdBond(i,first?T.A:applyGate(eH.A(i)*eH.A(i+1),T.A),Fromleft,{"Cutoff",err});
        }
    else eH.Anc(i) = first?T.A:applyGate(eH.A(i),T.A);
    return;
    }

template<class Tensor>
void TrotterExp(MPOt<Tensor>& eH , double t , int M , vector<SiteOp<Tensor> >& terms , Real err) {
    const SiteSet& hs = eH.sites();
    vector<SiteOp<Tensor> > eHevn , eHodd , eH2odd;
    int N = hs.N();
    Real tstep = 1.0/(t*(double)M);
    int i,n;

    for(const auto& it : terms) {
        if(!it.A) continue;
        i = it.i;
        if(i%2) {
             eHodd.push_back(SiteOp<Tensor>(i,expHermitian(-tstep*it.A)));
            eH2odd.push_back(SiteOp<Tensor>(i,expHermitian(-0.5*tstep*it.A)));
            }
        else eHevn.push_back(SiteOp<Tensor>(i,expHermitian(-tstep*it.A)));
        } 
    
    eH.position(1,{"Cutoff",1e-16});
    eH.Aref(1) *= 1e10/norm(eH.A(1));
    
    for(const auto& g : eH2odd) applyToMPO(eH,g,N,1,err);
    
    for(n = 0 ; n < M ; ++n) {
        for(const auto& g : eHevn) applyToMPO(eH,g,N,0,err);
        if(n != M-1) for(const auto& g : eHodd) applyToMPO(eH,g,N,0,err); 
        eH.Aref(N) *= 1e10/norm(eH.A(N));
        }
    
    for(const auto& g : eH2odd) applyToMPO(eH,g,N,0,err);
    eH.Aref(1) *= 1e10/norm(eH.A(1));
    
    return;
    }

template<class MPOType>
void twoLocalTrotter(MPOType& eH , double t , int M , AutoMPO& ampo , Real err) {
    using Tensor = typename MPOType::TensorT;
    const auto& hs = eH.sites();
    int N = hs.N();
 
    vector<SiteOp<Tensor> > tns(N);
    
    for(const HTerm& ht : ampo.terms()) {
        Tensor term;
        int i;
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
    eH.position(1,{"Cutoff",1e-16});

    return;
    }
template void twoLocalTrotter(MPO& eH , double t , int M , AutoMPO& ampo , Real err);
template void twoLocalTrotter(IQMPO& eH , double t , int M , AutoMPO& ampo , Real err);
