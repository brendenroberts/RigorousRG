#include "rrg.h"

ITensor apply(const ITensor a , const ITensor b) {
    auto ret = a;
    ret *= prime(b,Site);
    ret.mapprime(2,1,Site);
    return ret;
    }

void applyToMPO(MPO& eH , const SiteITensor& T , int N , int first , Real err) {
    auto i = T.i;
    eH.position(i,{"Cutoff",0.0});
    if(i != N) {
        ITensor U,S,V;
        if(first || i == 1) U = ITensor(eH.sites().si(i),eH.sites().siP(i));
        else U = ITensor(eH.sites().si(i),eH.sites().siP(i),leftLinkInd(eH,i));
        svdL(first?T.A:apply(eH.A(i)*eH.A(i+1),T.A),U,S,V,{"Cutoff",err});
        eH.Aref(i) = U;
        eH.Aref(i+1) = S*V;
        //eH.svdBond(i,first?T.A:apply(eH.A(i)*eH.A(i+1),T.A),Fromleft,{"Cutoff",err*1e-2});
        }
    else eH.Anc(i) = first?T.A:apply(eH.A(i),T.A);
    return;
    }

void TrotterExp(MPO& eH , double t , int M , vector<SiteITensor>& terms , Real err) {
    const SiteSet& hs = eH.sites();
    vector<SiteITensor> eHevn , eHodd , eH2odd;
    int N = hs.N();
    Real tstep = 1.0/(t*(double)M);
    int i,n;

    for(const auto& it : terms) {
        if(!it.A) continue;
        i = it.i;
        if(i%2) {
             eHodd.push_back(SiteITensor(i,expHermitian(-tstep*it.A)));
            eH2odd.push_back(SiteITensor(i,expHermitian(-0.5*tstep*it.A)));
            }
        else eHevn.push_back(SiteITensor(i,expHermitian(-tstep*it.A)));
        } 
    
    eH.position(1,{"Cutoff",0.0});
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

void twoLocalTrotter(MPO& eH , double t , int M , AutoMPO& ampo , Real err) {
    const auto& hs = eH.sites();
    int N = hs.N();
 
    vector<SiteITensor> tns(N);
    
    for(const HTerm& ht : ampo.terms()) {
        ITensor term;
        int i;
        auto st = ht.ops;
        if(st.size() == 1) { // onsite term
            i = st[0].i;
            term = ht.coef*ITensor(hs.op(st[0].op,i));
            if(i != N) term *= ITensor(hs.op("Id",i+1));
        } else { // pairing term
            i = std::min(st[0].i,st[1].i);
            term = ht.coef*ITensor(hs.op(st[0].op,st[0].i))*ITensor(hs.op(st[1].op,st[1].i));
            if(i != N && st[0].i == st[1].i) // user not very familiar with AutoMPO
                term *= ITensor(hs.op("Id",i+1)); 
            }
        tns[i-1] = SiteITensor(i,(tns[i-1].i == 0 ? term : tns[i-1].A+term));
        }

    TrotterExp(eH,t,M,tns,err);
    eH.position(1,{"Cutoff",0.0});

    return;
    }
