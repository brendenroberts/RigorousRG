#include "rrg.h"
#include <cblas.h>
#include <iostream>
#include <fstream>

ITensor apply(const ITensor a , const ITensor b) {
    auto ret = a;
    ret *= prime(b,Site);
    ret.mapprime(2,1,Site);
    return ret;
    }

void applyToMPO(MPO& eH , const SiteITensor& T , int N) {
    auto i = T.i;
    eH.position(i,{"Cutoff",eps});
    if(i != N) eH.svdBond(i,apply(eH.A(i)*eH.A(i+1),T.A),Fromleft,{"Cutoff",eps});
    else eH.Anc(i) = apply(eH.A(i),T.A);
    return;
    }

void TrotterExp(MPO& eH , double t , int M , vector<SiteITensor>& terms) {
    const SiteSet& hs = eH.sites();
    vector<SiteITensor> eHevn , eHodd , eH2odd;
    int N = hs.N();
    Real tstep = 1.0/(t*(double)M);
    int i,n;

    for(const auto& it : terms) {
        if(!it.A) continue;
        i = it.i;
        if(i%2) {
             eHodd.push_back(SiteITensor(i,expHermitian(    -tstep*it.A)));
            eH2odd.push_back(SiteITensor(i,expHermitian(-0.5*tstep*it.A)));
            }
        else eHevn.push_back(SiteITensor(i,expHermitian(-tstep*it.A)));
        } 
    
    eH.position(1,{"Cutoff",eps});
    eH.Aref(1) *= 1e4/norm(eH.A(1));
    //eH.Aref(1).scaleTo(1e4);
    
    for(const auto& g : eH2odd) applyToMPO(eH,g,N);
        
    for(n = 0 ; n < M ; ++n) {
        for(const auto& g : eHevn) applyToMPO(eH,g,N);
        if(n != M-1) for(const auto& g : eHodd) applyToMPO(eH,g,N); 
        //Print(eH.A(N).scale().real());
        }
    
    for(const auto& g : eH2odd) applyToMPO(eH,g,N);
    
    return;
    }

void twoLocalTrotter(MPO& eH , double t , int M , AutoMPO& ampo) {
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

    TrotterExp(eH,t,M,tns);
    eH.position(1,{"Cutoff",eps});

    return;
    }
