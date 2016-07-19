#include "itensor/all.h"
#include "itensor/mps/mpo.h"
#include "davidson2.h"
#include <cmath>
#include <sstream>
#include <iostream>
#include <vector>
#include <time.h>

using namespace itensor;
using std::vector;

void plussers(Index const& l1, 
         Index const& l2, 
         Index      & sumind, 
         ITensor    & first, 
         ITensor    & second)
    {
    auto m = l1.m()+l2.m();
    if(m <= 0) m = 1;
    sumind = Index(sumind.rawname(),m);

    first = delta(l1,sumind);
    auto S = Matrix(l2.m(),sumind.m());
    for(auto i : range(l2.m()))
        {
        S(i,l1.m()+i) = 1;
        }
    second = matrixTensor(std::move(S),l2,sumind);
    }

MPO& MPOadd(MPO & L, MPO const& R, Args const& args) {
    using Tensor = typename MPO::TensorT;

    auto N = L.N();
    if(R.N() != N) Error("Mismatched MPS sizes");

    L.primelinks(0,4);

    auto first = vector<Tensor>(N);
    auto second = vector<Tensor>(N);

    for(auto i : range1(N-1))
        {
        auto l1 = rightLinkInd(L,i);
        auto l2 = rightLinkInd(R,i);
        auto r = l1;
        plussers(l1,l2,r,first[i],second[i]);
        }

    L.Anc(1) = L.A(1) * first[1] + R.A(1) * second[1];
    for(auto i : range1(2,N-1))
        {
        L.Anc(i) = dag(first[i-1]) * L.A(i) * first[i] 
                     + dag(second[i-1]) * R.A(i) * second[i];
        }
    L.Anc(N) = dag(first[N-1]) * L.A(N) + dag(second[N-1]) * R.A(N);

    L.noprimelink();

    L.orthogonalize(args);

    return L;
    }

void MPSfromITensor(const ITensor& psi , MPS& state , double eps) {
    ITensor U,S,T;
    auto N = state.N();
    Index linkInd;
    auto W = psi;
    for(int i = 1 ; i < N ; ++i) {
        if(i == 1) 
            U = state.A(i);
        else
            U = ITensor(findtype(state.A(i),Site),linkInd);
        svd(W,U,S,T,{"Cutoff",eps});
        linkInd = commonIndex(U,S);
        state.setA(i,U);
        W = S*T;
        }
    state.setA(N,W);
    
    return;
    }

vector<Real> exactDiagonalizeMPO(const MPO& H , vector<MPS>& states , int n , double eps) {
    vector<Real> evals;
    // contract MPO into single tensor A
    auto N = H.N();
    auto A = H.A(1);
    A *= -1.0;
    for(int i = 2 ; i <= N ; ++i)
        A *= H.A(i);

    // these hold eigenvalues in ascending order, and vectors
    // (minus signs flip the spectrum, which is ordinarily descending) 
    auto D = ITensor(A.inds());
    auto V = ITensor(A.inds());
    diagHermitian(A,V,D);
    D *= -1.0;

    for(int i = 1 ; i <= n ; ++i) {
        auto psi = MPS(H.sites());
        // make a projector into the state we want
        auto proj = ITensor(noprime(D.inds()[0]));
        proj.set(proj.inds()[0](i),1.0);
        evals.push_back((prime(proj)*D*proj).real());
        auto Vp = V*proj;
        MPSfromITensor(Vp,psi,eps);
        states.push_back(psi);
        }
    
    return evals;
    }

vector<Real> dmrgThatStuff(const MPO& H , vector<MPS>& states , double penalty , double eps) {
    vector<Real> evals;
    vector<MPS> excl;
    int num_sw = 20;
    for(auto psi = states.begin() ; psi != states.end() ; ++psi) {
        auto swp = Sweeps(num_sw);
        num_sw = 50;
        swp.maxm() = 50,50,100,100,500;
        swp.cutoff() = eps;
        swp.niter() = 3;
        swp.noise() = 1E-7,1E-8,0.0;
 
        // dmrg call won't shut up
        std::stringstream ss;
        auto out = std::cout.rdbuf(ss.rdbuf()); 
        auto e = dmrg(*psi,H,excl,swp,{"Quiet",true,"PrintEigs",false,"Weight",penalty});
        std::cout.rdbuf(out);

        excl.push_back(*psi);
        evals.push_back(e);
        }

    return evals;
    }

ITensor Apply(const ITensor a , const ITensor b) {
    auto ret = a;
    ret *= prime(b,Site);
    ret.mapprime(2,1,Site);
    return ret;
    }

double boundMPOnorm(const MPO& mpo) {
    const SiteSet& hs = mpo.sites();
    int N = mpo.N();
    double result = 0.0;
    for(int i = 1 ; i <= N ; ++i) {
        auto sym = mpo.A(i);
        ITensor site(hs.si(i));
        randomize(site);
        sym = Apply(sym,dag(sym));
        result += sqrt(-1.0*davidson2(-1.0*sym,site));
        }
    
    return result;
    }

MPO ExactH(const SiteSet& hs , double offset) {
    int N = hs.N();
    AutoMPO ampo(hs);
    for(int i = 1; i <= N ; ++i) {
        if(i != N) ampo += 2.0,"Sz",i,"Sz",i+1;
        ampo += 5.0,"Sx",i;
        }
    ampo += offset,"Id",1;

    return MPO(ampo);
    }

vector<ITensor> TwoSiteH(const SiteSet& hs , double offset) {
    int N = hs.N();
    auto H = vector<ITensor>(N-1);
    for(int i = 1 ; i <= N-1 ; ++i) {
        H.at(i-1) = 2.0*ITensor(hs.op("Sz",i)*hs.op("Sz",i+1));
        if(i % 2 != 0) {
            H.at(i-1) += 5.0*ITensor(hs.op("Sx",i))*ITensor(hs.op("Id",i+1));
            H.at(i-1) += 5.0*ITensor(hs.op("Id",i))*ITensor(hs.op("Sx",i+1));
            }
        }
    H.at(0) += offset*ITensor(hs.op("Id",1)*hs.op("Id",2));
    if(N % 2 != 0) H.at(N-2) += 5.0*ITensor(hs.op("Id",N-1))*ITensor(hs.op("Sx",N));
    return H;
    }

// TEBD way to get MPO exp(-bH), kinda sucks because H is hardcoded
void TrotterExp(MPO& eH , double t , int M , double E , Real eps) {
    const SiteSet& hs = eH.sites();
    vector<ITensor>::const_iterator g;
    vector<ITensor> eH_evn , eH_odd;
    int N = hs.N();
    int M2 = M/2;
    Real tstep = 1.0/(t*(double)(2*M2));
    eH_evn.reserve(N/2);
    eH_odd.reserve(N/2);
    auto H2 = TwoSiteH(hs,E);
    int i,n;

    // go halfway, then square
    auto eH2 = eH;

    for(i = 1 ; i < N ; ++i) {
        if(i % 2 == 0)  eH_evn.push_back(expHermitian(-tstep*H2.at(i-1)));
        else            eH_odd.push_back(expHermitian(-0.5*tstep*H2.at(i-1)));
        }

    for(n = 0 ; n < M2 ; ++n) {
        for(g = eH_odd.cbegin() , i = 1 ; g != eH_odd.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = Apply(eH2.A(i)*eH2.A(i+1),*g);
            eH2.svdBond(i,site,Fromleft,{"Cutoff",eps});
            }
        
        for(g = eH_evn.cbegin() , i = 2 ; g != eH_evn.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = Apply(eH2.A(i)*eH2.A(i+1),*g);
            eH2.svdBond(i,site,Fromleft,{"Cutoff",eps});
            }
        
        for(g = eH_odd.cbegin() , i = 1 ; g != eH_odd.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = Apply(eH2.A(i)*eH2.A(i+1),*g);
            eH2.svdBond(i,site,Fromleft,{"Cutoff",eps});
            } 
        }
    
    nmultMPO(eH2,eH2,eH,{"Cutoff",eps});
    return;
    }

double ApproxH(const MPO& eH , MPO& Ha , double ej , double t , Real eps) {
    Ha *= (ej+t);
    auto E = eH;
    E *= -t*exp(ej/t);
    Ha.orthogonalize();
    E.orthogonalize();
    //Ha.plusEq(E,{"Cutoff",eps});
    MPOadd(Ha,E,{"Cutoff",eps});
    
    return ej+t;
    }

void ShiftH(MPO& H , double normH , double eta1) {
    const SiteSet& hs = H.sites();
    MPO I1(hs);
    MPO I2(hs);
    I1 *= -eta1;
    I2 *= -1.0;
    I1.orthogonalize();
    I2.orthogonalize();
    H.orthogonalize();
    MPOadd(H,I1,{});
    //H.plusEq(I1,{"Cutoff",1E-8,"Maxm",500});
    H *= 2.0/(normH-eta1);
    H.orthogonalize();
    MPOadd(H,I2,{}); 
    
    return;
    }

void NormalizedCheby(const MPO& H , MPO& K , int k , double eta0 , double eta1 , double normH , Real eps) {
    MPO Pm2(H.sites());
    MPO Pm1 = H;

    // build normalization through same process as MPO construction
    //double narg = 2.0*(eta0 - eta1)/(normH - eta1) - 1.0;
    //double Pnm2 = 1.0;
    //double Pnm1 = narg;
    //double Pn = 0.0;

    // do recurrence to appropriate order k
    for(int i = 2 ; i <= k ; ++i) {
        nmultMPO(H,2.0*Pm1,K,{"Cutoff",eps});
        K.orthogonalize();
        Pm2 *= -1.0;
        Pm2.orthogonalize();
        MPOadd(K,Pm2,{"Cutoff",eps});
        K.orthogonalize();
        
        // rescaling needed for numerical stability
        K *= 1.0/norm(K.A(1));
        Pm1 *= 1.0/norm(K.A(1));
        Pm2 = Pm1;
        Pm1 = K;
        } 
    return;
    }
