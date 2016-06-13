#include "itensor/all.h"
#include "itensor/mps/mpo.h"
#include <cmath>
#include <vector>

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

MPO dag(const MPO& mpo) {
    int N = mpo.N();
    auto dmpo = mpo;
    for(int i = 1 ; i <= N ; ++i)
        dmpo.Anc(i) = dag(dmpo.A(i));
    return dmpo;
    }

double MPOnorm(const MPO& mpo) {
    int N = mpo.N();
    auto dmpo = dag(mpo);
    ITensor result = mpo.A(1)*dmpo.A(1);
    for(int i = 2 ; i <= N ; ++i)
        result *= mpo.A(i)*dmpo.A(i);
    return sqrt((double)result.real());
    }

MPO ExactH(const SiteSet& hs) {
    int N = hs.N();
    AutoMPO ampo(hs);
    for(int i = 1; i <= N ; ++i) {
        if(i != N) ampo += 0.25,"Sz",i,"Sz",i+1;
        ampo += "Sx",i;
        }

    return MPO(ampo);
    }

vector<ITensor> TwoSiteH(const SiteSet& hs) {
    int N = hs.N();
    auto H = vector<ITensor>(N-1);
    for(int i = 1 ; i <= N-1 ; ++i) {
        H.at(i-1) = 0.25*ITensor(hs.op("Sz",i)*hs.op("Sz",i+1));
        H.at(i-1) += (i==1 ? 1.0:0.5)*ITensor(hs.op("Sx",i))*ITensor(hs.op("Id",i+1));
        H.at(i-1) += (i==N-1 ? 1.0:0.5)*ITensor(hs.op("Id",i))*ITensor(hs.op("Sx",i+1));
        }
    return H;
    }

ITensor Apply(ITensor a , const ITensor b) {
    a *= prime(b);
    a.mapprime(2,1,Site);
    a /= norm(a);
    return a;
    }
/*
void MPO_SVD(MPO& mpo , int i , ITensor a , Direction dir , Real d) {
    auto U = mpo.A(i);
    ITensor D,V;
    svd(a,U,D,V,{"Cutoff",d});
    mpo.Anc(i) = U;
    mpo.Anc(i+1) = V;
    dir == Fromleft ? mpo.Anc(i+1) = D*mpo.A(i+1) : mpo.Anc(i) = mpo.A(i)*D;

    return;
    }
*/
// TEBD way to get MPO exp(-bH), kinda sucks because H is hardcoded
// This apparently returns a bad construction of the exponential!
// TODO: take MPO H as argument (AutoMPO)?, pass to TwoSiteH which extracts
void TrotterExp(MPO& eH , double t , int Nt , Real eps) {
    const SiteSet& hs = eH.sites();
    vector<ITensor>::const_iterator g;
    vector<ITensor> eH_evn , eH_odd;
    int N = hs.N();
    Real tstep = 1.0/(2.0*t*(double)Nt);
    eH_evn.reserve(N/2);
    eH_odd.reserve(N/2);
    auto H2 = TwoSiteH(hs);
    int i,n;

    // go halfway, then square
    auto eH2 = eH;

    for(i = 1 ; i < N ; ++i)
        if(i % 2 == 0)  eH_evn.push_back(expHermitian(-tstep*H2.at(i-1)));
        else            eH_odd.push_back(expHermitian(-tstep/2.0*H2.at(i-1)));

    for(n = 0 ; n < Nt ; ++n) {
        for(g = eH_odd.cbegin() , i = 1 ; g != eH_odd.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = eH2.A(i)*eH2.A(i+1);
            site = Apply(site,*g);
            eH2.svdBond(i,site,Fromleft,{"Cutoff",eps});
            }
        
        for(g = eH_evn.cbegin() , i = 2 ; g != eH_evn.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = eH2.A(i)*eH2.A(i+1);
            site = Apply(site,*g);
            eH2.svdBond(i,site,Fromleft,{"Cutoff",eps});
            }
        
        for(g = eH_odd.cbegin() , i = 1 ; g != eH_odd.cend() ; ++g , i+=2) {
            eH2.position(i);
            auto site = eH2.A(i)*eH2.A(i+1);
            site = Apply(site,*g);
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
    
    return MPOnorm(Ha);
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
    double narg = 2.0*(eta0 - eta1)/(normH - eta1) - 1.0;
    double Pnm2 = 1.0;
    double Pnm1 = narg;
    double Pn = 0.0;

    // do recurrence to appropriate order k
    for(int i = 2 ; i <= k ; ++i) {
        Pn = 2.0 * narg * Pnm1 - Pnm2;
        Pnm2 = Pnm1;
        Pnm1 = Pn;
         
        nmultMPO(H,2.0*Pm1,K,{"Cutoff",eps,"Maxm",500});
        K.orthogonalize();
        Pm2 *= -1.0;
        Pm2.orthogonalize();
        MPOadd(K,Pm2,{"Cutoff",eps,"Maxm",500});
        Pm2 = Pm1;
        Pm1 = K;
        
        K /= Pn;
        } 
    
    return;
    }
