#include "rrg.h"
#include <lapacke.h>
#include <random>
#include <fstream>

int main(int argc, char *argv[]) {
    if(argc != 5 && argc != 6) {
        printf("usage: rrg N n s D (seed)\n");
        return 1;
        }
    time_t t1,t2,tI,tF;
    Real delt;
    MPO rhoG,rhoGA;
    ITensor U,Dg,G;
    Index ei;

    // RRG structure parameters
    const int    N  = atoi(argv[1]); // should be n*(power of 2)
    const int    n  = atoi(argv[2]); // initial blocking size
    int          w  = n;             // block size (scales with m)
    int          ll = 0;             // lambda block index
    int          m  = 0;             // RG scale factor

    // AGSP and subspace parameters
    const double t = 5.0;            // Trotter temperature
    const int    M = 50;             // num Trotter steps
    const int    k = 10;             // power of Trotter op
    const int    s = atoi(argv[3]);  // formal s param
    const int    D = atoi(argv[4]);  // formal D param

    // Hamitonian parameters
    const Real   J = 1.0;
    const Real   h = 0.5;
    const Real   g = -1.05;
    //const Real   P = 0.5;

    // computational settings
    const int    e   = s; // number of DMRG eigenstates to compute
    const int    doI = 1; // diag restricted Hamiltonian iteratively?
    const int    doV = 1; // compute viability from DMRG gs?

    // setup random sampling
    std::random_device r;
    int seed = (argc == 6 ? atoi(argv[5]) : r());
    fprintf(stderr,"seed is %d\n",seed);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> udist(0.0,1.0);

    // random Hamiltonian parameters
    const double J0 = 1.0;
    const double x0 = 1.0;
    const double x1 = 0.0;
    const double PW  = -0.5;
    vector<double> Js(N-1);
    for(int i = 1 ; i < N ; ++i)
        Js[i-1] = J0*pow((pow(x1,PW+1.0)-pow(x0,PW+1.0))*udist(gen)+pow(x0,PW+1.0),1.0/(PW+1.0));

    // generate Hilbert subspaces for each level m = 0,...,log(N/n)
    vector<SpinHalf> hsps;
    for(int x = n ; x <= N ; x *= 2) hsps.push_back(SpinHalf(x));
    SpinHalf hs = hsps.back();
   
    // compute restricted H, boundary terms to use for each level
    AutoMPO autoH(hs);
    std::stringstream sts;
    auto out = std::cout.rdbuf(sts.rdbuf());
    vector< vector<MPO> > Hs;
    vector< vector< vector<MPOPair> > > bndterms;
    for(const auto& it : hsps) {
        vector<MPO> Hcur;
        vector< vector<MPOPair> > bcur;
        int L = it.N();
        for(int o = 0 ; o < N ; o += L) {
            AutoMPO ampo(it);
            for(int i = 1 ; i <= L ; ++i) {
                if(i != L) ampo += -J*4.0,"Sz",i,"Sz",i+1;
                ampo += -h*2.0,"Sz",i;
                ampo += -g*2.0,"Sx",i;
                }
            Hcur.push_back(MPO(ampo));

            if(L == N) autoH = ampo; 

            if(o+L < N) {
                vector<MPOPair> cur(1);
                AutoMPO bl1(it),br1(it);
                bl1 += -J*4.0,"Sz",L;     br1 += 1.0,"Sz",1;
                cur[0] = MPOPair(MPO(bl1),MPO(br1));
                bcur.push_back(cur);
                }
            }
        Hs.push_back(Hcur);
        bndterms.push_back(bcur);
        }
    auto H = Hs.back()[0];
    std::cout.rdbuf(out);

    // use DMRG to find good gs and try to approximate excited states
    time(&t1);
    vector<MPS> evecs;
    for(int i = 0 ; i < e ; ++i) evecs.push_back(MPS(hs));    
    auto evals = dmrgMPO(H,evecs,10.0);
    auto mn = std::distance(evals.begin(),std::min_element(evals.begin(),evals.end()));
    auto gs = evecs[mn];
    fprintf(stderr,"BD ");
    for(const auto& it : evecs) fprintf(stderr,"%d\t",maxM(it));
    fprintf(stderr,"\n");
    pvec(evals,e);
    time(&t2);
    fprintf(stderr,"DMRG time: %.f s\n",difftime(t2,t1));
 
    // approximate the thermal operator exp(-H/t)^k using Trotter
    // and MPO multiplication; temperature of K is k/t
    time(&tI);
    MPO eH(hs);
    twoLocalTrotter(eH,t,M,autoH);
    auto K = eH;
    for(int i = 1 ; i < k ; ++i) nmultMPO(eH,K,K,{"Cutoff",eps});
   
    // INITIALIZATION: generate complete product basis over m=0 Hilbert space
    int p = (int)pow(2,n);
    vector<MPS> V1;
    
    for(int i = 0 ; i < p ; ++i) {
        InitState istate(hsps[0],"Dn");
        for(int j = 1 ; j <= n ; ++j)
            if(i/(int)pow(2,j-1)%2) istate.set(j,"Up");
        V1.push_back(MPS(istate));
        }
    MPS bSpaceL(hsps[0]);
    MPS bSpaceR(hsps[0]);
    combineMPS(V1,bSpaceL,LEFT);
    combineMPS(V1,bSpaceR,RIGHT);
    
    // reduce dimension by sampling from initial basis, either
    // bSpaceL or bSpaceR depending on how the merge will work
    vector<MPS> Spre;
    for(ll = 0 ; ll < N/n ; ll++) {
        int ss = (ll%2 ? 1 : n); // site housing the dangling Select index
        auto curSS = (ll%2 ? bSpaceR : bSpaceL);
        ITensor P,S;
        Index si("ext",s,Select);
       
        // return orthonormal basis of eigenstates
        auto Hproj = overlapT(curSS,Hs[0][ll],curSS);
        auto eigs = diagHermitian(-Hproj,P,S,{"Maxm",s});
        curSS.Aref(ss) *= P*delta(commonIndex(P,S),si);
        regauge(curSS,ss);
 
        Spre.push_back(curSS);
        }
    time(&t2);
    fprintf(stderr,"initialization: %.f s\n",difftime(t2,tI));

    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPS> Spost;
    for(m = 0 ; (int)Spre.size() > 1 ; ++m,w*=2) {
        fprintf(stderr,"Level %d (w = %d)\n",m,w);
        auto hs = hsps[m];
        if(doV) rhoGA = MPO(hs);
        Spost.clear();
        // STEP 1: for each block, expand dimension of subspace with AGSP operators
        for(ll = 0 ; ll < N/w ; ++ll) {
            MPO A(hs);
            MPO Hc = Hs[m][ll];
            MPS pre = Spre[ll];
            int ss = (ll%2 ? 1 : w);
            auto pi = findtype(pre.A(ss),Select);

            // STEP 1a: extract filtering operators A from AGSP K
            time(&t1);
            restrictMPO(K,A,w*ll+1,D,ll%2);
            time(&t2);
            fprintf(stderr,"truncate AGSP: %.f s\n",difftime(t2,t1));
            
            // STEP 1b: expand subspace using the mapping A:pre->ret
            time(&t1);
            MPS ret(hs);
            applyMPO(pre,A,ret,ll%2);
            time(&t2);
            fprintf(stderr,"apply AGSP: %.f s\n",difftime(t2,t1));
            fprintf(stderr,"max m: %d\n",maxM(ret));

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H,
            // which is necessary for the iterative solver in the Merge step
            G = overlapT(ret,ret);
            ITensor U,Dg,P,S;
            diagHermitian(G,U,Dg,{"Cutoff",1E-9});
            Dg.apply(invsqrt);
            auto Hproj = prime(Dg*U*overlapT(ret,Hs[m][ll],ret)*prime(dag(U)*Dg),-1);
            ei = Index("ext",int(commonIndex(Dg,U)),Select);
            auto eigs = diagHermitian(-Hproj,P,S);
            ret.Aref(ss) *= prime(P)*Dg*U*delta(prime(commonIndex(P,S)),ei);
            regauge(ret,ss);

            if(doV) {
                reducedDM(gs,rhoGA,w*ll+1);
                delt = (overlapT(pre,rhoGA,pre)*delta(pi,prime(pi))).real();
                fprintf(stderr,"1-delta (pre): %e\n",max(eps,1.0-delt));
                fprintf(stdout,"%18.15e,",max(eps,1.0-delt));
                delt = (overlapT(ret,rhoGA,ret)*delta(ei,prime(ei))).real();
                fprintf(stderr,"1-delta (ret): %.10e\n",max(eps,1.0-delt));
                fprintf(stdout,"%18.15e\n",max(eps,1.0-delt));
                }

            Spost.push_back(ret);
            }

        // STEP 2: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(ll = 0 ; ll < N/w ; ll+=2) {
            auto spL = Spost[ll];                // L subspace
            auto spR = Spost[ll+1];              // R subspace
            auto sL = findtype(spL.A(w),Select); // L dangling index
            auto sR = findtype(spR.A(1),Select); // R dangling index
            ITensor P;
            Index si("ext",s,Select);

            auto HL = overlapT(spL,Hs[m][ll],spL);
            auto HR = overlapT(spR,Hs[m][ll+1],spR);
            auto H1 = HL*delta(sR,prime(sR)) + HR*delta(sL,prime(sL));
            for(auto& bb : bndterms[m][ll])
                H1 += overlapT(spL,bb.L,spL)*overlapT(spR,bb.R,spR);
            
            time(&t1);
            if(doI) {
                auto C = combiner(sL,sR);
                H1 = prime(C)*H1*C;
                auto ci = commonIndex(H1,C);
                fprintf(stderr,"dim H = %d\n",int(ci));
                vector<ITensor> ret;
                vector<Real> eigs;
                for(int i = 0 ; i < s ; ++i) ret.push_back(randomTensor(ci));
                eigs = davidsonT(H1,ret,{"ErrGoal",1e-10,"MaxIter",100*s});
                P = ITensor(si,ci);
                combineVectors(ret,P);
                P *= C;
                pvec(eigs,s);
            } else {
                ITensor Dg;
                auto eigs = diagHermitian(-H1,P,Dg,{"Maxm",s});
                P *= delta(commonIndex(P,Dg),si);
                }
            time(&t2);
            fprintf(stderr,"diag restricted H: %.f s\n",difftime(t2,t1));

            MPS ret(hsps[m+1]);
            time(&t1);
            tensorProduct(spL,spR,ret,P,(ll/2)%2);
            time(&t2);
            fprintf(stderr,"tensor product (ll=%d): %.f s\n",ll,difftime(t2,t1));
            fprintf(stderr,"max m: %d\n",maxM(ret));
            
            if(N/w > 2) {    
                int ss = ((ll/2)%2 ? 1 : 2*w);
                G = overlapT(ret,ret);
                diagHermitian(G,U,Dg,{"Cutoff",eps});
                Dg.apply(invsqrt);
                ret.Aref(ss) *= U*Dg;
                regauge(ret,ss);
                ei = Index("ext",int(commonIndex(Dg,U)),Select);
                ret.Aref(ss) *= delta(ei,prime(commonIndex(Dg,U)));
                }
 
            Spre.push_back(ret);
            }
        }

    // EXIT: compute eigenvalues of low-energy subspace using H
    auto res = Spre[0];
    auto fi = findtype(res.A(N),Select);
    vector<MPS> eigenstates(s);
    for(int i = 1 ; i <= s ; ++i) {
        auto fc = res;
        fc.Aref(N) *= setElt(fi(i));
        regauge(fc,1);
        fc.normalize();
        if(i == 1) {
            delt = fabs(overlap(fc,gs));
            fprintf(stderr,"Viability: %e\n",1.0-delt*delt);
            fprintf(stdout,"%18.15e,0\n",1.0-delt*delt);
            }
        Real eng = overlap(fc,H,fc);
        eigenstates[i-1] = fc;
        }
    fprintf(stderr,"BD ");
    for(const auto& it : eigenstates) fprintf(stderr,"%d\t",maxM(it));
    fprintf(stderr,"\n");
    
    time(&tF);
    fprintf(stderr,"RRG elapsed time: %.f s\n",difftime(tF,tI));

/*  
    fprintf(stdout,"# DMRG\n");
    for(i = 1 ; i < N ; ++i) {
        gs.position(i,{"Cutoff",eps});
        auto SzA = hs.op("Sz",i);
	for(j = i+1 ; j <= N ; ++j) { 
            auto SzB = hs.op("Sz",j);
            double ccD = twopoint(gs,SzA,SzB,i,j);
            fprintf(stdout,"%9.6f\t",ccD);
            }
        fprintf(stdout,"\n");
        }
    fprintf(stdout,"# RRG\n");
    for(i = 1 ; i < N ; ++i) {
        res.position(i,{"Cutoff",eps});
        auto SzA = hs.op("Sz",i);
	for(j = i+1 ; j <= N ; ++j) { 
            auto SzB = hs.op("Sz",j);
            double ccR = twopoint(res,SzA,SzB,i,j);
            fprintf(stdout,"%9.6f\t",ccR);
            }
        fprintf(stdout,"\n");
        }
*/
    return 0;
    }

double twopoint(const MPS& psi , const ITensor& A , const ITensor& B , int a , int b) {
    auto ir = commonIndex(psi.A(a),psi.A(a+1),Link);

    auto C = psi.A(a)*A*dag(prime(psi.A(a),Site,ir));
    for(int k = a+1; k < b; ++k) {
        C *= psi.A(k);
        C *= dag(prime(psi.A(k),Link));
        }

    C *= psi.A(b);
    C *= B;
    auto jl = commonIndex(psi.A(b),psi.A(b-1),Link);
    C *= dag(prime(psi.A(b),jl,Site));

    return C.real();
    }

