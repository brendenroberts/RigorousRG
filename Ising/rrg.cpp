#include "rrg.h"
#include <fstream>

int main(int argc, char *argv[]) {
    if(argc != 5 && argc != 6) {
        printf("usage: rrg N n s D (id_string)\n");
        return 1;
        }
    time_t t1,t2,tI,tF;
    Real delt;
    MPO rhoG,rhoGA;
    ITensor U,Dg,P,S;
    Index ei;

    // RRG structure parameters
    const int    N  = atoi(argv[1]); // should be n*(power of 2)
    const int    n  = atoi(argv[2]); // initial blocking size
    int          w  = n;             // block size (scales with m)
    int          ll = 0;             // lambda block index
    int          m  = 0;             // RG scale factor

    // AGSP and subspace parameters
    const double t = 0.4;            // Trotter temperature
    const int    M = 100;             // num Trotter steps
    const int    k = 1;              // power of Trotter op (should be 1)
    const int    s = atoi(argv[3]);  // formal s param
    const int    D = atoi(argv[4]);  // formal D param
    
    // computational settings
    const int    e   = 2; // number of DMRG states to compute, e>2 may be slow
    const int    doI = 1; // diag restricted Hamiltonian iteratively?
    const int    doV = 1; // compute viability from DMRG gs?

    // Hamitonian parameters
    const Real   J = 1.0;
    const Real   h = 0.5;
    const Real   g = -1.05;

    FILE *sxfl,*syfl,*szfl,*gsfl;
    char id[128],sxnm[256],synm[256],sznm[256],gsnm[256];
    if(argc == 5) sprintf(id,"rrg-L%d-s%d-D%d",N,s,D);
    else sprintf(id,"%s",argv[5]);
    strcat(sxnm,id); strcat(sxnm,"-sx.dat");
    strcat(synm,id); strcat(synm,"-sy.dat");
    strcat(sznm,id); strcat(sznm,"-sz.dat");
    strcat(gsnm,id); strcat(gsnm,"-gs.dat");
    sxfl = fopen(sxnm,"a");
    syfl = fopen(synm,"a");
    szfl = fopen(sznm,"a");
    gsfl = fopen(gsnm,"a");

    // initialize Hilbert subspaces for each level m = 0,...,log(N/n)
    vector<SpinHalf> hsps;
    for(int x = n ; x <= N ; x *= 2) hsps.push_back(SpinHalf(x));
    SpinHalf hs = hsps.back();
 
    // generate product basis over m=0 Hilbert space
    auto p = int(pow(2,n));
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

    // initialize block H, and boundary terms used in reduction step
    AutoMPO autoH(hs);
    std::stringstream sts;
    auto out = std::cout.rdbuf(sts.rdbuf());
    vector<vector<MPO> > Hs(hsps.size());
    for(int i = 1 ; i <= N ; ++i) {
        if(i != N) autoH += -J*4.0,"Sz",i,"Sz",i+1;
        autoH += -h*2.0,"Sz",i;
        autoH += -g*2.0,"Sx",i;
        }
    auto H = toMPO<ITensor>(autoH,{"Exact",true});
    std::cout.rdbuf(out);

    for(auto i : args(hsps)) extractBlocks(autoH,Hs[i],hsps[i]);

    // use DMRG to find guess at gs energy, gap
    vector<MPS> dvecs;
    for(int i = 0 ; i < e ; ++i) dvecs.push_back(MPS(hs));    
    time(&t1);
    auto dvals = dmrgMPO(H,dvecs,6,{"Penalty",10.0,"Cutoff",epx});
    time(&t2);
    auto gsD = dvecs[0];
    fprintf(stderr,"DMRG BD ");
    for(const auto& it : dvecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));
    fprintf(stderr,"gs: %17.14f gap: %10.9e\n",dvals[0],dvals[1]-dvals[0]);
    pvec(dvals,e);

    // approximate the thermal operator exp(-H/t)^k using Trotter
    // and MPO multiplication; temperature of K is t/k
    time(&tI);
    MPO eH(hs);
    twoLocalTrotter(eH,t,M,autoH);
    auto K = eH;
    for(int i = 1 ; i < k ; ++i) {
        nmultMPO(eH,K,K,{"Cutoff",eps,"Maxm",MAXBD});
        K.Aref(1) *= 1.0/norm(K.A(1));
        }

    // INITIALIZATION: reduce dimension by sampling from initial basis, either
    // bSpaceL or bSpaceR depending on how the merge will work
    vector<MPS> Spre;
    for(ll = 0 ; ll < N/n ; ll++) {
        int xs = ll % 2 ? 1 : n; // site housing the dangling Select index
        auto cur = ll % 2 ? bSpaceR : bSpaceL;
        Index si("ext",s,Select);
       
        // return orthonormal basis of eigenstates
        auto eigs = diagHermitian(-overlapT(cur,Hs[0][ll],cur),P,S,{"Maxm",s});
        cur.Aref(xs) *= P*delta(commonIndex(P,S),si);
        regauge(cur,xs,{"Truncate",false});

        Spre.push_back(cur);
        }
    time(&t2);
    fprintf(stderr,"initialization: %.f s\n",difftime(t2,tI));

    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPS> Spost;
    for(m = 0 ; (int)Spre.size() > 1 ; ++m,w*=2) {
        fprintf(stderr,"Level %d (w = %d)\n",m,w);
        auto hs = hsps[m];
        auto thr = 1e-9;
        if(doV) rhoGA = MPO(hs);
        Spost.clear();
        
        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        for(ll = 0 ; ll < N/w ; ++ll) {
            MPO A(hs) , Hc = Hs[m][ll];
            MPS pre = Spre[ll] , ret(hs);
            auto xs = ll % 2 ? 1 : w;
            auto pi = findtype(pre.A(xs),Select);

            // STEP 1: extract filtering operators A from AGSP K
            time(&t1);
            restrictMPO(K,A,w*ll+1,D,ll%2);
            time(&t2);
            fprintf(stderr,"trunc AGSP: %.f s\n",difftime(t2,t1));
 
            // STEP 2: expand subspace using the mapping A:pre->ret
            time(&t1);
            ret = applyMPO(A,pre,ll%2,{"Cutoff",eps,"Maxm",MAXBD});
            time(&t2);
            fprintf(stderr,"apply AGSP: %.f s\n",difftime(t2,t1));

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H
            diagHermitian(overlapT(ret,ret),U,Dg,{"Cutoff",thr});
            ei = Index("ext",int(commonIndex(Dg,U)),Select);
            Dg.apply(invsqrt);
            ret.Aref(xs) *= dag(U)*Dg*delta(prime(commonIndex(Dg,U)),ei);

            auto eigs = diagHermitian(-overlapT(ret,Hs[m][ll],ret),P,S);
            ret.Aref(xs) *= P*delta(commonIndex(P,S),ei);
            ret.Aref(xs) *= 1.0/sqrt(overlapT(ret,ret).real(ei(1),prime(ei)(1)));
            regauge(ret,xs,{"Cutoff",eps});

            if(doV) {
                reducedDM(gsD,rhoGA,w*ll+1);
                delt = (overlapT(pre,rhoGA,pre)*delta(pi,prime(pi))).real();
                fprintf(stderr,"1-delta (pre): %.10e\n",1.0-delt);
                delt = (overlapT(ret,rhoGA,ret)*delta(ei,prime(ei))).real();
                fprintf(stderr,"1-delta (ret): %.10e\n",1.0-delt);
                }
            
            fprintf(stderr,"max m: %d\n",maxM(ret));
            Spost.push_back(ret);
            
            }

        // MERGE/REDUCE STEP: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(ll = 0 ; ll < N/w ; ll+=2) {
            auto spL = Spost[ll];                // L subspace
            auto spR = Spost[ll+1];              // R subspace
            auto sL = findtype(spL.A(w),Select); // L dangling index
            auto sR = findtype(spR.A(1),Select); // R dangling index
            auto tpH = tensorProdContract(spL,spR,Hs[m+1][ll/2]);
            bool toobig = (int(sL)*int(sR) >= 15000);
            Index si("ext",s,Select);
            Real tol = 1e-16;

            // STEP 1: find s lowest eigenpairs of restricted H
            time(&t1);
            fprintf(stderr,"dim H = %d... ",int(sL)*int(sR));
            if(doI || toobig) { // iterative function-based diag, ARPACK (best for large s,D, or N) or ITensor 
                if(toobig && !doI) fprintf(stderr,"H too large, iterative diag\n");
                tensorProdH resH(tpH);

                #ifdef USE_ARPACK                   
                auto nn = int(sL)*int(sR) , nev = min(s,int(sL)*int(sR)-2);
                ARSymStdEig<Real, tensorProdH> tprob;
                tprob.DefineParameters(nn,nev,&resH,&tensorProdH::MultMv,"SA", min(2*nev,nn-1),tol,10000*nev);
                auto nconv = tprob.FindEigenvectors();
                fprintf(stderr,"nconv = %d (tol %1.0e)\n",nconv,tol);

                vector<Real> vdat;
                vdat.reserve(nconv*nn);
                auto vraw = tprob.RawEigenvectors();
                vdat.assign(vraw,vraw+nconv*nn);
                if(nconv != s) si = Index("ext",nconv,Select);
                P = ITensor({sL,sR,si},Dense<Real>(std::move(vdat)));
                #else
                vector<ITensor> ret;
                for(int i = 0 ; i < s ; ++i) ret.push_back(randomTensor(sL,sR));
                davidsonT(resH,ret,{"ErrGoal",tol,"MaxIter",10000*s,"DebugLevel",-1});
                fprintf(stderr,"done\n");
                P = ITensor(sL,sR,si);
                for(auto i : args(ret)) P += ret[i]*setElt(si(i+1));
                #endif
            } else { // full matrix diag routine, limited to small parameters (s,D)
                diagHermitian(-tpH.L*tpH.R,P,Dg,{"Maxm",s});
                fprintf(stderr,"done\n");
                P *= delta(commonIndex(P,Dg),si);
                }
            // the following line can replace the entire above if/else block to randomly sample
            // P = randomTensor(sL,sR,si);
            time(&t2);
            fprintf(stderr,"diag restricted H: %.f s\n",difftime(t2,t1));

            // STEP 2: tensor viable sets on each side and reduce dimension
            MPS ret(hsps[m+1]);
            time(&t1);
            tensorProduct(spL,spR,ret,P,(ll/2)%2);
            time(&t2);
            fprintf(stderr,"tensor product (ll=%d): %.f s\n",ll,difftime(t2,t1));
            
            // orthogonalize viable set for next iteration, if ON basis needed 
            if(N/w > 2 && doV) {    
                int xs = (ll/2) % 2 ? 1 : 2*w;
                diagHermitian(overlapT(ret,ret),U,Dg);
                ei = Index("ext",int(commonIndex(Dg,U)),Select);
                Dg.apply(invsqrt);
                ret.Aref(xs) *= U*Dg*delta(ei,prime(commonIndex(Dg,U)));
                regauge(ret,xs,{"Cutoff",1e-16});
                }
 
            Spre.push_back(ret);
            }
        }

    // EXIT: extract two lowest energy candidate states to determine gap
    auto res = Spre[0];
    auto fi = findtype(res.A(N),Select);
    
    vector<MPS> evecs(s);
    for(int i : range1(s)) {
        auto fc = res;
        fc.Aref(N) *= setElt(fi(i));
        fc.orthogonalize({"Cutoff",epx,"Maxm",MAXBD});
        fc.normalize();
        if(i == 1)
            fprintf(stderr,"Overlap with DMRG gs: %e\n",fabs(overlap(fc,gsD)));
        evecs[i-1] = fc;
        }
    time(&t2);
   
    fprintf(stderr,"gs candidate energy: %17.14f\nRRG BD ",overlap(evecs[0],H,evecs[0]));
    for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,tI));

    // CLEANUP: use DMRG to improve discovered eigenstates
    vector<Real> evals(s);
    time(&t1);
    evals = dmrgMPO(H,evecs,2,{"Penalty",10.0,"Cutoff",epx});
    time(&t2);
     
    fprintf(stderr,"DMRG BD ");
    for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));     
    time(&tF);

    pvec(evals,s);
    auto gsR = evecs[0];
    fprintf(stderr,"gs: %17.14f gap: %15.9e\n",evals[0],evals[1]-evals[0]);
    fprintf(gsfl,"# GS data (L=%d s=%d D=%d (J,h,g)=(%.2f,%.2f,%.2f) time=%.f)\n",N,s,D,J,h,g,difftime(tF,tI));
    fprintf(gsfl,"%17.14f\t%15.9e\n",evals[0],evals[1]-evals[0]);

    // Compute two-point correlation functions in ground state via usual MPS method
    fprintf(sxfl,"# SxSx corr matrix (L=%d s=%d D=%d J=%.2f h=%.2f g=%.2f)\n",N,s,D,J,h,g);
    fprintf(syfl,"# SySy corr matrix (L=%d s=%d D=%d J=%.2f h=%.2f g=%.2f)\n",N,s,D,J,h,g);
    fprintf(szfl,"# SzSz corr matrix (L=%d s=%d D=%d J=%.2f h=%.2f g=%.2f)\n",N,s,D,J,h,g);
    for(int i = 1 ; i <= N ; ++i) {
        gsR.position(i,{"Cutoff",0.0});
        auto SxA = hs.op("Sx",i); auto SyA = hs.op("Sy",i); auto SzA = hs.op("Sz",i);
        for(int j = 1 ; j <= N ; ++j) {
            if(j <= i) {
                fprintf(sxfl,"%15.12f\t",0.0);
                fprintf(syfl,"%15.12f\t",0.0);
                fprintf(szfl,"%15.12f\t",0.0); 
            } else {
                auto SxB = hs.op("Sx",j); auto SyB = hs.op("Sy",j); auto SzB = hs.op("Sz",j);
                fprintf(sxfl,"%15.12f\t",measOp(gsR,SxA,i,SxB,j));
                fprintf(syfl,"%15.12f\t",measOp(gsR,SyA,i,SyB,j));
                fprintf(szfl,"%15.12f\t",measOp(gsR,SzA,i,SzB,j));
                }
            }
        fprintf(sxfl,"\n");
        fprintf(syfl,"\n");
        fprintf(szfl,"\n");
        }

    fclose(sxfl);
    fclose(syfl);
    fclose(szfl);
    fclose(gsfl);

    return 0;
    
    }
