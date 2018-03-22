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
    const int    M = 140;            // num Trotter steps
    const int    k = 1;              // power of Trotter op
    const int    s = atoi(argv[3]);  // formal s param
    const int    D = atoi(argv[4]);  // formal D param
    
    // computational settings
    const int    e   = 2; // number of DMRG states to compute (should be 2)
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
    vector<MPO> Hs;
    vector< vector<ITPair> > bndterms;
    for(const auto& it : hsps) {
        int L = it.N();
        AutoMPO ampo(it);
        for(int i = 1 ; i <= L ; ++i) {
            if(i != L) ampo += -J*4.0,"Sz",i,"Sz",i+1;
            ampo += -h*2.0,"Sz",i;
            ampo += -g*2.0,"Sx",i;
            }
        Hs.push_back(MPO(ampo));

        if(L == N) autoH = ampo;
        else {
            vector<ITPair> bcur(1);
            bcur[0] = ITPair(-J*4.0*it.op("Sz",L),it.op("Sz",1));
            bndterms.push_back(bcur);
            }
        }
    auto H = toMPO<ITensor>(autoH,{"Exact",true});
    std::cout.rdbuf(out);

    // use DMRG to find guexs at gs energy, gap
    vector<MPS> evecs;
    for(int i = 0 ; i < e ; ++i) evecs.push_back(MPS(hs));    
    time(&t1);
    auto evals = dmrgMPO(H,evecs,9,10.0,1e-16);
    time(&t2);
    auto mn = std::distance(evals.begin(),std::min_element(evals.begin(),evals.end()));
    auto gs = evecs[mn];
    fprintf(stderr,"DMRG BD ");
    for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));
    fprintf(stderr,"gs: %17.14f gap: %10.9e\n",evals[mn],evals[1-mn]-evals[mn]);

    // approximate the thermal operator exp(-H/t)^k using Trotter
    // and MPO multiplication; temperature of K is k/t
    time(&tI);
    MPO eH(hs);
    twoLocalTrotter(eH,t,M,autoH,eps);
    auto K = eH;
    for(int i = 1 ; i < k ; ++i) {
        nmultMPO(eH,K,K,{"Cutoff",eps,"Maxm",MAXBD});
        fprintf(stderr,"norm K: %e\n",norm(K.A(1)));
        K.Aref(1) *= 1e10/norm(K.A(1));
        } 

    // INITIALIZATION: reduce dimension by sampling from initial basis, either
    // bSpaceL or bSpaceR depending on how the merge will work
    vector<MPS> Spre;
    for(ll = 0 ; ll < N/n ; ll++) {
        int xs = ll % 2 ? 1 : n; // site housing the dangling Select index
        auto cur = ll % 2 ? bSpaceR : bSpaceL;
        Index si("ext",s,Select);
       
        // return orthonormal basis of eigenstates
        auto eigs = diagHermitian(-overlapT(cur,Hs[0],cur),P,S,{"Maxm",s});
        cur.Aref(xs) *= P*delta(commonIndex(P,S),si);
        regauge(cur,xs,{"Cutoff",1e-16});

        Spre.push_back(cur);
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
        
        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        for(ll = 0 ; ll < N/w ; ++ll) {
            MPO A(hs),Hc = Hs[m];
            MPS pre = Spre[ll],ret(hs);
            auto xs = ll % 2 ? 1 : w;
            auto pi = findtype(pre.A(xs),Select);
            Real thr = 1e-8;

            // STEP 1: extract filtering operators A from AGSP K
            time(&t1);
            restrictMPO(K,A,w*ll+1,D,ll%2);
            time(&t2);
            fprintf(stderr,"trunc AGSP: %.f s\n",difftime(t2,t1));
           
            // STEP 2: expand subspace using the mapping A:pre->ret
            time(&t1);
            applyMPO(pre,A,ret,ll%2,{"Cutoff",eps,"Maxm",MAXBD});
            time(&t2);
            fprintf(stderr,"apply AGSP: %.f s\n",difftime(t2,t1));
            fprintf(stderr,"max m: %d\n",maxM(ret));

            // rotate into principal components of subspace, poxsibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H,
            // which is necexsary for the iterative solver in the Merge step
            diagHermitian(overlapT(ret,ret),U,Dg,{"Cutoff",thr});
            ei = Index("ext",int(commonIndex(Dg,U)),Select);
            Dg.apply(invsqrt);
            ret.Aref(xs) *= dag(U)*Dg*delta(prime(commonIndex(Dg,U)),ei);
         
            auto eigs = diagHermitian(-overlapT(ret,Hs[m],ret),P,S);
            ret.Aref(xs) *= P*delta(commonIndex(P,S),ei);
            regauge(ret,xs,{"Cutoff",eps});
            
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

        // MERGE/REDUCE STEP: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(ll = 0 ; ll < N/w ; ll+=2) {
            auto spL = Spost[ll];                // L subspace
            auto spR = Spost[ll+1];              // R subspace
            auto sL = findtype(spL.A(w),Select); // L dangling index
            auto sR = findtype(spR.A(1),Select); // R dangling index
            Index si("ext",s,Select);
            bool toobig = (int(sL)*int(sR) >= 15000);

            // STEP 1: find s lowest eigenpairs of restricted H
            time(&t1);
            fprintf(stderr,"dim H = %d... ",int(sL)*int(sR));
            if(doI || toobig) { // iterative function-based diag, ARPACK (best for large s,D, or N) or ITensor 
                if(toobig && !doI) fprintf(stderr,"H too large, iterative diag\n");
                auto C = combiner(sL,sR);
                auto ci = findtype(C,Link);
                vector<ITPair> Hp;
                Real tol = 1e-16;

                Hp.push_back(ITPair(overlapT(spL,Hs[m],spL),delta(sR,prime(sR))));
                Hp.push_back(ITPair(delta(sL,prime(sL)),overlapT(spR,Hs[m],spR)));
                for(auto& bb : bndterms[m])
                    Hp.push_back(ITPair(
                         (prime(spL.A(w))*bb.L)*(spL.A(w)*delta(leftLinkInd(spL,w),prime(leftLinkInd(spL,w))))
                        ,(prime(spR.A(1))*bb.R)*(spR.A(1)*delta(rightLinkInd(spR,1),prime(rightLinkInd(spR,1))))));
                tensorProdH resH(Hp,C);

                #ifdef USE_ARPACK                   
                auto nn = int(ci);
                ARSymStdEig<Real, tensorProdH> tprob;
                for(int i = 0 , nconv = 0 ; nconv < s ; ++i) {
                    if(i != 0) tol *= 1e1;
                    tprob.DefineParameters(nn,s,&resH,&tensorProdH::MultMv,"SA", min(2*s,nn-1),tol,10000*s);
                    nconv = tprob.FindEigenvectors();
                    fprintf(stderr,"nconv = %d (tol %1.0e)\n",nconv,tol);
                    }

                vector<Real> vdat;
                vdat.reserve(s*nn);
                auto vraw = tprob.RawEigenvectors();
                vdat.assign(vraw,vraw+s*nn);
                P = ITensor({ci,si},Dense<Real>(std::move(vdat)));
                #else
                vector<ITensor> ret;
                ret.reserve(s);
                for(int i = 0 ; i < s ; ++i) ret.push_back(setElt(ci(i+1)));
                davidsonT(resH,ret,{"ErrGoal",tol,"MaxIter",100*s,"DebugLevel",-1});
                fprintf(stderr,"\n");
                P = ITensor(si,ci);
                combineVectors(ret,P);
                #endif

                P *= C;
            } else { // full matrix diag routine, limited to small parameters (s,D)
                auto HL = overlapT(spL,Hs[m],spL);
                auto HR = overlapT(spR,Hs[m],spR);
                auto HH = HL*delta(sR,prime(sR)) + HR*delta(sL,prime(sL));
                for(auto& bb : bndterms[m])
                    HH += ((spL.A(w)*delta(leftLinkInd(spL,w),prime(leftLinkInd(spL,w))))
                        *prime(spL.A(w))*bb.L)*(prime(spR.A(1))*bb.R
                        *(spR.A(1)*delta(rightLinkInd(spR,1),prime(rightLinkInd(spR,1)))));
                diagHermitian(-HH,P,Dg,{"Maxm",s});
                fprintf(stderr,"\n");
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
                diagHermitian(overlapT(ret,ret),U,Dg,{"Cutoff",eps});
                ei = Index("ext",int(commonIndex(Dg,U)),Select);
                Dg.apply(invsqrt);
                ret.Aref(xs) *= U*Dg;
                ret.Aref(xs) *= delta(ei,prime(commonIndex(Dg,U)));
                regauge(ret,xs,{"Cutoff",eps});
                }
 
            Spre.push_back(ret);
            }
        }

    // EXIT: extract two lowest energy candidate states to determine gap
    auto res = Spre[0];
    auto fi = findtype(res.A(N),Select);
    vector<MPS> eigenstates(2);
    for(int i = 1 ; i <= 2 ; ++i) {
        auto fc = res;
        fc.Aref(N) *= setElt(fi(i));
        fc.orthogonalize({"Cutoff",epx,"Maxm",MAXBD});
        fc.normalize();
        if(i == 1)
            fprintf(stderr,"Overlap with DMRG gs: %e\n",fabs(overlap(fc,gs)));
        eigenstates[i-1] = fc;
        }
    time(&tF);
    
    fprintf(stderr,"RRG BD ");
    for(const auto& it : eigenstates) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(tF,tI));

    // CLEANUP: use DMRG to improve discovered eigenstates
    vector<Real> eigenvalues(2);
    time(&t1);
    eigenvalues = dmrgMPO(H,eigenstates,9,10.0,1e-16);
    time(&t2);
     
    fprintf(stderr,"DMRG BD ");
    for(const auto& it : eigenstates) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));     
    mn = std::distance(eigenvalues.begin(),std::min_element(eigenvalues.begin(),eigenvalues.end()));

    auto gsR = eigenstates[mn];
    fprintf(stderr,"gs: %17.14f gap: %10.9e\n",eigenvalues[mn],eigenvalues[1-mn]-eigenvalues[mn]);
    fprintf(gsfl,"# GS data (L=%d s=%d D=%d J=%.2f h=%.2f g=%.2f)\n",N,s,D,J,h,g);
    fprintf(gsfl,"%17.14f\t%10.9e\n",eigenvalues[mn],eigenvalues[1-mn]-eigenvalues[mn]);

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
