#include "rrg.h"
#include <fstream>
#include <random>

int main(int argc, char *argv[]) {
    if(argc != 6 && argc != 7) {
        printf("usage: rrg N n s D seed (id_string)\n");
        return 1;
        }
    time_t t1,t2,tI,tF;
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
    const int    k = 1;              // power of Trotter op (just use 1)
    const int    s = atoi(argv[3]);  // formal s param
    const int    D = atoi(argv[4]);  // formal D param
    
    // computational settings
    const int    seed = atoi(argv[5]);
    const int    doI  = 1; // diag restricted Hamiltonian iteratively?

    // setup random sampling
    fprintf(stderr,"seed is %d\n",seed);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> udist(0.0,1.0);

    FILE *sxfl,*syfl,*szfl,*gsfl;
    char id[128],sxnm[256],synm[256],sznm[256],gsnm[256];
    if(argc == 6) sprintf(id,"rrg-L%d-s%d-D%d",N,s,D);
    else sprintf(id,"%s",argv[6]);
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
    
    // Hamiltonian parameters
    const double Gamma = 2.0;
    vector<double> J(2*(N-1));
    fprintf(stdout,"# Hamiltonian terms Jx1,Jy1,Jx2,... (seed=%d)\n",seed);
    for(int i = 0 ; i < N-1 ; ++i) {
        J[2*i+0] = pow(udist(gen),Gamma);
        J[2*i+1] = pow(udist(gen),Gamma);
        fprintf(stdout,"%16.14f,%16.14f",J[2*i],J[2*i+1]);
        if(i != N-2) fprintf(stdout,",");
        }
    fprintf(stdout,"\n");
    fflush(stdout);

    // initialize block H, and boundary terms used in reduction step
    AutoMPO autoH(hs);
    std::stringstream sts;
    auto out = std::cout.rdbuf(sts.rdbuf());
    vector< vector<MPO> > Hs;
    vector< vector< vector<ITPair> > > bndterms;
    for(const auto& it : hsps) {
        vector<MPO> Hcur;
        vector< vector<ITPair> > bcur;
        int L = it.N();
        for(int o = 0 ; o < N ; o += L) {
            AutoMPO ampo(it);
            for(int i = 1 ; i < L ; ++i) {
                auto nn = o+i-1;
                ampo += (J[2*nn]-J[2*nn+1]),"S+",i,"S+",i+1;
                ampo += (J[2*nn]-J[2*nn+1]),"S-",i,"S-",i+1;
                ampo += (J[2*nn]+J[2*nn+1]),"S+",i,"S-",i+1;
                ampo += (J[2*nn]+J[2*nn+1]),"S-",i,"S+",i+1;
                }
            auto tmp = toMPO<ITensor>(ampo,{"Exact",true});
            Hcur.push_back(tmp);

            if(L == N) autoH = ampo;

            if(o+L <= N) {
                auto nL = o+L-1;
                vector<ITPair> cur(4);
                cur[0] = ITPair((J[2*nL]-J[2*nL+1])*it.op("S+",L),it.op("S+",1));
                cur[1] = ITPair((J[2*nL]-J[2*nL+1])*it.op("S-",L),it.op("S-",1));
                cur[2] = ITPair((J[2*nL]+J[2*nL+1])*it.op("S+",L),it.op("S-",1));
                cur[3] = ITPair((J[2*nL]+J[2*nL+1])*it.op("S-",L),it.op("S+",1));
                bcur.push_back(cur);
                }
            }
        Hs.push_back(Hcur);
        bndterms.push_back(bcur);
        }
    auto H = toMPO<ITensor>(autoH,{"Exact",true});
    std::cout.rdbuf(out);

    vector<MPO> prodSz,prodSx,projSzUp,projSzDn,projSxUp,projSxDn;
    for(auto& it : hsps) { 
        auto curSz = sysOp(it,"Sz",2.0); prodSz.push_back(curSz);
        auto curSx = sysOp(it,"Sx",2.0); prodSx.push_back(curSx);
        auto curSzUp = sysOp(it,"Id"); curSzUp.plusEq(curSz); curSzUp /= 2.0;
        auto curSzDn = sysOp(it,"Id"); curSzDn.plusEq(-1.0*curSz); curSzDn /= 2.0;
        auto curSxUp = sysOp(it,"Id"); curSxUp.plusEq(curSx); curSxUp /= 2.0;
        auto curSxDn = sysOp(it,"Id"); curSxDn.plusEq(-1.0*curSx); curSxDn /= 2.0;
        projSzUp.push_back(curSzUp); projSzDn.push_back(curSzDn);
        projSxUp.push_back(curSxUp); projSxDn.push_back(curSxDn);
        }   
 
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
        auto xs = ll % 2 ? 1 : n; // location of dangling Select index
        auto cur = ll % 2 ? bSpaceR : bSpaceL;
        Index si("ext",s,Select);
       
        // return orthonormal basis of evecs
        auto eigs = diagHermitian(-overlapT(cur,Hs[0][ll],cur),P,S,{"Maxm",s});
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
        auto DD = max(4,D/(int(log2(N/n)-m)));
        auto thr = 1e-8;
        Spost.clear();

        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        for(ll = 0 ; ll < N/w ; ++ll) {
            MPO A(hs) , Hc = Hs[m][ll];
            MPS pre = Spre[ll] , ret(hs);
            int xs = ll % 2 ? 1 : w;

            // STEP 1: extract filtering operators A from AGSP K
            time(&t1);
            restrictMPO(K,A,w*ll+1,DD,ll%2);
            time(&t2);
            fprintf(stderr,"trunc AGSP: %.f s\n",difftime(t2,t1));
           
            // STEP 2: expand subspace using the mapping A:pre->ret
            time(&t1);
            applyMPO(pre,A,ret,ll%2,{"Cutoff",eps,"Maxm",MAXBD});
            time(&t2);
            fprintf(stderr,"apply AGSP: %.f s\n",difftime(t2,t1));

            // rotate into principal components of subspace, poxsibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H,
            // which is necexsary for the iterative solver in the Merge step
            time(&t1); 
            diagHermitian(overlapT(ret,ret),U,Dg,{"Cutoff",thr});
            time(&t2);
            ei = Index("ext",int(commonIndex(Dg,U)),Select);
            Dg.apply(invsqrt);
            ret.Aref(xs) *= dag(U)*Dg*delta(prime(commonIndex(Dg,U)),ei);
            fprintf(stderr,"rotate MPS: %.f s\n",difftime(t2,t1));

            auto eigs = diagHermitian(-overlapT(ret,Hs[m][ll],ret),P,S);
            ret.Aref(xs) *= P;
            ret.Aref(xs) *= delta(commonIndex(P,S),ei);
            regauge(ret,xs,{"Cutoff",eps});

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
            Index si("ext",s,Select);
            bool toobig = (int(sL)*int(sR) >= 15000);

            // STEP 1: find s lowest eigenpairs of restricted H
            time(&t1);
            fprintf(stderr,"dim H = %d... ",int(sL)*int(sR));
            if(doI || toobig) { // iterative diag: ARPACK++ (best for large problems) or ITensor
                if(toobig && !doI) fprintf(stderr,"H too large, iterative diag\n");
                auto C = combiner(sL,sR);
                auto ci = findtype(C,Link);
                vector<ITPair> Hp;
                Real tol = 1e-16;

                Hp.push_back(ITPair(overlapT(spL,Hs[m][ll],spL),delta(sR,prime(sR))));
                Hp.push_back(ITPair(delta(sL,prime(sL)),overlapT(spR,Hs[m][ll+1],spR)));
                for(auto& bb : bndterms[m][ll])
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
                P = ITensor(si,ci);
                combineVectors(ret,P);
                #endif

                P *= C;
            } else { // full matrix diag routine, limited to small parameters (s,D)
                auto HL = overlapT(spL,Hs[m][ll],spL);
                auto HR = overlapT(spR,Hs[m][ll+1],spR);
                auto HH = HL*delta(sR,prime(sR)) + HR*delta(sL,prime(sL));
                for(auto& bb : bndterms[m][ll])
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
 
            Spre.push_back(ret);
            }
        }

    // EXIT: extract two lowest energy candidate states to determine gap
    auto res = Spre[0];
    auto fi = Index("ext",s/2,Select);
    vector<MPS> resSz = {res,res};
    
    // project to Sz sectors of the eigenspace
    diagHermitian(overlapT(res,prodSz[m],res),U,Dg);
    resSz[0].Aref(N) *= U*delta(commonIndex(U,Dg),fi);
    diagHermitian(overlapT(res,-1.0*prodSz[m],res),U,Dg);
    resSz[1].Aref(N) *= U*delta(commonIndex(U,Dg),fi);
   
    vector<MPS> evecs(2);
    for(int i = 0 ; i < 2 ; ++i) {
        auto fc = resSz[i];
        
        // diagonalize H within the Sz sectors
        auto eigs = diagHermitian(-overlapT(fc,H,fc),P,S);
        fc.Aref(N) *= (P*setElt(commonIndex(P,S)(1)));
        
        fc.orthogonalize({"Cutoff",epx,"Maxm",MAXBD});
        fc.normalize();
        if(i == 0)
            fprintf(stderr,"RRG gs energy: %17.14f\n",overlap(fc,H,fc));
        evecs[i] = fc;
        }
    time(&t2);

    Real vz,vx;
    vz = overlap(evecs[0],prodSz[m],evecs[0]); vx = overlap(evecs[0],prodSx[m],evecs[0]);
    fprintf(stderr,"Vz,vx of 0 is: %17.14f,%17.14f\n",vz,vx);
    vz = overlap(evecs[1],prodSz[m],evecs[1]); vx = overlap(evecs[1],prodSx[m],evecs[1]);
    fprintf(stderr,"Vz,vx of 1 is: %17.14f,%17.14f\n",vz,vx);
    int x1_up = (vx > 0.0 ? 1 : 0);

    evecs[0] = exactApplyMPO(evecs[0],projSzUp[m],{"Cutoff",epx});
    evecs[0] = exactApplyMPO(evecs[0],projSxUp[m],{"Cutoff",epx});
    evecs[1] = exactApplyMPO(evecs[1],projSzDn[m],{"Cutoff",epx});
    evecs[1] = exactApplyMPO(evecs[1],(x1_up?projSxUp[m]:projSxDn[m]),{"Cutoff",epx});
    for(auto& it : evecs) it.normalize();

    fprintf(stderr,"gs candidate energy: %17.14f\nRRG BD ",overlap(evecs[0],H,evecs[0]));
    for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,tI));

    // CLEANUP: use DMRG to improve discovered evecs
    vector<Real> evals(2),e_prev(2);
    int max_iter = 20 , used_max = 0;
    Real flr = 1e-13 , over_conv = 1e-1 , gap = 1.0 , conv = over_conv*gap , max_conv = 1.0;
    for(int i = 0 ; i < (int)evecs.size() ; ++i) evals[i] = overlap(evecs[i],H,evecs[i]);
    for(int i = 0 ; (i < 2 || conv < max_conv) && i < max_iter ; ++i) {
        e_prev = evals;

        time(&t1);
        evals = dmrgMPO(H,evecs,8,0.1,epx);
        time(&t2);
        
        gap = evals[1]-evals[0];

        max_conv = 0.0;
        for(auto& j : range(2))
            if(fabs(e_prev[j]-evals[j]) > max_conv) max_conv = e_prev[j]-evals[j];
        
        fprintf(stderr,"DMRG BD ");
        for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
        fprintf(stderr,"\tgap: %e\tconv=%9.2e,%9.2e\telapsed: %.f s\n",gap,
            e_prev[0]-evals[0],e_prev[1]-evals[1],difftime(t2,t1));
        conv = max(over_conv*gap,flr);
        if(i == max_iter) used_max = 1;
        }

    for(int i = 0 ; i < (int)evecs.size() ; ++i) {
        vz = overlap(evecs[i],prodSz[m],evecs[i]); vx = overlap(evecs[i],prodSx[m],evecs[i]);
        fprintf(stderr,"Vz,vx of %d is: %12.9f,%12.9f\n",i,vz,vx);
        }

    evecs[0] = exactApplyMPO(evecs[0],projSzUp[m],{"Cutoff",1e-16});
    evecs[0] = exactApplyMPO(evecs[0],projSxUp[m],{"Cutoff",1e-16});
    evecs[1] = exactApplyMPO(evecs[1],projSzDn[m],{"Cutoff",1e-16});
    evecs[1] = exactApplyMPO(evecs[1],(x1_up?projSxUp[m]:projSxDn[m]),{"Cutoff",1e-16});
    for(auto& it : evecs) it.normalize();
    for(auto i : range(evecs.size())) evals[i] = overlap(evecs[i],H,evecs[i]);
    time(&tF);

    auto gsR = evecs[0];
    auto ee = measEE(gsR,N/2);
    gap = evals[1]-evals[0];

    fprintf(stderr,"gs: %17.14f gap: %15.9e ee: %10.8f\n",evals[0],gap,ee);
    fprintf(gsfl,"# GS data (L=%d s=%d D=%d seed=%d time=%.f)\n",N,s,D,seed,difftime(tF,tI));
    if(used_max) fprintf(gsfl,"# WARNING max iterations reached\n");
    fprintf(gsfl,"%17.14f\t%15.9e\t%10.8f\n",evals[0],gap,ee);

    // Compute two-point correlation functions in ground state via usual MPS method
    fprintf(sxfl,"# SxSx corr matrix (L=%d s=%d D=%d seed=%d)\n",N,s,D,seed);
    fprintf(syfl,"# SySy corr matrix (L=%d s=%d D=%d seed=%d)\n",N,s,D,seed);
    fprintf(szfl,"# SzSz corr matrix (L=%d s=%d D=%d seed=%d)\n",N,s,D,seed);
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
