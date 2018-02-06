#include "rrg.h"
#include <fstream>
#include <random>

int main(int argc, char *argv[]) {
    if(argc != 6 && argc != 7) {
        printf("usage: rrg N n s D Jz (id_string)\n");
        return 1;
        }
    time_t t1,t2,tI,tF;
    ITensor U,Dg,G;
    Index ei;

    // RRG structure parameters
    const int    N  = atoi(argv[1]); // should be n*(power of 2)
    const int    n  = atoi(argv[2]); // initial blocking size
    int          w  = n;             // block size (scales with m)
    int          ll = 0;             // lambda block index
    int          m  = 0;             // RG scale factor

    // AGSP and subspace parameters
    const double t = 0.2;            // Trotter temperature
    const int    M = 140;            // num Trotter steps
    const int    k = 1;              // power of Trotter op
    const int    s = atoi(argv[3]);  // formal s param
    const int    D = atoi(argv[4]);  // formal D param
    
    // computational settings
    //const int    e   = 2; // number of DMRG states to compute (should be 2)
    const int    doI = 1; // diag restricted Hamiltonian iteratively?

    // setup random sampling
    std::random_device r;
    int seed = r();
    fprintf(stderr,"seed is %d\n",seed);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> udist(0.0,1.0);

    FILE *sxfl,*syfl,*szfl,*gsfl;
    char id[128],sxnm[256],synm[256],sznm[256],gsnm[256];
    if(argc == 6) sprintf(id,"rrg-L%d-s%d-D%d-z%.2f",N,s,D,atof(argv[5]));
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
    const double Jz_t  = atof(argv[5]);
    const double Gamma = 2.0;
    vector<double> J(3*(N-1));
    fprintf(stdout,"# Hamiltonian terms Jx1,Jy1,Jz1,Jx2,... (Jz=%.2f seed=%d)\n",Jz_t,seed);
    for(int i = 0 ; i < N-1 ; ++i) {
        J[3*i+0] = pow(udist(gen),Gamma);
        J[3*i+1] = pow(udist(gen),Gamma);
        J[3*i+2] = Jz_t*pow(udist(gen),Gamma);
        fprintf(stdout,"%16.14f,%16.14f,%16.14f",J[3*i],J[3*i+1],J[3*i+2]);
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
                ampo += (J[3*nn]-J[3*nn+1]),"S+",i,"S+",i+1;
                ampo += (J[3*nn]-J[3*nn+1]),"S-",i,"S-",i+1;
                ampo += (J[3*nn]+J[3*nn+1]),"S+",i,"S-",i+1;
                ampo += (J[3*nn]+J[3*nn+1]),"S-",i,"S+",i+1;
                ampo += 4.0*J[3*nn+2],"Sz",i,"Sz",i+1;
                }
            auto tmp = toMPO<ITensor>(ampo,{"Exact",true});
            Hcur.push_back(tmp);

            if(L == N) autoH = ampo;

            if(o+L <= N) {
                auto nL = o+L-1;
                vector<ITPair> cur(5);
                cur[0] = ITPair((J[3*nL]-J[3*nL+1])*it.op("S+",L),it.op("S+",1));
                cur[1] = ITPair((J[3*nL]-J[3*nL+1])*it.op("S-",L),it.op("S-",1));
                cur[2] = ITPair((J[3*nL]+J[3*nL+1])*it.op("S+",L),it.op("S-",1));
                cur[3] = ITPair((J[3*nL]+J[3*nL+1])*it.op("S-",L),it.op("S+",1));
                cur[4] = ITPair(4.0*J[3*nL+2]*it.op("Sz",L),it.op("Sz",1));
                bcur.push_back(cur);
                }
            }
        Hs.push_back(Hcur);
        bndterms.push_back(bcur);
        }
    auto H = toMPO<ITensor>(autoH,{"Exact",true});
    std::cout.rdbuf(out);
/*
    // use DMRG to make a guess at gs energy, gap
    // WARNING: can be very slow!
    vector<MPS> evecs;
    for(int i = 0 ; i < e ; ++i) evecs.push_back(MPS(hs));    
    time(&t1);
    auto evals = dmrgMPO(H,evecs,20,0.1,epx);
    time(&t2);
    if(evals[0] > evals[1]) {
        std::swap(evals[0],evals[1]);
        std::swap(evecs[0],evecs[1]);
        }
    auto gs = evecs[0];
    fprintf(stderr,"DMRG BD ");
    for(const auto& it : evecs) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));
    fprintf(stderr,"gs: %17.14f gap: %10.9e\n",evals[0],evals[1]-evals[0]);
*/
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
        int ss = (ll%2 ? 1 : n); // location of dangling Select index
        auto curSS = (ll%2 ? bSpaceR : bSpaceL);
        ITensor P,S;
        Index si("ext",s,Select);
       
        // return orthonormal basis of eigenstates
        auto Hproj = overlapT(curSS,Hs[0][ll],curSS);
        auto eigs = diagHermitian(-Hproj,P,S,{"Maxm",s});
        curSS.Aref(ss) *= P*delta(commonIndex(P,S),si);
        regauge(curSS,ss,0.0);

        Spre.push_back(curSS);
        }
    time(&t2);
    fprintf(stderr,"initialization: %.f s\n",difftime(t2,tI));

    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPS> Spost;
    for(m = 0 ; (int)Spre.size() > 1 ; ++m,w*=2) {
        fprintf(stderr,"Level %d (w = %d)\n",m,w);
        auto hs = hsps[m];
        Spost.clear();
        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        for(ll = 0 ; ll < N/w ; ++ll) {
            MPO A(hs);
            MPO Hc = Hs[m][ll];
            MPS pre = Spre[ll];
            MPS ret(hs);
            int ss = (ll%2 ? 1 : w);

            // STEP 1: extract filtering operators A from AGSP K
            time(&t1);
            restrictMPO(K,A,w*ll+1,D,ll%2);
            time(&t2);
            fprintf(stderr,"trunc AGSP: %.f s\n",difftime(t2,t1));
           
            pre.position(ll%2?w:1);
            A.position(ll%2?w:1);

            // STEP 2: expand subspace using the mapping A:pre->ret
            time(&t1);
            applyMPO(pre,A,ret,ll%2,{"Cutoff",eps,"Maxm",MAXBD});
            time(&t2);
            fprintf(stderr,"apply AGSP: %.f s\n",difftime(t2,t1));

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H,
            // which is necessary for the iterative solver in the Merge step
            time(&t1);
            G = overlapT(ret,ret);
            ITensor P,S;
            diagHermitian(G,U,Dg,{"Cutoff",eps});
            ei = Index("ext",int(commonIndex(Dg,U)),Select);
            Dg.apply(invsqrt);
            ret.Aref(ss) *= dag(U)*Dg*delta(prime(commonIndex(Dg,U)),ei);
            regauge(ret,ss,eps);
         
            auto Hproj = overlapT(ret,Hs[m][ll],ret);
            auto eigs = diagHermitian(-Hproj,P,S);
            ret.Aref(ss) *= P;
            ret.Aref(ss) *= delta(commonIndex(P,S),ei);
            regauge(ret,ss,eps);
            time(&t2);
            fprintf(stderr,"rotate MPS: %.f s\n",difftime(t2,t1));
            
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
            ITensor P,P2;
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
                Real tol = (N/w > 2 ? 1e-7 : 1e-8);

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
                    tprob.DefineParameters(nn,s,&resH,&tensorProdH::MultMv,"SA", min(2*s,nn-1),tol,1000*s);
                    nconv = tprob.FindEigenvectors();
                    fprintf(stderr,"nconv = %d (tol %1.0e)\n",nconv,tol);
                    }
                auto eigs2 = tprob.RawEigenvalues();
                pvec(eigs2,s);

                vector<Real> vdat;
                vdat.reserve(s*nn);
                auto vraw = tprob.RawEigenvectors();
                vdat.assign(vraw,vraw+s*nn);
                P = ITensor({ci,si},Dense<Real>(std::move(vdat)));
                #else
                vector<ITensor> ret;
                ret.reserve(s);
                for(int i = 0 ; i < s ; ++i) ret.push_back(setElt(ci(i+1)));
                auto eigs = davidsonT(resH,ret,{"ErrGoal",tol,"MaxIter",100*s,"DebugLevel",-1});
                pvec(eigs,s);
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
                auto eigs = diagHermitian(-HH,P,Dg,{"Maxm",s});
                pvec(eigs.eigs().data(),s);
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
    auto fi = findtype(res.A(N),Select);
    vector<MPS> eigenstates(2);
    for(int i = 1 ; i <= 2 ; ++i) {
        auto fc = res;
        fc.Aref(N) *= setElt(fi(i));
        fc.orthogonalize({"Cutoff",epx,"Maxm",MAXBD});
        fc.normalize();
        if(i == 1)
            fprintf(stderr,"RRG gs energy: %17.14f\n",overlap(fc,H,fc));
        eigenstates[i-1] = fc;
        }
    time(&tF);

    fprintf(stderr,"RRG BD ");
    for(const auto& it : eigenstates) fprintf(stderr,"%d ",maxM(it));
    fprintf(stderr,"\telapsed: %.f s\n",difftime(tF,tI));

    // CLEANUP: use DMRG to improve discovered eigenstates
    vector<Real> eigenvalues(2),e_prev(2);
    int max_iter = 10;
    Real over_conv = 1e-1;
    Real gap = 1.0;
    for(int i = 0 ; i < 2 || (over_conv*gap < e_prev[0]-eigenvalues[0] 
                && over_conv*gap < e_prev[1]-eigenvalues[1] && i < max_iter) ; ++i) {
        e_prev = eigenvalues;
        
        time(&t1);
        eigenvalues = dmrgMPO(H,eigenstates,10,0.1,1e-16);
        time(&t2);
         
        fprintf(stderr,"DMRG BD ");
        for(const auto& it : eigenstates) fprintf(stderr,"%d ",maxM(it));
        fprintf(stderr,"\telapsed: %.f s\n",difftime(t2,t1));     
        if(eigenvalues[0] > eigenvalues[1]) {
            std::swap(eigenvalues[0],eigenvalues[1]);
            std::swap(eigenstates[0],eigenstates[1]);
            }
        gap = eigenvalues[1]-eigenvalues[0];
        }

    auto gsR = eigenstates[0];
    fprintf(stderr,"gs: %17.14f gap: %10.9e\n",eigenvalues[0],eigenvalues[1]-eigenvalues[0]);
    fprintf(gsfl,"# GS data (L=%d s=%d D=%d Jz=%.2f seed=%d)\n",N,s,D,Jz_t,seed);
    if(gap < e_prev[1]-eigenvalues[1]) fprintf(gsfl,"#WARNING gap not converged!\n");
    fprintf(gsfl,"%17.14f\t%10.9e\n",eigenvalues[0],eigenvalues[1]-eigenvalues[0]);

    // Compute two-point correlation functions in ground state via usual MPS method
    fprintf(sxfl,"# SxSx corr matrix (L=%d s=%d D=%d Jz=%.2f seed=%d)\n",N,s,D,Jz_t,seed);
    fprintf(syfl,"# SySy corr matrix (L=%d s=%d D=%d Jz=%.2f seed=%d)\n",N,s,D,Jz_t,seed);
    fprintf(szfl,"# SzSz corr matrix (L=%d s=%d D=%d Jz=%.2f seed=%d)\n",N,s,D,Jz_t,seed);
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
