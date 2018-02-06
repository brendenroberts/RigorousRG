#include "rrg.h"

int nels(const ITensor& A) {
    int n = 1;
    for(const auto& i : A.inds()) n *= int(i);
    return n;
    }

void reducedDM(const MPS& psi , MPO& rho , int ls) {
    const auto& hs  = psi.sites();
    const auto& sub = rho.sites();
    auto N = psi.N();
    auto n = rho.N();
    auto rs = ls + n - 1;
    auto psip = psi;
    psip.mapprime(0,1);
    ITensor L,R,C;
    Index ai,bi,ci;

    for(int i = 1 ; i < ls ; ++i) {
        auto sp = hs.si(i);
        L = (i == 1 ? psi.A(i) : L*psi.A(i))*psip.A(i)*delta(sp,prime(sp));
        }

    for(int i = ls ; i <= rs ; ++i) {
        int r = i-ls+1;
        auto T = psi.A(i)*psip.A(i);
        if(i == ls && L) T *= L;
        if(i != ls) {
            C = combiner(leftLinkInd(psi,i),leftLinkInd(psip,i));
            T *= C;
            ci = commonIndex(T,C);
            T *= delta(ci,bi);
            }
        if(i != rs) {
            C = combiner(rightLinkInd(psi,i),rightLinkInd(psip,i));
            T *= C;
            bi = commonIndex(T,C);
            }
        rho.Aref(r) = T*delta(hs.si(i),sub.si(r))*delta(hs.siP(i),sub.siP(r));
        }

    for(int i = rs+1 ; i <= N ; ++i) {
        auto sp = hs.si(i);
        R = (i == rs+1 ? psi.A(i) : R*psi.A(i))*psip.A(i)*delta(sp,prime(sp));
        }
    
    if(R) rho.Aref(n) *= R;

    return;
    }

template<class MPSLike>
void regauge(MPSLike& psi , int o, Real thr) {
    psi.orthogonalize({"Cutoff",thr});
    psi.position(o,{"Cutoff",0.0});

    return;
    }
template void regauge(MPS& , int , Real);
template void regauge(MPO& , int , Real);

template<class MPSLike>
void regauge(MPSLike& psi , int o) { regauge(psi,o,eps); }
template void regauge(MPS& , int);
template void regauge(MPO& , int);

Real measEE(const MPS& state , int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});

    ITensor U = psi.A(a),S,V;
    auto spectrum = svd(U*psi.A(a+1),U,S,V);

    Real ret = 0.0;
    for(auto p : spectrum.eigs()) if(p > 1e-18) ret += -p*log(p);
    
    return ret;
    }

Real measOp(const MPS& state, const ITensor& A, int a, const ITensor& B, int b) {
    if(b <= a) Error("measOp requires a < b");
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});
    auto C = psi.A(a)*A*dag(prime(psi.A(a),Site,commonIndex(psi.A(a),psi.A(a+1),Link)));
    
    for(int k = a+1; k < b; ++k) {
        C *= psi.A(k);
        C *= dag(prime(psi.A(k),Link));
        }

    C *= psi.A(b);
    C *= B;
    C *= dag(prime(psi.A(b),Site,commonIndex(psi.A(b),psi.A(b-1),Link)));

    return C.real();
    }

Real measOp(const MPS& state, const ITensor& A, int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});
    auto C = psi.A(a)*A*dag(prime(psi.A(a),Site));
    return C.real();
    }

void combineVectors(const vector<ITensor>& vecs , ITensor& ret) {
    auto chi = commonIndex(ret,vecs[0]);
    auto ext = uniqueIndex(ret,vecs[0],Select);
    if(ext.m() != int(vecs.size())) Error("index/vector mismatch in combineVectors");

    for(int i = 1 ; i <= chi.m() ; ++i)
        for(int j = 1 ; j <= ext.m() ; ++j)
            ret.set(chi(i),ext(j),vecs[j-1].real(chi(i)));

    }

vector<Real> dmrgMPO(const MPO& H , vector<MPS>& states , int num_sw , double penalty, double err) {
    vector<Real> evals;
    vector<MPS> exclude;
    for(auto& psi : states) {
        auto swp = Sweeps(num_sw);
        swp.maxm() = 100,100,500;
        swp.cutoff() = err;
        swp.niter() = 8;
        swp.noise() = 0.0;
 
        // dmrg call won't shut up
        std::stringstream ss;
        auto out = std::cout.rdbuf(ss.rdbuf()); 
        auto e = dmrg(psi,H,exclude,swp,{"Quiet",true,"PrintEigs",false,"Weight",penalty});
        std::cout.rdbuf(out);

        exclude.push_back(psi);
        evals.push_back(e);
        }

    return evals;
    }

ITensor splitMPO(const MPO& O, MPO& P, int lr) {
    auto N = O.N();
    auto n = P.N();
    const auto HS = O.sites();
    const auto hs = P.sites();
    auto M = O;
    int t = (lr ? N-n+1 : n);
    ITensor U,S,V,R;
    Index sp,sq,ai,ei;

    M.position(t,{"Cutoff",0.0});
    int a[2] = {t-lr,t+1-lr};
    auto B = M.A(a[0])*M.A(a[1]);
    U = ITensor(HS.si(a[0]),HS.siP(a[0]),leftLinkInd(M,a[0]));
    svdL(B,U,S,V,{"Cutoff",0.0,"LeftIndexType",Select,"RightIndexType",Select});
    R = (lr ? V : U);
    sp = hs.si((lr ? 1 : n));
    sq = HS.si(t);
    P.Aref((lr ? 1 : n)) = R*delta(sp,sq)*delta(prime(sp),prime(sq));
    
    for(int i = 1 ; i < n ; ++i) {
        auto x = (lr ? i+1 : n-i); sp = hs.si(x);
        auto y = (lr ? t+x-1 : x); sq = HS.si(y);
        P.Aref(x) = M.A(y)*delta(sp,sq)*delta(prime(sp),prime(sq));
        }
    
    return S;
    }

struct LRVal {
    Real val;
    int L,R;

    LRVal(Real v , int l, int r) : val(v),L(l),R(r) {}
};

bool vcomp(LRVal& A , LRVal& B) { return A.val < B.val; }
bool lcomp(LRVal& A , LRVal& B) { return A.L < B.L; }
bool rcomp(LRVal& A , LRVal& B) { return A.R < B.R; }

int argmax(vector<LRVal> vec) { return std::distance(vec.begin(),std::max_element(vec.begin(),vec.end(),vcomp));}
int argmaxL(vector<LRVal> vec) { return std::distance(vec.begin(),std::max_element(vec.begin(),vec.end(),lcomp));}
int argmaxR(vector<LRVal> vec) { return std::distance(vec.begin(),std::max_element(vec.begin(),vec.end(),rcomp));}

void restrictMPO(const MPO& O , MPO& res , int ls , int D, int lr) {
    auto N = O.N();
    auto n = res.N();
    if(N == n) {res = O; return;}
    const auto& sub = res.sites();
    auto M = O;
    ITensor U,V,SB;
    int rs = ls+n-1;
   
    if(ls == 1) { // easy case: only dangling bond already at R end
        auto SS = splitMPO(M,res,LEFT);
        auto ei = Index("ext",min(D,int(findtype(res.A(n),Select))),Select);
        res.Aref(n) *= delta(ei,commonIndex(SS,res.A(n)));
        regauge(res,n,epx);
        return;
    } else if(rs == N) { // easy case: only dangling bond already at L end
        auto SS = splitMPO(M,res,RIGHT);
        auto ei = Index("ext",min(D,int(findtype(res.A(1),Select))),Select);
        res.Aref(1) *= delta(ei,commonIndex(SS,res.A(1)));
        regauge(res,1,epx);
        return;
        }

    // setup for moving external bond to correct end 
    ITensor SL,SR;
    if(lr) {
        SpinHalf htmp(rs);
        MPO tmp(htmp);
        SR = splitMPO(M,tmp,LEFT);
        SL = splitMPO(tmp,res,RIGHT);
    } else {
        SpinHalf htmp(N-ls+1);
        MPO tmp(htmp);
        SL = splitMPO(M,tmp,RIGHT);
        SR = splitMPO(tmp,res,LEFT);
        }
 
    Index li = commonIndex(SL,res.A(1));
    Index ri = commonIndex(SR,res.A(n));

    auto ldat = doTask(getReal{},SL.store());
    cblas_dscal(int(li),SL.scale().real(),ldat.data(),1);
    auto rdat = doTask(getReal{},SR.store());
    cblas_dscal(int(ri),SR.scale().real(),rdat.data(),1);

    vector<LRVal> args,wk;
    args.reserve(D*D);
    wk.reserve(2*D);
   
    for(int i = 0 ; i < std::min(2*D,int(li)) ; ++i)
        wk.push_back(LRVal(ldat[i]*rdat[0],i+1,1));
    
    for(int i = 0 ; i < std::min(D*D,int(li)*int(ri)) ; ++i) {  
        int amax = argmax(wk);
        if(amax == int(wk.size())-1)
            for(int j = 0,pln = wk.size() ; j < D && pln+j < int(li) ; ++j)
                wk.push_back(LRVal(ldat[pln+j]*rdat[0],pln+j+1,1));
        amax = argmax(wk);
        if(wk[amax].val < eps) break;
        args.push_back(wk[amax]);
        wk[amax] = LRVal(ldat[wk[amax].L-1]*rdat[wk[amax].R],wk[amax].L,wk[amax].R+1);
        }

    auto lt = Index("L",args[argmaxL(args)].L,Select);
    auto rt = Index("R",args[argmaxR(args)].R,Select);
    auto ei = Index("ext",args.size(),Select);
    auto UU = ITensor({lt,rt,ei});
    int count = 1;
    for(auto& it : args)
        UU.set(lt(it.L),rt(it.R),ei(count++),1);
    res.Aref(1) *= delta(lt,li);
    res.Aref(n) *= delta(rt,ri);

    ITensor S;
    if(lr)
        for(int i = n-1 ; i >= 1 ; --i) {
            auto B = res.A(i)*res.A(i+1);
            U = ITensor(sub.si(i),sub.siP(i),rt,(i == 1 ? lt : leftLinkInd(res,i)));
            V = ITensor();
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromright,{"Cutoff",eps,"Maxm",MAXBD});
            } else {
                svdL(B,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
                U *= S;
                }
            res.Aref(i) = U;
            res.Aref(i+1) = V;
            }
    else
        for(int i = 2 ; i <= n ; ++i) {
            auto B = res.A(i-1)*res.A(i);
            U = ITensor();
            V = ITensor(sub.si(i),sub.siP(i),lt,(i == n ? rt : rightLinkInd(res,i)));
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromleft,{"Cutoff",eps,"Maxm",MAXBD});
            } else {
                svdL(B,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
                V *= S;
                }
                res.Aref(i-1) = U;
                res.Aref(i) = V;
            }

    res.Aref(lr?1:n) *= UU;
    regauge(res,lr?1:n,eps);
    
    return; 
    }

template<class Tensor>
void applyMPO(MPSt<Tensor> const& psi, MPOt<Tensor> const& K, MPSt<Tensor>& res, int lr, Args const& args) {
    using IndexT = typename Tensor::index_type;
    auto N = psi.N();
    if(K.N() != N) Error("Mismatched N in applyMPO");
    res = psi; 
    res.mapprime(0,1,Site);
    int ss = (lr ? 1 : N);
    auto trunK = args.getBool("TruncateMPO",false);
    vector<Index> ext;
    if(findtype(psi.A(ss),Select)) ext.push_back(findtype(psi.A(ss),Select));
    if(findtype(K.A(ss),Select)) ext.push_back(findtype(K.A(ss),Select));

    Index iA,iK,iT;
    Tensor clust,nfork,S,tA,tK;
    for(int i = 0; i < N-1; i++)
        {
        int x = (lr ? N-i : i+1);
        if(trunK) {
            iT = lr ? rightLinkInd(K,x) : leftLinkInd(K,x);
            tK = i != 0 && iT != iK ? delta(iK,iT)*K.A(x) : K.A(x);
            tA = psi.A(x);
        } else {
            iT = lr ? rightLinkInd(psi,x) : leftLinkInd(psi,x);
            tA = i != 0 && iT != iA ? delta(iA,iT)*psi.A(x) : psi.A(x);
            tK = K.A(x);
            }
        iA = lr ? leftLinkInd(psi,x) : rightLinkInd(psi,x);
        iK = lr ? leftLinkInd(K,x) : rightLinkInd(K,x);
        
        if(i == 0) clust = psi.A(x) * K.A(x);
        else clust = int(iK) > int(iA) ? (nfork * tK) * tA : (nfork * tA) * tK;
        if(i == N-2) break; //No need to SVD for i == N-1

        if(int(iK)*int(iA) > MAXDIM) { // truncate bond for memory stability
            auto newlink = Index("nl",MAXDIM/int(trunK?iA:iK),Link);
            fprintf(stderr,"truncating %s bond %d->%d...\n",trunK?"MPO":"MPS",int(trunK?iK:iA),int(newlink));
            clust *= delta(trunK?iK:iA,newlink);
            (trunK?iK:iA) = newlink;
            }
        nfork = Tensor(iA,iK);
        if(int(iK)*int(iA) < 10000) {
            denmatDecomp(clust,res.Anc(x),nfork,Fromleft,args);
        } else {
            svdL(clust,res.Anc(x),S,nfork,args);
            nfork *= S;
            }
        IndexT mid = commonIndex(res.A(x),nfork,Link);
        mid.dag();
        if(lr)
            res.Anc(x-1) = Tensor(mid,prime(res.sites()(x-1)));
        else
            res.Anc(x+1) = Tensor(mid,prime(res.sites()(x+1)));
        }
    nfork = clust * psi.A(ss) * K.A(ss);

    svdL(nfork,res.Anc(lr?ss+1:ss-1),S,res.Anc(ss),args);
    res.Aref(ss) *= S;
    if(ext.size() > 1) res.Aref(ss) *= combiner(ext,{"IndexType",Select});
    res.mapprime(1,0,Site);
    res.position(ss,{"Cutoff",0.0});
    }
template void applyMPO(const MPS& , const MPO& , MPS& , int , const Args&);

template<class Tensor>
void applyMPO(MPOt<Tensor> const& psi, MPOt<Tensor> const& K, MPOt<Tensor>& res, int lr, Args const& args) {
    using IndexT = typename Tensor::index_type;
    auto N = psi.N();
    if(K.N() != N) Error("Mismatched N in applyMPO");
    res = psi; 
    res.mapprime(1,2,Site);
    int ss = (lr ? 1 : N);
    auto trunK = args.getBool("TruncateMPO",false);
    vector<Index> ext;
    if(findtype(psi.A(ss),Select)) ext.push_back(findtype(psi.A(ss),Select));
    if(findtype(K.A(ss),Select)) ext.push_back(findtype(K.A(ss),Select));

    Index iA,iK,iT;
    Tensor clust,nfork,S,tA,tK;
    for(int i = 0; i < N-1; i++)
        {
        int x = (lr ? N-i : i+1);
        if(trunK) {
            iT = lr ? rightLinkInd(K,x) : leftLinkInd(K,x);
            tK = i != 0 && iT != iK ? delta(iK,iT)*prime(K.A(x),Site) : prime(K.A(x),Site);
            tA = psi.A(x);
        } else {
            iT = lr ? rightLinkInd(psi,x) : leftLinkInd(psi,x);
            tA = i != 0 && iT != iA ? delta(iA,iT)*psi.A(x) : psi.A(x);
            tK = prime(K.A(x),Site);
            }
        iA = lr ? leftLinkInd(psi,x) : rightLinkInd(psi,x);
        iK = lr ? leftLinkInd(K,x) : rightLinkInd(K,x);
        
        if(i == 0) clust = psi.A(x) * prime(K.A(x),Site);
        else clust = int(iK) > int(iA) ? (nfork * tK) * tA : (nfork * tA) * tK;
        if(i == N-2) break; //No need to SVD for i == N-1
        //Print(clust);

        if(int(iK)*int(iA) > MAXDIM) { // truncate bond for memory stability
            auto newlink = Index("nl",MAXDIM/int(trunK?iA:iK),Link);
            fprintf(stderr,"truncating %s bond %d->%d...\n",trunK?"MPO":"MPS",int(trunK?iK:iA),int(newlink));
            clust *= delta(trunK?iK:iA,newlink);
            (trunK?iK:iA) = newlink;
            }
        nfork = Tensor(iA,iK);
        if(int(iK)*int(iA) < 10000) {
            denmatDecomp(clust,res.Anc(x),nfork,Fromleft,args);
        } else {
            svdL(clust,res.Anc(x),S,nfork,args);
            nfork *= S;
            }
        IndexT mid = commonIndex(res.A(x),nfork,Link);
        mid.dag();
        if(lr)
            res.Anc(x-1) = Tensor(mid,res.sites()(x-1),prime(res.sites()(x-1),2));
        else
            res.Anc(x+1) = Tensor(mid,res.sites()(x+1),prime(res.sites()(x+1),2));
        }
    nfork = clust * psi.A(ss) * prime(K.A(ss),Site);

    svdL(nfork,res.Anc(lr?ss+1:ss-1),S,res.Anc(ss),args);
    res.Aref(ss) *= S;
    if(ext.size() > 1) res.Aref(ss) *= combiner(ext,{"IndexType",Select});
    res.mapprime(2,1,Site);
    res.position(ss,{"Cutoff",0.0});
    }
template void applyMPO(const MPO& , const MPO& , MPO& , int , const Args&);

ITensor overlapT(const MPS& phi, const MPS& psi) {
    auto N = phi.N();
    if(psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    ITensor L;

    for(int i = 0 ; i < N ; ++i) {
        int x = (lr ? N-i : i+1);
        L = (i ? L*phi.A(x) : phi.A(x));
        L *= dag(primeExcept(psi.A(x),Site));
        }
    
    return L;
    }

ITensor overlapT(const MPS& phi, const MPO& H, const MPS& psi) {
    auto N = H.N();
    if(phi.N() != N || psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    ITensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = (i ? L*phi.A(x) : phi.A(x));
        L *= H.A(x);
        L *= dag(prime(psi.A(x)));
        }
    
    return L;
    }

void tensorProduct(const MPS& psiA, const MPS& psiB, MPS& ret, const ITensor& W, int lr) {
    const int N = ret.N();
    const int n = psiA.N();
    const auto& hs  = ret.sites();
    const auto& hsA = psiA.sites();
    const auto& hsB = psiB.sites();
    Index ai,ei,sp;
    ITensor T,U,S,V;
    
    for(int i = 1 ; i <= n ; ++i) {
        Index spA  = hsA.si(i); Index spB  = hsB.si(i);
        Index spAr = hs.si(i);  Index spBr = hs.si(n+i);
        ret.Aref(i)   = psiA.A(i)*delta(spA,spAr);
        ret.Aref(n+i) = psiB.A(i)*delta(spB,spBr);
        }

    // Move selection index from middle to edge
    for(int i = 0 ; i < n ; ++i) {
        int x = (lr ? n-i : n+i);
        sp = hs.si(x);
        ai = commonIndex(ret.A(x-1),ret.A(x));
        T = ret.A(x)*(i == 0 ? W*ret.A(x+1) : ret.A(x+1));
        if(i == 0) ei = findtype(T,Select);
        U = (lr ? ITensor(sp,ai,ei) : ITensor(sp,ai));
        svdL(T,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
        ret.Aref(x)   = (lr ? U*S : U);
        ret.Aref(x+1) = (lr ? V : S*V);
        }
    regauge(ret,(lr?1:N),eps);
   
    return; 
    }

void combineMPS(vector<MPS>& vecs , MPS& ret, int lr) {
    const int N = ret.N();
    const int nvc = vecs.size();
    const auto& hs = ret.sites();
    Index ak,bk,vi;
    ITensor U,S,V;
    vector<Index> inds;
    
    if(lr == LEFT) {
        // Do first tensor
        int bm = 0;
        for(auto& v : vecs) bm += rightLinkInd(v,1).m();
        bk = Index("link",bm);
        auto sp = hs.si(1);
        ITensor A(bk,sp);
        int bsum = 0;
        for(auto& v : vecs) {
            auto bi = rightLinkInd(v,1);
            for(int b = 1 ; b <= bi.m() ; ++b) {
                A.set(bk(bsum+b),sp(1),v.A(1).real(sp(1),bi(b)));
                A.set(bk(bsum+b),sp(2),v.A(1).real(sp(2),bi(b)));
                }
            bsum += bi.m();
            }
        U = ITensor(sp);
        svdL(A,U,S,V,{"Cutoff",0.0});
        inds.push_back(bk);
        ret.Aref(1) = U;
        
        // Do middle tensors
        for(int i = 2 ; i < N ; ++i) {
            int bm = 0;
            for(auto& v : vecs) bm += rightLinkInd(v,i).m();
            ak = inds.back();
            bk = Index("link",bm);
            sp = hs.si(i);
            A = ITensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = leftLinkInd(v,i);
                auto bi = rightLinkInd(v,i);
                for(int a = 1 ; a <= ai.m() ; ++a)
                    for(int b = 1 ; b <= bi.m() ; ++b) {
                        A.set(ak(asum+a),bk(bsum+b),sp(1),v.A(i).real(sp(1),ai(a),bi(b)));
                        A.set(ak(asum+a),bk(bsum+b),sp(2),v.A(i).real(sp(2),ai(a),bi(b)));
                        }
                asum += ai.m();
                bsum += bi.m();
                }
            A *= S*V;
            U = ITensor(commonIndex(U,S),sp);
            svdL(A,U,S,V,{"Cutoff",0.0});
            inds.push_back(bk);
            ret.Aref(i) = U;
            }
        
        // Do last tensor
        vi = Index("ext",nvc,Select);
        ak = inds.back();
        sp = hs.si(N);
        A = ITensor(vi,ak,sp);
        int asum = 0 , ct = 1;
        for(auto& v : vecs) {
            auto ai = leftLinkInd(v,N);
            for(int a = 1 ; a <= ai.m() ; ++a) {
                A.set(vi(ct),ak(asum+a),sp(1),v.A(N).real(sp(1),ai(a)));
                A.set(vi(ct),ak(asum+a),sp(2),v.A(N).real(sp(2),ai(a)));
                }
            asum += ai.m();
            ct++;
            }
        A *= S*V;
        ret.Aref(N) = A;
        regauge(ret,N,0.0);
    } else if(lr == RIGHT) { 
        // Do last tensor
        int bm = 0;
        for(auto& v : vecs) bm += leftLinkInd(v,N).m();
        bk = Index("link",bm);
        auto sp = hs.si(N);
        ITensor A(bk,sp);
        int bsum = 0;
        for(auto& v : vecs) {
            auto bi = leftLinkInd(v,N);
            for(int b = 1 ; b <= bi.m() ; ++b) {
                A.set(bk(bsum+b),sp(1),v.A(N).real(sp(1),bi(b)));
                A.set(bk(bsum+b),sp(2),v.A(N).real(sp(2),bi(b)));
                }
            bsum += bi.m();
            }
        V = ITensor(sp);
        svdL(A,U,S,V,{"Cutoff",0.0});
        inds.push_back(bk);
        ret.Aref(N) = V;

        // Do middle tensors
        for(int i = N-1 ; i > 1 ; --i) {
            int bm = 0;
            for(auto& v : vecs) bm += leftLinkInd(v,i).m();
            ak = inds.back();
            bk = Index("link",bm);
            sp = hs.si(i);
            A = ITensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = rightLinkInd(v,i);
                auto bi = leftLinkInd(v,i);
                for(int a = 1 ; a <= ai.m() ; ++a)
                    for(int b = 1 ; b <= bi.m() ; ++b) {
                        A.set(ak(asum+a),bk(bsum+b),sp(1),v.A(i).real(sp(1),ai(a),bi(b)));
                        A.set(ak(asum+a),bk(bsum+b),sp(2),v.A(i).real(sp(2),ai(a),bi(b)));
                        }
                asum += ai.m();
                bsum += bi.m();
                }
            A *= U*S;
            V = ITensor(commonIndex(V,S),sp);
            U = ITensor();
            svdL(A,U,S,V,{"Cutoff",0.0});
            inds.push_back(bk);
            ret.Aref(i) = V;
            }
        
        // Do first tensor
        vi = Index("ext",nvc,Select);
        ak = inds.back();
        sp = hs.si(1);
        A = ITensor(vi,ak,sp);
        int asum = 0 , ct = 1;
        for(auto& v : vecs) {
            auto ai = rightLinkInd(v,1);
            for(int a = 1 ; a <= ai.m() ; ++a) {
                A.set(vi(ct),ak(asum+a),sp(1),v.A(1).real(sp(1),ai(a)));
                A.set(vi(ct),ak(asum+a),sp(2),v.A(1).real(sp(2),ai(a)));
                }
            asum += ai.m();
            ct++;
            }
        A *= U*S;
        ret.Aref(1) = A;
        regauge(ret,1,0.0);
        }

    return; 
    }
