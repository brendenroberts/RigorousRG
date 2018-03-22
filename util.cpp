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

    for(int i = 1 ; i < ls ; ++i)
        L = (i == 1 ? psi.A(i) : L*psi.A(i))*psip.A(i)*delta(hs.si(i),hs.siP(i));

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

    for(int i = rs+1 ; i <= N ; ++i)
        R = (i == rs+1 ? psi.A(i) : R*psi.A(i))*psip.A(i)*delta(hs.si(i),hs.siP(i));
    
    if(R) rho.Aref(n) *= R;

    return;
    }

template<class MPSLike>
void regauge(MPSLike& psi , int o, Args const& args) {
    psi.orthogonalize(args);
    psi.position(o,args);

    return;
    }
template void regauge(MPS& , int , Args const&);
template void regauge(MPO& , int , Args const&);

Real measEE(const MPS& state , int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});

    ITensor U = psi.A(a),S,V;
    auto spectrum = svd(U*psi.A(a+1),U,S,V);

    Real ret = 0.0;
    for(auto p : spectrum.eigs()) if(p > 1e-18) ret += -p*log(p);
    
    return ret;
    }

MPO sysOp(const SiteSet& hs, const char* op_name, const Real scale) {
    auto ret = MPO(hs);    
    auto N = hs.N();

    for(int i = 1 ; i <= N ; ++i) {
        auto cur = scale*ITensor(hs.op(op_name,i));
        ret.Aref(i) = cur;
        }

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

ITensor measBd(const MPS& psi, const MPS& phi, const ITensor& A, int lr) {
    auto N = psi.N();
    auto xs = lr ? 1 : N;
    auto xt = lr ? N : 1;

    if(findtype(psi.A(xs),Link) == findtype(phi.A(xs),Link))
        return (psi.A(xs)*A)*primeExcept(psi.A(xs),Link);

    ITensor T = psi.A(xt)*phi.A(xt);

    for(int i = 1; i < N-1; i++) {
        int x = (lr ? N-i : i+1);
        T *= psi.A(x);
        T *= phi.A(x);
        }   
    T *= psi.A(xs);
    T *= A;
    T *= primeExcept(phi.A(xs),Link);
    
    return T;
    }

vector<Real> dmrgMPO(const MPO& H , vector<MPS>& states , int num_sw , double penalty, double err) {
    vector<Real> evals;
    vector<MPS> exclude;
    Real e;
    for(auto& psi : states) {
        auto swp = Sweeps(num_sw);
        //swp.maxm() = 150,150,150,200,200,200;
        swp.cutoff() = err;
        swp.niter() = 4;
        swp.noise() = 0.0;

        // dmrg call won't shut up
        std::stringstream ss;
        auto out = std::cout.rdbuf(ss.rdbuf()); 
        e = dmrg(psi,H,exclude,swp,{"Quiet",true,"PrintEigs",false,"Weight",penalty});
        std::cout.rdbuf(out);

        if(exclude.size() == 0)
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

double restrictMPO(const MPO& O , MPO& res , int ls , int D, int lr) {
    auto N = O.N();
    auto n = res.N();
    if(N == n) {res = O; return 0.0;}
    const auto& sub = res.sites();
    auto M = O;
    ITensor U,V,SB;
    int rs = ls+n-1;
    time_t t1,t2;
    double ctime = 0.0;
   
    if(ls == 1) { // easy case: only dangling bond already at R end
        auto SS = splitMPO(M,res,LEFT);
        auto ei = Index("ext",min(D,int(findtype(res.A(n),Select))),Select);
        res.Aref(n) *= delta(ei,commonIndex(SS,res.A(n)));
        regauge(res,n,{"Cutoff",epx});
        return 0.0;
    } else if(rs == N) { // easy case: only dangling bond already at L end
        auto SS = splitMPO(M,res,RIGHT);
        auto ei = Index("ext",min(D,int(findtype(res.A(1),Select))),Select);
        res.Aref(1) *= delta(ei,commonIndex(SS,res.A(1)));
        regauge(res,1,{"Cutoff",epx});
        return 0.0;
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
   
    for(int i = 0 ; i < min(2*D,int(li)) ; ++i)
        wk.push_back(LRVal(ldat[i]*rdat[0],i+1,1));
    
    for(int i = 0 ; i < min(D*D,int(li)*int(ri)) ; ++i) {  
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
    for(auto& it : args) UU.set(lt(it.L),rt(it.R),ei(count++),1);
    res.Aref(1) *= delta(lt,li);
    res.Aref(n) *= delta(rt,ri);

    ITensor S;
    if(lr)
        for(int i = n-1 ; i >= 1 ; --i) {
            auto B = res.A(i)*res.A(i+1);
            U = ITensor(sub.si(i),sub.siP(i),rt,(i == 1 ? lt : leftLinkInd(res,i)));
            V = ITensor();
            time(&t1);
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromright,{"Cutoff",eps,"Maxm",MAXBD});
            } else {
                svdL(B,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
                U *= S;
                }
            time(&t2); ctime += difftime(t2,t1);
            res.Aref(i) = U;
            res.Aref(i+1) = V;
            }
    else
        for(int i = 2 ; i <= n ; ++i) {
            auto B = res.A(i-1)*res.A(i);
            U = ITensor();
            V = ITensor(sub.si(i),sub.siP(i),lt,(i == n ? rt : rightLinkInd(res,i)));
            time(&t1);
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromleft,{"Cutoff",eps,"Maxm",MAXBD});
            } else {
                svdL(B,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
                V *= S;
                }
            time(&t2); ctime += difftime(t2,t1);
                res.Aref(i-1) = U;
                res.Aref(i) = V;
            }

    res.Aref(lr?1:n) *= UU;
    res.leftLim(lr?0:n-1);
    res.rightLim(lr?2:n+1);
    regauge(res,lr?1:n,{"Cutoff",eps});
    
    return ctime; 
    }

template<class Tensor>
double applyMPO(MPSt<Tensor> const& psi_in, MPOt<Tensor> const& K_in, MPSt<Tensor>& res, int lr, Args const& args) {
    using IndexT = typename Tensor::index_type;
    auto psi = psi_in;
    auto K = K_in;
    auto N = psi.N();
    if(K.N() != N) Error("Mismatched N in applyMPO");
    auto trunK = args.getBool("TruncateMPO",false);
    int xs = lr ? 1 : N , xt = lr ? N : 1;
    time_t t1,t2;
    double ctime = 0.0;

    vector<Index> ext;
    if(findtype(psi.A(xs),Select)) ext.push_back(findtype(psi.A(xs),Select));
    if(findtype(K.A(xs),Select)) ext.push_back(findtype(K.A(xs),Select));
    res = psi; res.mapprime(0,1,Site);
    if((int)ext.size() > 1) res.Aref(xs) *= setElt(ext[1](1));

    psi.position(xs); K.position(xs);

    Index iA,iK,iT;
    Tensor clust,nfork,S,tA,tK;
    for(int i = 0; i < N-1; i++) {
        int x = (lr ? i+1 : N-i);
        if(trunK) {
            iT = lr ? leftLinkInd(K,x) : rightLinkInd(K,x);
            tK = i != 0 && iT != iK ? delta(iK,iT)*K.A(x) : K.A(x);
            tA = psi.A(x);
        } else {
            iT = lr ? leftLinkInd(psi,x) : rightLinkInd(psi,x);
            tA = i != 0 && iT != iA ? delta(iA,iT)*psi.A(x) : psi.A(x);
            tK = K.A(x);
            }
        iA = lr ? rightLinkInd(psi,x) : leftLinkInd(psi,x);
        iK = lr ? rightLinkInd(K,x) : leftLinkInd(K,x);
 
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
        time(&t1);
        denmatDecomp(clust,nfork,res.Anc(x),Fromright,args);
        time(&t2); ctime += difftime(t2,t1);
        IndexT mid = commonIndex(res.A(x),nfork,Link);
        mid.dag();
        if(lr)
            res.Anc(x+1) = Tensor(mid,prime(res.sites()(x+1)));
        else
            res.Anc(x-1) = Tensor(mid,prime(res.sites()(x-1)));
        }
    nfork = clust * psi.A(xt) * K.A(xt);

    time(&t1);
    denmatDecomp(nfork,res.Aref(lr?xt-1:xt+1),res.Aref(xt),Fromleft,args);
    time(&t2); ctime += difftime(t2,t1);
    res.leftLim(xt-1);
    res.rightLim(xt+1);
    res.position(xs,{"Cutoff",eps});
    if(ext.size() > 1) res.Aref(xs) *= combiner(ext,{"IndexType",Select});
    res.mapprime(1,0,Site);

    return ctime;
    }
template double applyMPO(const MPS& , const MPO& , MPS& , int , const Args&);

ITensor overlapT(const MPS& phi, const MPO& H, const MPS& psi) {
    auto N = H.N();
    if(phi.N() != N || psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    ITensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = i ? L*phi.A(x) : phi.A(x);
        L *= H.A(x);
        L *= dag(prime(psi.A(x)));
        }
    
    return L;
    }

ITensor overlapT(const MPS& phi, const MPS& psi) { return overlapT(phi,sysOp(phi.sites(),"Id"),psi); }

double tensorProduct(const MPS& psiA, const MPS& psiB, MPS& ret, const ITensor& W, int lr) {
    const int N = ret.N();
    const int n = psiA.N();
    const auto& hs  = ret.sites();
    Index ai,ei;
    ITensor T,U,S,V;
    time_t t1,t2;
    double ctime = 0.0;
    
    for(int i = 1 ; i <= n ; ++i) {
        ret.Aref(i)   = psiA.A(i)*delta(psiA.sites().si(i),hs.si(i));
        ret.Aref(n+i) = psiB.A(i)*delta(psiB.sites().si(i),hs.si(n+i));
        }

    ai = commonIndex(ret.A(n-1),ret.A(n));
    T = lr ? (ret.A(n)*W)*ret.A(n+1) : ret.A(n)*(W*ret.A(n+1));
    ei = findtype(T,Select);
    U = lr && ei ? ITensor(hs.si(n),ai,ei) : ITensor(hs.si(n),ai);
    time(&t1);
    svdL(T,U,S,V,{"Cutoff",eps});
    time(&t2); ctime += difftime(t2,t1);
    ret.Aref(n)   = lr ? U : U*S;
    ret.Aref(n+1) = lr ? S*V : V;
    ret.leftLim(lr?n:n-1);
    ret.rightLim(lr?n+2:n+1);
    ret.position(lr?N:1,{"Cutoff",eps,"Maxm",MAXBD});
    ret.position(lr?n:n+1,{"Cutoff",eps,"Maxm",MAXBD});

    // Move selection index from middle to edge
    for(int i = 0 ; i < n-1 ; ++i) {
        int x = (lr ? n-1-i : n+1+i);
        ai = commonIndex(ret.A(x-1),ret.A(x));
        T = ret.A(x)*ret.A(x+1);
        U = lr && ei ? ITensor(hs.si(x),ai,ei) : ITensor(hs.si(x),ai);
        time(&t1);
        svdL(T,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
        time(&t2); ctime += difftime(t2,t1);
        ret.Aref(x)   = lr ? U*S : U;
        ret.Aref(x+1) = lr ? V : S*V;
        ret.leftLim(lr?x-1:x);
        ret.rightLim(lr?x+1:x+2);
        }
 
    return ctime; 
    }

template<class Tensor> 
double combineMPS(const vector<MPSt<Tensor> >& v_in , MPSt<Tensor>& ret, int lr) {
    auto n = (int)v_in.size(); 
    double ctime = 0.0;
    if(n == 1) {
        ret = v_in[0];
        return ctime;
    } else if(n > 2) { // might not suck??
        auto aMPS = ret,bMPS = ret; 
        vector<MPSt<Tensor> > a(v_in.begin(),v_in.begin() + v_in.size()/2);
        vector<MPSt<Tensor> > b(v_in.begin() + v_in.size()/2,v_in.end());
        ctime += combineMPS(a,aMPS,lr);
        ctime += combineMPS(b,bMPS,lr);
        vector<MPSt<Tensor> > c = {aMPS,bMPS};
        ctime += combineMPS(c,ret,lr);
        return ctime;
        }
        
    using IndexT = typename Tensor::index_type;
    const int N = ret.N();
    const auto& hs = ret.sites();
    const int xs = (lr ? 1 : N);
    auto vecs = v_in;
    vector<IndexT> inds,ext;
    IndexT ak,bk,ci,vi,sp;
    Tensor T,U,S,V;
    time_t t1,t2;

    int nx = 0;
    for(auto& v : vecs) {
        nx += (ci = findtype(v.A(xs),Select)) ? int(ci) : 1;
        v.position(xs,{"Cutoff",1e-16});
        }

    if(lr == LEFT) {
        // Do first tensor
        int bm = 0;
        for(auto& v : vecs) bm += int(leftLinkInd(v,N));
        vi = Index("ext",nx,Select);
        bk = IndexT("li",bm);
        sp = hs.si(N);
        Tensor A(vi,bk,sp);
        int bsum = 0, nsum = 0;
        for(auto& v : vecs) {
            auto bi = leftLinkInd(v,N);
            T = v.A(N);
            if(!(ci = findtype(T,Select))) {
                ci = Index("dummy",1,Select);
                T *= setElt(ci(1));
                }
            for(int n = 1 ; n <= int(ci) ; ++n)
                for(int b = 1 ; b <= int(bi) ; ++b)
                    for(int s = 1 ; s <= int(sp) ; ++s)
                        A.set(vi(nsum+n),bk(bsum+b),sp(s),T.real(sp(s),bi(b),ci(n)));
            bsum += int(bi);
            nsum += int(ci);    
            }
        U = Tensor(bk);
        V = Tensor(vi,sp);
        time(&t1);
        svdL(A,U,S,V,{"Cutoff",eps});
        time(&t2); ctime += difftime(t2,t1);
        inds.push_back(bk);
        ret.Aref(N) = V;

        // Do middle tensors
        for(int i = N-1 ; i > 1 ; --i) {
            int bm = 0;
            for(auto& v : vecs) bm += int(leftLinkInd(v,i));
            ak = inds.back();
            bk = IndexT("li",bm);
            sp = hs.si(i);
            A = Tensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = rightLinkInd(v,i);
                auto bi = leftLinkInd(v,i);
                for(int a = 1 ; a <= int(ai) ; ++a)
                    for(int b = 1 ; b <= int(bi) ; ++b)
                        for(int s = 1 ; s <= int(sp) ; ++s)
                            A.set(ak(asum+a),bk(bsum+b),sp(s),v.A(i).real(sp(s),ai(a),bi(b)));
                asum += int(ai);
                bsum += int(bi);
                }
            A *= U*S;
            U = Tensor(bk);
            V = Tensor(commonIndex(S,V),sp);
            time(&t1);
            svdL(A,U,S,V,{"Cutoff",eps});
            time(&t2); ctime += difftime(t2,t1);
            inds.push_back(bk);
            ret.Aref(i) = V;
            }
       
        // Do last tensor
        ak = inds.back();
        sp = hs.si(1);
        A = Tensor(ak,sp);
        int asum = 0;
        for(auto& v : vecs) {
            auto ai = rightLinkInd(v,1);
            for(int a = 1 ; a <= int(ai) ; ++a)
                for(int s = 1 ; s <= int(sp) ; ++s)
                    A.set(ak(asum+a),sp(s),v.A(1).real(sp(s),ai(a)));
            asum += int(ai);
            }
        A *= U*S;
        ret.Aref(1) = A;
    } else if(lr == RIGHT) { 
        // Do last tensor
        int bm = 0;
        for(auto& v : vecs) bm += int(rightLinkInd(v,1));
        vi = Index("ext",nx,Select);
        bk = IndexT("li",bm);
        sp = hs.si(1);
        Tensor A(vi,bk,sp);
        int bsum = 0 , nsum = 0;
        for(auto& v : vecs) {
            auto bi = rightLinkInd(v,1);
            T = v.A(1);
            if(!(ci = findtype(T,Select))) {
                ci = Index("dummy",1,Select);
                T *= setElt(ci(1));
                }
            for(int n = 1 ; n <= int(ci) ; ++n)
                for(int b = 1 ; b <= int(bi) ; ++b)
                    for(int s = 1 ; s <= int(sp) ; ++s)
                        A.set(vi(nsum+n),bk(bsum+b),sp(s),T.real(sp(s),bi(b),ci(n)));
            bsum += int(bi);
            nsum += int(ci);
            }
        U = Tensor(vi,sp);
        V = Tensor(bk);
        time(&t1);
        svdL(A,U,S,V,{"Cutoff",eps});
        time(&t2); ctime += difftime(t2,t1);
        inds.push_back(bk);
        ret.Aref(1) = U;

        // Do middle tensors
        for(int i = 2 ; i < N ; ++i) {
            int bm = 0;
            for(auto& v : vecs) bm += int(rightLinkInd(v,i));
            ak = inds.back();
            bk = IndexT("li",bm);
            sp = hs.si(i);
            A = Tensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = leftLinkInd(v,i);
                auto bi = rightLinkInd(v,i);
                for(int a = 1 ; a <= int(ai) ; ++a)
                    for(int b = 1 ; b <= int(bi) ; ++b)
                        for(int s = 1 ; s <= int(sp) ; ++s)
                            A.set(ak(asum+a),bk(bsum+b),sp(s),v.A(i).real(sp(s),ai(a),bi(b)));
                asum += int(ai);
                bsum += int(bi);
                }
            A *= S*V;
            U = Tensor(commonIndex(U,S),sp);
            V = Tensor();
            time(&t1);
            svdL(A,U,S,V,{"Cutoff",eps});
            time(&t2); ctime += difftime(t2,t1);
            inds.push_back(bk);
            ret.Aref(i) = U;
            }
        
        // Do first tensor
        ak = inds.back();
        sp = hs.si(N);
        A = Tensor(ak,sp);
        int asum = 0;
        for(auto& v : vecs) {
            auto ai = leftLinkInd(v,N);
            for(int a = 1 ; a <= int(ai) ; ++a)
                for(int s = 1 ; s <= int(sp) ; ++s)
                    A.set(ak(asum+a),sp(s),v.A(N).real(sp(s),ai(a)));
            asum += int(ai);
            }
        A *= S*V;
        ret.Aref(N) = A;
        }
    ret.leftLim(lr?N-1:0);    
    ret.rightLim(lr?N+1:2);    

    time(&t1);
    ret.position(xs,{"Cutoff",eps,"Maxm",MAXBD});
    time(&t2); ctime += difftime(t2,t1);
    return ctime; 
    }
template double combineMPS(const vector<MPS>& vecs , MPS& ret, int lr);
//template double combineMPS(const vector<IQMPS>& vecs , IQMPS& ret, int lr);
