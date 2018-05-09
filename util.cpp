#include "rrg.h"
#define T_TOL 1e-18

template<class Tensor>
int nels(const Tensor& A) {
    int n = 1;
    for(const auto& i : A.inds()) n *= int(i);
    return n;
    }
template int nels(const ITensor&);
template int nels(const IQTensor&);

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

template<class Tensor>
Tensor overlapT(const MPSt<Tensor>& phi, const MPOt<Tensor>& H, const MPSt<Tensor>& psi) {
    auto N = H.N(); if(phi.N() != N || psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    Tensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = i ? L*phi.A(x) : phi.A(x);
        L *= H.A(x);
        L *= dag(prime(psi.A(x)));
        }
    
    return L;
    }
template ITensor overlapT(const MPS& , const MPO& , const MPS&);
template IQTensor overlapT(const IQMPS& , const IQMPO& , const IQMPS&);

template<> ITensor overlapT(const MPS& phi, const MPS& psi) { return overlapT(phi,sysOp(phi.sites(),"Id").toMPO(),psi); }
template<> IQTensor overlapT(const IQMPS& phi, const IQMPS& psi) { return overlapT(phi,sysOp(phi.sites(),"Id"),psi); }

template<class Tensor>
Tensor overlapT(const MPOt<Tensor>& A, const MPOt<Tensor>& B) {
    auto N = A.N(); if(B.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(A.A(N),Select) ? LEFT : RIGHT);
    Tensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = i ? L*mapprime(A.A(x),Site,0,2) : mapprime(A.A(x),Site,0,2);
        L *= dag(mapprime(B.A(x),Site,1,2,Site,0,1,Link,0,1,Select,0,1));
        }
    
    return L;
    }
template ITensor overlapT(const MPO& , const MPO&);
template IQTensor overlapT(const IQMPO& , const IQMPO&);

template<class MPSLike>
void regauge(MPSLike& psi , int o, Args const& args) {
    psi.orthogonalize(args);
    psi.position(o,{"Cutoff",1e-16});

    return;
    }
template void regauge(MPS& , int , Args const&);
template void regauge(MPO& , int , Args const&);
template void regauge(IQMPS& , int , Args const&);
template void regauge(IQMPO& , int , Args const&);

template<class Tensor>
Real measEE(const MPSt<Tensor>& state , int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});

    Tensor U = psi.A(a),S,V;
    auto spectrum = svd(U*psi.A(a+1),U,S,V);

    Real ret = 0.0;
    for(auto p : spectrum.eigs()) if(p > 1e-18) ret += -p*log(p);
    
    return ret;
    }
template Real measEE(const MPS& , int);
template Real measEE(const IQMPS& , int);

IQMPO sysOp(const SiteSet& hs, const char* op_name, const Real scale) {
    auto ret = IQMPO(hs);    
    auto N = hs.N();

    for(int i = 1 ; i <= N ; ++i) {
        auto cur = scale*IQTensor(hs.op(op_name,i));
        ret.Aref(i) = cur;
        }

    return ret;
    }

template<class Tensor>
Real measOp(const MPSt<Tensor>& state, const IQTensor& A, int a, const IQTensor& B, int b) {
    if(b <= a) Error("measOp requires a < b");
    auto psi = state;
    psi.position(a,{"Cutoff",1e-24});
    
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
template Real measOp(const MPS& , const IQTensor& , int , const IQTensor& , int);
template Real measOp(const IQMPS& , const IQTensor& , int , const IQTensor& , int);

template<class Tensor>
Real measOp(const MPSt<Tensor>& state, const IQTensor& A, int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",0.0});
    auto C = psi.A(a)*A*dag(prime(psi.A(a),Site));
    return C.real();
    }
template Real measOp(const MPS& , const IQTensor& , int);
template Real measOp(const IQMPS& , const IQTensor& , int);

template<typename Tensor>
MPSt<Tensor> opFilter(MPSt<Tensor> const& st, vector<MPOt<Tensor> > const& ops, Real thr , int lr) {
    using IndexT = typename Tensor::index_type;
    int tr, ss = (lr ? 1 : st.N());
    Tensor U,Dg,G;
    IndexT fi;
    auto ret = st;

    for(auto& op : ops) {
        G = overlapT(ret,op,ret);
        auto vals = diagHermitian(G,U,Dg);
        tr = 0; for(auto v : vals.eigs()) if(v > thr) tr++; else break;
        if(tr > 0) {
            fi = Index("ext",tr,Select);
            ret.Aref(ss) *= dag(U)*Dg*delta(prime(commonIndex(U,Dg)),fi);
        } else ret.Aref(ss).prime(Select);
        }

    return ret;
    }
template MPS opFilter(MPS const& , vector<MPO> const& , Real , int);

template<class Tensor>
vector<Real> dmrgMPO(const MPOt<Tensor>& H , vector<MPSt<Tensor> >& states , int num_sw, Args const& args) {
    auto do_exclude = args.getBool("Exclude",true);
    auto penalty = args.getReal("Penalty",1.0);
    auto err = args.getReal("Cutoff",1e-16);
    vector<MPSt<Tensor> > exclude;
    vector<Real> evals;
    
    for(auto& psi : states) {
        auto swp = Sweeps(num_sw);
        swp.maxm() = 150,150,200,200;
        swp.cutoff() = err;
        swp.niter() = 4;
        swp.noise() = 0.0;

        std::stringstream ss;
        auto out = std::cout.rdbuf(ss.rdbuf()); 
        auto e = dmrg(psi,H,exclude,swp,{"Quiet",true,"PrintEigs",false,"Weight",penalty});
        std::cout.rdbuf(out);

        if(do_exclude) exclude.push_back(psi);
        evals.push_back(e);
        }

    return evals;
    }
template vector<Real> dmrgMPO(const MPO& , vector<MPS>& , int , Args const&);
template vector<Real> dmrgMPO(const IQMPO& , vector<IQMPS>& , int , Args const&);

template<class MPOType>
void twoLocalTrotter(MPOType& eH , double t , int M , AutoMPO& ampo) {
    using Tensor = typename MPOType::TensorT;
    const auto& hs = eH.sites();
    Tensor term;

    AutoMPO Hevn(hs),Hodd(hs);
    for(const auto& ht : ampo.terms()) {
        auto st = ht.ops;
        if(st.size() == 1) st[0].i%2==0?Hevn.add(ht):Hodd.add(ht);
        else  min(st[0].i,st[1].i)%2==0?Hevn.add(ht):Hodd.add(ht);
        }

    auto Uevn  = toExpH<Tensor>(Hevn,t/(double)M);
    auto Uodd  = toExpH<Tensor>(Hodd,t/(double)M);
    auto Uodd2 = toExpH<Tensor>(Hodd,t/(2.0*(double)M));

    auto evOp = Uodd2;
    nmultMPO(evOp,Uevn,evOp,{"Truncate",false});
    nmultMPO(evOp,Uodd2,evOp,{"Truncate",false});

    eH = evOp;
    for(int i : range(M-1)) {
        nmultMPO(eH,evOp,eH,{"Cutoff",T_TOL});
        eH.Aref(1) /= norm(eH.A(1));
        (void)i;
        }

    return;
    }
template void twoLocalTrotter(MPO& eH , double t , int M , AutoMPO& ampo);
template void twoLocalTrotter(IQMPO& eH , double t , int M , AutoMPO& ampo);

template<class Tensor>
void extractBlocks(AutoMPO const& H , vector<MPOt<Tensor> >& Hs , const SiteSet& HH) {
    std::stringstream sts;
    auto out = std::cout.rdbuf(sts.rdbuf()); 
    
    auto N = H.sites().N() , n = HH.N();
    if(n == N) Hs.push_back(toMPO<Tensor>(H,{"Exact",true}));
   
    for(auto k : range(N/n)) {
        AutoMPO Hcur(HH);
        for(const auto& term : H.terms()) {
            int mn = N, mx = 1;
            for(const auto& op : term.ops) {
                if(op.i > mx) mx = op.i;
                if(op.i < mn) mn = op.i;
                }
            if(mn > k*n && mx <= (k+1)*n) {
                auto tcur = term;
                for(auto& op : tcur.ops) op.i -= k*n;
                Hcur.add(tcur);
                }
            }
        Hs.push_back(toMPO<Tensor>(Hcur,{"Exact",true}));
        }
    std::cout.rdbuf(out);

    return;
    }
template void extractBlocks(AutoMPO const& , vector<MPO>& , const SiteSet&);
template void extractBlocks(AutoMPO const& , vector<IQMPO>& , const SiteSet&);

template<class Tensor>
Tensor splitMPO(const MPOt<Tensor>& O, MPOt<Tensor>& P, int lr) {
    using IndexT = typename Tensor::index_type;
    auto N = O.N() , n = P.N();
    const auto HS = O.sites() , hs = P.sites();
    auto M = O;
    int t = lr ? N-n : n;
    Tensor S,V;
    IndexT sp,sq;

    M.position(t,{"Truncate",false});
    auto B = M.A(t)*M.A(t+1), U = M.A(t);
    svdL(B,U,S,V,{"Cutoff",T_TOL,"LeftIndexType",Select,"RightIndexType",Select});
    auto R = lr ? V : U;
    sp = hs.si(lr ? 1 : n);
    sq = HS.si(lr ? t+1 : t);
    P.Aref(lr ? 1 : n) = R*delta(sp,sq)*delta(prime(sp),prime(sq));
    
    for(int i = 1 ; i < n ; ++i) {
        auto x = lr ? i+1 : n-i; sp = hs.si(x);
        auto y = lr ? N-n+x : x; sq = HS.si(y);
        P.Aref(x) = M.A(y)*delta(sp,sq)*delta(prime(sp),prime(sq));
        }
    
    return S;
    }
template ITensor splitMPO(const MPO& , MPO& , int);
template IQTensor splitMPO(const IQMPO& , IQMPO& , int);

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

template<class Tensor>
double restrictMPO(const MPOt<Tensor>& O , MPOt<Tensor>& res , int ls , int D, int lr) {
    auto N = O.N();
    auto n = res.N();
    if(N == n) {res = O; return 0.0;}
    const auto& sub = res.sites();
    auto M = O;
    Tensor U,V,SB;
    int rs = ls+n-1;
    time_t t1,t2;
    double ctime = 0.0;
    
    // easy case: one dangling bond already at correct end
    if(ls == 1) {
        auto SS = splitMPO(M,res,LEFT);
        auto ei = Index("ext",min(D,int(findtype(res.A(n),Select))),Select);
        res.Aref(n) *= delta(ei,commonIndex(SS,res.A(n)));
        regauge(res,n,{"Cutoff",epx});
        return 0.0;
    } else if(rs == N) {
        auto SS = splitMPO(M,res,RIGHT);
        auto ei = Index("ext",min(D,int(findtype(res.A(1),Select))),Select);
        res.Aref(1) *= delta(ei,commonIndex(SS,res.A(1)));
        regauge(res,1,{"Cutoff",epx});
        return 0.0;
        }

    // setup for moving external bond to correct end 
    Tensor SL,SR;
    if(lr) {
        SpinHalf htmp(rs);
        MPOt<Tensor> tmp(htmp);
        SR = splitMPO(M,tmp,LEFT);
        SL = splitMPO(tmp,res,RIGHT);
    } else {
        SpinHalf htmp(N-ls+1);
        MPOt<Tensor> tmp(htmp);
        SL = splitMPO(M,tmp,RIGHT);
        SR = splitMPO(tmp,res,LEFT);
        }
 
    Index li = commonIndex(SL,res.A(1));
    Index ri = commonIndex(SR,res.A(n));

    auto ldat = doTask(getReal{},SL.store());
    dscal_wrapper(int(li),SL.scale().real(),ldat.data());
    auto rdat = doTask(getReal{},SR.store());
    dscal_wrapper(int(ri),SR.scale().real(),rdat.data());

    vector<LRVal> args,work;
    args.reserve(D*D); work.reserve(int(li));
   
    for(int i = 0 ; i < int(li) ; ++i)
        work.push_back(LRVal(ldat[i]*rdat[0],i+1,1));
    
    for(int i = 0 ; i < min(D*D,int(li)*int(ri)) ; ++i) {  
        int amax = argmax(work);
        auto& next = work[amax];
        if(next.val < eps) break;
        args.push_back(next);
        if(next.R == int(ri)) work.erase(work.begin()+amax);
        else work[amax] = LRVal(ldat[next.L-1]*rdat[next.R],next.L,next.R+1);
        }

    auto lt = Index("L",args[argmaxL(args)].L,Select);
    auto rt = Index("R",args[argmaxR(args)].R,Select);
    auto ei = Index("ext",args.size(),Select);
    auto UU = ITensor({lt,rt,ei});
    int count = 1;
    for(auto& it : args) UU.set(lt(it.L),rt(it.R),ei(count++),1);
    res.Aref(1) *= delta(lt,li);
    res.Aref(n) *= delta(rt,ri);

    Tensor S;
    if(lr)
        for(int i = n-1 ; i >= 1 ; --i) {
            auto B = res.A(i)*res.A(i+1);
            U = Tensor(sub.si(i),sub.siP(i),rt,(i == 1 ? lt : leftLinkInd(res,i)));
            V = Tensor();
            time(&t1);
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromright,{"Cutoff",1e-16});
            } else {
                svdL(B,U,S,V,{"Cutoff",1e-16});
                U *= S;
                }
            time(&t2); ctime += difftime(t2,t1);
            res.Aref(i) =   U;
            res.Aref(i+1) = V;
            }
    else
        for(int i = 2 ; i <= n ; ++i) {
            auto B = res.A(i-1)*res.A(i);
            U = Tensor();
            V = Tensor(sub.si(i),sub.siP(i),lt,(i == n ? rt : rightLinkInd(res,i)));
            time(&t1);
            if(nels(B) < 1e7) {
                denmatDecomp(B,U,V,Fromleft,{"Cutoff",1e-16});
            } else {
                svdL(B,U,S,V,{"Cutoff",1e-16});
                V *= S;
                }
            time(&t2); ctime += difftime(t2,t1);
            res.Aref(i-1) = U;
            res.Aref(i) =   V;
            }

    res.Aref(lr?1:n) *= UU;
    regauge(res,lr?1:n,{"Cutoff",eps,"Maxm",MAXBD});
    
    return ctime; 
    }
template double restrictMPO(const MPO& , MPO& , int , int , int);
//template double restrictMPO(const IQMPO& , IQMPO& , int , int , int);

template<class Tensor>
MPSt<Tensor> applyMPO(MPOt<Tensor> const& K, MPSt<Tensor> const& psi, int lr , Tensor flTen , Args const& args) {
    auto cutoff = args.getReal("Cutoff",epx);
    auto dargs = Args{"Cutoff",cutoff};
    auto maxm_set = args.defined("Maxm");
    if(maxm_set) dargs.add("Maxm",args.getInt("Maxm"));
    auto verbose = args.getBool("Verbose",false);
    auto siteType = getIndexType(args,"SiteType",Site);
    auto linkType = getIndexType(args,"LinkType",Link);
    int plev = 14741;
    auto res = psi;
    auto N = psi.N();
    int xs = lr ? 1 : N;
    int xt = lr ? N : 1;

    //Set up conjugate psi and K
    auto psic = psi;
    auto Kc = K;
    for(int j : range1(N)) {
        psic.Aref(j) = dag(mapprime(psi.A(j),siteType,0,2,linkType,0,plev));
        Kc.Aref(j) = dag(mapprime(K.A(j),siteType,0,2,linkType,0,plev));
        }

    //Build environment tensors from the left/right
    if(verbose) print("Building environment tensors...");
    auto E = std::vector<Tensor>(N+1);
    if(flTen) E.at(1) = ((psi.A(xs)*K.A(xs))*flTen)*((Kc.A(xs)*psic.A(xs))*dag(flTen));
    else E.at(1) = psi.A(xs)*K.A(xs)*Kc.A(xs)*psic.A(xs);
    for(int j = 2; j < N; ++j) {
        int x = lr ? j : N+1-j;
        E.at(j) = E.at(j-1)*psi.A(x)*K.A(x)*Kc.A(x)*psic.A(x);
        assert(rank(E[j])==4);
        }
    if(verbose) println("done");

    //O is the representation of the product of K*psi in the new MPS basis
    auto O = psi.A(xt)*K.A(xt);
    O.noprime(siteType);

    auto rho = E.at(N-1) * O * dag(prime(O,plev));
    Tensor U,D;
    dargs.add("IndexName=",nameint("a",xt));
    auto spec = diagHermitian(rho,U,D,dargs);
    if(verbose) printfln("  j=%02d truncerr=%.2E m=%d",lr?xt-1:xt+1,spec.truncerr(),commonIndex(U,D).m());
    res.Aref(xt) = dag(U);

    O = O*U*psi.A(lr?xt-1:xt+1)*K.A(lr?xt-1:xt+1);
    O.noprime(siteType);

    for(int j = N-1; j > 1; --j) {
        int x = lr ? j : N+1-j;
        if(not maxm_set) {
            //Infer maxm from bond dim of original MPS
            //times bond dim of MPO
            //i.e. upper bound on rank of rho
            auto cip = commonIndex(psi.A(x),E.at(j-1));
            auto ciw = commonIndex(K.A(x),E.at(j-1));
            auto maxm = (cip) ? cip.m() : 1l;
            maxm *= (ciw) ? ciw.m() : 1l;
            dargs.add("Maxm",maxm);
            }
        rho = E.at(j-1) * O * dag(prime(O,plev));
        dargs.add("IndexName=",nameint("a",x));
        auto spec = diagHermitian(rho,U,D,dargs);
        O = O*U*psi.A(lr?x-1:x+1)*K.A(lr?x-1:x+1);
        O.noprime(siteType);
        res.Aref(x) = dag(U);
        if(verbose) printfln("  j=%02d truncerr=%.2E m=%d",x,spec.truncerr(),commonIndex(U,D).m());
        }

    res.Aref(xs) = O;

    if(flTen) res.Aref(xs) *= flTen;
    res.leftLim(xs-1);
    res.rightLim(xs+1);

    return res;
    }
template MPS applyMPO(MPO const&, MPS const&, int, ITensor, Args const&);
//template IQMPS applyMPO(IQMPO const&, IQMPS const&, int, IQTensor, Args const&);

template<class Tensor>
MPSt<Tensor> applyMPO(MPOt<Tensor> const& K, MPSt<Tensor> const& psi, int lr , Args const& args) {
    using IndexT = typename Tensor::index_type;
    int xs = lr ? 1 : K.N();
    auto px = findtype(psi.A(xs),Select) , Kx = findtype(K.A(xs),Select);
    return applyMPO(K,psi,lr,px&&Kx ? combiner(vector<IndexT>({px,Kx}),{"IndexType",Select}):Tensor(),args);
    }
template MPS applyMPO(MPO const&, MPS const&, int, Args const&);
//template IQMPS applyMPO(IQMPO const&, IQMPS const&, int, Args const&);

template<class Tensor>
LRPair<Tensor> tensorProdContract(MPSt<Tensor> const& psiL, MPSt<Tensor> const& psiR, MPOt<Tensor> const& H) {
    auto N = H.N() , n = psiL.N();
    if(n != N/2 || psiR.N() != n) Error("tensorProdContract mismatched N");
    const auto& lsp = psiL.sites() , rsp = psiR.sites() , hsp = H.sites(); 
    Tensor L,R;

    for(int i = 0; i < n; ++i) {
        int x = i+1 , y = N-i , z = n-i;
        L = i ? L*psiL.A(x) : psiL.A(x);
        R = i ? R*psiR.A(z) : psiR.A(z);
        L *= H.A(x)*delta(lsp.si(x),hsp.si(x))*delta(lsp.siP(x),hsp.siP(x));
        R *= H.A(y)*delta(rsp.si(z),hsp.si(y))*delta(rsp.siP(z),hsp.siP(y));
        L *= dag(prime(psiL.A(x)));
        R *= dag(prime(psiR.A(z)));
        }
    
    return LRPair<Tensor>(L,R);
    }
template LRPair<ITensor> tensorProdContract(MPS const&, MPS const&, MPO const&);
template LRPair<IQTensor> tensorProdContract(IQMPS const&, IQMPS const&, IQMPO const&);

template<class Tensor>
double tensorProduct(const MPSt<Tensor>& psiA, const MPSt<Tensor>& psiB, MPSt<Tensor>& ret, const Tensor& W, int lr) {
    const int N = ret.N();
    const int n = psiA.N();
    const auto& hs  = ret.sites();
    Index ai,ei;
    Tensor T,U,S,V;
    time_t t1,t2;
    double ctime = 0.0;
    
    for(int i = 1 ; i <= n ; ++i) {
        ret.Aref(i)   = psiA.A(i)*delta(psiA.sites().si(i),hs.si(i));
        ret.Aref(n+i) = psiB.A(i)*delta(psiB.sites().si(i),hs.si(n+i));
        }

    // Move selection index from middle to edge
    for(int i = 0 ; i < n ; ++i) {
        int x = (lr ? n-i : n+i);
        ai = commonIndex(ret.A(x-1),ret.A(x));
        T = i == 0 ? ret.A(x)*W*ret.A(x+1) : ret.A(x)*ret.A(x+1);
        if(i == 0) ei = findtype(T,Select);
        U = lr && ei ? Tensor(hs.si(x),ai,ei) : Tensor(hs.si(x),ai);
        time(&t1);
        svdL(T,U,S,V,{"Cutoff",1e-16});
        time(&t2); ctime += difftime(t2,t1);
        ret.Aref(x)   = lr ? U*S : U;
        ret.Aref(x+1) = lr ? V : S*V;
        //ret.leftLim(lr?x-1:x);
        //ret.rightLim(lr?x+1:x+2);
        }
    regauge(ret,lr?1:N,{"Cutoff",eps});
 
    return ctime; 
    }
template double tensorProduct(const MPS& , const MPS& , MPS& , const ITensor& , int);
//template double tensorProduct(const IQMPS& , const IQMPS& , IQMPS& , const IQTensor& , int);

template<class MPSLike>
double makeVS(const vector<MPSLike>& v_in , MPSLike& ret, int lr) {
    double ctime = 0.0;
    auto n = (int)v_in.size(); 
    if(n == 1) {
        ret = v_in[0];
        return ctime;
    } else if(n > 2) {
        auto aVS = ret , bVS = ret; 
        vector<MPSLike> avec(v_in.begin(),v_in.begin() + v_in.size()/2);
        vector<MPSLike> bvec(v_in.begin() + v_in.size()/2,v_in.end());
        ctime += makeVS(avec,aVS,lr);
        ctime += makeVS(bvec,bVS,lr);
        vector<MPSLike> cvec = {aVS,bVS};
        ctime += makeVS(cvec,ret,lr);
        return ctime;
        }

    using Tensor = typename MPSLike::TensorT;
    using IndexT = typename Tensor::index_type;
    auto vecs = v_in;
    time_t t1,t2;
    IndexT ci;

    int xs = lr ? 1 : ret.N() , ntot = 0 , nprev = 0;

    for(auto& v : vecs) ntot += (ci = findtype(v.A(xs),Select)) ? int(ci) : 1;
    auto ei = extIndex::gen(ci,ntot,0);
    for(auto& v : vecs) {
        if(!(ci = findtype(v.A(xs),Select))) v.Aref(xs) *= setElt(ei(++nprev));
        else {
            auto map = Tensor(ei,dag(ci)); 
            for(int i : range1(int(ci))) map.set(dag(ci)(i),ei(i+nprev),1.0);
            v.Aref(xs) *= map;
            nprev += int(ci);
            }
        }

    ret = sum(vecs,{"Cutoff",1e-24});

    time(&t1);
    regauge(ret,xs,{"Cutoff",eps});
    time(&t2); ctime += difftime(t2,t1);
    return ctime; 
    }
template double makeVS(const vector<MPS>& , MPS& , int);
template double makeVS(const vector<MPO>& , MPO& , int);
//template double makeVS(const vector<IQMPS>& , IQMPS& , int);
//template double makeVS(const vector<IQMPO>& , IQMPO& , int);
