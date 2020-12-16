#include "rrg.h"
#include "itensor/mps/mps.cc"
#include "itensor/mps/mpoalgs.cc"

struct getReal {};
vector<Real,itensor::uninitialized_allocator<Real>> doTask(getReal, QDiag<Real> const& d) { return d.store; }

Index extIndex(ITensor const& A , string tag) {
    if(hasQNs(A))
        return Index(QN({-div(A)}),1,tag);
    else
        return Index(1,tag);
    }

template<class MPSLike>
tuple<Index,int> findExt(MPSLike const& psi) {
    auto ret = Index();
    auto i = 0;

    for(i = 1 ; i <= length(psi) ; ++i)
        if(ret = findIndex(psi(i),"Ext"))
            break;

    return {ret,i};
    }
template tuple<Index,int> findExt(MPS const&);
template tuple<Index,int> findExt(MPO const&);

void parse_config(std::ifstream &cstrm , std::map<string,string> &props) {
    string key, val;
    for(string ln ; std::getline(cstrm, ln) ; ) {
        auto eqpos = ln.find('=');
        if(ln[0] == '#' || eqpos == string::npos) continue;
        key = ln.substr(0,eqpos) , val = ln.substr(eqpos+1,ln.length());

        key.erase(key.find_last_not_of(" \t")+1);
        key.erase(0, key.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t")+1);
        val.erase(0, val.find_first_not_of(" \t"));

        props.emplace(key,val);
        }
    
    return;
    }

vector<vector<size_t> > block_sizes(string const& spec_in) {
    vector<vector<size_t> > ns;
    auto spec = spec_in;
    ns.push_back(vector<size_t>());

    while(size_t pos = spec.find_first_of(" ;,\t")) {
        ns.back().push_back(stoul(spec.substr(0,pos)));
        if(pos == string::npos) break;
        spec.erase(0,pos+1);
        }

    auto l = ns.back().size();
    auto nLevels = static_cast<size_t>(log2(l));
    if(!l || l & (l-1)) { Error("must be power-of-two number of initial blocks"); }
    
    while(ns.size() <= nLevels) {
        auto np = ns.back();
        ns.push_back(vector<size_t>());
        for(auto it = np.begin() ; it != np.end() ; it += 2) {
            size_t sz = *it + *(it+1);
            ns.back().push_back(sz);
            }
        }

    return ns;
    }

void init_H_blocks(AutoMPO const& H , vector<MPO>& Hs , vector<SiteSet> const& HH) {
    if(static_cast<int>(HH.size()) == 1) Hs.push_back(toMPO(H,{"Exact",true}));
    const int N = length(H.sites());

    int offset = 0; 
    for(const auto& k : HH) {
        int n = length(k);
        AutoMPO Hcur(k);
        for(const auto& term : H.terms()) {
            int mn = N, mx = 1;
            for(const auto& op : term.ops) {
                if(op.i > mx) mx = op.i;
                if(op.i < mn) mn = op.i;
                }
            if(mn > offset && mx <= offset+n) {
                auto tcur = term;
                for(auto& op : tcur.ops) op.i -= offset;
                Hcur.add(tcur);
                }
            }
        Hs.push_back(toMPO(Hcur,{"Exact",true}));
        }

    return;
    }

MPVS sum(MPVS const& L , MPVS const& R , Args const& args = Args::global()) {
    auto lr = args.getInt("Direction",RIGHT);

    int xs = lr == RIGHT ? 1 : length(L);
    MPVS res(L,lr) , Rxt(R,lr);
    ITensor E1,E2;
    auto e1 = findIndex(res.A(xs),"Ext");
    auto e2 = findIndex(Rxt.A(xs),"Ext");
    auto ti = e1;
    itensor::plussers(e1,e2,ti,E1,E2);
    res.ref(xs) *= E1;
    Rxt.ref(xs) *= E2;

    res.plusEq(Rxt,args);
    return res;
    }

void sortExt(ITensor &A) {
    auto [C,c] = combiner(findInds(A,"Ext"),{"Tags","Ext"});
    A *= C;
    return;
    }

MPVS::MPVS(vector<MPS> const& v_in , int dir) : lr(dir) {
    auto n = static_cast<int>(v_in.size()) , xs = this->lr ? 1 : itensor::length(v_in.at(0));
    vector<MPVS> vecs(v_in.begin(), v_in.end());

    for(auto& v : vecs) {
        v.position(xs,{"Cutoff",1e-18});
        v.Aref(xs) *= setElt(extIndex(v.A(xs))(1));
        }
    *this = sum(vecs,{"Cutoff",1e-18,"Direction",this->lr});
    sortExt(this->A_[xs]);
    this->position(xs);
    }

void MPVS::reverse() {
    std::reverse(std::begin(this->A_),std::end(this->A_));
    this->l_orth_lim_ = this->N_ - 1 - this->l_orth_lim_;
    this->r_orth_lim_ = this->N_ + 3 - this->r_orth_lim_;
    this->lr = 1 - this->lr;
    }


void MPVS::position(int i , Args const& args) {
    auto args2 = args;
    args2.add("Truncate",false);
    this->MPS::position(i,args2);
    }

void MPOS::reverse() {
    std::reverse(std::begin(this->MPS::A_),std::end(this->MPS::A_));
    this->MPS::l_orth_lim_ = this->MPS::N_ - 1 - this->MPS::l_orth_lim_;
    this->MPS::r_orth_lim_ = this->MPS::N_ + 3 - this->MPS::r_orth_lim_;
    }

void MPOS::position(int i , Args const& args) {
    auto args2 = args;
    args2.add("Truncate",false);
    this->MPS::position(i,args2);
    }

Index siteIndex(MPVS const& psi, int j) {
    return findIndex(psi(j),"Site");
    }

IndexSet siteInds(MPVS const& x) {
    auto N = length(x);
    auto inds = IndexSetBuilder(N);
    for( auto n : range1(N) ) {
      auto s = siteIndex(x,n);
      inds.nextIndex(std::move(s));
      }
    return inds.build();
    }

Index uniqueSiteIndex(MPO const& W, MPVS const& A, int b) {
    return uniqueIndex(W(b),{W(b-1),W(b+1),A(b)},"Site");
    }

IndexSet siteInds(MPO const& x) {
    auto N = length(x);
    auto inds = IndexSetBuilder(N);
    for( auto n : range1(N) ) {
      auto s = siteIndex(x,n,"Site,0");
      inds.nextIndex(std::move(s));
      }
    return inds.build();
    }

IndexSet
uniqueSiteInds(MPO const& A, MPVS const& x)
    {
    auto N = length(x);
    if( N!=length(x) ) Error("In uniqueSiteInds(MPO,MPS), lengths of MPO and MPS do not match");
    auto inds = IndexSetBuilder(N);
    for( auto n : range1(N) )
      {
      auto s = uniqueSiteIndex(A,x,n);
      inds.nextIndex(std::move(s));
      }
    return inds.build();
    }

MPVS& MPVS::replaceSiteInds(IndexSet const& sites) {
    auto& x = *this;
    auto N = itensor::length(x);
    if( itensor::length(sites)!=N ) Error("In replaceSiteInds(MPS,IndexSet), number of site indices not equal to number of MPS tensors");
    auto sx = siteInds(x);
    if( equals(sx,sites) ) return x;
    for( auto n : range1(N) )
      {
      auto sn = sites(n);
      A_[n].replaceInds({sx(n)},{sn});
      }
    return x;
    }

MPVS replaceSiteInds(MPVS& x, IndexSet const& sites) {
    x.replaceSiteInds(sites);
    return x;
    }

ITensor inner(MPVS const& phi, MPO const& H, MPVS const& psi) {
    auto N = length(H); if(length(phi) != N || length(psi) != N) Error("inner mismatched N");
    auto lr = phi.parity();
    ITensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = i ? L*phi(x) : phi(x);
        L *= H(x);
        L *= dag(prime(psi(x)));
        }
    
    return L;
    }

ITensor inner(const MPVS& phi, const MPVS& psi) { return inner(phi,MPO(siteInds(phi)),psi); }

template<class MPSLike>
void regauge(MPSLike& psi , int o, Args const& args) {
    psi.orthogonalize(args);
    psi.position(o);

    return;
    }
template void regauge(MPVS& , int , Args const&);
template void regauge(MPOS& , int , Args const&);

void multiplyMPO(MPO const& Aorig, MPO const& Borig, MPO& res, Args args) {
    if(length(Aorig) != length(Borig)) Error("nmultMPO(MPO): Mismatched MPO length");
    const int N = length(Aorig);

    if(!args.defined("Cutoff")) args.add("Cutoff",1E-14);
    if(!args.defined("RespectDegenerate")) args.add("RespectDegenerate",true);
    auto args_inner = args;
    args_inner.add("Cutoff",args.getReal("Cutoff")/N);

    auto A = Aorig;
    A.position(1,{"Cutoff",1e-18});

    MPO B;
    if(&Borig == &Aorig)
        {
        B = A;
        }
    else
        {
        B = Borig;
        B.position(1,{"Cutoff",1e-18});
        }

    auto lA = linkInds(A);
    auto lB = linkInds(A);
    auto sA = uniqueSiteInds(A,B);
    auto sB = uniqueSiteInds(B,A);

    // Check that A and B have unique indices
    for(int i = 1; i <= N; ++i)
      {
      if(!sA(i)) throw ITError("Error in nmultMPO(A,B): MPO tensor A("+str(i)+") does not have a unique site index. You may have meant to call nmultMPO(A,prime(B)).");
      if(!sB(i)) throw ITError("Error in nmultMPO(A,B): MPO tensor B("+str(i)+") does not have a unique site index. You may have meant to call nmultMPO(A,prime(B)).");
      }

    res=A;
    res.ref(1) = ITensor(sA(1),sB(1),lA(1));

    ITensor clust,nfork;
    for(int i = 1; i < N; ++i)
        {
        if(i == 1) clust = A(i) * B(i);
        else clust = nfork * A(i) * B(i);
        if(i == N-1) break;

        nfork = ITensor(lA(i),lB(i),linkIndex(res,i));
        denmatDecomp(clust,res.ref(i),nfork,Fromleft,{args_inner,"Tags=",tags(lA(i))});
        auto mid = commonIndex(res(i),nfork);
        mid.dag();
        res.ref(i+1) = ITensor(mid,sA(i+1),sB(i+1),rightLinkIndex(res,i+1));
        }

    nfork = clust * A(N) * B(N);
    res.svdBond(N-1,nfork,Fromright,args_inner);
    res.orthogonalize(args);
    }

Real cutEE(MPS const& state , int a) {
    auto psi = state;
    psi.position(a,{"Cutoff",1e-18});

    auto [U,S,V] = svd(psi(a)*psi(a+1),inds(psi(a)));
    auto eigs = doTask(getReal{},S.store());

    Real ret = 0.0;
    for(auto p : eigs) if(p > 1e-18) ret += -p*p*log(p*p);
    
    return ret;
    }

// I(a:b) = S(a) + S(b) - S(a u b)
Real mutualInfoTwoSite(MPS const& state , int a , int b) {
    Real Sa = 0.0 , Sb = 0.0 , Sab = 0.0;
    auto psi = state;
    ITensor T;
    if(a > b) {
        auto c = a;
        a = b;
        b = c;
        }
    
    psi.position(a,{"Cutoff",1e-18});
    T = psi(a)*dag(prime(psi(a),"Site"));
    auto [U,D] = diagHermitian(T);
    auto eigs = doTask(getReal{},D.store());
    for(auto p : eigs) if(p > 1e-18) Sa += -p*log(p);
    
    psi.position(b,{"Cutoff",1e-18});
    T = psi(b)*dag(prime(psi(b),"Site"));
    std::tie(U,D) = diagHermitian(T);
    eigs = doTask(getReal{},D.store());
    for(auto p : eigs) if(p > 1e-18) Sb += -p*log(p);

    auto psiC = prime(psi);
    T = a == 1 ? psi(a)*dag(psiC(a)) 
               : psi(a)*dag(noPrime(psiC(a),leftLinkIndex(psiC,a)));

    for(int i : range1(b-a-1)) {
        T *= psi(a+i);
        T *= dag(noPrime(psiC(a+i),"Site"));
        }
    
    T *= b == length(psi) ? psi(b)*dag(psiC(b))
                          : psi(b)*dag(noPrime(psiC(b),rightLinkIndex(psiC,b)));
    
    std::tie(U,D) = diagHermitian(T);
    eigs = doTask(getReal{},D.store());
    for(auto p : eigs) if(p > 1e-18) Sab += -p*log(p);
 
    return Sa+Sb-Sab;
    }

void Trotter(MPO& eH , double t , size_t M , AutoMPO& ampo) {
    auto evOp = toExpH(ampo,1.0/(t*(double)M));

    eH = evOp;
    evOp.prime();
    eH.ref(1) /= norm(eH(1));
    for(auto i = 2u ; i <= M ; ++i) {
        multiplyMPO(eH,evOp,eH,{"Cutoff",epx});
        eH.mapPrime(2,1);
        eH.ref(1) /= norm(eH(1));
        (void)i;
        }

    return;
    }

void dmrgMPO(MPO const& H , vector<pair<double,MPS> >& eigen , int num_sw, Args const& args) {
    auto do_exclude = args.getBool("Exclude",true);
    auto penalty = args.getReal("Penalty",1.0);
    auto err = args.getReal("Cutoff",epx);
    vector<MPS> exclude;

    for(auto& evPair : eigen) {
        auto psi = evPair.second;
        auto swp = Sweeps(num_sw);
        swp.maxdim() = MAX_BOND;
        swp.cutoff() = err;
        swp.niter() = 2;
        swp.noise() = 0.0;

        auto [en,res] = dmrg(H,exclude,psi,swp,{"Silent",true,"Weight",penalty});

        if(do_exclude) exclude.push_back(psi);
        evPair = make_pair<double,MPS>(std::move(en),std::move(res));
        }

    return;
    }

tuple<ITensor,ITensor> sliceMPO(MPO const& O, MPOS& P, int ls , int rs , int m = 0) {
    const auto N = length(O) , n = length(P);
    MPOS M(O);
    ITensor T,U,S,V,SL,SR;
    auto sRes = siteInds(P);

    auto args = Args("Cutoff",epx,"LeftTags","Ext,R","RightTags","Ext,L","RespectDegenerate",true);
    if(m > 0) args.add("MaxDim",m);

    if(ls != 1) {
        M.position(ls);
        T = M(ls-1)*M(ls), U = M(ls-1);
        svd(T,U,S,V,args);
        M.set(ls,V);
        SL = S;
        M.leftLim(ls-2);
        M.rightLim(ls);
        }
    
    if(rs != N) {
        M.position(rs);
        T = M(rs)*M(rs+1), U = M(rs);
        svd(T,U,S,V,args);
        M.set(rs,U);
        SR = S;
        M.leftLim(rs);
        M.rightLim(rs+2);
        } 

    for(int i : range1(n))
        P.set(i,M(ls+i-1));

    auto sFull = siteInds(P);
    P.replaceSiteInds(sFull,sRes);
    P.replaceSiteInds(prime(sFull),prime(sRes));
    P.orthogonalize({"Cutoff",epx});
 
    return {SL,SR};
    }

void restrictMPO(MPO const& O , MPOS& res , int ls , int D, int lr) {
    auto N = length(O) , n = length(res); if(N == n) { res = MPOS(O); return; }
    ITensor U,V;
    int rs = ls+n-1;
    auto M = O;
    
    sliceMPO(M,res,ls,rs,D);

    if(ls == 1 || rs == N) {
        res.position(lr?1:n);
        return;
        }

    auto lInd = siteIndex(res,1,"Ext,L") , rInd = siteIndex(res,n,"Ext,R");
    auto [C,cInd] = combiner(IndexSet(lInd,rInd),{"Tags","Ext"});

    // move external bond to correct end 
    if(lr == RIGHT) {
        res.position(n);
        for(int i = n-1 ; i >= 1 ; --i) {
            auto si = siteIndex(res,i,"Site,0");
            auto B = res(i)*res(i+1);
            U = ITensor(si,prime(si),rInd,(i == 1 ? lInd : leftLinkIndex(res,i)));
            V = ITensor();
            denmatDecomp(B,U,V,Fromright,{"Cutoff",eps});
            res.set(i,U);
            res.set(i+1,V);
            }
        res.leftLim(0);
        res.rightLim(2);
    } else {
        res.position(1);
        for(int i = 2 ; i <= n ; ++i) {
            auto si = siteIndex(res,i,"Site,0");
            auto B = res(i-1)*res(i);
            U = ITensor();
            V = ITensor(si,prime(si),lInd,(i == n ? rInd : rightLinkIndex(res,i)));
            denmatDecomp(B,U,V,Fromleft,{"Cutoff",eps});
            res.set(i-1,U);
            res.set(i,V);
            }
        res.leftLim(n-1);
        res.rightLim(n+1);
        }
    
    res.ref(lr == RIGHT ? 1 : n) *= C;
    
    return; 
    }

pair<ITensor,ITensor> tensorProdContract(MPVS const& psiL, MPVS const& psiR, MPO const& H_in) {
    const int N = length(H_in) , nL = length(psiL) , nR = length(psiR);
    if(nL + nR != N) Error("tensorProdContract mismatched N");
    ITensor L,R;

    auto si = unionInds(siteInds(psiL),siteInds(psiR));
    auto hi = siteInds(H_in);
    auto H = replaceSiteInds(H_in,hi,si.dag());
    H = replaceSiteInds(H,hi.prime(),si.prime());

    for(int i : range1(nL)) {
        L = i == 1 ? psiL(i) : L*psiL(i);
        L *= H(i);
        L *= dag(prime(psiL(i)));
        }
    L = dag(L);
    
    for(int i : range(nR)) {
        int y = N-i , z = nR-i;
        R = i == 0 ? psiR(z) : R*psiR(z);
        R *= H(y);
        R *= dag(prime(psiR(z)));
        }
    R = dag(R); 
 
    return std::make_pair(L,R);
    }

void tensorProduct(MPVS const& psiL,
                   MPVS const& psiR,
                   MPVS& ret,
                   ITensor const& W,
                   int lr,
                   bool move) {
    const int N = length(ret) , nL = length(psiL) , nR = length(psiR);
    const int n = lr==RIGHT ? nL : nR; 
    Index ai,ei;
    ITensor T,U,S,V;
    time_t t1,t2;
    double ctime = 0.0;

    for(int i : range1(nL))
        ret.set(i,replaceInds(psiL(i),{siteIndex(psiL,i)},{siteIndex(ret,i)}));
    for(int i : range1(nR))
        ret.set(nL+i,replaceInds(psiR(i),{siteIndex(psiR,i)},{siteIndex(ret,nL+i)}));
    ret.position(nL);

    // Move selection index from middle to edge
    for(int i : range(n)) {
        int x = lr==RIGHT ? nL-i : nL+i;
        ai = leftLinkIndex(ret,x);
        T = i == 0 ? ret(x)*W*ret(x+1) : ret(x)*ret(x+1);
        if(i == 0) { sortExt(T); ei = findIndex(T,"Ext"); }
        U = lr==RIGHT && ei ? ITensor(siteIndex(ret,x),ai,ei) : ITensor(siteIndex(ret,x),ai);
        svd(T,U,S,V,{"Cutoff",eps,"RespectDegenerate",true});
        ret.set(x,lr ? U*S : U);
        ret.set(x+1,lr ? V : S*V);
        if(!move) return;
        }
    ret.leftLim(lr == RIGHT ? 0 : N-1);
    ret.rightLim(lr == RIGHT ? 2 : N+1);

    return; 
    }

MPVS
densityMatrixApplyMPOImpl(MPO const& K,
                          MPVS const& x,
                          Args args = Args::global());

void
fitApplyMPOImpl(MPVS const& psi,
                MPO const& K,
                MPVS & res,
                Args const& args = Args::global());

void
zipUpApplyMPOImpl(MPVS const& psi, 
                  MPOS const& K, 
                  MPVS& res, 
                  Args const& args = Args::global());

MPVS
applyMPO(MPOS const& K_in,
         MPVS const& x_in,
         Args const& args)
    {
    if( !x_in ) Error("Error in applyMPO, MPS is uninitialized.");
    if( !K_in ) Error("Error in applyMPO, MPO is uninitialized.");

    auto args2 = args;
    auto method = args.getString("Method","DensityMatrix");
    if(!args2.defined("RespectDegenerate")) args2.add("RespectDegenerate",true);

    auto x = x_in;
    auto K = K_in;
    auto doRev = x_in.parity() == LEFT;

    if(doRev) {
        x.reverse();
        K.reverse();
        }

    if(method != "Fit")
        for(auto j = 1 ; j < length(x) ; ++j)
            if(rightLinkIndex(K,j).dim()*rightLinkIndex(x,j).dim() > MAX_TEN_DIM) {
                method = "Fit";
                if(!args2.defined("Nsweep")) args2.add("Nsweep",20);
                std::cout << "Switching to fit method" << std::endl;
                break;
                }

    MPVS res;
    if(method == "DensityMatrix")
        {
        res = densityMatrixApplyMPOImpl(K,x,args2);
        }
    else if(method == "Fit")
        {
        //auto sites = uniqueSiteInds(K,x);
        //res = replaceSiteInds(x,sites);
        auto coarseEps = sqrt(args.getReal("Cutoff",eps));
        auto coarseK = K; coarseK.orthogonalize({"Cutoff",coarseEps});
        auto coarsex = x; coarsex.orthogonalize({"Cutoff",coarseEps});
        zipUpApplyMPOImpl(coarsex,coarseK,res,{"Cutoff",coarseEps});
        args2.add("NCenterSites",1);
        fitApplyMPOImpl(x,K,res,args2);
        }
    else
        {
        Error("applyMPO currently supports the following methods: 'DensityMatrix', 'Fit'");
        }

    res.ref(1) *= std::get<0>(combiner(findInds(res(1),"Ext"),{"Tags","Ext"}));
    if(doRev) res.reverse();

    return res;
    }

MPVS
applyMPO(MPO const& K_in,
         MPVS const& x_in,
         Args const& args)
    {
    if( !x_in ) Error("Error in applyMPO, MPS is uninitialized.");
    if( !K_in ) Error("Error in applyMPO, MPO is uninitialized.");

    auto args2 = args;
    auto method = args.getString("Method","DensityMatrix");
    if(!args2.defined("RespectDegenerate")) args2.add("RespectDegenerate",true);

    auto x = x_in;
    auto K = K_in;

    if(method != "Fit")
        for(auto j = 1 ; j < length(x) ; ++j)
            if(rightLinkIndex(K,j).dim()*rightLinkIndex(x,j).dim() > MAX_TEN_DIM) {
                method = "Fit";
                if(!args2.defined("Nsweep")) args2.add("Nsweep",20);
                std::cout << "Switching to fit method" << std::endl;
                break;
                }

    MPVS res;
    if(method == "DensityMatrix")
        {
        res = densityMatrixApplyMPOImpl(K,x,args2);
        }
    else if(method == "Fit")
        {
        // Use the input MPS x to be applied as the
        // default starting state
        auto sites = uniqueSiteInds(K,x);
        res = replaceSiteInds(x,sites);
        args2.add("NCenterSites",1);
        fitApplyMPOImpl(x_in,K,res,args2);
        }
    else
        {
        Error("applyMPO currently supports the following methods: 'DensityMatrix', 'Fit'");
        }

    return res;
    }

MPVS
densityMatrixApplyMPOImpl(MPO const& K,
                          MPVS const& psi,
                          Args args)
    {
    if( args.defined("Maxm") )
      {
      if( args.defined("MaxDim") )
        {
        Global::warnDeprecated("Args Maxm and MaxDim are both defined. Maxm is deprecated in favor of MaxDim, MaxDim will be used.");
        }
      else
        {
        Global::warnDeprecated("Arg Maxm is deprecated in favor of MaxDim.");
        args.add("MaxDim",args.getInt("Maxm"));
        }
      }

    auto cutoff = args.getReal("Cutoff",epx);
    auto dargs = Args{"Cutoff",cutoff};
    auto maxdim_set = args.defined("MaxDim");
    if(maxdim_set) dargs.add("MaxDim",args.getInt("MaxDim"));
    dargs.add("RespectDegenerate",args.getBool("RespectDegenerate",true));
    auto verbose = args.getBool("Verbose",false);
    auto normalize = args.getBool("Normalize",false);

    auto N = length(psi);

    auto [eIndP,eSiteP] = findExt(psi);
    auto [eIndK,eSiteK] = findExt(K);

    for( auto n : range1(N) )
      {
      if( commonIndex(psi(n),K(n)) != siteIndex(psi,n) )
          Error("MPS and MPO have different site indices in applyMPO method 'DensityMatrix'");
      }

    auto rand_plev = 14741;

    auto res = psi;

    //Set up conjugate psi and K
    auto psic = psi;
    auto Kc = K;
    //TODO: use sim(linkInds), sim(siteInds)
    psic.dag().prime(rand_plev);
    Kc.dag().prime(rand_plev);
    if(eIndP) psic.ref(eSiteP).noPrime("Ext");
    if(eIndK) Kc.ref(eSiteK).noPrime("Ext");

    // Make sure the original and conjugates match
    for(auto j : range1(N-1))
        Kc.ref(j).prime(-rand_plev,uniqueSiteIndex(Kc,psic,j));

    //Build environment tensors from the left
    if(verbose) print("Building environment tensors...");
    auto E = std::vector<ITensor>(N+1);
    E[1] = psi(1)*K(1)*Kc(1)*psic(1);
    for(int j = 2; j < N; ++j)
        {
        E[j] = E[j-1]*psi(j)*K(j)*Kc(j)*psic(j);
        }
    if(verbose) println("done");

    //O is the representation of the product of K*psi in the new MPS basis
    auto O = psi(N)*K(N);

    auto rho = E[N-1] * O * dag(prime(O,rand_plev));

    ITensor U,D;
    auto ts = tags(linkIndex(psi,N-1));
    auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
    if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",N-1,spec.truncerr(),dim(commonIndex(U,D)));

    res.ref(N) = dag(U);

    O = O*U*psi(N-1)*K(N-1);

    for(int j = N-1; j > 1; --j)
        {
        if(not maxdim_set)
            {
            //Infer maxdim from bond dim of original MPS
            //times bond dim of MPO
            //i.e. upper bound on order of rho
            auto cip = commonIndex(psi(j),E[j-1]);
            auto ciw = commonIndex(K(j),E[j-1]);
            auto maxdim = (cip) ? dim(cip) : 1l;
            maxdim *= (ciw) ? dim(ciw) : 1l;
            dargs.add("MaxDim",maxdim);
            }
        rho = E[j-1] * O * dag(prime(O,rand_plev));
        ts = tags(linkIndex(psi,j-1));
        auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
        O = O*U*psi(j-1)*K(j-1);
        res.ref(j) = dag(U);
        if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",j,spec.truncerr(),dim(commonIndex(U,D)));
        }

    if(normalize) O /= norm(O);
    res.ref(1) = O;
    res.leftLim(0); res.rightLim(2);
    res.position(N);
    res.noPrime();

    return res;
    }

void
oneSiteFitApply(vector<ITensor> & E,
                Real fac,
                MPVS const& x,
                MPO const& K,
                MPVS & Kx,
                Args const& args)
    {
    auto N = length(x);
    auto verbose = args.getBool("Verbose",false);
    auto normalize = args.getBool("Normalize",false);
    auto sw = args.getInt("Sweep",1);

    for(int s = 1, ha = 1; ha <= 2; sweepnext1(s,ha,N))
        {
        if(verbose)
            {
            printfln("Sweep=%d, HS=%d, Site=%d",sw,ha,s);
            }
        auto ds = (ha==1 ? +1 : -1);

        auto nE = E[s-ds]*x(s)*K(s);
        auto P = nE*E[s+ds];
/*
        if(order(P) > 3)
            {
            Print(P);
            Error("order > 3 of P");
            }
*/
        P *= fac;
        if(normalize) P /= norm(P);

        if(s+ds >= 1 && s+ds <= N)
            {
            auto ci = commonIndex(Kx(s),Kx(s+ds));
            if(!hasIndex(P,ci))
                {
                Print(ci);
                Print(inds(P));
                Error("P does not have Index ci");
                }
            auto [U,S,V] = svd(P,{ci},args);
            (void)U;
            (void)S;
            Kx.ref(s) = dag(V);
            }
        else
            {
            Kx.ref(s) = dag(P);
            }

        // Update environment:
        E[s] = nE * Kx(s);
        }
    }

void
fitApplyMPOImpl(Real fac,
                MPVS const& x,
                MPO const& K,
                MPVS& Kx,
                Sweeps const& sweeps,
                Args args)
    {
    auto N = length(x);
    auto NCenterSites = args.getInt("NCenterSites",1);//2);

    Kx.dag();
    // Make the indices of |Kx> and K|x> match
    Kx.replaceSiteInds(uniqueSiteInds(K,x));
    // Replace the link indices of |Kx> with similar ones
    // so they don't clash with the links of |x>
    Kx.replaceLinkInds(sim(linkInds(Kx)));
    Kx.position(1);

    auto E = vector<ITensor>(N+2,ITensor(1.));
    for(auto n = N; n > NCenterSites; --n)
        {
        E[n] = E[n+1]*x(n)*K(n)*Kx(n);
        }

    for(auto sw : range1(sweeps.nsweep()))
        {
        args.add("Sweep",sw);
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("MinDim",sweeps.mindim(sw));
        args.add("MaxDim",sweeps.maxdim(sw));
        args.add("Noise",sweeps.noise(sw));

        if(NCenterSites == 2)
            {
            twoSiteFitApply(E,fac,x,K,Kx,args);
            }
        else if(NCenterSites == 1)
            {
            oneSiteFitApply(E,fac,x,K,Kx,args);
            }
        else
            {
            Error(format("NCenterSites = %d not supported",NCenterSites));
            }
        }
    Kx.dag();
    Kx.noPrime();
    }

void
fitApplyMPOImpl(Real fac,
                MPVS const& psi,
                MPO const& K,
                MPVS& res,
                Args args)
    {
    if( args.defined("Maxm") )
      {
      if( args.defined("MaxDim") )
        {
        Global::warnDeprecated("Args Maxm and MaxDim are both defined. Maxm is deprecated in favor of MaxDim, MaxDim will be used.");
        }
      else
        {
        Global::warnDeprecated("Arg Maxm is deprecated in favor of MaxDim.");
        args.add("MaxDim",args.getInt("Maxm"));
        }
      }
    auto nsweep = args.getInt("Nsweep",1);
    Sweeps sweeps(nsweep);
    auto cutoff = args.getReal("Cutoff",-1);
    if(cutoff >= 0) sweeps.cutoff() = cutoff;
    auto maxdim = args.getInt("MaxDim",-1);
    if(maxdim >= 1) sweeps.maxdim() = maxdim;
    fitApplyMPOImpl(fac,psi,K,res,sweeps,args);
    }

void
fitApplyMPOImpl(MPVS const& psi,
                MPO const& K,
                MPVS& res,
                Args const& args)
    {
    fitApplyMPOImpl(1.,psi,K,res,args);
    }

// assumes that external indices of both objects are located at site 1 (also the OC)
void
zipUpApplyMPOImpl(MPVS const& psi, 
                  MPOS const& K, 
                  MPVS& res, 
                  Args const& args)
    {
    const bool allow_arb_position = args.getBool("AllowArbPosition",false);
    auto [extIndexK,extSiteK] = findExt(K);

    if(&psi == &res)
        Error("psi and res must be different MPS instances");

    auto N = length(psi);
    if(length(K) != N) 
        Error("Mismatched N in ApplyMPO() ZipUp method");

    if(!itensor::isOrtho(psi) || itensor::orthoCenter(psi) != 1)
        Error("Ortho center of psi must be site 1");

    if(!allow_arb_position && (!itensor::isOrtho(K) || itensor::orthoCenter(K) != 1))
        Error("Ortho center of K must be site 1");

    res = psi;
    res.replaceTags("0","4","Link");
    res.replaceTags("0","1","Site");
    if(extIndexK) res.ref(1) *= setElt(extIndexK(1));
    ITensor clust,nfork;
    vector<int> midsize(N);
    int maxdim = 1;
    for(int i = 1; i < N; i++)
        {
        if(i == 1) { clust = psi(i) * K(i); }
        else { clust = nfork * (psi(i) * K(i)); }
        if(i == N-1) break; //No need to SVD for i == N-1

        Index oldmid = linkIndex(res,i); assert(oldmid.dir() == Out);
        nfork = ITensor(linkIndex(psi,i),linkIndex(K,i-1),oldmid);
        denmatDecomp(clust, res.ref(i), nfork,Fromleft,args);
        Index mid = commonIndex(res(i),nfork);
        mid.dag();
        midsize[i] = dim(mid);
        maxdim = std::max(midsize[i],maxdim);
        assert(linkIndex(res,i+1).dir() == Out);
        res.ref(i+1) = ITensor(mid,siteIndex(res,i+1),linkIndex(res,i+1));
        }
    nfork = clust * psi(N) * K(N);

    res.svdBond(N-1,nfork,Fromright,args);
    res.noPrime("Link");
    res.replaceTags("1","0","Site");
    res.position(1);
    }
