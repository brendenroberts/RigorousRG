#include "rrg.h"
#include "itensor/mps/mps.cc"
#include "itensor/mps/mpoalgs.cc"

namespace itensor {
void plussers(Index const& , Index const& , Index& , ITensor& , ITensor&);
}

struct getReal {};
vector<Real,itensor::uninitialized_allocator<Real>> doTask(getReal, QDiag<Real> const& d) { return d.store; }

size_t size(ITensor const& A) {
    return std::accumulate(A.inds().begin(),A.inds().end(),1lu,
                            [](size_t prod , Index const& index) { return prod*index.dim();});
    }

Index extIndex(ITensor const& A , string tag) {
    if(hasQNs(A))
        return Index(QN({-div(A)}),1,tag);
    else
        return Index(1,tag);
    }

template<class MPSLike>
std::pair<Index,size_t> findExt(MPSLike const& psi) {
    for(auto j = 1 ; j <= length(psi) ; ++j)
        if(auto ret = findIndex(psi(j),"Ext"))
            return {ret,j};
    
    return {Index(),0lu};
    }
template std::pair<Index,size_t> findExt(MPS const&);
template std::pair<Index,size_t> findExt(MPO const&);

Index sortExt(ITensor &A) {
    auto [C,c] = combiner(findInds(A,"Ext"),{"Tags","Ext"});
    A *= C;
    return c;
    }

void parseConfig(std::ifstream &cstrm , std::map<string,string> &props) {
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

vector<vector<size_t> > parseBlockSizes(string spec) {
    vector<vector<size_t> > ns;
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

void blockHs(vector<MPO>& Hs , AutoMPO const& H , vector<SiteSet> const& HH , Args const& args) {
    if(HH.size() == 1lu) Hs.push_back(toMPO(H,args));
    const auto N = length(H.sites());

    auto offset = 0; 
    for(const auto& k : HH) {
        auto n = length(k);
        AutoMPO Hcur(k);
        for(const auto& term : H.terms()) {
            auto mn = N, mx = 1;
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
        offset += n;
        Hs.push_back(toMPO(Hcur,args));
        }

    return;
    }

MPVS sum(MPVS const& L , MPVS const& R , Args const& args = Args::global()) {
    auto lr = args.getBool("Direction",RIGHT);

    auto xs = lr == RIGHT ? 1lu : length(L);
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

MPVS::MPVS(vector<MPS> const& v_in , bool dir) : lr(dir) {
    auto n = v_in.size() , xs = this->lr == RIGHT ? 1lu : itensor::length(v_in.at(0));
    vector<MPVS> vecs(v_in.begin(), v_in.end());

    for(auto& v : vecs) {
        v.position(xs,{"Truncate",false});
        v.ref(xs) *= setElt(extIndex(v(xs))(1));
        }
    *this = sum(vecs,{"Truncate",false,"Direction",this->lr});
    sortExt(this->A_[xs]);
    this->position(xs);
    }

void MPVS::reverse() {
    std::reverse(std::begin(this->A_),std::end(this->A_));
    this->l_orth_lim_ = this->N_ - 1 - this->l_orth_lim_;
    this->r_orth_lim_ = this->N_ + 3 - this->r_orth_lim_;
    this->lr = !this->lr;
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

IndexSet siteInds(MPVS const& x) {
    auto N = length(x);
    auto inds = IndexSetBuilder(N);
    for( auto n : range1(N) ) {
      auto s = siteIndex(x,n);
      inds.nextIndex(std::move(s));
      }
    return inds.build();
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

Index uniqueSiteIndex(MPO const& W, MPVS const& A, int b) {
    return uniqueIndex(W(b),{W(b-1),W(b+1),A(b)},"Site");
    }

IndexSet
uniqueSiteInds(MPO const& A, MPVS const& x) {
    auto N = length(x);
    if(N != length(A)) Error("In uniqueSiteInds(MPO,MPVS), lengths of MPO and MPVS do not match");
    auto inds = IndexSetBuilder(N);
    for(auto n : range1(N)) {
        auto s = uniqueSiteIndex(A,x,n);
        inds.nextIndex(std::move(s));
        }
    return inds.build();
    }

MPVS& MPVS::replaceSiteInds(IndexSet const& sites) {
    auto& x = *this;
    auto N = itensor::length(x);
    if( itensor::length(sites)!=N ) Error("In replaceSiteInds(MPVS,IndexSet), number of site indices not equal to number of MPS tensors");
    auto sx = siteInds(x);
    if( equals(sx,sites) ) return x;
    for( auto n : range1(N) )
      {
      auto sn = sites(n);
      A_[n].replaceInds({sx(n)},{sn});
      }
    return x;
    }

MPVS replaceSiteInds(MPVS const& x, IndexSet const& sites) {
    auto y = x;
    y.replaceSiteInds(sites);
    return y;
    }

ITensor inner(MPVS const& phi, MPO const& H, MPVS const& psi) {
    auto N = length(H); if(length(phi) != N || length(psi) != N) Error("inner mismatched N");
    auto lr = phi.parity();
    ITensor L;

    for(auto i = 0; i < N; ++i) {
        auto x = (lr == RIGHT ? N-i : i+1);
        L = i ? L*phi(x) : phi(x);
        L *= H(x);
        L *= dag(prime(psi(x)));
        }
    
    return L;
    }

ITensor inner(MPVS const& phi, MPO const& H, MPVS const& psi , size_t eSite) {
    size_t N = length(H);
    if(static_cast<size_t>(length(phi)) != N || static_cast<size_t>(length(psi)) != N) Error("inner mismatched N");
    ITensor L,R;

    for(size_t i = 1; i <= eSite; ++i) {
        L = i == 1 ? phi(i) : L*phi(i);
        L *= H(i);
        if(i < eSite) L *= dag(prime(psi(i)));
        }

    for(size_t i = N ; i >= eSite ; --i) {
        if(i > eSite) {
            R = i == N ? phi(i) : R*phi(i);
            R *= H(i);
            }
        R *= dag(prime(psi(i)));
        }
    
    return L*R;
    }

MPO multiplyMPO(MPO const& A , MPO const& B , Args args) {
    auto cutoff = args.getReal("Cutoff",epx);
    auto dargs = Args{"Cutoff",cutoff};
    auto maxdim_set = args.defined("MaxDim");
    if(maxdim_set) dargs.add("MaxDim",args.getInt("MaxDim"));
    dargs.add("RespectDegenerate",args.getBool("RespectDegenerate",true));
    auto verbose = args.getBool("Verbose",false);

    auto N = length(A);
    if(N != length(B)) Error("nmultMPO(MPO): Mismatched MPO length");

    auto rand_plev = 14741;

    auto res = A;

    //Set up conjugate A and B
    auto Ac = A;
    auto Bc = B;
    //TODO: use sim(linkInds), sim(siteInds)
    Ac.dag().prime(rand_plev);
    Bc.dag().prime(rand_plev);

    // Make sure the original and conjugates match
    for(auto j : range1(N-1)) {
        Bc.ref(j).prime(-rand_plev,uniqueSiteIndex(Bc,Ac,j));
        Ac.ref(j).prime(-rand_plev,uniqueSiteIndex(Ac,Bc,j));
        }

    //Build environment tensors from the left
    if(verbose) print("Building environment tensors...");
    auto E = std::vector<ITensor>(N+1);
    E[1] = A(1)*B(1)*Bc(1)*Ac(1);
    for(int j = 2; j < N; ++j)
        {
        E[j] = E[j-1]*A(j)*B(j)*Bc(j)*Ac(j);
        }
    if(verbose) println("done");

    //O is the representation of the product of A*B in the new MPO basis
    auto O = A(N)*B(N);

    auto rho = E[N-1] * O * dag(prime(O,rand_plev));

    ITensor U,D;
    auto ts = tags(linkIndex(A,N-1));
    auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
    if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",N,spec.truncerr(),dim(commonIndex(U,D)));

    res.ref(N) = dag(U);

    O = O*U*A(N-1)*B(N-1);

    for(int j = N-1; j > 1; --j)
        {
        if(not maxdim_set)
            {
            //Infer maxdim from bond dim of original MPS
            //times bond dim of MPO
            //i.e. upper bound on order of rho
            auto cip = commonIndex(A(j),E[j-1]);
            auto ciw = commonIndex(B(j),E[j-1]);
            auto maxdim = (cip) ? dim(cip) : 1l;
            maxdim *= (ciw) ? dim(ciw) : 1l;
            dargs.add("MaxDim",maxdim);
            }
        rho = E[j-1] * O * dag(prime(O,rand_plev));
        ts = tags(linkIndex(A,j-1));
        auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
        O = O*U*A(j-1)*B(j-1);
        res.ref(j) = dag(U);
        if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",j,spec.truncerr(),dim(commonIndex(U,D)));
        }

    res.ref(1) = O;
    res.leftLim(0); res.rightLim(2);
    res.mapPrime(2,1);

    return res;
    }

MPO Trotter(double t , size_t M , AutoMPO const& ampo , double eps) {
    auto evOp = toExpH(ampo,1.0/(t*M));
    evOp.orthogonalize();

    auto eH = evOp;
    evOp.prime();
    for(auto i : range(M-1)) {
        eH = multiplyMPO(eH,evOp,{"Cutoff",eps/length(eH),"RespectDegenerate",true});
        eH.ref(1) *= exp(length(eH))/norm(eH(1));
        }

    return eH;
    }

void dmrgMPO(MPO const& H , vector<std::pair<double,MPS> >& eigen , int num_sw, Args const& args) {
    auto do_exclude = args.getBool("Exclude",true);
    auto maxDim = args.getInt("MaxDim",MAX_BOND);
    auto penalty = args.getReal("Penalty",1.0);
    auto err = args.getReal("Cutoff",epx);
    vector<MPS> exclude;

    for(auto& evPair : eigen) {
        auto psi = evPair.second;
        auto swp = Sweeps(num_sw);
        swp.maxdim() = maxDim;
        swp.cutoff() = err;
        swp.niter() = 3;
        swp.noise() = 0.0;

        auto [en,res] = dmrg(H,exclude,psi,swp,{"Silent",true,"Weight",penalty});

        if(do_exclude) exclude.push_back(res);
        evPair = {std::move(en),std::move(res)};
        }

    return;
    }

void sliceMPO(MPO const& O, MPOS& res, int ls , size_t D) {
    const auto N = length(O) , n = length(res) , rs = ls+n-1;
    if(N == n) { res = MPOS(O); return; }
    auto sRes = siteInds(res);
    auto M = O;
    ITensor S;

    auto args = Args("Cutoff",epx,"RespectDegenerate",true,"LeftTags","Ext,R","RightTags","Ext,L");
    if(D > 0) args.add("MaxDim",static_cast<int>(D));

    if(ls != 1) {
        M.position(ls,{"Truncate",false});
        svd(M(ls-1)*M(ls),M.ref(ls-1),S,M.ref(ls),args);
        M.ref(ls-1) *= S;
        M.leftLim(ls-2);
        M.rightLim(ls);
        }
 
    if(rs != N) {
        M.position(rs,{"Truncate",false});
        svd(M(rs)*M(rs+1),M.ref(rs),S,M.ref(rs+1),args);
        M.ref(rs+1) *= S;
        M.leftLim(rs);
        M.rightLim(rs+2);
        } 

    for(int i : range1(n))
        res.set(i,M(ls+i-1));

    auto sFull = siteInds(res);
    res.replaceSiteInds(sFull,sRes);
    res.replaceSiteInds(prime(sFull),prime(sRes));
    res.orthogonalize({"Cutoff",epx,"MaxDim",MAX_BOND,"RespectDegenerate",true});
 
    return;
    }

std::pair<ITensor,ITensor> tensorProdContract(MPVS const& psiL, MPVS const& psiR, MPO const& H_in) {
    const size_t N = length(H_in) , nL = length(psiL) , nR = length(psiR);
    if(nL + nR != N) Error("tensorProdContract mismatched N");
    ITensor L,R;

    auto si = unionInds(siteInds(psiL),siteInds(psiR));
    auto hi = siteInds(H_in);
    auto H = replaceSiteInds(H_in,hi,si.dag());
    H = replaceSiteInds(H,hi.prime(),si.prime());

    for(auto i : range1(nL)) {
        L = L ? L*psiL(i) : psiL(i);
        L *= H(i);
        L *= dag(prime(psiL(i)));
        }
    L = dag(L);
    
    for(auto i : range(nR)) {
        auto y = N-i , z = nR-i;
        R = R ? R*psiR(z) : psiR(z);
        R *= H(y);
        R *= dag(prime(psiR(z)));
        }
    R = dag(R); 
 
    return {L,R};
    }

void tensorProduct(MPVS const& psiL,
                   MPVS const& psiR,
                   MPVS& ret,
                   ITensor const& W,
                   Args const& args) {
    const auto N = length(ret) , nL = length(psiL) , nR = length(psiR);
    const auto ei = uniqueIndex(W,{psiL(nL),psiR(1)},"Ext");
    const auto cutoff = args.getReal("Cutoff",epx);
    const auto maxBd = args.getInt("MaxDim",MAX_BOND);
    const auto move = args.getBool("Move",true);
    ITensor T,U,S,V;

    for(auto i : range1(nL))
        ret.set(i,replaceInds(psiL(i),{siteIndex(psiL,i)},{siteIndex(ret,i)}));
    for(auto i : range1(nR))
        ret.set(nL+i,replaceInds(psiR(i),{siteIndex(psiR,i)},{siteIndex(ret,nL+i)}));
    ret.position(nL);

    // Move selection index from middle to edge
    for(auto i : range(nL)) {
        auto ai = leftLinkIndex(ret,nL-i);
        T = i == 0 ? ret(nL-i)*W*ret(nL-i+1) : ret(nL-i)*ret(nL-i+1);
        U = ITensor(siteIndex(ret,nL-i),ai,ei);
        //ret.svdBond(nL-i,T,Fromright,{"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",true,"UseSVD",true});
        svd(T,U,S,V,{"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",true});
        ret.set(nL-i,U*S);
        ret.set(nL-i+1,V);
        if(!move) return;
        }

    ret.orthogonalize({"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",true});
    return; 
    }

MPVS
densityMatrixApplyMPOImpl(MPO const& K,
                          MPVS const& psi,
                          Args args)
    {
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
    auto eIndK2 = findIndex(K(N),"Ext");

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
    if(eIndK2) rho *= delta(dag(eIndK2),prime(eIndK2,rand_plev));

    ITensor U,D;
    auto ts = tags(linkIndex(psi,N-1));
    auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
    if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",N,spec.truncerr(),dim(commonIndex(U,D)));

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
        if(eIndK2) rho *= delta(dag(eIndK2),prime(eIndK2,rand_plev));
        ts = tags(linkIndex(psi,j-1));
        auto spec = diagPosSemiDef(rho,U,D,{dargs,"Tags=",ts});
        O = O*U*psi(j-1)*K(j-1);
        res.ref(j) = dag(U);
        if(verbose) printfln("  j=%02d truncerr=%.2E dim=%d",j,spec.truncerr(),dim(commonIndex(U,D)));
        }

    if(normalize) O /= norm(O);
    res.ref(1) = O;
    res.leftLim(0);
    res.rightLim(2);

    return res;
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
    auto NCenterSites = args.getInt("NCenterSites",2);

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
                Args args)
    {
    fitApplyMPOImpl(1.,psi,K,res,args);
    }

MPVS
applyMPO(MPO const& K,
         MPVS const& x,
         Args args)
    {
    auto method = args.getString("Method","DensityMatrix");
    auto doApprox = args.getBool("DoApprox",true);
    if(!args.defined("RespectDegenerate")) args.add("RespectDegenerate",true);

    auto N = length(x);

    if(method != "Fit")
        for(auto j = 1 ; j < N ; ++j)
            if(rightLinkIndex(K,j).dim()*rightLinkIndex(x,j).dim() > sqrt(nBlocks(siteIndex(x,j)))*MAX_TEN_DIM) {
                method = "Fit";
                if(!args.defined("Nsweep")) args.add("Nsweep",16);
                std::cout << "Switching to fit method" << std::endl;
                break;
                }

    MPVS res;
    if(method == "DensityMatrix")
        {
        res = densityMatrixApplyMPOImpl(K,x,args);
        }
    else if(method == "Fit")
        {
        if(doApprox) {
            auto roughArgs = args;
            roughArgs.add("Cutoff",sqrt(args.getReal("Cutoff")));
            auto roughK = K; roughK.orthogonalize(roughArgs);
            auto maxDim = MAX_BOND;
            for(auto j : range1(N-1))
                maxDim = std::min(static_cast<size_t>(std::floor(sqrt(nBlocks(siteIndex(x,j)))*MAX_TEN_DIM
                                                                        /rightLinkIndex(roughK,j).dim())),maxDim);
            roughArgs = args; roughArgs.add("MaxDim",static_cast<int>(maxDim));
            auto roughx = x; roughx.orthogonalize(roughArgs);
            res = densityMatrixApplyMPOImpl(roughK,roughx,args);
        } else {
            auto sites = uniqueSiteInds(K,x);
            res = replaceSiteInds(x,sites);
            }

        args.add("NCenterSites",2);
        fitApplyMPOImpl(x,K,res,args);
        }
    else
        {
        Error("applyMPO currently supports the following methods: 'DensityMatrix', 'Fit'");
        }

    if(length(findInds(res(1),"Ext")) > 1) sortExt(res.ref(1));
    return res;
    }
