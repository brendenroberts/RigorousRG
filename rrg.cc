#include "rrg.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

// some forward declarations
vector<vector<size_t> > parseBlockSizes(string);
void blockHs(vector<MPO>& , AutoMPO const& , vector<SiteSet> const& , Args const& = Args::global());
void sliceMPO(MPO const& , MPOS& , int , size_t = 0lu);
pair<ITensor,ITensor> tensorProdContract(MPVS const&, MPVS const&, MPO const&);
void tensorProduct(MPVS const& , MPVS const& , MPVS& , ITensor const& , Args const& = Args::global());

pair<MPVS,double> rrg(AutoMPO const& autoH , MPO const& K , string const& blockStr , function<SiteSet(size_t)> const& hFunc , vector<int> const& targetQNs , Args const& args) {
    const auto N = length(autoH.sites());
    const auto cutoff = args.getReal("Cutoff",epx);
    const auto s = args.getInt("ExtDim",1);
    const auto D = args.getInt("OpDim",1);
    const auto sLast = args.getInt("ExtDimLast",s);
    const auto maxBd = args.getInt("MaxDim",MAX_BOND);
    const auto truncateQNs = args.getBool("TruncateQNs",false);
    const auto qnSpread = args.getInt("QNSpread",1);
    const auto nSweep = args.getInt("Nsweep",16);
    const auto keepDeg = args.getBool("RespectDegenerate",true);
    const auto doLanczos = args.getBool("Iterative",true);
    const auto verbose = args.getBool("Verbose",true);

    std::streambuf* old_cout = std::cout.rdbuf();
    if(!verbose) std::cout.rdbuf(NULL);

    // INITIALIZE: create hierarchy structure, generate product-state basis for initial blocking
    auto tI = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto blockNs = parseBlockSizes(blockStr);
    vector<vector<SiteSet> > hsps;
    for(auto const& v : blockNs) {
        hsps.push_back(vector<SiteSet>());
        for(auto const & n : v) {
            SiteSet cur = hFunc(n);
            hsps.back().push_back(cur);
            }
        }

    if(length(hsps.back().back()) != N) { std::cerr << "sum(n) not equal to N" << std::endl; return {MPVS(),0.0}; }

    // Produce block Hamiltonians from AutoMPO
    // TODO: do partial trace of MPO object instead
    vector<vector<MPO> > Hs(hsps.size());
    for(auto i : args(hsps)) blockHs(Hs.at(i),autoH,hsps.at(i));

    // Generate classical basis for exact diagonalization of initial blocks
    vector<MPVS> Spre;
    for(auto a : args(hsps.front())) {
        auto hCur = hsps.front().at(a);
        auto n = length(hCur);
        auto p = vector<size_t>(n,1lu);
        for(auto i : range1(n-1))
            p.at(i) = p.at(i-1)*dim(hCur(i));
        auto nSt = p.back()*dim(hCur(n));

        vector<MPS> V;
        V.reserve(nSt);
        for(auto d : range(nSt)) {
            auto basisVector = MPS(hCur);
            auto links = linkInds(basisVector);
            basisVector.ref(1).set(hCur(1)=d/p.front()%dim(hCur(1))+1,links(1)=1,1.0);
            for(auto j = 2 ; j < n ; ++j)
                basisVector.ref(j).set(hCur(j)=d/p.at(j-1)%dim(hCur(j))+1,links(j-1)=1,links(j)=1,1.0);
            basisVector.ref(n).set(hCur(n)=d/p.back()%dim(hCur(n))+1,links(n-1)=1,1.0);
            V.push_back(basisVector);
            }

        Spre.push_back(MPVS(V,a%2==1?RIGHT:LEFT));
        }

    // Reduce dimension by sampling from initial eigenbasis
    for(auto ll : args(Spre)) {
        auto& pcur = Spre.at(ll);
        auto Hcur = MPOS(Hs.at(0).at(ll));
        auto parity = pcur.parity();
        if(parity == LEFT) { pcur.reverse(); Hcur.reverse(); }

        vector<int> localQNs(targetQNs.size());
        std::transform(targetQNs.begin(),targetQNs.end(),localQNs.begin(),
                       [&Hcur,&N](auto &val){ return divRoundClosest(val*length(Hcur),N); });

        // Generate block eigenbasis by hijacking tensorProdH code
        auto di = hasQNs(pcur(1)) ? Index(QN(),1,"Ext") : Index(1,"Ext");
        tensorProdH init({setElt(di=1,dag(prime(di))=1),inner(pcur,Hcur,pcur)});
        init.diag(localQNs,{"ExtDim",s,"QNSpread",qnSpread,"Iterative",false,"Verbose",false});

        pcur.ref(1) *= init.eigenvectors()*setElt(di=1);
        pcur.orthogonalize({"Cutoff",cutoff,"MaxDim",MAX_BOND,"RespectDegenerate",true});
        if(parity == LEFT) pcur.reverse();
        }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tInit = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "initialization: " << std::fixed <<std::setprecision(0) << tInit.count() << " s" << std::endl;

    // Do the inner loop over scale parameter w
    vector<MPVS> Spost;
    for(auto w  = 0u ; Spre.size() > 1 ; ++w) {
        std::cout << "Level " << w << std::endl;
        const auto& Hl = Hs.at(w);
        auto offset = 0u;

        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        Spost.clear();
        for(auto ll : args(Hl)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            const auto& hs = SiteSet(siteInds(Hs.at(w).at(ll)));
            auto Hc = MPOS(Hl.at(ll));
            auto pre = Spre.at(ll) , ret = MPVS(hs);
            auto parity = pre.parity();
            MPOS A(hs);

            // STEP 1: extract filtering operators A from AGSP K
            sliceMPO(K,A,offset+1,D);
            if(parity == LEFT) { A.reverse(); pre.reverse(); Hc.reverse(); }
            A.orthogonalize({"Cutoff",epx,"MaxDim",maxBd,"RespectDegenerate",keepDeg});

            // STEP 2: expand subspace using the mapping A : pre -> ret
            ret = applyMPO(A,pre,{"Cutoff",cutoff,"MaxDim",maxBd,"Nsweep",nSweep,"UseSVD",true});
            ret.noPrime();

            // Rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H
            auto [U,Dg] = diagPosSemiDef(inner(ret,ret),{"Cutoff",cutoff,"Tags","Ext"});
            Dg.apply([](Real r) {return 1.0/sqrt(r);});
            ret.ref(1) *= U*dag(Dg);
            auto [P,S] = diagPosSemiDef(-inner(ret,Hc,ret),{"Tags","Ext"});
            ret.ref(1) *= P;
            ret.orthogonalize({"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",keepDeg});

            if(parity == LEFT) { ret.reverse(); }
            Spost.push_back(std::move(ret));
            offset += length(pre);

            auto t2 = std::chrono::high_resolution_clock::now();
            auto tExpand = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            std::cout << "expand block " << std::setw(2) << ll << ": "
                      << std::fixed << std::setprecision(0) << tExpand.count() << " s" << std::endl;
            }

        // MERGE/REDUCE STEP: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(auto ll : range(Spost.size()/2)) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto parity = ll%2 ? RIGHT : LEFT , last = Spost.size() == 2;
            auto spL = Spost.at(parity == RIGHT ? 2*ll : 2*ll+1); // L subspace
            auto spR = Spost.at(parity == RIGHT ? 2*ll+1 : 2*ll); // R subspace
            auto Hcur = MPOS(Hs.at(w+1).at(ll)); // tensor prod Hamiltonian 
            if(parity == LEFT) { spL.reverse(); spR.reverse(); Hcur.reverse(); }
 
            vector<int> localQNs;
            if(truncateQNs) {
                localQNs.resize(targetQNs.size());
                std::transform(targetQNs.begin(),targetQNs.end(),localQNs.begin(),
                               [&Hcur,&N](auto &val){ return divRoundClosest(val*length(Hcur),N); });
                }

            // STEP 1: find s lowest eigenpairs of restricted H
            auto tpH = tensorProdContract(spL,spR,Hcur);
            tensorProdH resH(tpH);
            resH.diag(localQNs,{"ExtDim",last?sLast:s,"QNSpread",last?0:qnSpread,"Iterative",doLanczos,
                                "ErrGoal",cutoff*(last?1:1e2),"MaxIter",500*s,"DebugLevel",0});

            // STEP 2: tensor viable sets on each side and reduce dimension
            MPVS ret(SiteSet(siteInds(Hcur)));
            tensorProduct(spL,spR,ret,resH.eigenvectors(),{"Cutoff",cutoff,"MaxDim",maxBd,"Move",!last});
            if(parity == LEFT) ret.reverse();
            Spre.push_back(std::move(ret));
            
            auto t2 = std::chrono::high_resolution_clock::now();
            auto tMerge = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            std::cout << "merge pair " << std::setw(2) << ll << ": "
                      << std::fixed << std::setprecision(0) << tMerge.count() << " s" << std::endl;
            }
        }
    auto tF = std::chrono::high_resolution_clock::now();
    auto tRRG = std::chrono::duration_cast<std::chrono::duration<double>>(tF - tI);
    std::cout << "rrg elapsed: " << std::fixed << std::setprecision(0) << tRRG.count() << " s" << std::endl;
    if(!verbose) std::cout.rdbuf(old_cout);

    return {Spre.at(0),tRRG.count()};
    }

pair<MPVS,double> rrg(AutoMPO const& autoH , MPO const& K , string const& blockStr , function<SiteSet(size_t)> const& hFunc , Args const& args) {
    auto args2 = args; args2.add("TruncateQNs",false);
    vector<int> dummyQNs;
    return rrg(autoH , K , blockStr , hFunc , dummyQNs , args2);
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

pair<ITensor,ITensor> tensorProdContract(MPVS const& psiL, MPVS const& psiR, MPO const& H_in) {
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
        auto ai = i < nL-1 ? commonIndex(ret(nL-i-1),ret(nL-i),"Link") : Index();
        T = i == 0 ? ret(nL-i)*W*ret(nL-i+1) : ret(nL-i)*ret(nL-i+1);
        U = ITensor(siteIndex(ret,nL-i),ai,ei);
        svd(T,U,S,V,{"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",true});
        ret.set(nL-i,U*S);
        ret.set(nL-i+1,V);
        if(!move) return;
        }

    ret.orthogonalize({"Cutoff",cutoff,"MaxDim",maxBd,"RespectDegenerate",true});
    return;
    }
