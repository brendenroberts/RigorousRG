#include "rrg.h"
#include "itensor/mps/sites/fermion.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char *argv[]) {
    if(argc != 2) { std::cerr << "usage: " << argv[0] << " config_file" << std::endl; return 1; }
    std::ostringstream ss;
    std::map<string,string> configParams;
    std::ofstream logFile;
    std::cout.fill('0');
    
    std::ifstream configFile;
    configFile.open(argv[1]);
    if(configFile.is_open()) parseConfig(configFile,configParams);
    else { std::cerr << "error opening config file" << std::endl; return 1; }
    configFile.close();

    // RRG & AGSP parameters
    const size_t N   = stoul(configParams.at("N")); // system size
    const double t   = stod(configParams.at("t"));  // AGSP temperature
    const size_t M   = stoul(configParams.at("M")); // num Trotter steps
    const size_t s   = stoul(configParams.at("s")); // formal s param
    const size_t D   = stoul(configParams.at("D")); // formal D param
    const double eps = stod(configParams.at("eps")); // MPVS error tolerance
    
    // computational settings
    const bool doI = true; // diag restricted Hamiltonian iteratively?
    const auto thr = epx;  // PCA threshold for sampled states

    // generate Hamiltonian couplings using C++ random library
    std::random_device rd;
    const unsigned long seed = (configParams.find("seed") == configParams.end() ? rd() : stoul(configParams.at("seed")));
    std::linear_congruential_engine<unsigned long, 6364136223846793005ul, 1442695040888963407ul, 0ul> gen(seed);
    std::uniform_real_distribution<double> udist(0.0,1.0);
    
    // IO stream stuff for setting up output filenames
    auto configId = configParams.find("id");
    ss.setf(std::ios::fixed);
    ss.fill('0');
    if(configId == configParams.end())
        ss << argv[0] << "_N" << std::setw(3) << N << "_s" << std::setw(2) << s << "_D" << std::setw(2) << D;
    else ss << (*configId).second;
    auto dbFilename = ss.str();
    ss << "_seed" << std::setw(10) << seed;
    auto id = ss.str();
    if(configParams.find("log") != configParams.end())
        if(auto doLog = configParams.at("log") ; doLog == "true" || doLog == "True" || doLog == "1") {
            ss << ".log";
            auto logFilename = ss.str();
            std::cout << "writing output to " << logFilename << std::endl;
            logFile.open(logFilename);
            std::cout.rdbuf(logFile.rdbuf()); // redirect cout to log file buffer
            }
    std::ostringstream().swap(ss);
    std::cout << "seed is " << seed << std::endl;

    vector<double> J(2*(N-1));
    const double Gamma = stod(configParams.at("Gamma")); // disorder strength
    for(auto i = 0u ; i < N-1 ; ++i) {
        J.at(2*i+0) = pow(udist(gen),Gamma);
        J.at(2*i+1) = pow(udist(gen),Gamma);
        }

    // initialize hierarchy structure, generate product basis for initial blocking
    // use Fermion local Hilbert space, which is natural to conserve parity
    auto tI = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto blockNs = parseBlockSizes(configParams.at("n"));
    if(blockNs.back().back() != N) { std::cerr << "sum(n) not equal to N" << std::endl; return 1; }
    vector<vector<SiteSet> > hsps;
    for(auto const& v : blockNs) {
        hsps.push_back(vector<SiteSet>());
        for(auto const & n : v) {
            SiteSet cur = Fermion(n,{"ConserveQNs",true,"ConserveNf",false});
            hsps.back().push_back(cur);
            }
        }

    // create MPO for H with open boundary conditions, also block Hamiltonians
    auto const& hs = hsps.back().back();
    AutoMPO autoH(hs);
    for(auto i = 0 ; static_cast<size_t>(i) < N-1 ; ++i) {
        autoH += (J.at(2*i+0)+J.at(2*i+1)),"Cdag",i+1,   "C",i+2;
        autoH += (J.at(2*i+0)+J.at(2*i+1)),"Cdag",i+2,   "C",i+1;
        autoH += (J.at(2*i+0)-J.at(2*i+1)),"Cdag",i+1,"Cdag",i+2;
        autoH += (J.at(2*i+0)-J.at(2*i+1)),   "C",i+2,   "C",i+1;
        }
    auto H = toMPO(autoH,{"Exact",true});
    vector<vector<MPO> > Hs(hsps.size());
    for(auto i : args(hsps)) blockHs(Hs.at(i),autoH,hsps.at(i));

    // generate complete basis for exact diagonalization under initial blocking
    vector<MPVS> Spre;
    for(auto a : args(hsps.front())) {
        auto n = length(hsps.front().at(a));
        auto p = static_cast<int>(pow(2,n));
        vector<MPS> V;
        for(auto i : range(p)) {
            InitState istate(hsps.front().at(a),"Emp");
            for(auto j : range1(n))
                if(i/static_cast<int>(pow(2,j-1))%2 == 1) istate.set(j,"Occ");
            auto st = MPS(istate);
            V.push_back(st);
            }
        
        Spre.push_back(MPVS(V,a%2 ? RIGHT : LEFT));
        }

    // generate AGSP thermal operator exp(-H/t) using Trotter
    auto K = Trotter(t,M,autoH,eps);
    std::cout << "maximum AGSP bond dim = " << maxLinkDim(K) << std::endl;
    
    // INITIALIZATION: reduce dimension by sampling from initial basis
    for(auto ll : args(Spre)) {
        auto& pcur = Spre.at(ll);
        auto Hcur = MPOS(Hs.at(0).at(ll));
        auto parity = pcur.parity();
        if(parity == LEFT) { pcur.reverse(); Hcur.reverse(); }

        // return orthonormal basis of evecs
        auto [P,S] = diagPosSemiDef(-inner(pcur,Hcur,pcur),{"MaxDim",2*s,"Tags","Ext"});
        pcur.ref(1) *= P;
        pcur.orthogonalize({"Cutoff",eps,"MaxDim",MAX_BOND,"RespectDegenerate",true});
        if(parity == LEFT) pcur.reverse();
        }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tInit = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "initialization: " << std::fixed <<std::setprecision(0) << tInit.count() << " s" << std::endl;
 
    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPVS> Spost;
    auto nLevels = blockNs.size();
    for(auto w  = 0u ; w < nLevels-1 ; ++w) {
        std::cout << "Level " << w << std::endl;
        const auto& Hl = Hs.at(w);
        auto offset = 0u;

       // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        Spost.clear();
        for(auto ll : args(Hl)) {
            t1 = std::chrono::high_resolution_clock::now();
            const auto& hs = hsps.at(w).at(ll);
            auto Hc = MPOS(Hl.at(ll));
            auto pre = Spre.at(ll) , ret = MPVS(hs);
            auto parity = pre.parity();
            MPOS A(hs);

            // STEP 1: extract filtering operators A from AGSP K
            sliceMPO(K,A,offset+1,D);
            if(parity == LEFT) { A.reverse(); pre.reverse(); Hc.reverse(); }
            A.orthogonalize({"Cutoff",epx,"MaxDim",MAX_BOND,"RespectDegenerate",true});

            // STEP 2: expand subspace using the mapping A:pre->ret
            ret = applyMPO(A,pre,{"Cutoff",eps,"MaxDim",MAX_BOND,"Nsweep",16,"UseSVD",true});
            ret.noPrime();

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H
            auto [U,Dg] = diagPosSemiDef(inner(ret,ret),{"Cutoff",thr,"Tags","Ext"});
            Dg.apply([](Real r) {return 1.0/sqrt(r);});
            ret.ref(1) *= U*dag(Dg);
            auto [P,S] = diagPosSemiDef(-inner(ret,Hc,ret),{"Tags","Ext"});
            ret.ref(1) *= P;
            ret.orthogonalize({"Cutoff",eps,"MaxDim",MAX_BOND,"RespectDegenerate",true});

            if(parity == LEFT) { ret.reverse(); }
            Spost.push_back(std::move(ret));
            offset += length(pre);

            t2 = std::chrono::high_resolution_clock::now();
            auto tExpand = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            std::cout << "expand block " << std::setw(2) << ll << ": "
                      << std::fixed << std::setprecision(0) << tExpand.count() << " s" << std::endl;
            }

        // MERGE/REDUCE STEP: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(auto ll : range(Spost.size()/2)) {
            t1 = std::chrono::high_resolution_clock::now();
            auto parity = ll%2 ? RIGHT : LEFT , last = Spost.size() == 2;
            auto spL = Spost.at(parity == RIGHT ? 2*ll : 2*ll+1); // L subspace
            auto spR = Spost.at(parity == RIGHT ? 2*ll+1 : 2*ll); // R subspace
            auto Hcur = MPOS(Hs.at(w+1).at(ll)); // tensor prod Hamiltonian
            auto si = Index(QN({"Pf",0,-2}),last?2:s,QN({"Pf",1,-2}),last?2:s,"Ext");
            if(parity == LEFT) { spL.reverse(); spR.reverse(); Hcur.reverse(); }

            // STEP 1: find s lowest eigenpairs of restricted H
            auto tpH = tensorProdContract(spL,spR,Hcur);
            tensorProdH resH(tpH);
            resH.diag(si,{"Iterative",doI,"ErrGoal",eps,"MaxIter",500*s,"DebugLevel",0});

            // STEP 2: tensor viable sets on each side and reduce dimension
            MPVS ret(SiteSet(siteInds(Hcur)));
            tensorProduct(spL,spR,ret,resH.eigenvectors(),eps,!last);
            if(parity == LEFT) ret.reverse();
            Spre.push_back(std::move(ret));

            t2 = std::chrono::high_resolution_clock::now();
            auto tMerge = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            std::cout << "merge pair " << std::setw(2) << ll << ": "
                      << std::fixed << std::setprecision(0) << tMerge.count() << " s" << std::endl;
            }
        }
    auto tF = std::chrono::high_resolution_clock::now();
    auto tRRG = std::chrono::duration_cast<std::chrono::duration<double>>(tF - tI);
    std::cout << "rrg elapsed: " << std::fixed << std::setprecision(0) << tRRG.count() << " s" << std::endl;

    // CLEANUP: do some rounds of TEBD, followed by DMRG
    auto res = Spre[0];
    auto [extIndex,eSite] = findExt(res);
    auto nTEBD = 10lu , tSteps = M/2 , nDMRG = 2*N; 
    auto eGS = 0.0 , minGap = 0.0 , convGS = 1.0;
    auto eH = toExpH(autoH,1.0/(N/2*tSteps),{"Cutoff",epx});

    for(auto j : range1(N-1)) std::cout << rightLinkIndex(res,j).dim() << " ";
    std::cout << std::endl;
 
    auto [U,Dg] = diagPosSemiDef(inner(res,res),{"Truncate",false,"Tags","Ext"});
    Dg.apply([](Real r) {return 1.0/sqrt(r);});
    res.ref(eSite) *= U*dag(Dg);
    res.ref(eSite).noPrime();
    auto [P,S] = diagPosSemiDef(-inner(res,H,res),{"Tags","Ext"});
    res.ref(eSite) *= P;
    extIndex = findIndex(res(eSite),"Ext");
    
    eGS = -S.elt(1,1);
    std::cout << std::fixed << std::setprecision(14) << eGS << " (q:" << 0 << " gap:"
              << std::scientific << std::setprecision(4) << -S.elt(2,2)-eGS << ") ";
    for(auto q : range(nblock(extIndex)))
        if(q != 0)
            std::cout << std::scientific << std::setprecision(6) << -S.elt(2*q+1,2*q+1)-eGS << " (q:" << q << " gap:"
                      << std::scientific << std::setprecision(4) << -S.elt(2*q+2,2*q+2)+S.elt(2*q+1,2*q+1) << ") ";
    std::cout << std::endl;

    std::cout << "TEBD steps:" << std::endl; 
    for(auto i : range(nTEBD)) {
        for(auto mval : range(tSteps)) {
            res = applyMPO(eH,res,{"Cutoff",eps,"MaxDim",MAX_BOND,"DoApprox",false});
            res.noPrime();
            }
        auto [U,Dg] = diagPosSemiDef(inner(res,res),{"Truncate",false,"Tags","Ext"});
        Dg.apply([](Real r) {return 1.0/sqrt(r);});
        res.ref(eSite) *= U*dag(Dg);
        res.ref(eSite).noPrime();
        auto [P,S] = diagPosSemiDef(-inner(res,H,res),{"Tags","Ext"});
        res.ref(eSite) *= P;
        extIndex = findIndex(res(eSite),"Ext");
        
        eGS = -S.elt(1,1);
        std::cout << std::fixed << std::setprecision(14) << eGS << " (q:" << 0 << " gap:"
                  << std::scientific << std::setprecision(4) << -S.elt(2,2)-eGS << ") ";
        for(auto q : range(nblock(extIndex)))
            if(q != 0)
                std::cout << std::scientific << std::setprecision(6) << -S.elt(2*q+1,2*q+1)-eGS << " (q:" << q << " gap:"
                          << std::scientific << std::setprecision(4) << -S.elt(2*q+2,2*q+2)+S.elt(2*q+1,2*q+1) << ") ";
        std::cout << std::endl;
        }

    using ePair = std::pair<double,MPS>;
    vector<vector<ePair> > eigenspace;
    for(int q : range(nblock(extIndex))) {
        eigenspace.push_back(vector<ePair>());
        for(auto i : range1(2)) {
            auto fc = MPS(res);
            fc.ref(eSite) *= setElt(dag(extIndex)(2*q+i));
            fc.orthogonalize({"Cutoff",epx,"RespectDegenerate",true});
            fc.normalize();
            eigenspace.back().push_back({inner(fc,H,fc),fc});
            }
        }

    auto count = 0u;
    std::cout << "DMRG steps:" << std::endl;
    vector<double> ePrev(eigenspace.size());
    do {
        for(auto q : args(eigenspace)) {
            auto& sector = eigenspace.at(q);
            ePrev.at(q) = sector.front().first;
            dmrgMPO(H,sector,6,{"Cutoff",epx});
            std::sort(sector.begin(),sector.end(),[](ePair const& a , ePair const& b) {return a.first < b.first;});
            }
        ++count;
        convGS = std::inner_product(eigenspace.begin(),eigenspace.end(),ePrev.begin(),0.0,
                                    [](double const& a , double const& b) { return std::max(a,b); }, // std::max<double> overloaded
                                    [](vector<ePair> const& a , double const& b) { return abs(b - a.front().first); });

        eGS = eigenspace.at(0).front().first;
        std::cout << std::fixed << std::setprecision(14) << eGS << " (q:" << 0 << " gap:"
                  << std::scientific << std::setprecision(4) << eigenspace.at(0).at(1).first-eGS << " m:"
                  << maxLinkDim(eigenspace.at(0).front().second) << ") ";
        minGap = eigenspace.at(0).at(1).first-eGS;
        for(auto q : args(eigenspace))
            if(q != 0) {
                minGap = std::min(minGap,std::min(eigenspace.at(q).front().first-eGS,eigenspace.at(q).at(1).first-eigenspace.at(q).front().first));
                std::cout << std::scientific << std::setprecision(6) << eigenspace.at(q).front().first-eGS << " (q:" << q << " gap:"
                          << std::scientific << std::setprecision(4) << eigenspace.at(q).at(1).first-eigenspace.at(q).front().first << " m:"
                          << maxLinkDim(eigenspace.at(q).front().second) << ") ";
                }
        std::cout << std::endl;
    } while(convGS > std::max(epx*N/4,std::min(eps,minGap*1e-2)) && count < nDMRG);
    if(logFile.is_open()) logFile.close();

    // EXIT: write out spectral data, save ground state in each sector to disk
    std::ostringstream().swap(ss);
    ss << id << "_sites.dat";
    auto sitesFilename = ss.str();
    writeToFile(sitesFilename,hs);
    std::ostringstream().swap(ss);

    std::ostringstream dbEntry;
    dbEntry.setf(std::ios::fixed);
    dbEntry.fill('0');
    for(auto q : args(eigenspace)) {
        dbEntry << "# N G s D t M seed conv q E0 ..." << std::endl;
        dbEntry << std::setw(2) << N << " " << std::setprecision(2) << Gamma << " " << std::setw(2) << s
                << " " << std::setw(2) << D << " " << std::setprecision(3) << t << " " << std::setw(4) << M
                << " " << std::setw(10) << seed << " " << std::setw(1) << (count == nDMRG ? 0 : 1)
                << " " << std::setw(1) << q << " ";
        auto const& sector = eigenspace.at(q);
        for(auto j : range(2)) {
            dbEntry << std::setprecision(16) << sector.at(j).first << " ";

            std::ostringstream().swap(ss);
            ss << id << "_q" << q << "_e" << j << ".dat";
            auto stateFilename = ss.str();
            writeToFile(stateFilename,sector.at(j).second);
            }
        dbEntry << std::endl;
        }
    
    // Minimize amount of time spent with db file open (but don't bother with locking)
    std::ofstream dbFile(dbFilename,std::fstream::app);
    dbFile << dbEntry.str();
    dbFile.flush();
    dbFile.close();

    return 0;
    }
