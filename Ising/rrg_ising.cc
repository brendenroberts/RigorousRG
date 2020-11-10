#include "rrg.h"
#include "itensor/mps/sites/spinhalf.h"
#include <thread>

int main(int argc, char *argv[]) {
    if(argc != 2) { fprintf(stderr,"usage: %s config_file\n",argv[0]); return 1; }
    std::ostringstream ss;
    std::map<string,string> inp;
    
    std::ifstream cfile;
    cfile.open(argv[1],std::ios::in);
    if(cfile.is_open()) parse_config(cfile,inp);
    else { fprintf(stderr,"error opening config file\n"); return 1; }
    cfile.close();

    // RRG & AGSP parameters
    const size_t N = stoul(inp.at("N")); // system size
    const double t = stod(inp.at("t"));  // Trotter temperature
    const size_t M = stoul(inp.at("M")); // num Trotter steps
    const size_t s = stoul(inp.at("s")); // formal s param
    const size_t D = stoul(inp.at("D")); // formal D param
   
    // Hamitonian parameters
    const double J = stod(inp.at("J")); // Ising interaction strength
    const double g = stod(inp.at("g")); // transverse field strength
    const double h = stod(inp.at("h")); // longitudinal field strength
 
    // computational settings
    const bool   doI = false; // diag restricted Hamiltonian iteratively?
    const auto   thr = 1e-8; // PCA threshold for sampled states

    // output filenames
    auto inputId = inp.find("id");
    ss.setf(std::ios::fixed);
    ss.fill('0');
    if(inputId == inp.end())
        ss << argv[0] << "_N" << std::setw(3) << N << "_s" << std::setw(2) << s << "_D" << std::setw(2) << D;
    else ss << (*inputId).second;
    auto id = ss.str();
    std::ostringstream().swap(ss);

    // initialize hierarchy structure, generate product basis for initial blocking
    auto tI = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto blockNs = block_sizes(inp.at("n"));
    if(blockNs.back().back() != N) { Error("sum(n) not equal to N"); }
    vector<vector<SiteSet> > hsps;
    for(auto const& v : blockNs) {
        hsps.push_back(vector<SiteSet>());
        for(auto const & n : v) {
            SiteSet cur = SpinHalf(n,{"ConserveQNs",false});
            hsps.back().push_back(cur);
            }
        }
    
    // create MPO for H with open boundary conditions, also block Hamiltonians
    auto const& hs = hsps.back().back();
    AutoMPO autoH(hs);
    for(auto i = 1 ; static_cast<size_t>(i) <= N ; ++i) {
        if(static_cast<size_t>(i) != N)
            autoH += -J*4.0,"Sz",i,"Sz",i+1;
        autoH += -g*2.0,"Sx",i;
        autoH += -h*2.0,"Sz",i;
        }
    auto H = toMPO(autoH,{"Exact",true});
    vector<vector<MPO> > Hs(hsps.size());
    for(auto i : args(hsps)) init_H_blocks(autoH,Hs.at(i),hsps.at(i));

    // generate complete basis for exact diagonalization under initial blocking
    // TODO: this could probably be generic, moved to util.cc
    vector<MPVS> Spre;
    for(auto a : args(hsps.front())) {
        auto n = int(length(hsps.front().at(a)));
        auto p = int(pow(2,n));
        vector<MPS> V;
        for(int i : range(p)) {
            InitState istate(hsps.front().at(a),"Up");
            for(int j : range1(n))
                if(i/(int)pow(2,j-1)%2) istate.set(j,"Dn");
            auto st = MPS(istate);
            V.push_back(st);
            }
        
        Spre.push_back(MPVS(V,a%2==1?RIGHT:LEFT));
        }

    // generate AGSP thermal operator exp(-H/t) using Trotter
    MPO K(hs);
    Trotter(K,t,M,autoH);    
    K.ref(1) *= pow(2,N/2);
    fprintf(stderr,"maximum AGSP bond dim = %d\n",maxLinkDim(K));

    // INITIALIZATION: reduce dimension by sampling from initial basis
    for(auto ll : args(Spre)) {
        auto& cur = Spre.at(ll); 
        auto xs = cur.parity() == RIGHT ? 1 : length(cur);
       
        // return orthonormal basis of evecs
        auto [P,S] = diagPosSemiDef(-inner(cur,Hs.at(0).at(ll),cur),{"MaxDim",2*s,"Tags","Ext"});
        cur.ref(xs) *= P;
        regauge(cur,xs,{"Cutoff",1e-18});
        }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tInit = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    fprintf(stdout,"initialization: %.f s\n",tInit.count());
 
    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPVS> Spost;
    auto nLevels = blockNs.size();
    for(auto w  = 0u ; w < nLevels-1 ; ++w) {
        fprintf(stdout,"Level %u\n",w);
        const auto& Hl = Hs.at(w);
        int offset = 0;

        // EXPAND STEP: for each block, expand dimension of subspace with AGSP operators
        Spost.clear();
        for(auto ll : args(Hl)) {
            const auto& hs = hsps.at(w).at(ll);
            MPO  Hc = Hl.at(ll);
            MPVS pre = Spre.at(ll) , ret(hs);
            MPOS A(hs);
            auto xs = pre.parity() == RIGHT ? 1 : length(pre);

            // STEP 1: extract filtering operators A from AGSP K
            t1 = std::chrono::high_resolution_clock::now();
            restrictMPO(K,A,offset+1,D,pre.parity());
            t2 = std::chrono::high_resolution_clock::now();
            auto tRes = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            fprintf(stdout,"trunc AGSP: %.f s ",tRes.count());

            // STEP 2: expand subspace using the mapping A:pre->ret
            t1 = std::chrono::high_resolution_clock::now();
            ret = applyMPO(A,pre,{"Cutoff",eps,"MaxDim",MAXBD});
            t2 = std::chrono::high_resolution_clock::now();
            auto tApp = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            fprintf(stdout,"apply AGSP: %.f s ",tApp.count());

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H
            t1 = std::chrono::high_resolution_clock::now();
            auto [U,Dg] = diagPosSemiDef(inner(ret,ret),{"Cutoff",thr,"Tags","Ext"});
            Dg.apply([](Real r) {return 1.0/sqrt(r);});
            ret.ref(xs) *= U*dag(Dg);
            auto [P,S] = diagPosSemiDef(-inner(ret,Hc,ret),{"Tags","Ext"});
            ret.ref(xs) *= P;
            regauge(ret,xs,{"Cutoff",eps});
            t2 = std::chrono::high_resolution_clock::now();
            auto tRot = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            fprintf(stdout,"rotate MPS: %.f s\n",tRot.count());

            offset += length(pre);
            Spost.push_back(ret);
            }
        fflush(stdout);

        // MERGE/REDUCE STEP: construct tensor subspace, sample to reduce dimension
        Spre.clear();
        for(auto ll : range(Spost.size()/2)) {
            auto spL = Spost.at(2*ll);    // L subspace
            auto spR = Spost.at(2*ll+1);  // R subspace
            auto Htp = Hs.at(w+1).at(ll); // tensor prod Hamiltonian
            auto si = Index(s,"Ext");

            // STEP 1: find s lowest eigenpairs of restricted H
            t1 = std::chrono::high_resolution_clock::now();
            auto tpH = tensorProdContract(spL,spR,Htp);
            tensorProdH resH(tpH);
            if(resH.size() < s) si = Index(resH.size()-1,"Ext");
            resH.diag(si,{"Iterative",w == nLevels-2 ? false : doI,"ErrGoal",1e-8,"MaxIter",200*s});
            t2 = std::chrono::high_resolution_clock::now();
            auto tDiag = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            fprintf(stdout,"(ll=%lu) diag restricted H: %.f s ",2*ll,tDiag.count());

            // STEP 2: tensor viable sets on each side and reduce dimension
            t1 = std::chrono::high_resolution_clock::now();
            MPVS ret(SiteSet(siteInds(Htp)) , ll%2 == 1 ? RIGHT : LEFT);
            tensorProduct(spL,spR,ret,resH.eigenvectors(),ll%2,w!=nLevels-2);
            t2 = std::chrono::high_resolution_clock::now();
            auto tTens = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            fprintf(stdout,"tensor product: %.f s\n",tTens.count());

            Spre.push_back(ret);
            }
        }
    auto tF = std::chrono::high_resolution_clock::now();
    auto tRRG = std::chrono::duration_cast<std::chrono::duration<double>>(tF - tI);
    fprintf(stdout,"rrg elapsed: %.f s\n",tRRG.count());
    fflush(stdout);

    // CLEANUP: do rounds of TEBD in order to improve final states
    auto res = Spre[0];
    auto [ei,eSite] = findExt(res);
    auto e0 = 0.0 , e1 = 0.0;
    for(auto i : range(8)) {
        res = applyMPO(K,res,{"Cutoff",eps,"MaxDim",MAXBD});
        auto [U,Dg] = diagPosSemiDef(inner(res,res),{"Truncate",false,"Tags","Ext"});
        Dg.apply([](Real r) {return 1.0/sqrt(r);});
        res.ref(eSite) *= U*dag(Dg);
        res.ref(eSite).noPrime();
        auto [P,S] = diagPosSemiDef(-inner(res,H,res),{"Tags","Ext"});
        res.ref(eSite) *= P;
        ei = findIndex(res(eSite),"Ext");
        e0 = -S.elt(ei(1),prime(ei)(1)) , e1 = -S.elt(ei(2),prime(ei)(2));
        fprintf(stderr,"gs: %.8f gap: %.8f\n",e0,e1-e0);
        }

    // EXIT: write out spectral data, save low-energy states to disk
    ss << id << "_sites.dat";
    auto sitesFilename = ss.str();
    std::ostringstream().swap(ss);
    writeToFile(sitesFilename,hs);
    std::cout << "Low-energy spectrum:" << std::endl;
    for(auto i : range1(int(ei))) {
        auto fc = MPS(res);
        fc.ref(eSite) *= setElt(dag(ei)(i));
        fc.orthogonalize({"Cutoff",epx,"RespectDegenerate",true,"MaxDim",MAXBD});
        fc.normalize();
        ss.fill('0');
        std::cout << std::setprecision(10) << inner(fc,H,fc) << std::endl;
        ss << id << "_state" << std::setw(2) << i-1 << ".dat";
        auto stateFilename = ss.str();
        std::ostringstream().swap(ss);
        writeToFile(stateFilename,fc);
        }

    return 0;  
    }
