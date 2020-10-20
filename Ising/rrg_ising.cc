#include "rrg.h"
#include "itensor/mps/sites/spinhalf.h"
#include <sys/file.h>
#include <chrono>
#include <thread>

int main(int argc, char *argv[]) {
    if(argc != 2) { fprintf(stderr,"usage: %s config_file\n",argv[0]); return 1; }
    std::map<string,string> inp;
    time_t t1,t2,tI,tF;
    
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
    const bool   doI = true; // diag restricted Hamiltonian iteratively?
    const auto   thr = 1e-14; // PCA threshold for sampled states

    // file output (TODO: change to C++ style io)
    FILE *gsfl;
    char id[256],gsnm[256];
    auto id_inp = inp.find("id");
    if(id_inp == inp.end()) sprintf(id,"rrg-L%lu-s%lu-D%lu",N,s,D);
    else sprintf(id,"%s",(*id_inp).second.c_str());
    strcpy(gsnm,id); strcat(gsnm,"_gs.dat");

    // initialize hierarchy structure, generate product basis for initial blocking
    // use Fermion Hilbert space for sites, since we only conserve parity
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
    time(&tI);
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
    time(&t2);
    fprintf(stdout,"initialization: %.f s\n",difftime(t2,tI));
 
    // ITERATION: proceed through RRG hierarchy, increasing the scale m
    vector<MPVS> Spost;
    //for(auto itH = Hs.begin() ; itH != Hs.end()-1 ; ++itH) {
    for(auto w : args(Hs)) {
        if(w == Hs.size()-1) break;
        fprintf(stdout,"Level %lu\n",w);
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
            time(&t1);
            restrictMPO(K,A,offset+1,D,pre.parity());
            time(&t2);
            fprintf(stdout,"trunc AGSP: %.f s ",difftime(t2,t1));
 
            // STEP 2: expand subspace using the mapping A:pre->ret
            time(&t1);
            ret = applyMPO(A,pre,{"Cutoff",eps,"MaxDim",MAXBD});
            time(&t2);
            fprintf(stdout,"apply AGSP: %.f s ",difftime(t2,t1));

            // rotate into principal components of subspace, possibly reducing dimension
            // and stabilizing numerics, then store subspace in eigenbasis of block H
            time(&t1);
            auto [U,Dg] = diagPosSemiDef(inner(ret,ret),{"Cutoff",thr,"Tags","Ext"});
            Dg.apply([](Real r) {return 1.0/sqrt(r);});
            ret.ref(xs) *= U*dag(Dg);
            auto [P,S] = diagPosSemiDef(-inner(ret,Hc,ret),{"Tags","Ext"});
            ret.ref(xs) *= P;
            //PrintData(inner(ret,ret));
            //ret.ref(xs) *= 1.0/sqrt(inner(ret,ret).real(ei(1),prime(ei)(1)));
            regauge(ret,xs,{"Cutoff",eps});

            time(&t2);
            fprintf(stdout,"rotate MPS: %.f s\n",difftime(t2,t1));

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
            time(&t1);
            auto tpH = tensorProdContract(spL,spR,Htp);
            tensorProdH resH(tpH);
            if(resH.size() < s) si = Index(resH.size()-1,"Ext");
            resH.diag(si,doI);
            time(&t2);
            fprintf(stdout,"(ll=%lu) diag restricted H: %.f s ",2*ll,difftime(t2,t1));

            // STEP 2: tensor viable sets on each side and reduce dimension
            MPVS ret(SiteSet(siteInds(Htp)) , ll%2 == 1 ? RIGHT : LEFT);
            time(&t1);
            tensorProduct(spL,spR,ret,resH.eigenvectors(),ll%2);
            time(&t2);
            fprintf(stdout,"tensor product: %.f s\n",difftime(t2,t1));

            Spre.push_back(ret);
            }
        }
    time(&t2);
    fflush(stdout);

    // EXIT: save numerical low-energy spectrum
    auto res = Spre[0];
    auto [P,S] = diagHermitian(-inner(res,H,res),{"Tags","Ext"});
    res.ref(N) *= P;
    auto ei = findIndex(res.ref(N),"Ext");

    vector<MPS> evecs(dim(ei));
    vector<double> evals;
    for(auto i : range1(dim(ei))) {
        auto fc = MPS(res);
        fc.ref(N) *= setElt(ei(i));
        fc.position(1);
        fc.orthogonalize({"Cutoff",1e-16,"RespectDegenerate",true,"MaxDim",MAXBD});
        fc.normalize();
        evecs.at(i-1) = fc;
        }
    for(auto const& psi : evecs) evals.push_back(inner(psi,H,psi));

    fprintf(stdout,"rrg elapsed: %.f s\ngs energy %17.14f\n",difftime(t2,tI),evals.front());
    fflush(stdout);

    gsfl = fopen_safe(gsnm);
    flock(fileno(gsfl),LOCK_EX);
    fprintf(gsfl,"# RRG spectrum (L=%lu s=%lu D=%lu)\n",N,s,D);
    for(auto const& v : evals) fprintf(gsfl,"%14.12f ",v);
    fprintf(gsfl,"\n");
    flock(fileno(gsfl),LOCK_UN);
    fclose(gsfl);

    return 0;
    
    }
