#include "rrg.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

MPVS rrg(vector<MPVS>& Spre , MPO const& K , vector<vector<MPO> > const& Hs , vector<int> const& targetQNs , Args const& args) {
    const auto N = length(Hs.back().front());
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

            // STEP 2: expand subspace using the mapping A:pre->ret
            ret = applyMPO(A,pre,{"Cutoff",cutoff,"MaxDim",maxBd,"Nsweep",nSweep,"UseSVD",true});
            ret.noPrime();

            // rotate into principal components of subspace, possibly reducing dimension
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

    if(!verbose) std::cout.rdbuf(old_cout);

    return Spre.at(0);
    }

MPVS rrg(vector<MPVS>& Spre , MPO const& K , vector<vector<MPO> > const& Hs , Args const& args) {
    auto args2 = args; args2.add("TruncateQNs",false);
    vector<int> dummyQNs;
    return rrg(Spre , K , Hs , dummyQNs , args2);
    }
