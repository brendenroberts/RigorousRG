#include "rrg.h"

void tensorProdH::product(ITensor const& A, ITensor& B) const {
    B = (ten.first*A*ten.second).noPrime("Ext");
    return;
    }

vector<ITensor> diagTen(tensorProdH const& H , Index si , IndexSet iset , Args const& args) {
    auto ret = vector<ITensor>(int(si));
 
    if(hasQNs(si)) {
        auto offset = 0;   
        for(auto q : range1(nblock(si))) {
            vector<ITensor> cur;
            for(auto i : range(blocksize(si,q)))
                cur.push_back(randomITensor(qn(si,q),iset));
            davidson(H,cur,args);
            std::swap_ranges(cur.begin(),cur.end(),ret.begin()+offset);
            offset += blocksize(si,q);
            }
    } else {
        for(auto i : range(int(si))) ret.at(i) = randomITensor(iset);
        davidson(H,ret,args);
        return ret;
        }
    
    return ret;
    }

void tensorProdH::diag(Index si , Args const& args) {
    auto N = size();
    auto s = int(si);
    auto doI = args.getBool("Iterative",true);
    auto nBlocks = hasQNs(si) ? static_cast<size_t>(nblock(si)) : 1lu;
    
    std::cout << "dim H = " << N << "..." << std::endl;
    if(doI || N >= MAX_TEN_DIM*nBlocks) { // iterative diag
        if(N >= MAX_TEN_DIM*nBlocks && !doI)
            std::cout << "H too large, iterative diag" << std::endl;
        
        auto iset = IndexSet(dag(ind.first),dag(ind.second)); 
	    auto ret = diagTen(*this,si,iset,args);
        for(int i : range(s)) {
            auto A = ret.at(i); // verbose way to do this, but there's a bug in setElt
            A *= setElt(dag(si(i+1)));
            evc += A;
            }
    } else { // dense matrix diag routine, limited to low dimensional local spaces
        auto bigTensor = -(ten.first*ten.second);
        evc = std::get<0>(diagPosSemiDef(bigTensor,{"Truncate",false,"Tags","Ext"}));
        auto eI = uniqueIndex(evc,bigTensor);
        auto A = ITensor(si,dag(eI));
        if(hasQNs(si)) {
            auto offsetS = 0;
            for(auto q : range1(nblock(si))) {
                auto offsetE = 0;
                for(auto r : range1(nblock(eI))) {
                    if(qn(si,q) == qn(eI,r))
                        for(auto i : range1(blocksize(si,q)))
                            A.set(si(offsetS+i),dag(eI)(offsetE+i),1.0);
                    offsetE += blocksize(eI,r);
                    }
                offsetS += blocksize(si,q);
                }
        } else
            for(auto i : range1(s))
                A.set(si(i),dag(eI)(i),1.0);

        evc *= A;
        }
    evc.dag();

    return;
    }
