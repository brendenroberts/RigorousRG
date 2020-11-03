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

    fprintf(stdout,"dim H = %lu... \n",N);
    if(doI || N >= 15000) { // iterative diag
        if(N >= 15000 && !doI) fprintf(stderr,"H too large, iterative diag\n");
        
        auto iset = IndexSet(dag(ind.first),dag(ind.second)); 
	    auto ret = diagTen(*this,si,iset,args);
        for(int i : range(s)) {
            auto A = ret.at(i); // verbose way to do this, but there's a bug in setElt
            A *= setElt(dag(si(i+1)));
            evc += A;
            }
    } else { // dense matrix diag routine, limited to low dimensional local spaces
        evc = std::get<0>(diagPosSemiDef(-(ten.first*ten.second),{"MaxDim",s,"Tags","Ext"}));
        }
    evc.dag();

    return;
    }