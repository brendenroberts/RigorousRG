#include "rrg.h"

void tensorProdH::product(ITensor const& A, ITensor& B) const {
    B = (ten.first*A*ten.second).noPrime("Ext");
    return;
    }

vector<ITensor> diagTen(tensorProdH const& H , Index si , IndexSet iset , Args const& args) {
    auto ret = vector<ITensor>(dim(si));

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

void tensorProdH::diag(vector<int> const& localQNs , Args const& args) {
    const auto N = size();
    const auto s = args.getInt("ExtDim",1);
    const auto qnSpread = args.getInt("QNSpread",1);
    const auto doI = args.getBool("Iterative",true);
    auto ci = get<1>(combiner({ind.first,ind.second},{"Tags","Ext"})) , si = Index();

    if(hasQNs(ci)) {
        vector<std::pair<QN,long> > siQNs;
        for(auto q : range1(nBlocks(ci))) {
            auto qnCur = qn(ci,q);
            auto qnOK = true;
            for(auto i = 0lu ; i < localQNs.size() ; ++i)
                if(-qnCur.val(i+1) > localQNs.at(i)+qnSpread || -qnCur.val(i+1) < localQNs.at(i)-qnSpread) {
                    qnOK = false; break;
                    }
            if(qnOK) siQNs.push_back({qnCur,std::min(s,blocksize(ci,q))});
            }
        si = Index(std::move(siQNs),"Ext");    
    } else
        si = Index(s,"Ext");
    
    const auto blNum = nBlocks(si);

    std::cout << "dim H = " << N << "..." << std::endl;
    if(doI || N >= sqrt(blNum)*MAX_TEN_DIM) { // iterative diag
        if(N >= sqrt(blNum)*MAX_TEN_DIM && !doI)
            std::cout << "H too large, iterative diag" << std::endl;
        
        auto iset = IndexSet(dag(ind.first),dag(ind.second)); 
	    auto ret = diagTen(*this,si,iset,args);
        for(int i : range(dim(si))) {
            auto A = ret.at(i); // verbose way to do this, but there's a bug in setElt
            A *= setElt(dag(si(i+1)));
            evc += A;
            }
        evc.dag();
    } else { // dense matrix diag routine, limited to low dimensional local spaces
        auto bigTensor = -(ten.first*ten.second);
        evc = std::get<0>(diagPosSemiDef(bigTensor,{"Truncate",false,"Tags","Ext"}));
        auto eI = uniqueIndex(evc,bigTensor);
        auto A = ITensor(si,dag(eI));
        if(hasQNs(si)) {
            auto offsetS = 0;
            for(auto q : range1(nBlocks(si))) {
                auto offsetE = 0;
                for(auto r : range1(nBlocks(eI))) {
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

    return;
    }
