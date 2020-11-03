#ifndef RRG_H
#define RRG_H

#include "itensor/util/print_macro.h"
#include "itensor/mps/autompo.h"
#include "itensor/mps/dmrg.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <map>

#define LEFT  0
#define RIGHT 1
#define MAXBD 800
#define args(x) range(x.size())

using namespace itensor;
using std::vector;
using std::string;
using std::pair;

// error threshold for most dangling-bond operations
const Real eps = 1E-10;
// more sensitive threshold for single MPS or MPO
const Real epx = 1E-14;

inline Index extIndex(ITensor const& A , string tag = "Ext") {
    if(hasQNs(A))
        return Index(QN({-div(A)}),1,tag);
    else
        return Index(1,tag);
    }

// class used for generalized ''dangling-bond'' MPS (matrix product vector space)
class MPVS : public MPS {
protected:
    int lr;
public:
    MPVS() {}
    MPVS(int N , int dir = RIGHT) : MPS(N) , lr(dir) {}
    MPVS(SiteSet const& sites , int dir = RIGHT) : MPS(sites) , lr(dir) {}
    MPVS(InitState const& initState , int dir = RIGHT) : MPS(initState) , lr(dir) {}
    MPVS(MPS const& in , int dir = RIGHT) : MPS(in) , lr(dir) {}
    MPVS(vector<MPS> const& , int dir = RIGHT);
    int parity() const { return lr; }
    void reverse();
};

// class used for generalized ''dangling-bond'' MPO (matrix product operator space)
class MPOS : public MPO {
public:
    MPOS(MPO const& in) : MPO(in) {}
    MPOS(SiteSet const& sites) : MPO(sites) {}
    void reverse();
};

// class used as interface for iterative solver
class tensorProdH {
using LRTen = pair<ITensor,ITensor>;
using LRInd = pair<Index,Index>;

protected:
    const LRTen ten;
    const LRInd ind;
    ITensor evc;

public:
    tensorProdH(LRTen& HH) : ten(HH),ind(std::pair(findIndex(HH.first, "Ext,0"),
                                                   findIndex(HH.second,"Ext,0"))) { }
    void product(ITensor const& , ITensor&) const;
    void diag(Index , Args const& = Args::global());
    long unsigned int size() const { return int(ind.first)*int(ind.second); }
    ITensor eigenvectors() const { return evc; }
};

namespace itensor {
void plussers(Index const& , Index const& , Index& , ITensor& , ITensor&);
}

inline FILE* fopen_safe(const char name[]) {
    FILE *fl;
    for(int count = 0 ; !(fl = fopen(name,"a")) && count < 100 ; ++count) {
        fprintf(stderr,"unable to open output file %s, retrying...\n",name);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

    return fl;
    }

// util.cc
void parse_config(std::ifstream& , std::map<string,string>&);

vector<vector<size_t> > block_sizes(string const&);

//template<class Sites>
void init_H_blocks(AutoMPO const& , vector<MPO>& , vector<SiteSet> const&);

Index siteIndex(MPVS const&, int);

IndexSet siteInds(MPVS const&);

IndexSet siteInds(MPO const&);

MPO sysOp(SiteSet const&, const char*, const Real = 1.);

ITensor inner(MPVS const& , MPO const& , MPVS const&);

ITensor inner(MPVS const& , MPVS const&);

template<class MPSLike>
void regauge(MPSLike& , int , Args const& = Args::global());

Real cutEE(MPS const& , int);

Real mutualInfoTwoSite(MPS const& , int , int);
/*
template<class Tensor>
Real measOp(const MPSt<Tensor>& , const ITensor& , int , const ITensor& , int);

template<class Tensor>
Real measOp(const MPSt<Tensor>& , const ITensor& , int);
*/
vector<Real> dmrgMPO(MPO const& , vector<MPS>& , int , Args const& = Args::global()); 

void Trotter(MPO& , double , size_t , AutoMPO&); 

double restrictMPO(MPO const& , MPOS& , int , int , int);

pair<ITensor,ITensor> tensorProdContract(MPVS const&, MPVS const&, MPO const&);

double tensorProduct(MPVS const& , MPVS const& , MPVS& , ITensor const& , int);

MPVS applyMPO(MPOS const&, MPVS const&, Args = Args::global());

#endif
