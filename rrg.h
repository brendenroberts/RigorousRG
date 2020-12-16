#ifndef RRG_H
#define RRG_H

#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <map>
#include "itensor/util/print_macro.h"
#include "itensor/mps/autompo.h"
#include "itensor/mps/dmrg.h"

#define LEFT  0
#define RIGHT 1
#define MAX_BOND 2000
#define MAX_TEN_DIM 14000 // ~3GB at double precision
#define args(x) range(x.size())

using namespace itensor;
using std::vector;
using std::string;
using std::tuple;
using std::pair;

// error threshold for most dangling-bond operations
const double eps = 1E-10;
// more sensitive threshold for single MPS or MPO
const double epx = 1E-13;

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
    void position(int , Args const& = Args::global());
    MPVS& replaceSiteInds(IndexSet const&);
};

// class used for generalized ''dangling-bond'' MPO (matrix product operator space)
class MPOS : public MPO {
public:
    MPOS(MPO const& in) : MPO(in) {}
    MPOS(SiteSet const& sites) : MPO(sites) {}
    void reverse();
    void position(int , Args const& = Args::global());
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

// util.cc
Index extIndex(ITensor const& , string = "Ext");

template<class MPSLike>
tuple<Index,int> findExt(MPSLike const&);

void parse_config(std::ifstream& , std::map<string,string>&);

vector<vector<size_t> > block_sizes(string const&);

void init_H_blocks(AutoMPO const& , vector<MPO>& , vector<SiteSet> const&);

Index siteIndex(MPVS const&, int);

IndexSet siteInds(MPVS const&);

IndexSet siteInds(MPO const&);

ITensor inner(MPVS const& , MPO const& , MPVS const&);

ITensor inner(MPVS const& , MPVS const&);

template<class MPSLike>
void regauge(MPSLike& , int , Args const& = Args::global());
/*
Real cutEE(MPS const& , int);

Real mutualInfoTwoSite(MPS const& , int , int);
*/
void dmrgMPO(MPO const& , vector<pair<double,MPS> >& , int , Args const& = Args::global()); 

void Trotter(MPO& , double , size_t , AutoMPO&); 

void restrictMPO(MPO const& , MPOS& , int , int , int);

pair<ITensor,ITensor> tensorProdContract(MPVS const&, MPVS const&, MPO const&);

void tensorProduct(MPVS const& , MPVS const& , MPVS& , ITensor const& , int, bool = true);

MPVS applyMPO(MPOS const&, MPVS const&, Args const& = Args::global());

MPVS applyMPO(MPO const&, MPVS const&, Args const& = Args::global());

#endif
