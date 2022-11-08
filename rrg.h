#ifndef RRG_H
#define RRG_H

#include "itensor/util/print_macro.h"
#include "itensor/mps/autompo.h"
#include "itensor/mps/dmrg.h"
#include <functional>
#include <map>

#define LEFT false
#define RIGHT true
#define MAX_BOND 3000lu
#define MAX_TEN_DIM 7500lu
#define args(x) range(x.size())

using namespace itensor;
using std::vector;
using std::string;
using std::function;
using std::pair;

// more sensitive threshold for single MPS or MPO
const double epx = 1E-14;

// class used for generalized "dangling-bond" MPS (matrix product vector space):
// typically the dangling bond is located at site 1 (indicated by RIGHT parity),
// and many MPVS subroutines expect this; member function reverse() can be used
// to flip the parity if the external index is located at site N (LEFT parity) 
class MPVS : public MPS {
protected:
    bool lr;

public:
    MPVS() {}
    MPVS(size_t N , bool dir = RIGHT) : MPS(N) , lr(dir) {}
    MPVS(SiteSet const& sites , bool dir = RIGHT) : MPS(sites) , lr(dir) {}
    MPVS(MPS const& in , bool dir = RIGHT) : MPS(in) , lr(dir) {}
    MPVS(vector<MPS> const& , bool = RIGHT);
    void position(int , Args const& = Args::global());
    MPVS& replaceSiteInds(IndexSet const&);
    bool parity() const { return lr; }
    void reverse();
};

// class used for generalized "dangling-bond" MPO (matrix product operator space):
// much less built in here, as MPOS in RRG typically have danglers at both ends;
// main functionality is to be able to call reverse() in order to match MPVS parity
class MPOS : public MPO {
public:
    MPOS(MPO const& in) : MPO(in) {}
    void reverse();
};

// interface for solver (ITensor Davidson algorithm), implemented in tensorProdH.cc
class tensorProdH {
using LRTen = pair<ITensor,ITensor>;
using LRInd = pair<Index,Index>;

protected:
    const LRTen ten;
    const LRInd ind;
    ITensor evc;

public:
    tensorProdH(LRTen HH) : ten(HH),ind({findIndex(HH.first,"Ext,0"),findIndex(HH.second,"Ext,0")}) { }
    void product(ITensor const& , ITensor&) const;
    void diag(vector<int> const& , Args const& = Args::global());
    void diag(Args const& = Args::global());
    size_t size() const { return ind.first.dim()*ind.second.dim(); }
    ITensor eigenvectors() const { return evc; }
};

// main loop implemented in rrg.cc
pair<MPVS,double> rrg(AutoMPO const& , MPO const& , string const& , function<SiteSet(size_t)> const& , vector<int> const& , Args const& = Args::global());

pair<MPVS,double> rrg(AutoMPO const& , MPO const& , string const& , function<SiteSet(size_t)> const& , Args const& = Args::global());

// auxiliary subroutines implemented in util.cc
Index extIndex(ITensor const& , string = "Ext");

template<class MPSLike>
pair<Index,size_t> findExt(MPSLike const&);

void parseConfig(std::ifstream& , std::map<string,string>&);

IndexSet siteInds(MPVS const&);

IndexSet siteInds(MPO const&);

ITensor inner(MPVS const& , MPO const& , MPVS const&);

ITensor inner(MPVS const& , MPO const& , MPVS const& , size_t);

void dmrgMPO(MPO const& , vector<pair<double,MPS> >& , int , Args const& = Args::global()); 

MPO Trotter(double , size_t , AutoMPO const& , double); 

MPVS applyMPO(MPO const&, MPVS const&, Args = Args::global());

// some one-liners
inline int divRoundClosest(const int n, const int d) { return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d); }

inline size_t nBlocks(Index const& index) { return std::max(static_cast<size_t>(nblock(index)),1lu); }

inline Index siteIndex(MPVS const& psi, int j) { return findIndex(psi(j),"Site"); }

inline ITensor inner(MPVS const& phi, MPVS const& psi) { return inner(phi,MPO(siteInds(phi)),psi); }

inline ITensor inner(MPVS const& phi, MPVS const& psi , size_t eSite) { return inner(phi,MPO(siteInds(phi)),psi,eSite); }

#endif
