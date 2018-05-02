#ifndef RRG__H
#define RRG__H

#ifdef USE_ARPACK 
#include "arssym.h"
#undef F77NAME
#endif

#include "itensor/mps/sites/spinhalf.h"
#include "itensor/mps/autompo.h"
#include "itensor/mps/dmrg.h"
#include "itensor/util/print_macro.h"
#include <vector>
#include <ctime>
#include <cmath>
#include <string>

#define LEFT   0
#define RIGHT  1
#define MAXBD  800
#define args(x) range(x.size())

using namespace itensor;
using std::vector;
using std::min;
using std::max;

// "catchall" error threshold for most dangling-bond MPS/MPO operations
const Real eps = 1E-10;
// more sensitive threshold for single MPS or MPO
const Real epx = 1E-14;

const auto Select = IndexType("Select");
struct getReal {};
const auto invsqrt = [](Real r) { return 1.0/sqrt(r); };
inline vector<Real> doTask(getReal, Dense<Real> const& d) { return d.store; }
inline vector<Real> doTask(getReal, Diag<Real> const& d) {
    return d.allSame() ? vector<Real>({d.val}) : d.store;
    }

// general-purpose struct useful in merging step (L,R tensor product) 
template<typename T>
struct LRPair {
    T L,R;
    LRPair() {}
    LRPair(const T& LL , const T& RR): L(LL) , R(RR) {}
};

// class used to interface with ARPACK++ or Davidson solver
template<class Tensor>
class tensorProdH {
private:
    LRPair<Tensor>        ten;
    LRPair<vector<Real> > dat;
    LRPair<Real>          scl;
    LRPair<int>           dim;

public:
    tensorProdH(LRPair<Tensor> HH) : ten(HH) {
        dat = LRPair<vector<Real> >(doTask(getReal{},ten.L.store()),doTask(getReal{},ten.R.store()));
        scl = LRPair<Real>(ten.L.scale().real(),ten.R.scale().real());
        dim = LRPair<int>(int(findtype(ten.L,Select)),int(findtype(ten.R,Select)));
        }
    void product(const Tensor& A , Tensor& B) const;
    void MultMv(Real* v , Real* w);
    int size() const;
};

// utility functions for printing matrices and vectors to stderr
inline void pvec(const double *vec , int n , int s = 1) {
    for(int i = 0 ; i < n*s ; i+=s) fprintf(stderr,"%17.14f\n",vec[i]);
    }

inline void pmat(const double *mat , int n , int m , int ld = 0) {
    if(ld == 0) ld = m;
    for(auto i : range(n)) {
        for(auto j : range(m)) fprintf(stderr,"%7.5f ",mat[i*ld+j]);
        fprintf(stderr,"\n");
        }
    }

inline void pvec(const vector<Real>& vec , int n , int s = 1) { pvec(&vec[0],n,s); }
inline void pmat(const vector<Real>& mat , int n , int m , int ld = 0) { pmat(&mat[0],n,m,ld); }

// util.cpp
void reducedDM(const MPS& , MPO& , int);

template<class Tensor>
Tensor overlapT(const MPSt<Tensor>& , const MPOt<Tensor>& , const MPSt<Tensor>&);

template<class Tensor>
Tensor overlapT(const MPSt<Tensor>& , const MPSt<Tensor>&);

template<class MPSLike>
void regauge(MPSLike& , int , Args const&);

template<class Tensor>
Real measEE(const MPSt<Tensor>& , int);

IQMPO sysOp(const SiteSet& , const char* , const Real = 1.0);

template<class Tensor>
Real measOp(const MPSt<Tensor>& , const IQTensor& , int , const IQTensor& , int);

template<class Tensor>
Real measOp(const MPSt<Tensor>& , const IQTensor& , int);

template<class Tensor>
void extractBlocks(AutoMPO const& , vector<MPOt<Tensor> >& ,  const SiteSet&);

template<class Tensor>
MPSt<Tensor> opFilter(MPSt<Tensor> const& , vector<MPOt<Tensor> > const& , Real , int);

template<class Tensor>
vector<Real> dmrgMPO(const MPOt<Tensor>& , vector<MPSt<Tensor> >& , int , Args const& = Args::global()); 

template<class MPOType>
void twoLocalTrotter(MPOType& , double , int , AutoMPO&); 

template<class Tensor>
double restrictMPO(const MPOt<Tensor>& , MPOt<Tensor>& , int , int , int);

template<class Tensor>
MPSt<Tensor> applyMPO(MPOt<Tensor> const& , MPSt<Tensor> const& , int , Args const& = Args::global());

template<class Tensor>
LRPair<Tensor> tensorProdContract(MPSt<Tensor> const&, MPSt<Tensor> const&, MPOt<Tensor> const&);

template<class Tensor>
double tensorProduct(const MPSt<Tensor>& , const MPSt<Tensor>& , MPSt<Tensor>& , const Tensor& , int);

template<class Tensor>
double combineMPS(const vector<MPSt<Tensor> >& , MPSt<Tensor>& , int);

// svdL.cpp
template<class Tensor>
Spectrum svdL(Tensor , Tensor& , Tensor& , Tensor& , Args = Args::global());

#ifndef USE_ARPACK
// davidson.cpp
template <class BigMatrixT , class Tensor>
Real davidsonT(BigMatrixT const& , Tensor& , Args const& = Args::global());

template <class BigMatrixT , class Tensor>
vector<Real> davidsonT(BigMatrixT const& , vector<Tensor>& phi , Args const& args = Args::global());
#endif

#endif
