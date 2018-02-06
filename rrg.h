#ifndef RRG__H
#define RRG__H

#ifdef USE_ARPACK 
#include "arssym.h"
#undef F77NAME
#endif

#include "itensor/all_mps.h"
#include <vector>
#include <ctime>
#include <cmath>
#include <string>

#define LEFT   0
#define RIGHT  1
#define MAXBD  300
#define MAXDIM 20000

using namespace itensor;
using std::vector;
using std::min;
using std::max;

// "catchall" error threshold for most dangling-bond MPS/MPO operations
const Real eps = 1E-10;
// more sensitive threshold, usually for single MPS or MPO
const Real epx = 1E-12;

struct getReal {};
const auto invsqrt = [](Real r) { return 1.0/sqrt(r); };
inline vector<Real> doTask(getReal, Dense<Real> const& d) { return d.store; }
inline vector<Real> doTask(getReal, Diag<Real> const& d) { return d.allSame() ? vector<Real>({d.val}) : d.store; }
const auto Select = IndexType("Select");

// templated struct useful in merging step (L,R tensor product) 
template<typename T>
struct LRPair {
    T L,R;
    LRPair() {}
    LRPair(const T& LL , const T& RR): L(LL) , R(RR) {}
};
using IntPair = LRPair<int>;
using RealPair = LRPair<Real>;
using MPOPair = LRPair<MPO>;
using ITPair = LRPair<ITensor>;
using RVPair = LRPair<vector<Real> >;

// class used to interface with ARPACK++ or Davidson solver
class tensorProdH {
private:
    vector<ITPair>      pairH;
    vector<RVPair>      pHvec;
    vector<RealPair>    pHscl;
    ITensor C;

public:
    tensorProdH() { }
    tensorProdH(vector<ITPair> HH , ITensor CC)
        : pairH(HH) , pHvec() , pHscl() , C(CC) {
        pHvec.reserve(pairH.size());
        pHscl.reserve(pairH.size());
        for(auto& Hpair : pairH) {
            pHvec.push_back(RVPair(doTask(getReal{},Hpair.L.store()),
                                   doTask(getReal{},Hpair.R.store())));
            pHscl.push_back(RealPair(Hpair.L.scale().real(),
                                     Hpair.R.scale().real()));
            }
        }
    void product(const ITensor& A , ITensor& B) const;
    void MultMv(Real* v , Real* w);
    int size() const;
};

// struct used to build MPO version of AGSP
struct SiteITensor
{
    int i;
    ITensor A;
    SiteITensor() : i(0) {}
    SiteITensor(int ii , ITensor AA) : i(ii), A(AA) {}
};

// utility functions for printing matrices and vectors to stderr
inline void pvec(const double *vec, int n , int s) {
    for(int i = 0 ; i < n*s ; i+=s) fprintf(stderr,"%17.14f\n",vec[i]);
    }

inline void pmat(const double *mat, int n , int m , int ld) {
    for(int i = 0 ; i < n ; ++i) {
        for(int j = 0 ; j < m ; ++j)
            fprintf(stderr,"%9.7e ",mat[i*ld+j]);
        fprintf(stderr,"\n");
        }
    }

inline void pvec(const double *vec, int n) { pvec(vec,n,1); }
inline void pvec(const vector<Real>& vec, int n, int s) { pvec(&vec[0],n,s); }
inline void pvec(const vector<Real>& vec, int n) { pvec(&vec[0],n,1); }
inline void pmat(const double *mat, int n , int m) { pmat(mat,n,m,m); }
inline void pmat(const vector<Real>& mat, int n, int m) { pmat(&mat[0],n,m); }
inline void pmat(const vector<Real>& mat, int n, int m , int ld) { pmat(&mat[0],n,m,ld); }

// util.cpp
void reducedDM(const MPS& , MPO& ,int);

template<class MPSLike>
void regauge(MPSLike& , int , Real);

template<class MPSLike>
void regauge(MPSLike& , int);

Real measEE(const MPS& , int);

Real measOp(const MPS& , const ITensor& , int , const ITensor& , int);

Real measOp(const MPS& , const ITensor& , int);

void combineVectors(const vector<ITensor>& , ITensor&);

vector<Real> dmrgMPO(const MPO& , vector<MPS>& , int , double , double); 

void restrictMPO(const MPO& , MPO& , int , int , int);

template<class Tensor>
void applyMPO(MPSt<Tensor> const& , MPOt<Tensor> const& , MPSt<Tensor>& , int , Args const&);

template<class Tensor>
void applyMPO(MPOt<Tensor> const& , MPOt<Tensor> const& , MPOt<Tensor>& , int , Args const&);

ITensor overlapT(const MPS& , const MPO& , const MPS&);

ITensor overlapT(const MPS& , const MPS&);

void tensorProduct(const MPS& , const MPS& , MPS& , const ITensor& , int);

void combineMPS(vector<MPS>& , MPS& , int);

// trotter.cpp
void twoLocalTrotter(MPO& , double , int , AutoMPO& , Real);

// svdL.cpp
template<class Tensor>
Spectrum svdL(Tensor , Tensor& , Tensor& , Tensor& , Args args = Args::global());

#ifndef USE_ARPACK
// davidson.cpp
template <class BigMatrixT , class Tensor>
Real davidsonT(BigMatrixT const& , Tensor& , Args const& args = Args::global());

template <class BigMatrixT , class Tensor>
vector<Real> davidsonT(BigMatrixT const& , vector<Tensor>& phi , Args const& args = Args::global());
#endif

#endif
