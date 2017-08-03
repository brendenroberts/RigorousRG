#include "itensor/all.h"
#include "itensor/mps/mpo.h"
#include <ctime>
#include <cmath>
#include <vector>
#include <string>

#define LEFT  0
#define RIGHT 1
#define MAXBD 1000

using namespace itensor;
using std::vector;
using std::min;
using std::max;

const Real eps = 1E-10;
const auto Select = IndexType("Select");
const auto invsqrt = [](Real r) { return 1.0/sqrt(r); };
const auto getdata = [](Dense<Real> const& d) { return d.store; };

// templated struct used for boundary terms from Hamiltonian 
template<typename T>
struct LRPair
{
    T L,R;
    LRPair() {}
    LRPair(const T& LL , const T& RR): L(LL) , R(RR) {}
};
using MPOPair = LRPair<MPO>;
using ITensorPair = LRPair<ITensor>;

// class used to interface with ARPACK++
class tensorProdH {
    vector<ITensorPair> pairedH;
    ITensor C;

public:
    tensorProdH() { }
    tensorProdH(vector<ITensorPair> HH , ITensor CC)
        : pairedH(HH) , C(CC) { }
    virtual void product(const ITensor& A , ITensor& B) const;
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
inline void pvec(const double *vec, int n) {
    for(int i = 0 ; i < n ; ++i) fprintf(stderr,"%17.14f\n",vec[i]);
    }

inline void pmat(const double *mat, int n , int m , int ld) {
    for(int i = 0 ; i < n ; ++i) {
        for(int j = 0 ; j < m ; ++j)
            fprintf(stderr,"%.8e ",mat[i*ld+j]);
        fprintf(stderr,"\n");
        }
    }

inline void pmat(const double *mat, int n , int m) { pmat(mat,n,m,m); }
inline void pvec(const vector<Real>& vec, int n) { pvec(&vec[0],n); }
inline void pmat(const vector<Real>& mat, int n , int m) { pmat(&mat[0],n,m); }
inline void pmat(const vector<Real>& mat, int n , int m , int ld) { pmat(&mat[0],n,m,ld); }

// util.cpp
void reducedDM(const MPS& , MPO& ,int);

template<class MPSLike>
void regauge(MPSLike& , int);

void combineVectors(const vector<ITensor>& , ITensor&);

vector<Real> dmrgMPO(const MPO& , vector<MPS>& , double , double); 

vector<Real> dmrgMPO(const MPO& , vector<MPS>& , double); 

ITensor svDense(const ITensor&);

void restrictMPO(const MPO& , MPO& , int , int , int);

void applyMPO(const MPS& , const MPO& , MPS& , int);

ITensor overlapT(const MPS& , const MPO& , const MPS&);

ITensor overlapT(const MPS& , const MPS&);

void tensorProduct(const MPS& , const MPS& , MPS& , const ITensor& , int);

void combineMPS(vector<MPS>& , MPS& , int);

// davidson.cpp
template <class BigMatrixT , class Tensor>
Real davidsonT(BigMatrixT const& , Tensor& , Args const& args = Args::global());

template <class BigMatrixT , class Tensor>
vector<Real> davidsonT(BigMatrixT const& , vector<Tensor>& phi , Args const& args = Args::global());

// trotter.cpp
void twoLocalTrotter(MPO& , double , int , AutoMPO&);

// svdL.cpp
template<class Tensor>
Spectrum svdL(Tensor , Tensor& , Tensor& , Tensor& , Args args = Args::global());

// rrg.cpp
double twopoint(const MPS& , const ITensor& , const ITensor& , int , int);

