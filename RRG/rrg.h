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

const Real eps = 1E-12;
const auto Select = IndexType("Select");
const auto invsqrt = [](Real r) { return 1.0/sqrt(r); };
const auto getdata = [](Dense<Real> const& d) { return d.store; };

struct MPOPair
{
    MPO L,R;
    MPOPair() {}
    MPOPair(MPO LL , MPO RR): L(LL) , R(RR) {}
};

struct SiteITensor
{
    int i;
    ITensor A;
    SiteITensor() : i(0) {}
    SiteITensor(int ii , ITensor AA) : i(ii), A(AA) {}
};

inline void pvec(const double *vec, int n) {
    for(int i = 0 ; i < n ; ++i) fprintf(stderr,"%17.14f\n",vec[i]);
    }

inline void pmat(const double *mat, int n , int m) {
    for(int i = 0 ; i < m ; ++i) {
        for(int j = 0 ; j < n ; ++j)
            fprintf(stderr,"%.8e ",mat[i*n+j]);
        fprintf(stderr,"\n");
        }
    }

inline void pvec(const vector<Real>& vec, int n) { pvec(&vec[0],n); }

inline void pmat(const vector<Real>& mat, int n , int m) { pmat(&mat[0],n,m); }

// main.cpp
double twopoint(MPS , ITensor , ITensor , int , int);

// util.cpp
void reducedDM(const MPS& , MPO& ,int);

template<class MPSLike>
void regauge(MPSLike& , int);

void combineVectors(const vector<ITensor>& , ITensor&);

vector<Real> dmrgMPO(const MPO& , vector<MPS>& , double); 

ITensor svDense(const ITensor&);

void restrictMPO(const MPO& , MPO& , int , int , int);

void applyMPO(const MPS& , const MPO& , MPS& , int);

ITensor overlapT(const MPS& , const MPO& , const MPS&);

ITensor overlapT(const MPS& , const MPS&);

void tensorProduct(const MPS& , const MPS& , MPS& , const ITensor& , int);

void combineMPS(vector<MPS>& , MPS& , int);

// davidson.cpp
template <class Tensor>
Real davidsonT(Tensor const& , Tensor& , Args const& args = Args::global());

template <class Tensor>
vector<Real> davidsonT(Tensor const& , vector<Tensor>& phi , Args const& args = Args::global());

// trotter.cpp
void twoLocalTrotter(MPO&, double, int, AutoMPO&);
