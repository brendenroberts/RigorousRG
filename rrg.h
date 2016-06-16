#include "itensor/all.h"
#include "itensor/mps/mpo.h"
#include "davidson2.h"
#include <cmath>
#include <vector>

using namespace itensor;
using std::vector;

// Miles originally wrote the following two func-
// tions in itensor/mps/mpsalgs.cc as plussers and
// addAssumeOrth, which I reproduce locally here...
// for some reason only this makes the code work
void plussers(Index const& l1, 
         Index const& l2, 
         Index      & sumind, 
         ITensor    & first, 
         ITensor    & second);

MPO& MPOadd(MPO & L, MPO const& R, Args const& args);

MPO dag(const MPO& mpo);

double MPOnorm(const MPO& mpo);

vector<Real> dmrgThatStuff(const MPO& H , vector<MPS>& states , double eps); 

MPO ExactH(const SiteSet& hs , double offset);

vector<ITensor> TwoSiteH(const SiteSet& hs);

ITensor Apply(ITensor a , const ITensor b);

void TrotterExp(MPO& eH , double tstep , int Nt , double ej , Real eps);

double ApproxH(const MPO& eH , MPO& Ha , double ej , double t , Real eps);

void ShiftH(MPO& H , double normH , double eta1);

void NormalizedCheby(const MPO& H , MPO& K , int k , double eta0 , double eta1 , double normH , Real eps);
