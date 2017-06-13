#include "itensor/all.h"
#include "itensor/mps/mpo.h"
#include "davidson2.h"
#include <cmath>
#include <vector>

using namespace itensor;
using std::vector;

// Miles originally wrote the following function
// in itensor/mps/mpsalgs.cc as plussers and
// addAssumeOrth, which I reproduce locally here...
// for some reason only this makes the code work
MPO& MPOadd(MPO &, MPO const&, Args const& args);

MPO dag(const MPO&);

double boundMPOnorm(const MPO&);

vector<Real> exactDiagonalizeMPO(const MPO&, vector<MPS>&, int, double);

vector<Real> dmrgThatStuff(const MPO&, vector<MPS>&, double, double); 

MPO ExactH(const SiteSet&, double);

void TrotterExp(MPO&, double, int, double, Real);

double ApproxH(const MPO&, MPO&, double, double, Real);

void ShiftH(MPO&, double, double);

void NormalizedCheby(const MPO&, MPO&, int, double, double, double, Real);
