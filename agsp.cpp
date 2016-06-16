#include "rrg.h"

using namespace itensor;
using std::vector;

int main(int argc, char *argv[]) {
    if(argc != 3 && argc != 4) {
        printf("usage: agsp t k (N=8)\n");
        return 1;
        }
    const int N = (argc == 4 ? atoi(argv[3]) : 8);
    const int Nt = 50;
    const int k = atoi(argv[2]);
    const int ED_MAX = 16;
    vector<MPS> evecs,evecsA;
    vector<Real> evals,evalsA;
    Real eps = 1E-12;
    int i,j;

    // the numbers
    int     l = 3;
    double  t = atof(argv[1]);
    double  ej = 0.25*sqrt(t)*N;
    double  eta0 = ej+t;
    double  eta1 = eta0+2.0*t;
    //fprintf(stderr,"l=%d , t=%5f , ej=%5f , eta0=%5f , eta1=%5f\n",l,t,ej,eta0,eta1);

    // make H as an MPO
    const auto hilbert_space = SpinHalf(N);
    auto H = ExactH(hilbert_space,0.0);

    // try to shift H to positive spectrum, assuming symmetric originally
    auto Hn = MPOnorm(H);
    H = ExactH(hilbert_space,Hn/2.0);
    
    // use DMRG to get H gs,1e for AGSP validation later
    evecs.push_back(MPS(hilbert_space));
    evecs.push_back(MPS(hilbert_space));
    evals = dmrgThatStuff(H,evecs,eps);
    double gap = evals[1] - evals[0];
    fprintf(stderr,"exact GS: E=%7f , gap=%7f\n",evals[0],gap);

    // cheat for demonstration's sake
    ej = evals[0]-1.0;
    eta0 = evals[0]+0.1*gap;
    eta1 = evals[0]+0.9*gap;
    fprintf(stderr,"l=%d , t=%5f , ej=%5f , eta0=%5f , eta1=%5f\n",l,t,ej,eta0,eta1);
   
    // make exp(-H/t) as an MPO
    MPO eH(hilbert_space);
    TrotterExp(eH,t,Nt,Hn/2.0,eps);
    eH.orthogonalize();
    
    // make reduced-norm approximation Ha
    // TODO: why doesn't plusEq work from itensor lib?
    MPO Ha(hilbert_space);
    auto normHa = ApproxH(eH,Ha,ej,t,eps);
    fprintf(stderr,"norm(exact)=%f, norm(approx)=%f (%E)\n",Hn,normHa,normHa/MPOnorm(H));

    evecsA.push_back(MPS(hilbert_space));
    evecsA.push_back(MPS(hilbert_space));
    evalsA = dmrgThatStuff(Ha,evecsA,eps);
    double gapA = evalsA[1] - evalsA[0];
    fprintf(stderr,"approx GS: E=%7f , gap=%7f\n",evalsA[0],gapA);

    fprintf(stderr,"Approx/exact GS fidelity: %f\n",fabs(overlap(evecs[0],evecsA[0])));

    // make shifted H for argument to initial Chebyshev polynomials
    auto Harg = Ha;
    ShiftH(Harg,normHa,eta1);

    // make order-k Chebyshev polynomial
    MPO K(hilbert_space);
    NormalizedCheby(Harg,K,k,eta0,eta1,normHa,eps);

    // validate K by DMRG
    double K0 = overlap(evecs[0],K,evecs[0]);
    double K1 = overlap(evecs[1],K,evecs[1]);
    fprintf(stderr,"Marix elements of K of gs=%7f , 1e=%7f (ratio %E)\n",K0,K1,K1/K0);
    
    return 0;
    }
