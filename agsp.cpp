#include "rrg.h"

using namespace itensor;
using std::vector;

int main(int argc, char *argv[]) {
    if(argc != 4 && argc != 5) {
        printf("usage: agsp t k N_Trotter (N=8)\n");
        return 1;
        }
    const int N = (argc == 5 ? atoi(argv[4]) : 8);
    const int M = atoi(argv[3]);
    const int k = atoi(argv[2]);
    vector<MPS> evecs,evecsA;
    vector<Real> evals,evalsA;
    Real eps = 1E-14;
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

    // try to shift H to positive spectrum, assuming roughly symmetric originally
    //auto Hn = boundMPOnorm(H)/1.5;
    //H = ExactH(hilbert_space,Hn);
    //fprintf(stderr,"bound on exact MPO norm: %11.9f\n",Hn);
    
    evals = exactDiagonalizeMPO(H,evecs,N+2,eps);
    double gap = evals.at(1) - evals.at(0);
    fprintf(stderr,"exact  gs energy: %11.9f , gap=%11.9f\n",evals.at(0),gap);

    // cheat for demonstration's sake
    ej = evals.at(0)-gap;
    
    // make exp(-H/t) as an MPO
    MPO eH(hilbert_space);
    TrotterExp(eH,t,M,0.0,eps);
    eH.orthogonalize();
    
    // make reduced-norm approximation Ha
    // TODO: why doesn't plusEq work from itensor lib?
    MPO Ha(hilbert_space);
    auto normHa = ApproxH(eH,Ha,ej,t,eps);
    fprintf(stderr,"bound on approx MPO norm: %11.9f\n",normHa);
    
    evalsA = exactDiagonalizeMPO(Ha,evecsA,2,eps);
    double gapA = evalsA.at(1) - evalsA.at(0);
    fprintf(stderr,"approx gs energy: %11.9f , gap=%11.9f\n",evalsA.at(0),gapA);
 
    double d0 = fabs(overlap(evecs.at(0),evecsA.at(0)));
    double d1 = fabs(overlap(evecs.at(N+1),evecsA.at(0)));
    fprintf(stderr,"overlaps : 1 - <gs|gs'>=%15.13e , <ex|gs'>=%15.13e\n",1.0-d0,d1);
    //fprintf(stdout,"%15.13E\n",d1);

    // more cheating; etas should properly be inside the approximate gap
    // but it can be very small, which makes K very difficult to construct
    eta0 = evals.at(0)+0.1*gap;
    eta1 = evals.at(1)+0.9*gap;
    fprintf(stderr,"l=%d , t=%5f , ej=%5f , eta0=%5f , eta1=%5f\n",l,t,ej,eta0,eta1);
   
    // make shifted H for argument to initial Chebyshev polynomials
    auto Harg = Ha;
    ShiftH(Harg,normHa,eta1);

    // make order-k Chebyshev polynomial
    MPO K(hilbert_space);
    NormalizedCheby(Harg,K,k,eta0,eta1,normHa,eps);
    
    // validate K
    MPS res0, res1;
    exactApplyMPO(evecs.at(0),K,res0);
    exactApplyMPO(evecs.at(N+1),K,res1);
    double K0 = norm(res0);
    double K1 = norm(res1);
    fprintf(stderr,"Norms of K*gs=%15.13f , K*1e=%15.13f (ratio=%E)\n",K0,K1,fabs(K1/K0));
    fprintf(stdout,"%15.13E",fabs(K1/K0));

    return 0;
    }
