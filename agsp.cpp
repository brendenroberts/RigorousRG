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
    vector<ITensor> evecs,evecsA;
    vector<Real> evals,evalsA;
    Real eps = 1E-12;
    int i,j;

    // the numbers
    int     l = 3;
    double  t = atof(argv[1]);
    double  ej = 0.25*sqrt(t)*N;
    double  eta0 = ej+t;
    double  eta1 = eta0+2.0*t;
    fprintf(stderr,"l=%d , t=%5f , ej=%5f , eta0=%5f , eta1=%5f\n",l,t,ej,eta0,eta1);

    // make H as an MPO
    const auto hilbert_space = SpinHalf(N);
    auto H = ExactH(hilbert_space,0.0);

    // try to shift H to positive spectrum, assuming symmetric originally
    auto Hn = MPOnorm(H);
    H = ExactH(hilbert_space,Hn/2.0);
    
    // exactly diagonalize H for AGSP validation later
    if(N <= ED_MAX) { 
        auto init_state = InitState(hilbert_space,"Up");
        auto P = MPS(init_state);

        auto tensorP = P.A(1);
        auto tensorH = H.A(1);
        for(i = 2 ; i <= N ; ++i) {
            tensorP *= P.A(i);
            tensorH *= H.A(i);
            }
        
        for(i = 0 ; i < 2 ; ++i) {
            randomize(tensorP);
            evecs.push_back(tensorP);
            }
        evals = davidson2(tensorH,evecs);
        fprintf(stderr,"exact GS: E=%7f , gap=%7f\n",evals[0],evals[1]-evals[0]);
        }
   
    // make exp(-H/t) as an MPO
    MPO eH(hilbert_space);
    TrotterExp(eH,t,Nt,Hn/2.0,eps);
    eH.orthogonalize();
    
    // make reduced-norm approximation Ha
    // TODO: why doesn't plusEq work from itensor lib?
    MPO Ha(hilbert_space);
    auto normHa = ApproxH(eH,Ha,ej,t,eps);
    fprintf(stderr,"norm(exact)=%f, norm(approx)=%f (%E)\n",Hn,normHa,normHa/MPOnorm(H));
    
    if(N <= ED_MAX) {
        auto tensorHa = Ha.A(1);
        for(i = 2 ; i <= N ; ++i) tensorHa *= Ha.A(i);
        double g0 = (dag(prime(evecs[0]))*tensorHa*evecs[0]).real();
        double g1 = (dag(prime(evecs[1]))*tensorHa*evecs[1]).real();
        fprintf(stderr,"(approx H) exact GS E=%7f , exact 1e E=%7f\n",g0,g1);
        }
    
    // make shifted H for argument to initial Chebyshev polynomials
    auto Harg = Ha;
    ShiftH(Harg,normHa,eta1);

    // make order-k Chebyshev polynomial
    MPO K(hilbert_space);
    NormalizedCheby(Harg,K,k,eta0,eta1,normHa,eps);

    // validate K by diagonalizing, test with H ground state
    if(N <= ED_MAX) { 
        auto tensorK = K.A(1);
        for(i = 2 ; i <= N ; ++i) tensorK *= K.A(i);

        auto l = (dag(prime(evecs[0]))*tensorK*evecs[0]).real();
        auto o = (dag(prime(evecs[1]))*tensorK*evecs[1]).real();
        fprintf(stderr,"Eigenvalues of K... gs: %7f , 1e: %7f (ratio %E)\n",l,o,o/l);
        }
    
    return 0;
    }
