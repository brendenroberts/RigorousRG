#include "rrg.h"

using namespace itensor;
using std::vector;

int main(int argc, char *argv[]) {
    if(argc != 2 && argc != 3) {
        printf("usage: agsp chebyshev_degree (N=8)\n");
        return 1;
        }
    const int N = (argc == 3 ? atoi(argv[2]) : 8);
    const int Nt = 40;
    const int k = atoi(argv[1]);
    vector<ITensor> evecs;
    vector<Real> evals;
    Real eps = 1E-12;
    int i,j;

    // the numbers
    int     l = 3;
    double  t = 0.1;
    double  tp = 4.0;
    double  ej = 0.0;
    double  eta0 = 0.1;
    double  eta1 = 0.6;

    // make H as an MPO
    const auto hilbert_space = SpinHalf(N);
    auto H = ExactH(hilbert_space);
    
    // exactly diagonalize H for AGSP validation--also use this for the gap
    if(N <= 16) { 
        auto init_state = InitState(hilbert_space,"Up");
        for(j = 2 ; j <= N ; j += 2)
            init_state.set(j,"Dn");
        auto P = MPS(init_state);

        auto tensorP = P.A(1);
        auto tensorQ = P.A(1);
        auto tensorH = H.A(1);
        for(i = 2 ; i <= N ; ++i) {
            tensorP *= P.A(i);
            tensorQ *= P.A(i);
            tensorH *= H.A(i);
            }

        evecs.push_back(tensorP);
        evecs.push_back(tensorQ);
        evals = davidson2(tensorH,evecs);
        fprintf(stderr,"exact GS: E=%7f , gap=%7f\n",evals[0],evals[1]-evals[0]);

        ej = evals[0];
        eta0 = evals[0];
        eta1 = evals[1];
        }
   
    // make exp(-H/t) as an MPO
    MPO eH(hilbert_space);
    TrotterExp(eH,t,Nt,eps);
    eH.orthogonalize();
    
    // make reduced-norm approximation Ha
    // TODO: why doesn't plusEq work from itensor lib?
    MPO Ha(hilbert_space);
    auto normHa = ApproxH(eH,Ha,ej,t,eps);
    fprintf(stderr,"fractional reduction of H: %E\n",normHa/MPOnorm(H));

    if(N <= 16) {
        auto tensorHa = Ha.A(1);
        for(i = 2 ; i <= N ; ++i) tensorHa *= Ha.A(i);
        double g0 = (dag(prime(evecs[0]))*tensorHa*evecs[0]).real();
        double g1 = (dag(prime(evecs[1]))*tensorHa*evecs[1]).real();
        fprintf(stderr,"(approx H) exact GS E=%7f , exact 1ex E=%7f\n",g0,g1);
        }
    
    // make shifted H for argument to initial Chebyshev polynomials
    //auto Harg = Ha;
    auto Harg = H;
    ShiftH(Harg,MPOnorm(H),eta1);

    // make order-k Chebyshev polynomial
    MPO K(hilbert_space);
    NormalizedCheby(Harg,K,k,eta0,eta1,normHa,eps);
    //NormalizedCheby(Harg,K,k,eta0,eta1,MPOnorm(H),eps);

    // validate K by diagonalizing, test with H ground state
    if(N <= 16) { 
        auto tensorK = K.A(1);
        for(i = 2 ; i <= N ; ++i) {
            tensorK *= K.A(i);
            }

        // diagonalize H for AGSP validation purposes
        auto l = dag(prime(evecs[0]))*tensorK*evecs[0];
        auto o = dag(prime(evecs[1]))*tensorK*evecs[1];
        fprintf(stderr,"Eigenvalues of K... gs: %7f , 1ex: %7f\n",l.real(),o.real());
        }
    
    return 0;
    }
