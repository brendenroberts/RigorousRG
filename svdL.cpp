#include "rrg.h"
#include <lapacke.h>
#include <limits>
#include <tuple>
#include <stdexcept>
#include "itensor/util/range.h"
#include "itensor/tensor/sliceten.h"

using std::move;
using std::tie;
using std::make_tuple;
using std::tuple;

template<typename V>
struct ToMatRef
    {
    using value_type = V;
    long nrows=0,
         ncols=0;
    bool transpose=false;
    ToMatRef(long nr, long nc, bool trans=false) 
        : nrows(nr), ncols(nc), transpose(trans)
        { }
    };
template<typename V>
MatRef<V>
doTask(ToMatRef<V> & T, 
       Dense<V> & d)
    {
    MatRef<V> res = makeMatRef(d.data(),d.size(),T.nrows,T.ncols);
    if(T.transpose) return transpose(res);
    return res;
    }

template<typename V>
MatRef<V>
toMatRef(ITensor & T, 
        Index const& i1, 
        Index const& i2)
    {
    if(i1 == T.inds().front())
        {
        return doTask(ToMatRef<V>{i1.m(),i2.m()},T.store());
        }
    return doTask(ToMatRef<V>{i2.m(),i1.m(),true},T.store());
    }
template MatRef<Real>
toMatRef(ITensor & T, Index const& i1, Index const& i2);
template MatRef<Cplx>
toMatRef(ITensor & T, Index const& i1, Index const& i2);

template<typename T>
void SVDRefL(MatRef<T> const& , MatRef<T> const& , VectorRef const& , MatRef<T> const& , Real);

template<typename T>
int SVDRefImplL(MatRef<T> const& M,
            MatRef<T>  const& U, 
            VectorRef  const& D, 
            MatRef<T>  const& V,
            Real thresh,
            int depth = 0)
    {
    auto Mr = nrows(M), 
         Mc = ncols(M);

    struct SVD {
        LAPACK_INT static call(LAPACK_INT M_ , LAPACK_INT N_ ,
            Real* Adata , Real* Sdata , Real* Udata , Real* Vdata) {
            LAPACK_INT LDA_=M_,LDU_=M_,LDVT_=N_;
            if(min(M_,N_) <= 5000 && max(M_,N_) <= 20000)
                return LAPACKE_dgesdd(LAPACK_COL_MAJOR,'S',M_,N_,
                    Adata,LDA_,Sdata,Udata,LDU_,Vdata,LDVT_);
            Real superb[min(M_,N_)-1];
            return LAPACKE_dgesvd(LAPACK_COL_MAJOR,'S','S',M_,N_,
                    Adata,LDA_,Sdata,Udata,LDU_,Vdata,LDVT_,superb);
            }
        LAPACK_INT static call(LAPACK_INT M_ , LAPACK_INT N_ ,
            Cplx* Adata , Real* Sdata , Cplx* Udata , Cplx* Vdata) {
            LAPACK_INT LDA_=M_,LDU_=M_,LDVT_=N_;
            auto pA = reinterpret_cast<LAPACK_COMPLEX*>(Adata); 
            auto pU = reinterpret_cast<LAPACK_COMPLEX*>(Udata); 
            auto pV = reinterpret_cast<LAPACK_COMPLEX*>(Vdata); 
            if(min(M_,N_) <= int(5000/sqrt(2)) && max(M_,N_) <= int(20000/sqrt(2)))
                return LAPACKE_zgesdd(LAPACK_COL_MAJOR,'S',M_,N_,
                    pA,LDA_,Sdata,pU,LDU_,pV,LDVT_);
            Real superb[min(M_,N_)-1];
            return LAPACKE_zgesvd(LAPACK_COL_MAJOR,'S','S',M_,N_,
                    pA,LDA_,Sdata,pU,LDU_,pV,LDVT_,superb);
            }
        };
    
    LAPACK_INT info;
    if(isTransposed(M))
        info = SVD::call(Mc,Mr,M.data(),D.data(),V.data(),U.data());
    else
        info = SVD::call(Mr,Mc,M.data(),D.data(),U.data(),V.data());
    
    return info;
    }

template<typename T>
void
SVDRefL(MatRef<T> const& M,
        MatRef<T>  const& U, 
        VectorRef  const& D, 
        MatRef<T>  const& V,
        Real thresh)
    {
    auto info = SVDRefImplL(M,U,D,V,thresh);
    if(info) {
        fprintf(stderr,"Error %d in LAPACK SVD call... retrying\n",info);
        info = SVDRefImplL(M,U,D,V,thresh);
        if(info) Error("Error in LAPACK SVD call");
        }
    }
template void SVDRefL(MatRef<Real> const&,MatRef<Real> const&, VectorRef const&, MatRef<Real> const& , Real);
template void SVDRefL(MatRef<Cplx> const&,MatRef<Cplx> const&, VectorRef const&, MatRef<Cplx> const& , Real);

template<class MatM, class MatU,class VecD,class MatV,
         class = stdx::require<
         hasMatRange<MatM>,
         hasMatRange<MatU>,
         hasVecRange<VecD>,
         hasMatRange<MatV>
         >>
void
SVDL(MatM && M,
    MatU && U, 
    VecD && D, 
    MatV && V,
    Real thresh = SVD_THRESH);

template<class MatM, 
         class MatU,
         class VecD,
         class MatV,
         class>
void
SVDL(MatM && M,
    MatU && U, 
    VecD && D, 
    MatV && V,
    Real thresh)
    {
    auto Mr = nrows(M),
         Mc = ncols(M);
    auto nsv = std::min(Mr,Mc);
    resize(U,Mr,Mr);
    resize(V,Mc,Mc);
    resize(D,nsv);
    SVDRefL(makeRef(M),makeRef(U),makeRef(D),makeRef(V),thresh);
    
    if(isTransposed(M)) {
        U = subMatrix(U,0,nsv,0,Mr);
        reduceCols(V,nsv);
    } else {
        reduceCols(U,nsv);
        V = subMatrix(V,0,nsv,0,Mc);
        }       
    }

template<typename T>
Spectrum
svdImplL(ITensor& A,
        Index const& ui, 
        Index const& vi,
        ITensor & U, 
        ITensor & D, 
        ITensor & V,
        Args const& args)
    {
    SCOPED_TIMER(7);
    auto do_truncate = args.getBool("Truncate");
    auto thresh = args.getReal("SVDThreshold",1E-3);
    auto cutoff = args.getReal("Cutoff",MIN_CUT);
    auto maxm = args.getInt("Maxm",MAX_M);
    auto minm = args.getInt("Minm",1);
    auto doRelCutoff = args.getBool("DoRelCutoff",true);
    auto absoluteCutoff = args.getBool("AbsoluteCutoff",false);
    auto lname = args.getString("LeftIndexName","ul");
    auto rname = args.getString("RightIndexName","vl");
    auto itype = getIndexType(args,"IndexType",Link);
    auto litype = getIndexType(args,"LeftIndexType",itype);
    auto ritype = getIndexType(args,"RightIndexType",itype);
    auto show_eigs = args.getBool("ShowEigs",false);

    auto M = toMatRef<T>(A,ui,vi);

    Mat<T> UU,VV;
    Vector DD;

    TIMER_START(6)
    SVDL(M,UU,DD,VV,thresh);
    TIMER_STOP(6)

    //conjugate VV so later we can just do
    //U*D*V to reconstruct ITensor A:
    conjugate(VV);

    //
    // Truncate
    //
    Vector probs;
    if(do_truncate || show_eigs)
        {
        probs = DD;
        for(auto j : range(probs)) probs(j) = sqr(probs(j));
        }

    Real truncerr = 0;
    Real docut = -1;
    long m = DD.size();
    if(do_truncate)
        {
        tie(truncerr,docut) = truncate(probs,maxm,minm,cutoff,
                                       absoluteCutoff,doRelCutoff);
        if(int(probs.size()) != m) {
            m = probs.size();
            resize(DD,m);
            if(isTransposed(M)) {
                UU = subMatrix(UU,0,m,0,ncols(UU));
                reduceCols(VV,m);
            } else {
                reduceCols(UU,m);
                VV = subMatrix(VV,0,m,0,ncols(VV));
                }       
            }
        }

    if(show_eigs) 
        {
        auto showargs = args;
        showargs.add("Cutoff",cutoff);
        showargs.add("Maxm",maxm);
        showargs.add("Minm",minm);
        showargs.add("Truncate",do_truncate);
        showargs.add("DoRelCutoff",doRelCutoff);
        showargs.add("AbsoluteCutoff",absoluteCutoff);
        showEigs(probs,truncerr,A.scale(),showargs);
        }
    
    Index uL(lname,m,litype),
          vL(rname,m,ritype);

    //Fix sign to make sure D has positive elements
    Real signfix = (A.scale().sign() == -1) ? -1 : +1;

    D = ITensor({uL,vL},
                Diag<Real>{DD.begin(),DD.end()},
                A.scale()*signfix);
    if(isTransposed(M)) {
        U = ITensor({uL,ui},Dense<T>(move(UU.storage())),LogNum(signfix));
        V = ITensor({vi,vL},Dense<T>(move(VV.storage())));
    } else {
        U = ITensor({ui,uL},Dense<T>(move(UU.storage())),LogNum(signfix));
        V = ITensor({vL,vi},Dense<T>(move(VV.storage())));
        }

    //Square all singular values
    //since convention is to report
    //density matrix eigs
    for(auto& el : DD) el = sqr(el);

    if(A.scale().isFiniteReal()) 
        {
        DD *= sqr(A.scale().real0());
        }
    else                         
        {
        println("Warning: scale not finite real after svd");
        }
    
    return Spectrum(move(DD),{"Truncerr",truncerr});
    }

template<typename IndexT>
Spectrum 
svdRank2L(ITensorT<IndexT>& A, 
         IndexT const& ui, 
         IndexT const& vi,
         ITensorT<IndexT> & U, 
         ITensorT<IndexT> & D, 
         ITensorT<IndexT> & V,
         Args args)
    {
    auto do_truncate = args.defined("Cutoff") 
                    || args.defined("Maxm");
    if(not args.defined("Truncate")) 
        {
        args.add("Truncate",do_truncate);
        }

    if(A.r() != 2) 
        {
        Print(A);
        Error("A must be matrix-like (rank 2)");
        }
    if(isComplex(A))
        {
        return svdImplL<Cplx>(A,ui,vi,U,D,V,args);
        }
    return svdImplL<Real>(A,ui,vi,U,D,V,args);
    }
template Spectrum 
svdRank2L(ITensor&,Index const&,Index const&,
         ITensor &,ITensor &,ITensor &,Args );

template<class Tensor>
Spectrum 
svdL(Tensor AA, 
    Tensor & U, 
    Tensor & D, 
    Tensor & V, 
    Args args)
    {
    using IndexT = typename Tensor::index_type;

#ifdef DEBUG
    if(!U && !V) 
        Error("U and V default-initialized in svd, must indicate at least one index on U or V");
#endif

    auto noise = args.getReal("Noise",0);
    auto useOrigM = args.getBool("UseOrigM",false);

    if(noise > 0)
        Error("Noise term not implemented for svd");
    
    //if(isZero(AA,Args("Fast"))) 
    //    throw ResultIsZero("svd: AA is zero");


    //Combiners which transform AA
    //into a rank 2 tensor
    std::vector<IndexT> Uinds, 
                        Vinds;
    Uinds.reserve(AA.r());
    Vinds.reserve(AA.r());
    //Divide up indices based on U
    //If U is null, use V instead
    auto &L = (U ? U : V);
    auto &Linds = (U ? Uinds : Vinds),
         &Rinds = (U ? Vinds : Uinds);
    for(const auto& I : AA.inds())
        { 
        if(hasindex(L,I)) Linds.push_back(I);
        else              Rinds.push_back(I);
        }
    Tensor Ucomb,
           Vcomb;
    if(!Uinds.empty())
        {
        Ucomb = combiner(std::move(Uinds),{"IndexName","uc"});
        AA *= Ucomb;
        }
    if(!Vinds.empty())
        {
        Vcomb = combiner(std::move(Vinds),{"IndexName","vc"});
        AA *= Vcomb;
        }

    if(useOrigM)
        {
        //Try to determine current m,
        //then set minm_ and maxm_ to this.
        args.add("Cutoff",-1);
        long minm = 1,
             maxm = MAX_M;
        if(D.r() == 0)
            {
            auto mid = commonIndex(U,V,Link);
            if(mid) minm = maxm = mid.m();
            else    minm = maxm = 1;
            }
        else
            {
            minm = maxm = D.inds().front().m();
            }
        args.add("Minm",minm);
        args.add("Maxm",maxm);
        }

    auto ui = commonIndex(AA,Ucomb);
    auto vi = commonIndex(AA,Vcomb);

    auto spec = svdRank2L(AA,ui,vi,U,D,V,args);

    U = dag(Ucomb) * U;
    V = V * dag(Vcomb);

    return spec;
    } //svd
template Spectrum svdL(ITensor, ITensor& , ITensor& , ITensor& , Args);
