#include "rrg.h"
#include <lapacke.h>
#include <limits>
#include <tuple>
#include <stdexcept>
#include "itensor/util/range.h"
#include "itensor/tensor/sliceten.h"
#include "itensor/itdata/qutil.h"

const auto MAX_INT = std::numeric_limits<int>::max();

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
int SVDRefImpl(MatRef<T> const& M,
            MatRef<T>  const& U, 
            VectorRef  const& D, 
            MatRef<T>  const& V,
            Real thresh,
            int depth = 0)
    {
    auto Mr = nrows(M), 
         Mc = ncols(M);
    auto nsv = min(Mr,Mc);

    struct SVD {
        LAPACK_INT static call(LAPACK_INT M_ , LAPACK_INT N_ , LAPACK_INT P_ ,
            Real* Adata , Real* Sdata , Real* Udata , Real* Vdata) {
            LAPACK_INT LDA_=M_,LDU_=M_,LDVT_=P_;
            if(min(M_,N_) <= 5000 && max(M_,N_) <= 20000)
                return LAPACKE_dgesdd(LAPACK_COL_MAJOR,'S',M_,N_,
                    Adata,LDA_,Sdata,Udata,LDU_,Vdata,LDVT_);
            Real superb[min(M_,N_)-1];
            return LAPACKE_dgesvd(LAPACK_COL_MAJOR,'S','S',M_,N_,
                    Adata,LDA_,Sdata,Udata,LDU_,Vdata,LDVT_,superb);
            }
        LAPACK_INT static call(LAPACK_INT M_ , LAPACK_INT N_ , LAPACK_INT P_ ,
            Cplx* Adata , Real* Sdata , Cplx* Udata , Cplx* Vdata) {
            LAPACK_INT LDA_=M_,LDU_=M_,LDVT_=P_;
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
        info = SVD::call(Mc,Mr,nsv,M.data(),D.data(),V.data(),U.data());
    else
        info = SVD::call(Mr,Mc,nsv,M.data(),D.data(),U.data(),V.data());
    
    return info;
    }

template<typename T>
void
SVDRef(MatRef<T> const& M,
        MatRef<T>  const& U, 
        VectorRef  const& D, 
        MatRef<T>  const& V,
        Real thresh)
    {
    auto info = SVDRefImpl(M,U,D,V,thresh);
    if(info) {
        fprintf(stderr,"Error %d in LAPACK SVD call... retrying\n",info);
        info = SVDRefImpl(M,U,D,V,thresh);
        if(info) Error("Error in LAPACK SVD call");
        }
    }
template void SVDRef(MatRef<Real> const&,MatRef<Real> const&, VectorRef const&, MatRef<Real> const& , Real);
template void SVDRef(MatRef<Cplx> const&,MatRef<Cplx> const&, VectorRef const&, MatRef<Cplx> const& , Real);

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

    if(isTransposed(M)) {
        resize(U,nsv,Mr);
        resize(V,Mc,nsv);
    } else {
        resize(U,Mr,nsv);
        resize(V,nsv,Mc);
        }
    resize(D,nsv);
    SVDRef(makeRef(M),makeRef(U),makeRef(D),makeRef(V),thresh);
 
//    if(isTransposed(M)) {
//        U = subMatrix(U,0,nsv,0,Mr);
//        reduceCols(V,nsv);
//    } else {
//        reduceCols(U,nsv);
//        V = subMatrix(V,0,nsv,0,Mc);
//        }       
    }

template<typename T>
Spectrum
svdImpl(ITensor& A,
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

template<typename T>
Spectrum
svdImpl(IQTensor A, 
        IQIndex const& uI, 
        IQIndex const& vI,
        IQTensor & U, 
        IQTensor & D, 
        IQTensor & V,
        Args const& args)
    {
    auto do_truncate = args.getBool("Truncate");
    auto thresh = args.getReal("SVDThreshold",1E-3);
    auto cutoff = args.getReal("Cutoff",0);
    auto maxm = args.getInt("Maxm",MAX_INT);
    auto minm = args.getInt("Minm",1);
    auto doRelCutoff = args.getBool("DoRelCutoff",true);
    auto absoluteCutoff = args.getBool("AbsoluteCutoff",false);
    auto show_eigs = args.getBool("ShowEigs",false);
    auto lname = args.getString("LeftIndexName","ul");
    auto rname = args.getString("RightIndexName","vl");
    auto itype = getIndexType(args,"IndexType",Link);
    auto litype = getIndexType(args,"LeftIndexType",itype);
    auto ritype = getIndexType(args,"RightIndexType",itype);
    auto compute_qn = args.getBool("ComputeQNs",false);

    auto blocks = doTask(GetBlocks<T>{A.inds(),uI,vI},A.store());

    auto Nblock = blocks.size();
    if(Nblock == 0) throw ResultIsZero("IQTensor has no blocks");

    //TODO: optimize allocation/lookup of Umats,Vmats
    //      etc. by allocating memory ahead of time (see algs.cc)
    //      and making Umats a vector of MatrixRef's to this memory
    auto Umats = vector<Mat<T>>(Nblock);
    auto Vmats = vector<Mat<T>>(Nblock);

    //TODO: allocate dvecs in a single allocation
    //      make dvecs a vector<VecRef>
    auto dvecs = vector<Vector>(Nblock);

    auto alleig = stdx::reserve_vector<Real>(std::min(uI.m(),vI.m()));

    auto alleigqn = vector<EigQN>{};
    if(compute_qn)
        {
        alleigqn = stdx::reserve_vector<EigQN>(std::min(uI.m(),vI.m()));
        }

    if(uI.m() == 0) throw ResultIsZero("uI.m() == 0");
    if(vI.m() == 0) throw ResultIsZero("vI.m() == 0");

    for(auto b : range(Nblock))
        {
        auto& M = blocks[b].M;
        auto& UU = Umats.at(b);
        auto& VV = Vmats.at(b);
        auto& d =  dvecs.at(b);

        SVDL(M,UU,d,VV,thresh);

        //conjugate VV so later we can just do
        //U*D*V to reconstruct ITensor A:
        conjugate(VV);

        alleig.insert(alleig.end(),d.begin(),d.end());
        if(compute_qn)
            {
            auto bi = blocks[b].i1;
            auto q = uI.qn(1+bi);
            for(auto sval : d)
                {
                alleigqn.emplace_back(sqr(sval),q);
                }
            }
        }

    //Square the singular values into probabilities
    //(density matrix eigenvalues)
    for(auto& sval : alleig) sval = sval*sval;

    //Sort all eigenvalues from largest to smallest
    //irrespective of quantum numbers
    stdx::sort(alleig,std::greater<Real>{});
    if(compute_qn) stdx::sort(alleigqn,std::greater<EigQN>{});

    auto probs = Vector(move(alleig),VecRange{alleig.size()});

    long m = probs.size();
    Real truncerr = 0;
    Real docut = -1;
    if(do_truncate)
        {
        tie(truncerr,docut) = truncate(probs,maxm,minm,cutoff,
                                       absoluteCutoff,doRelCutoff);
        m = probs.size();
        alleigqn.resize(m);
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

    auto Liq = IQIndex::storage{};
    auto Riq = IQIndex::storage{};
    Liq.reserve(Nblock);
    Riq.reserve(Nblock);

    for(auto b : range(Nblock))
        {
        auto& d = dvecs.at(b);
        auto& B = blocks[b];

        //Count number of eigenvalues in the sector above docut
        long this_m = 0;
        for(decltype(d.size()) n = 0; n < d.size() && sqr(d(n)) > docut; ++n)
            {
            this_m += 1;
            if(d(n) < 0) d(n) = 0;
            }

        if(m == 0 && d.size() >= 1) // zero mps, just keep one arb state
            { 
            this_m = 1; 
            m = 1; 
            docut = 1; 
            }

        if(this_m == 0) 
            { 
            d.clear();
            B.M.clear();
            assert(not B.M);
            continue; 
            }

        resize(d,this_m);

        Liq.emplace_back(Index("l",this_m,litype),uI.qn(1+B.i1));
        Riq.emplace_back(Index("r",this_m,ritype),vI.qn(1+B.i2));
        }
    
    auto L = IQIndex(lname,move(Liq),uI.dir());
    auto R = IQIndex(rname,move(Riq),vI.dir());

    auto Uis = IQIndexSet(uI,dag(L));
    auto Dis = IQIndexSet(L,R);
    auto Vis = IQIndexSet(vI,dag(R));

    auto Ustore = QDense<T>(Uis,QN());
    auto Vstore = QDense<T>(Vis,QN());
    auto Dstore = QDiagReal(Dis);

    long n = 0;
    for(auto b : range(Nblock))
        {
        auto& B = blocks[b];
        auto& UU = Umats.at(b);
        auto& VV = Vmats.at(b);
        auto& d = dvecs.at(b);
        //Default-constructed B.M corresponds
        //to this_m==0 case above
        if(not B.M) continue;

        auto uind = stdx::make_array(B.i1,n);
        auto pU = getBlock(Ustore,Uis,uind);
        assert(pU.data() != nullptr);
        assert(uI[B.i1].m() == long(nrows(UU)));
        auto Uref = makeMatRef(pU,uI[B.i1].m(),L[n].m());
        reduceCols(UU,L[n].m());
        Uref &= UU;

        auto dind = stdx::make_array(n,n);
        auto pD = getBlock(Dstore,Dis,dind);
        assert(pD.data() != nullptr);
        auto Dref = makeVecRef(pD.data(),d.size());
        Dref &= d;

        auto vind = stdx::make_array(B.i2,n);
        auto pV = getBlock(Vstore,Vis,vind);
        assert(pV.data() != nullptr);
        assert(vI[B.i2].m() == long(nrows(VV)));
        auto Vref = makeMatRef(pV.data(),pV.size(),vI[B.i2].m(),R[n].m());
        reduceCols(VV,R[n].m());
        Vref &= VV;

        ++n;
        }

    //Fix sign to make sure D has positive elements
    Real signfix = (A.scale().sign() == -1) ? -1. : +1.;

    U = IQTensor(Uis,move(Ustore));
    D = IQTensor(Dis,move(Dstore),A.scale()*signfix);
    V = IQTensor(Vis,move(Vstore),LogNum{signfix});

    //Originally eigs were found without including scale
    //so put the scale back in
    if(A.scale().isFiniteReal())
        {
        probs *= sqr(A.scale().real0());
        }
    else
        {
        println("Warning: scale not finite real after svd");
        }

    if(compute_qn)
        {
        auto qns = stdx::reserve_vector<QN>(alleigqn.size());
        for(auto& eq : alleigqn) qns.push_back(eq.qn);
        return Spectrum(move(probs),move(qns),{"Truncerr",truncerr});
        }

    return Spectrum(move(probs),{"Truncerr",truncerr});

    }

template<typename IndexT>
Spectrum 
svdRank2(ITensorT<IndexT>& A, 
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
        return svdImpl<Cplx>(A,ui,vi,U,D,V,args);
        }
    return svdImpl<Real>(A,ui,vi,U,D,V,args);
    }
template Spectrum 
svdRank2(ITensor&,Index const&,Index const&,ITensor &,ITensor &,ITensor &,Args );
template Spectrum 
svdRank2(IQTensor&,IQIndex const&,IQIndex const&,IQTensor &,IQTensor &,IQTensor &,Args );

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

    auto spec = svdRank2(AA,ui,vi,U,D,V,args);

    U = dag(Ucomb) * U;
    V = V * dag(Vcomb);
    
    return spec;
    } //svd
template Spectrum svdL(ITensor, ITensor& , ITensor& , ITensor& , Args);
template Spectrum svdL(IQTensor, IQTensor& , IQTensor& , IQTensor& , Args);
