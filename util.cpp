#include "rrg.h"
#include <lapacke.h>
#include <iostream>
#include <fstream>

int nels(const ITensor& A) {
    int n = 1;
    for(const auto& i : A.inds()) n *= int(i);
    return n;
    }

void reducedDM(const MPS& psi , MPO& rho , int ls) {
    const auto& hs  = psi.sites();
    const auto& sub = rho.sites();
    auto N = psi.N();
    auto n = rho.N();
    auto rs = ls + n - 1;
    auto psip = psi;
    psip.mapprime(0,1);
    ITensor L,R,C;
    Index ai,bi,ci;

    for(int i = 1 ; i < ls ; ++i) {
        auto sp = hs.si(i);
        L = (i == 1 ? psi.A(i) : L*psi.A(i))*psip.A(i)*delta(sp,prime(sp));
        }

    for(int i = ls ; i <= rs ; ++i) {
        int r = i-ls+1;
        auto T = psi.A(i)*psip.A(i);
        if(i == ls && L) T *= L;
        if(i != ls) {
            C = combiner(leftLinkInd(psi,i),leftLinkInd(psip,i));
            T *= C;
            ci = commonIndex(T,C);
            T *= delta(ci,bi);
            }
        if(i != rs) {
            C = combiner(rightLinkInd(psi,i),rightLinkInd(psip,i));
            T *= C;
            bi = commonIndex(T,C);
            }
        rho.Aref(r) = T*delta(hs.si(i),sub.si(r))*delta(hs.siP(i),sub.siP(r));
        }

    for(int i = rs+1 ; i <= N ; ++i) {
        auto sp = hs.si(i);
        R = (i == rs+1 ? psi.A(i) : R*psi.A(i))*psip.A(i)*delta(sp,prime(sp));
        }
    
    if(R) rho.Aref(n) *= R;

    return;
    }

template<class MPSLike>
void regauge(MPSLike& psi , int o) {
    psi.position((psi.N()-o+1),{"Cutoff",eps*1e-1});
    psi.position(o,{"Cutoff",eps});

    return;
    }
template void regauge(MPS& , int);
template void regauge(MPO& , int);

void combineVectors(const vector<ITensor>& vecs , ITensor& ret) {
    auto chi = commonIndex(ret,vecs[0]);
    auto ext = uniqueIndex(ret,vecs[0],Select);
    if(ext.m() != int(vecs.size())) Error("index/vector mismatch in combineVectors");

    for(int i = 1 ; i <= chi.m() ; ++i)
        for(int j = 1 ; j <= ext.m() ; ++j)
            ret.set(chi(i),ext(j),vecs[j-1].real(chi(i)));

    }

vector<Real> dmrgMPO(const MPO& H , vector<MPS>& states , double penalty, double err) {
    vector<Real> evals;
    vector<MPS> exclude;
    int num_sw = 40;
    for(auto& psi : states) {
        auto swp = Sweeps(num_sw);
        num_sw = 50;
        swp.maxm() = 50,50,100,100,500;
        swp.cutoff() = err;
        swp.niter() = 3;
        swp.noise() = 1E-7,1E-8,0.0;
 
        // dmrg call won't shut up
        std::stringstream ss;
        auto out = std::cout.rdbuf(ss.rdbuf()); 
        auto e = dmrg(psi,H,exclude,swp,{"Quiet",true,"PrintEigs",false,"Weight",penalty});
        std::cout.rdbuf(out);

        exclude.push_back(psi);
        evals.push_back(e);
        }

    return evals;
    }

vector<Real> dmrgMPO(const MPO& H , vector<MPS>& states , double penalty) {
    return dmrgMPO(H,states,penalty,eps);
    }

ITensor svDense(const ITensor& A) {
    auto a = A.inds()[0];
    auto b = A.inds()[1];
    ITensor ret(a,b);
    int nA = std::min(a.m(),b.m());
    for(int i = 1 ; i <= nA ; ++i)
        ret.set(a(i),b(i),A.real(a(i),b(i)));

    return ret;
    }

ITensor splitMPO(const MPO& O, MPO& P, int lr) {
    auto N = O.N();
    auto n = P.N();
    const auto HS = O.sites();
    const auto hs = P.sites();
    auto M = O;
    int t = (lr ? N-n+1 : n);
    ITensor U,S,V,R;
    Index sp,sq,ai,ei;

    M.position(t);
    int a[2] = {t-lr,t+1-lr};
    auto B = M.A(a[0])*M.A(a[1]);
    U = ITensor(HS.si(a[0]),HS.siP(a[0]),leftLinkInd(M,a[0]));
    svdL(B,U,S,V,{"Cutoff",eps*1e-1});
    R = (lr ? V : U);
    ai = commonIndex(S,R);
    sp = hs.si((lr ? 1 : n)); sq = HS.si(t);
    ei = Index("ext",ai.m(),Select);
    S = svDense(S);
    S *= delta(ai,ei);
    R *= delta(ai,ei);
    P.Aref((lr ? 1 : n)) = R*delta(sp,sq)*delta(prime(sp),prime(sq));
        
    for(int i = 1 ; i < n ; ++i) {
        int x = (lr ? i+1 : n-i);
        sp = hs.si(x); sq = HS.si((lr ? t+x-1 : x));
        P.Aref(x) = M.A((lr ? t+x-1 : x))*delta(sp,sq)*delta(prime(sp),prime(sq));
        }
    
    return S;
    }

void restrictMPO(const MPO& O , MPO& res , int ls , int D, int lr) {
    auto N = O.N();
    auto n = res.N();
    if(N == n) {res = O; return;}
    int rs = ls+n-1;
    const auto& hs = O.sites();
    const auto& sub = res.sites();
    auto M = O;
    ITensor U,V,SB;
   
    if(ls == 1) { // easy case: only dangling bond already at R end
        SB = splitMPO(M,res,LEFT);
        auto ei = Index("ext",std::min(D,(int)findtype(res.A(n),Select).m()),Select);
        res.Aref(n) *= delta(ei,commonIndex(SB,res.A(n)));
        regauge(res,n);
        return;
    } else if(rs == N) { // easy case: only dangling bond already at L end
        SB = splitMPO(M,res,RIGHT);
        auto ei = Index("ext",std::min(D,(int)findtype(res.A(1),Select).m()),Select);
        res.Aref(1) *= delta(ei,commonIndex(SB,res.A(1)));
        regauge(res,1);
        return;
        }

    // setup for moving external bond to correct end 
    ITensor SL,SR;
    if(lr) {
        SpinHalf htmp(rs);
        MPO tmp(htmp);
        SR = splitMPO(M,tmp,LEFT);
        SL = splitMPO(tmp,res,RIGHT);
    } else {
        SpinHalf htmp(N-ls+1);
        MPO tmp(htmp);
        SL = splitMPO(M,tmp,RIGHT);
        SR = splitMPO(tmp,res,LEFT);
        }
    SB = SL*SR;
    Index li = Index("ext",std::min(D+5,(int)findtype(res.A(1),Select).m()),Select);
    Index ri = Index("ext",std::min(D+5,(int)findtype(res.A(n),Select).m()),Select);
    Index li2 = commonIndex(SL,res.A(1));
    Index ri2 = commonIndex(SR,res.A(n));
    res.Aref(1) *= delta(li,li2);
    res.Aref(n) *= delta(ri,ri2);
    SB *= delta(li,li2);
    SB *= delta(ri,ri2);

    ITensor S;
    if(lr)
        for(int i = n-1 ; i >= 1 ; --i) {
            auto B = res.A(i)*res.A(i+1);
            U = ITensor(sub.si(i),sub.siP(i),ri,(i == 1 ? li : leftLinkInd(res,i)));
            V = ITensor();
            //denmatDecomp(B,U,V,Fromright,{"Cutoff",eps});
            svdL(B,U,S,V,{"Cutoff",eps});
            res.Aref(i) = U*S;
            res.Aref(i+1) = V;
            }
    else
        for(int i = 2 ; i <= n ; ++i) {
            auto B = res.A(i-1)*res.A(i);
            U = ITensor();
            V = ITensor(sub.si(i),sub.siP(i),li,(i == n ? ri : rightLinkInd(res,i)));
            //denmatDecomp(B,U,V,Fromleft,{"Cutoff",eps});
            svdL(B,U,S,V,{"Cutoff",eps});
            res.Aref(i-1) = U;
            res.Aref(i) = V*S;
            }
    
    // combine ext indices
    U = ITensor(li,ri);
    V = ITensor();
    denmatDecomp(SB,U,V,Fromleft,{"Maxm",D*D,"Cutoff",eps});
    auto ei = Index("ext",int(commonIndex(U,V)),Select);
    U *= delta(ei,commonIndex(U,V));
    res.Aref((lr ? 1 : n)) *= U;
    regauge(res,(lr ? 1 : n));

    return;
    }

void applyMPO(const MPS& psi, const MPO& K, MPS& res , int lr) {
    auto N = psi.N();
    if(K.N() != N) Error("applyMPO mismatched N");
    auto ss = (lr ? 1 : N);
    auto st = (lr ? N : 1);
    res = psi;
    vector<Index> ext;
    if(findtype(psi.A(ss),Select)) ext.push_back(findtype(psi.A(ss),Select));
    if(findtype(K.A(ss),Select)) ext.push_back(findtype(K.A(ss),Select));

    ITensor clust,nfork,temp,S;
    for(int i = 0; i < N-1; i++) {
        int x = (lr ? N-i : i+1);
        if(i == 0) { clust = psi.A(x) * K.A(x); }
        else { clust = (nfork * psi.A(x)) * K.A(x); }
        if(i == N-2) break;

        nfork = (lr ? ITensor(leftLinkInd(psi,x),leftLinkInd(K,x))
                    : ITensor(rightLinkInd(psi,x),rightLinkInd(K,x)));
        temp = ITensor();
        fprintf(stderr,"SVD mat dim: %dx%d\n",nels(nfork),nels(clust)/nels(nfork));
        svdL(clust,temp,S,nfork,{"Cutoff",eps,"Maxm",MAXBD});
        nfork *= S;
        res.Aref(x) = temp;
        }
    nfork = (clust * psi.A(ss)) * K.A(ss);
   
    // deal with multiple ext indices on final site
    ITensor B;
    auto itemp = ext;
    itemp.push_back(prime(findtype(psi.A(ss),Site)));
    auto A = ITensor(itemp);
    if(lr) svdL(nfork,A,S,B,{"Cutoff",eps});
    else   svdL(nfork,B,S,A,{"Cutoff",eps});
    res.Aref(ss) = A;
    res.Aref((lr ? ss+1 : ss-1)) = S*B;
    
    res.noprimelink();
    res.mapprime(1,0,Site);    
    regauge(res,ss);
    if(ext.size() == 2)
        res.Aref(ss) *= combiner(ext,{"IndexType",Select});
    }

ITensor overlapT(const MPS& phi, const MPS& psi) {
    auto N = phi.N();
    if(psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    ITensor L;

    for(int i = 0 ; i < N ; ++i) {
        int x = (lr ? N-i : i+1);
        L = (i ? L*phi.A(x) : phi.A(x));
        L *= dag(primeExcept(psi.A(x),Site));
        }
    
    return L;
    }

ITensor overlapT(const MPS& phi, const MPO& H, const MPS& psi) {
    auto N = H.N();
    if(phi.N() != N || psi.N() != N) Error("overlap mismatched N");
    auto lr = (findtype(phi.A(N),Select) ? LEFT : RIGHT);
    ITensor L;

    for(int i = 0; i < N; ++i) {
        int x = (lr ? N-i : i+1);
        L = (i ? L*phi.A(x) : phi.A(x));
        L *= H.A(x);
        L *= dag(prime(psi.A(x)));
        }
    
    return L;
    }

void tensorProduct(const MPS& psiA, const MPS& psiB, MPS& ret, const ITensor& W, int lr) {
    const int N = ret.N();
    const int n = psiA.N();
    const auto& hs  = ret.sites();
    const auto& hsA = psiA.sites();
    const auto& hsB = psiB.sites();
    Index ai,ei,sp;
    ITensor T,U,S,V;
    
    for(int i = 1 ; i <= n ; ++i) {
        Index spA  = hsA.si(i); Index spB  = hsB.si(i);
        Index spAr = hs.si(i);  Index spBr = hs.si(n+i);
        ret.Aref(i)   = psiA.A(i)*delta(spA,spAr);
        ret.Aref(n+i) = psiB.A(i)*delta(spB,spBr);
        }

    // Gauge merged MPS starting in the middle
    for(int i = 0 ; i < n ; ++i) {
        int x = (lr ? n-i : n+i);
        sp = hs.si(x);
        ai = commonIndex(ret.A(x-1),ret.A(x));
        T = ret.A(x)*(x == n ? W*ret.A(x+1) : ret.A(x+1));
        if(x == n) ei = findtype(T,Select);
        U = (lr ? ITensor(sp,ai,ei) : ITensor(sp,ai));
        svd(T,U,S,V,{"Cutoff",eps,"Maxm",MAXBD});
        ret.Aref(x)   = (lr ? U*S : U);
        ret.Aref(x+1) = (lr ? V : S*V);
        }
    regauge(ret,(lr ? 1 : N));
   
    return; 
    }

void combineMPS(vector<MPS>& vecs , MPS& ret, int lr) {
    const int N = ret.N();
    const int nvc = vecs.size();
    const auto& hs = ret.sites();
    Index ak,bk,vi;
    ITensor U,S,V;
    vector<Index> inds;
    
    if(lr == LEFT) {
        // Do first tensor
        int bm = 0;
        for(auto& v : vecs) bm += rightLinkInd(v,1).m();
        bk = Index("link",bm);
        auto sp = hs.si(1);
        ITensor A(bk,sp);
        int bsum = 0;
        for(auto& v : vecs) {
            auto bi = rightLinkInd(v,1);
            for(int b = 1 ; b <= bi.m() ; ++b) {
                A.set(bk(bsum+b),sp(1),v.A(1).real(sp(1),bi(b)));
                A.set(bk(bsum+b),sp(2),v.A(1).real(sp(2),bi(b)));
                }
            bsum += bi.m();
            }
        U = ITensor(sp);
        svd(A,U,S,V,{"Cutoff",eps*1e-1});
        inds.push_back(bk);
        ret.Aref(1) = U;
        
        // Do middle tensors
        for(int i = 2 ; i < N ; ++i) {
            int bm = 0;
            for(auto& v : vecs) bm += rightLinkInd(v,i).m();
            ak = inds.back();
            bk = Index("link",bm);
            sp = hs.si(i);
            A = ITensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = leftLinkInd(v,i);
                auto bi = rightLinkInd(v,i);
                for(int a = 1 ; a <= ai.m() ; ++a)
                    for(int b = 1 ; b <= bi.m() ; ++b) {
                        A.set(ak(asum+a),bk(bsum+b),sp(1),v.A(i).real(sp(1),ai(a),bi(b)));
                        A.set(ak(asum+a),bk(bsum+b),sp(2),v.A(i).real(sp(2),ai(a),bi(b)));
                        }
                asum += ai.m();
                bsum += bi.m();
                }
            A *= S*V;
            U = ITensor(commonIndex(U,S),sp);
            svd(A,U,S,V,{"Cutoff",eps*1e-1});
            inds.push_back(bk);
            ret.Aref(i) = U;
            }
        
        // Do last tensor
        vi = Index("ext",nvc,Select);
        ak = inds.back();
        sp = hs.si(N);
        A = ITensor(vi,ak,sp);
        int asum = 0 , ct = 1;
        for(auto& v : vecs) {
            auto ai = leftLinkInd(v,N);
            for(int a = 1 ; a <= ai.m() ; ++a) {
                A.set(vi(ct),ak(asum+a),sp(1),v.A(N).real(sp(1),ai(a)));
                A.set(vi(ct),ak(asum+a),sp(2),v.A(N).real(sp(2),ai(a)));
                }
            asum += ai.m();
            ct++;
            }
        A *= S*V;
        ret.Aref(N) = A;
        ret.position(N,{"Cutoff",eps});
    } else if(lr == RIGHT) { 
        // Do last tensor
        int bm = 0;
        for(auto& v : vecs) bm += leftLinkInd(v,N).m();
        bk = Index("link",bm);
        auto sp = hs.si(N);
        ITensor A(bk,sp);
        int bsum = 0;
        for(auto& v : vecs) {
            auto bi = leftLinkInd(v,N);
            for(int b = 1 ; b <= bi.m() ; ++b) {
                A.set(bk(bsum+b),sp(1),v.A(N).real(sp(1),bi(b)));
                A.set(bk(bsum+b),sp(2),v.A(N).real(sp(2),bi(b)));
                }
            bsum += bi.m();
            }
        V = ITensor(sp);
        svd(A,U,S,V,{"Cutoff",eps*1e-1});
        inds.push_back(bk);
        ret.Aref(N) = V;

        // Do middle tensors
        for(int i = N-1 ; i > 1 ; --i) {
            int bm = 0;
            for(auto& v : vecs) bm += leftLinkInd(v,i).m();
            ak = inds.back();
            bk = Index("link",bm);
            sp = hs.si(i);
            A = ITensor(ak,bk,sp);
            int asum = 0 , bsum = 0;
            for(auto& v : vecs) {
                auto ai = rightLinkInd(v,i);
                auto bi = leftLinkInd(v,i);
                for(int a = 1 ; a <= ai.m() ; ++a)
                    for(int b = 1 ; b <= bi.m() ; ++b) {
                        A.set(ak(asum+a),bk(bsum+b),sp(1),v.A(i).real(sp(1),ai(a),bi(b)));
                        A.set(ak(asum+a),bk(bsum+b),sp(2),v.A(i).real(sp(2),ai(a),bi(b)));
                        }
                asum += ai.m();
                bsum += bi.m();
                }
            A *= U*S;
            V = ITensor(commonIndex(V,S),sp);
            U = ITensor();
            svd(A,U,S,V,{"Cutoff",eps*1e-1});
            inds.push_back(bk);
            ret.Aref(i) = V;
            }
        
        // Do first tensor
        vi = Index("ext",nvc,Select);
        ak = inds.back();
        sp = hs.si(1);
        A = ITensor(vi,ak,sp);
        int asum = 0 , ct = 1;
        for(auto& v : vecs) {
            auto ai = rightLinkInd(v,1);
            for(int a = 1 ; a <= ai.m() ; ++a) {
                A.set(vi(ct),ak(asum+a),sp(1),v.A(1).real(sp(1),ai(a)));
                A.set(vi(ct),ak(asum+a),sp(2),v.A(1).real(sp(2),ai(a)));
                }
            asum += ai.m();
            ct++;
            }
        A *= U*S;
        ret.Aref(1) = A;
        ret.position(1,{"Cutoff",eps});
        }

    return; 
    }
