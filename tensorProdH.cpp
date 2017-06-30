#include "rrg.h"

void tensorProdH::product(const ITensor& A, ITensor& B) const {
    ITensor ret;
    for(auto& Hpair : pairedH) {
        auto cur = A*C;
        cur *= Hpair.L;
        cur *= Hpair.R;
        ret = (ret ? ret + noprime(cur)*C : noprime(cur)*C);
        }

    B = ret;

    return;
    }

int tensorProdH::size() const {
    return int(findtype(C,Link)); 
    }
