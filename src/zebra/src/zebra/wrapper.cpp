#include <wrapper.h>

void thetav_np(pbarray p, pbarray t, pbarray qv, pbarray out) {
    int m = t.shape(0), n = t.shape(1), o = t.shape(2);
    thetav(p.data(), t.data(), qv.data(), out.mutable_data(), m, n, o);
}

void brn_np(pbarray p, pbarray t, pbarray qv, pbarray u, pbarray v, pbarray hhl, pbarray hsurf, pbarray out) {
    int m = t.shape(0), n = t.shape(1), o = t.shape(2);
    brn(p.data(), t.data(), qv.data(), u.data(), v.data(), hhl.data(), hsurf.data(), out.mutable_data(), m, n, o);
}
