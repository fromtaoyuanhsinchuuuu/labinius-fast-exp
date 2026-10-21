use super::bitmat::{transpose_tile, untranspose_planes, GF_IDENT};
use super::scalar::{B128, F162};
use crate::simd::transpose_f162::transpose8x8_q;
use std::arch::x86_64::*;

pub const LOG_PACK: usize = 7;
pub const PACK: usize = 1 << LOG_PACK;

#[derive(Clone)]
struct Tensor {
    pub c: Vec<F162>,
}

impl Tensor {
    fn from_vertical(a: F162) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[0] = a;
        Self { c }
    }

    fn scale_vertical(&self, a: F162) -> Self {
        Self {
            c: self.c.iter().map(|&x| x * a).collect(),
        }
    }

    fn shift_x(&self) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[1..].copy_from_slice(&self.c[..127]);
        let top = self.c[127];
        for k in [0usize, 1, 2, 7] {
            c[k] += top;
        }
        Self { c }
    }

    fn scale_horizontal(&self, h: B128) -> Self {
        let mut acc = Self {
            c: vec![F162::ZERO; 128],
        };
        for k in (0..128).rev() {
            acc = acc.shift_x();
            if h.bit(k) {
                for i in 0..128 {
                    acc.c[i] += self.c[i];
                }
            }
        }
        acc
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            c: self.c.iter().zip(&o.c).map(|(&a, &b)| a + b).collect(),
        }
    }

    fn fold_vertical(&self, batch: &[F162]) -> F162 {
        self.c
            .iter()
            .zip(batch)
            .fold(F162::ZERO, |a, (&x, &y)| a + x * y)
    }
}

pub fn eq_expand_f162(r: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &ri in r {
        let mut n = Vec::with_capacity(t.len() * 2);
        for &x in &t {
            let hi = x * ri;
            n.push(x + hi);
            n.push(hi);
        }
        t = n;
    }
    t
}

pub fn eq_expand_b128(r: &[B128]) -> Vec<B128> {
    let mut t = vec![B128::ZERO; 1 << r.len()];
    eq_expand_b128_into(r, &mut t);
    t
}

/// The expansion in place, doubling from the back so no entry is read after it is written.
pub fn eq_expand_b128_into(r: &[B128], t: &mut [B128]) {
    assert_eq!(t.len(), 1 << r.len());
    t[0] = B128::ONE;
    for (round, &ri) in r.iter().enumerate() {
        for i in (0..1 << round).rev() {
            let hi = t[i] * ri;
            t[2 * i + 1] = hi;
            t[2 * i] = t[i] + hi;
        }
    }
}

pub fn transparent_coeff(r_hi: &[B128], r_pp: &[F162], batch: &[F162]) -> F162 {
    assert_eq!(r_hi.len(), r_pp.len());
    let mut t = Tensor::from_vertical(F162::ONE);
    for (&v, &h) in r_pp.iter().zip(r_hi) {
        let vs = t.scale_vertical(v);
        let hs = t.scale_horizontal(h);
        t = t.add(&vs).add(&hs);
    }
    t.fold_vertical(batch)
}

pub fn psi_table(batch: &[F162]) -> Vec<Vec<F162>> {
    (0..16)
        .map(|chunk| {
            let mut t = vec![F162::ZERO; 256];
            for m in 1..256usize {
                let b = m.trailing_zeros() as usize;
                t[m] = t[m ^ (1 << b)] + batch[chunk * 8 + b];
            }
            t
        })
        .collect()
}

#[inline]
pub fn psi(tab: &[Vec<F162>], x: B128) -> F162 {
    let mut acc = F162::ZERO;
    for (c, t) in tab.iter().enumerate() {
        acc += t[((x.0 >> (8 * c)) & 0xff) as usize];
    }
    acc
}

#[inline(always)]
unsafe fn affine(x: __m512i, a: __m512i) -> __m512i {
    _mm512_gf2p8affine_epi64_epi8::<0>(x, a)
}

#[inline(always)]
unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

const TILE: usize = 64;

/// `v[k] = sum_j bit_k(pi0[j]) eq_hi[j]`, a `128 x n` by `n x 128` bit product: per tile the
/// bit-transposed bytes of `pi0` are the matrices, those of `eq_hi` the vectors, and
/// `acc[h][kb].qword[i].byte[t].bit[k']` sums `V[8kb + 7 - k'][8(8h + i) + t]`.
fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let mut v = partial_evals_tiles(pi0, eq_hi, 0, tiles);
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

/// The tiles `t0 .. t1` of [`partial_evals`] — the 128 partials over the words
/// `64 t0 .. 64 t1` alone. The accumulation is GF(2)-linear, so the partials of a split range
/// are the XOR of the partials of its parts.
fn partial_evals_tiles(pi0: &[B128], eq_hi: &[B128], t0: usize, t1: usize) -> Vec<B128> {
    debug_assert!(TILE * t1 <= pi0.len() && pi0.len() == eq_hi.len());
    let mut acc = PartialAcc::new();
    unsafe {
        acc.add_tiles(
            pi0.as_ptr().add(TILE * t0),
            eq_hi.as_ptr().add(TILE * t0),
            t1 - t0,
        );
    }
    acc.finish()
}

/// The running accumulator of [`partial_evals`]: `acc[h][kb].qword[i].byte[t].bit[k']` sums
/// `V[8kb + 7 - k'][8(8h + i) + t]` over the tiles added so far. Tiles can be added in any
/// number of calls, which is what lets the expansion of a stretch of `eq` be consumed while it
/// is still in cache.
#[repr(C, align(64))]
struct PartialAcc {
    acc: [[__m512i; 16]; 2],
}

impl PartialAcc {
    fn new() -> Self {
        PartialAcc {
            acc: [[unsafe { _mm512_setzero_si512() }; 16]; 2],
        }
    }

    /// Add `tiles` tiles of 64 words at `pi0` and `eq`.
    ///
    /// # Safety
    /// Both pointers address `64 * tiles` words.
    unsafe fn add_tiles(&mut self, pi0: *const B128, eq: *const B128, tiles: usize) {
        let ident = _mm512_set1_epi64(GF_IDENT);
        let mut xe = [_mm512_setzero_si512(); 16];
        let mut xp = [_mm512_setzero_si512(); 16];
        let mut ap = [[0u64; 8]; 16];
        for t in 0..tiles {
            transpose_tile(eq.add(TILE * t) as *const u8, &mut xe);
            transpose_tile(pi0.add(TILE * t) as *const u8, &mut xp);
            for k in 0..16 {
                _mm512_storeu_si512(ap[k].as_mut_ptr() as *mut __m512i, affine(ident, xp[k]));
            }
            let te: [__m512i; 16] = core::array::from_fn(|b| affine(ident, xe[b]));
            for (h, acc) in self.acc.iter_mut().enumerate() {
                let tep = transpose8x8_q(core::array::from_fn(|i| te[8 * h + i]));
                for (k, a) in acc.iter_mut().enumerate() {
                    for j in (0..8).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(tep[j], _mm512_set1_epi64(ap[k][j] as i64)),
                            affine(tep[j + 1], _mm512_set1_epi64(ap[k][j + 1] as i64)),
                        );
                    }
                }
            }
        }
    }

    /// The 128 partials the accumulator holds.
    fn finish(&self) -> Vec<B128> {
        let mut v = vec![B128::ZERO; PACK];
        for (h, acc) in self.acc.iter().enumerate() {
            for (k, a) in acc.iter().enumerate() {
                let q: [u64; 8] = unsafe { core::mem::transmute(*a) };
                for (i, qi) in q.iter().enumerate() {
                    for t in 0..8 {
                        let byte = (qi >> (8 * t)) & 0xff;
                        for kp in 0..8 {
                            if (byte >> kp) & 1 == 1 {
                                v[8 * k + 7 - kp].0 ^= 1u128 << (8 * (8 * h + i) + t);
                            }
                        }
                    }
                }
            }
        }
        v
    }
}

/// [`partial_evals`] with the tiles split over `threads` threads and the per-thread partials
/// XORed together; the sub-tile tail is taken once, serially.
pub fn partial_evals_parallel(pi0: &[B128], eq_hi: &[B128], threads: usize) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let runs = crate::par::ranges(tiles, threads, 1);
    let parts: Vec<Vec<B128>> = std::thread::scope(|scope| {
        let handles: Vec<_> = runs
            .into_iter()
            .map(|run| scope.spawn(move || partial_evals_tiles(pi0, eq_hi, run.start, run.end)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut v = vec![B128::ZERO; PACK];
    for part in parts {
        for (vk, &x) in v.iter_mut().zip(&part) {
            *vk = *vk + x;
        }
    }
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

/// The eight `B128` at `x` as their low words and their high words.
#[inline(always)]
unsafe fn split_words(x: *const B128) -> (__m512i, __m512i) {
    let a = _mm512_loadu_si512(x as *const __m512i);
    let b = _mm512_loadu_si512(x.add(4) as *const __m512i);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    (
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
    )
}

/// `pi0` in the sumcheck's layout: the bits of a `B128` are the low coefficients of an `F162`.
fn poly_from_b128(pi0: &[B128], out: &mut super::sumcheck::Poly) {
    let n = pi0.len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    w[2].fill(0);
    let blocks = n / 8;
    unsafe {
        for b in 0..blocks {
            let (lo, hi) = split_words(pi0.as_ptr().add(8 * b));
            _mm512_storeu_si512(w[0].as_mut_ptr().add(8 * b) as *mut __m512i, lo);
            _mm512_storeu_si512(w[1].as_mut_ptr().add(8 * b) as *mut __m512i, hi);
        }
    }
    for j in 8 * blocks..n {
        w[0][j] = pi0[j].0 as u64;
        w[1][j] = (pi0[j].0 >> 64) as u64;
    }
}

/// `m[cb][kb].byte[7 - c].bit[k] = batch[8kb + k].bit(8cb + c)`: the matrix taking byte `kb` of
/// a word to byte `cb` of its `F162` image.
///
/// Per eight images and limb `l`: the limb's qwords gathered into one vector (two permutes),
/// one byte permute making qword `c` the byte-reversed `[byte 8l + c of image k]_k`, and one
/// bit transpose against the reversed identity, whose qword `c` is then `m[8l + c][kb]`.
fn eq_matrices(batch: &[F162]) -> [[u64; 16]; 24] {
    assert_eq!(batch.len(), PACK);
    #[repr(C, align(64))]
    struct Tables {
        ab: [[u64; 8]; 3],
        c: [[u64; 8]; 3],
        bytes: [u8; 64],
    }
    static T: Tables = {
        let mut ab = [[0u64; 8]; 3];
        let mut c = [[0u64; 8]; 3];
        let mut l = 0;
        while l < 3 {
            let mut k = 0;
            while k < 8 {
                let s = 3 * k + l;
                if s < 16 {
                    ab[l][k] = s as u64;
                    c[l][k] = k as u64;
                } else {
                    c[l][k] = (8 + s - 16) as u64;
                }
                k += 1;
            }
            l += 1;
        }
        // byte 8c + r of the result is byte c of qword 7 - r
        let mut bytes = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            let (cc, r) = (i / 8, i % 8);
            bytes[i] = (8 * (7 - r) + cc) as u8;
            i += 1;
        }
        Tables { ab, c, bytes }
    };
    let mut m = [[0u64; 16]; 24];
    unsafe {
        // x.byte[j] = 1 << (7 - j): affine(rev, t).byte[j].bit[i] = t.byte[7 - i].bit[7 - j]
        let rev = _mm512_set1_epi64(0x0102_0408_1020_4080);
        let bytes = _mm512_load_si512(T.bytes.as_ptr() as *const __m512i);
        for kb in 0..PACK / 8 {
            let base = batch.as_ptr().add(8 * kb) as *const __m512i;
            let (a, b, c) = (
                _mm512_loadu_si512(base),
                _mm512_loadu_si512((base as *const u64).add(8) as *const __m512i),
                _mm512_loadu_si512((base as *const u64).add(16) as *const __m512i),
            );
            for l in 0..3 {
                let ab = _mm512_permutex2var_epi64(a, _mm512_load_si512(T.ab[l].as_ptr() as *const __m512i), b);
                let limb = _mm512_permutex2var_epi64(ab, _mm512_load_si512(T.c[l].as_ptr() as *const __m512i), c);
                let out = affine(rev, _mm512_permutexvar_epi8(bytes, limb));
                let q: [u64; 8] = core::mem::transmute(out);
                for (cc, &v) in q.iter().enumerate() {
                    m[8 * l + cc][kb] = v;
                }
            }
        }
    }
    m
}

/// [`eq_matrices`] bit by bit, the reference the tests hold it to.
#[cfg(test)]
fn eq_matrices_ref(batch: &[F162]) -> [[u64; 16]; 24] {
    let mut m = [[0u64; 16]; 24];
    for (cb, row) in m.iter_mut().enumerate() {
        for (kb, q) in row.iter_mut().enumerate() {
            for c in 0..8 {
                let mut byte = 0u64;
                for k in 0..8 {
                    byte |= ((batch[8 * kb + k].0[cb / 8] >> (8 * (cb % 8) + c)) & 1) << k;
                }
                *q |= byte << (8 * (7 - c));
            }
        }
    }
    m
}

/// `a[j] = sum_i sum_k bit_k(eq[i][j]) batch[i][k]`, straight into the sumcheck's layout: per
/// tile the byte planes of `eq` through [`eq_matrices`], and the 24 output planes back to qwords.
fn poly_from_eq(eqs: &[&[B128]], batches: &[Vec<F162>], out: &mut super::sumcheck::Poly) {
    let n = eqs[0].len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    let tiles = n / TILE;
    let ms: Vec<[[u64; 16]; 24]> = batches.iter().map(|b| eq_matrices(b)).collect();
    unsafe {
        let mut xe = [_mm512_setzero_si512(); 16];
        for t in 0..tiles {
            let mut acc = [_mm512_setzero_si512(); 24];
            for (eq, m) in eqs.iter().zip(&ms) {
                transpose_tile(eq.as_ptr().add(TILE * t) as *const u8, &mut xe);
                for (a, row) in acc.iter_mut().zip(m) {
                    for k in (0..16).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                            affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
                        );
                    }
                }
            }
            for (l, wl) in w.iter_mut().enumerate() {
                let out = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
                for (g, o) in out.iter().enumerate() {
                    _mm512_storeu_si512(wl.as_mut_ptr().add(TILE * t + 8 * g) as *mut __m512i, *o);
                }
            }
        }
    }
    if TILE * tiles < n {
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for j in TILE * tiles..n {
            let x = eqs
                .iter()
                .zip(&tabs)
                .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
            for l in 0..3 {
                w[l][j] = x.0[l];
            }
        }
    }
}

pub struct SwitchProof {
    pub v: Vec<B128>,
    pub rounds: Vec<[F162; 2]>,
    pub final_eval: F162,
}

pub struct Transcript {
    pub r_prime: Vec<F162>,
    pub r_pp: Vec<F162>,
}

pub fn prove(pi0: &[B128], r_lo: &[B128], r_hi: &[B128], challenges: &Transcript) -> SwitchProof {
    assert_eq!(r_lo.len(), LOG_PACK);
    let l = r_hi.len();
    assert_eq!(pi0.len(), 1 << l);

    let eq_hi = eq_expand_b128(r_hi);
    let v = partial_evals(pi0, &eq_hi);

    let batch = eq_expand_f162(&challenges.r_prime);
    let mut ap = super::sumcheck::Poly::zero(pi0.len());
    let mut pp = super::sumcheck::Poly::zero(pi0.len());
    poly_from_eq(&[&eq_hi], &[batch], &mut ap);
    poly_from_b128(pi0, &mut pp);
    let mut rounds = Vec::with_capacity(l);
    let mut half = 1usize << l;
    for round in 0..l {
        half /= 2;
        rounds.push(super::sumcheck::round(
            &mut ap,
            &mut pp,
            half,
            challenges.r_pp[round],
        ));
    }
    let final_eval = pp.get(0);

    SwitchProof {
        v,
        rounds,
        final_eval,
    }
}

pub fn verify(
    proof: &SwitchProof,
    claim: B128,
    r_lo: &[B128],
    r_hi: &[B128],
    challenges: &Transcript,
) -> Result<F162, &'static str> {
    let eq_lo = eq_expand_b128(r_lo);
    let recomputed = proof
        .v
        .iter()
        .zip(&eq_lo)
        .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
    if recomputed != claim {
        return Err("partial evaluation mismatch");
    }

    let mut u = vec![0u128; PACK];
    for (i, &vi) in proof.v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    let batch = eq_expand_f162(&challenges.r_prime);
    let mut s = F162::ZERO;
    for k in 0..PACK {
        s += F162::from_b128(B128(u[k])) * batch[k];
    }

    for (round, msg) in proof.rounds.iter().enumerate() {
        let [e0, einf] = *msg;
        let e1 = s + e0;
        let r = challenges.r_pp[round];
        s = e0 + r * (e0 + e1 + einf) + r * r * einf;
    }

    let d = transparent_coeff(r_hi, &challenges.r_pp, &batch);
    if s != d * proof.final_eval {
        return Err("sumcheck final check failed");
    }
    Ok(proof.final_eval)
}

/// The switch's buffers, sized once and reused: fresh pages cost more than the kernels do.
pub struct Workspace {
    eq: Vec<Vec<B128>>,
    a: super::sumcheck::Poly,
    p: super::sumcheck::Poly,
}

impl Workspace {
    pub fn new(l: usize, claims: usize) -> Self {
        Self {
            eq: (0..claims).map(|_| vec![B128::ZERO; 1 << l]).collect(),
            a: super::sumcheck::Poly::zero(1 << l),
            p: super::sumcheck::Poly::zero(1 << l),
        }
    }

    pub fn fits(&self, l: usize, claims: usize) -> bool {
        self.eq.len() == claims && self.a.n == 1 << l
    }

    /// The partial evaluations of claim `i` at `r_hi`; its `eq` stays for the sumcheck.
    pub fn partial_evals(&mut self, i: usize, pi0: &[B128], r_hi: &[B128]) -> Vec<B128> {
        eq_expand_b128_into(r_hi, &mut self.eq[i]);
        partial_evals(pi0, &self.eq[i])
    }
}

pub struct SwitchProver {
    ws: Workspace,
    half: usize,
}

impl SwitchProver {
    pub fn partial_evals_and_eq(pi0: &[B128], r_hi: &[B128]) -> (Vec<B128>, Vec<B128>) {
        let eq_hi = eq_expand_b128(r_hi);
        let v = partial_evals(pi0, &eq_hi);
        (v, eq_hi)
    }

    pub fn new(pi0: &[B128], eq_hi: &[B128], batch: &[F162]) -> Self {
        Self::batched(pi0, &[eq_hi], &[F162::ONE], batch)
    }

    pub fn batched(pi0: &[B128], eq_his: &[&[B128]], gammas: &[F162], batch: &[F162]) -> Self {
        let mut ws = Workspace::new(pi0.len().trailing_zeros() as usize, eq_his.len());
        for (dst, src) in ws.eq.iter_mut().zip(eq_his) {
            dst.copy_from_slice(src);
        }
        Self::from_workspace(ws, pi0, gammas, batch)
    }

    /// The sumcheck over the `eq` the workspace holds from [`Workspace::partial_evals`].
    pub fn from_workspace(
        mut ws: Workspace,
        pi0: &[B128],
        gammas: &[F162],
        batch: &[F162],
    ) -> Self {
        let batches: Vec<Vec<F162>> = gammas
            .iter()
            .map(|&g| batch.iter().map(|&x| g * x).collect())
            .collect();
        let views: Vec<&[B128]> = ws.eq.iter().map(Vec::as_slice).collect();
        poly_from_eq(&views, &batches, &mut ws.a);
        poly_from_b128(pi0, &mut ws.p);
        Self {
            ws,
            half: pi0.len(),
        }
    }

    pub fn into_workspace(self) -> Workspace {
        self.ws
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        super::sumcheck::msg(&self.ws.a, &self.ws.p, self.half)
    }

    pub fn fold(&mut self, r: F162) {
        super::sumcheck::fold(&mut self.ws.a, &mut self.ws.p, self.half, r);
    }

    pub fn final_eval(&self) -> F162 {
        self.ws.p.get(0)
    }
}

pub fn slice_sum(v: &[B128], batch: &[F162]) -> F162 {
    let mut u = vec![0u128; PACK];
    for (i, &vi) in v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    (0..PACK).fold(F162::ZERO, |s, k| {
        s + F162::from_b128(B128(u[k])) * batch[k]
    })
}

pub struct SwitchVerifier {
    pub s: F162,
    round: usize,
}

impl SwitchVerifier {
    pub fn start(
        v: &[B128],
        claim: B128,
        r_lo: &[B128],
        batch: &[F162],
    ) -> Result<Self, &'static str> {
        let eq_lo = eq_expand_b128(r_lo);
        let recomputed = v
            .iter()
            .zip(&eq_lo)
            .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
        if recomputed != claim {
            return Err("partial evaluation mismatch");
        }
        Ok(Self::from_sum(slice_sum(v, batch)))
    }

    pub const fn from_sum(s: F162) -> Self {
        Self { s, round: 0 }
    }

    pub fn round(&mut self, msg: [F162; 2], r: F162) {
        let [e0, einf] = msg;
        let e1 = self.s + e0;
        self.s = e0 + r * (e0 + e1 + einf) + r * r * einf;
        self.round += 1;
    }

    pub fn finish(
        &self,
        r_hi: &[B128],
        r_pp: &[F162],
        batch: &[F162],
        opened: F162,
    ) -> Result<(), &'static str> {
        let d = transparent_coeff(r_hi, r_pp, batch);
        if d == F162::ZERO {
            return Err("the switch left the opened value unbound");
        }
        if self.s != d * opened {
            return Err("sumcheck final check failed");
        }
        Ok(())
    }
}

pub fn eval_pi1(pi0: &[B128], r_pp: &[F162]) -> F162 {
    let eq = eq_expand_f162(r_pp);
    pi0.iter()
        .zip(&eq)
        .fold(F162::ZERO, |a, (&p, &e)| a + F162::from_b128(p) * e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
        let mut v = vec![B128::ZERO; PACK];
        for (&p, &e) in pi0.iter().zip(eq_hi) {
            for (k, vk) in v.iter_mut().enumerate() {
                if p.bit(k) {
                    *vk = *vk + e;
                }
            }
        }
        v
    }

    #[test]
    fn poly_from_eq_matches_psi() {
        let mut x = 0x2545F4914F6CDD1Du128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let batches: Vec<Vec<F162>> = (0..2)
            .map(|_| {
                (0..PACK)
                    .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
                    .collect()
            })
            .collect();
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for n in [1usize, 8, 13, 64, 200, 1024] {
            let eqs: Vec<Vec<B128>> = (0..2)
                .map(|_| (0..n).map(|_| B128(next())).collect())
                .collect();
            let views: Vec<&[B128]> = eqs.iter().map(|e| e.as_slice()).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_eq(&views, &batches, &mut poly);
            for j in 0..n {
                let want = views
                    .iter()
                    .zip(&tabs)
                    .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
                assert_eq!(poly.get(j), want);
            }
            let p: Vec<B128> = (0..n).map(|_| B128(next())).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_b128(&p, &mut poly);
            for j in 0..n {
                assert_eq!(poly.get(j), F162::from_b128(p[j]));
            }
        }
    }

    const M34_TEST: u64 = (1 << 34) - 1;

    #[test]
    fn workspace_reuse_is_clean() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let l = 8;
        let pi0: Vec<B128> = (0..1 << l).map(|_| B128(next())).collect();
        let r_hi: Vec<B128> = (0..l).map(|_| B128(next())).collect();
        let batch: Vec<F162> = (0..PACK)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let r: Vec<F162> = (0..l)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let run = |ws: Workspace| {
            let mut ws = ws;
            let v = ws.partial_evals(0, &pi0, &r_hi);
            let mut p = SwitchProver::from_workspace(ws, &pi0, &[F162::ONE], &batch);
            let mut msgs = Vec::new();
            for &ri in &r {
                msgs.push(p.msg());
                p.fold(ri);
            }
            (v, msgs, p.final_eval(), p.into_workspace())
        };
        let (v1, m1, f1, ws) = run(Workspace::new(l, 1));
        let (v2, m2, f2, _) = run(ws);
        assert_eq!((v1, m1, f1), (v2, m2, f2));
    }

    #[test]
    fn eq_matrices_match_the_bit_loop() {
        let mut x = 0x5A5A_1234_F0F0_9876_0F0F_ABCD_A5A5_4321u128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..8 {
            let batch: Vec<F162> = (0..PACK)
                .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
                .collect();
            assert_eq!(eq_matrices(&batch), eq_matrices_ref(&batch));
        }
        let unit: Vec<F162> = (0..PACK).map(|i| F162([1u64 << (i % 64), 1u64 << ((i + 7) % 64), 1u64 << (i % 34)])).collect();
        assert_eq!(eq_matrices(&unit), eq_matrices_ref(&unit));
    }

    #[test]
    fn raw_finish_matches_the_bit_loop() {
        let mut x = 0x0F1E_2D3C_4B5A_6978_8796_A5B4_C3D2_E1F0u128;
        for _ in 0..8 {
            let mut acc = PartialAcc::new();
            let lanes: &mut [u64] = unsafe {
                core::slice::from_raw_parts_mut(acc.acc.as_mut_ptr() as *mut u64, 2 * 16 * 8)
            };
            for l in lanes.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *l = (x >> 32) as u64;
            }
            assert_eq!(finish_raw(&acc.acc), acc.finish());
        }
    }

    #[test]
    fn partial_evals_match_the_bitwise_sum() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            B128(x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (x >> 64))
        };
        for n in [1usize, 7, 8, 64, 1000, 4096] {
            let pi0: Vec<B128> = (0..n).map(|_| next()).collect();
            let eq: Vec<B128> = (0..n).map(|_| next()).collect();
            assert_eq!(partial_evals(&pi0, &eq), reference(&pi0, &eq));
        }
    }
}

// =============================================================================================
// buffers the big tables live in
// =============================================================================================

/// `len` elements of `T` on 2 MB-aligned storage with transparent huge pages requested where the
/// platform has them (a hint; the arithmetic is the same on 4 KB pages). Uninitialised: the
/// owner writes every element before reading any, through [`as_mut_ptr`](Self::as_mut_ptr).
pub struct HugeBuf<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}

unsafe impl<T: Send> Send for HugeBuf<T> {}
unsafe impl<T: Sync> Sync for HugeBuf<T> {}

#[cfg(target_os = "linux")]
extern "C" {
    fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
}

impl<T> HugeBuf<T> {
    pub fn new(len: usize) -> Self {
        let bytes = (len * core::mem::size_of::<T>()).max(1);
        let align = (1usize << 21).max(core::mem::align_of::<T>());
        let layout = std::alloc::Layout::from_size_align(bytes, align).expect("a valid layout");
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut T;
        assert!(!ptr.is_null(), "out of memory allocating {bytes} bytes");
        #[cfg(target_os = "linux")]
        unsafe {
            // MADV_HUGEPAGE = 14; the return value is advisory.
            madvise(ptr as *mut core::ffi::c_void, bytes, 14);
        }
        HugeBuf { ptr, len, layout }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// The elements as a slice.
    ///
    /// # Safety
    /// Every element must have been written.
    pub unsafe fn as_slice(&self) -> &[T] {
        core::slice::from_raw_parts(self.ptr, self.len)
    }

    /// The buffer's address and size in bytes, for [`touch_pages`] on another thread.
    pub fn region(&self) -> (usize, usize) {
        (self.ptr as usize, self.len * core::mem::size_of::<T>())
    }
}

/// Fault in every page of `[addr, addr + bytes)` by writing one zero byte per 4 KB (a huge
/// page faults on its first touch and the rest of its touches hit), so that the first real
/// writer finds the pages mapped. Meant for a background thread while a compute-bound phase
/// runs.
///
/// # Safety
/// The region must be writable, owned by the caller, and not read or written by anyone else
/// until this returns.
pub unsafe fn touch_pages(addr: usize, bytes: usize) {
    let p = addr as *mut u8;
    let mut off = 0;
    while off < bytes {
        core::ptr::write_volatile(p.add(off), 0);
        off += 4096;
    }
    if bytes > 0 {
        core::ptr::write_volatile(p.add(bytes - 1), 0);
    }
}

impl<T> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) };
    }
}


/// Scale a row by one fixed GHASH field element. Intel's precomputed-operand
/// multiply/reduce form (2023, section 2.2.1) removes one serial folding step:
/// K = b_hi * (x^7+x^2+x+1) + b_lo*x^64.
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
unsafe fn scale_b128_row(input: &[B128], factor: B128, out: *mut B128) {
    let f = _mm_set_epi64x((factor.0 >> 64) as i64, factor.0 as i64);
    let poly128 = _mm_set_epi64x(0, 0x87);
    let kh = _mm_clmulepi64_si128::<0x01>(f, poly128);
    let k = _mm_xor_si128(kh, _mm_slli_si128::<8>(f));
    let b = _mm512_broadcast_i32x4(f);
    let k = _mm512_broadcast_i32x4(k);
    let poly = _mm512_set1_epi64(0x87);
    for i in (0..input.len()/4*4).step_by(4) {
        let a = _mm512_loadu_si512(input.as_ptr().add(i) as *const __m512i);
        let y0 = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x00>(a, b),
            _mm512_clmulepi64_epi128::<0x01>(a, k),
        );
        let y1 = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x10>(a, b),
            _mm512_clmulepi64_epi128::<0x11>(a, k),
        );
        let lo = _mm512_xor_si512(y0, _mm512_bslli_epi128::<8>(y1));
        let hi = _mm512_bsrli_epi128::<8>(y1);
        let res = _mm512_xor_si512(lo, _mm512_clmulepi64_epi128::<0x00>(hi, poly));
        _mm512_storeu_si512(out.add(i) as *mut __m512i, res);
    }
    for i in input.len()/4*4..input.len() {
        out.add(i).write(input[i]*factor);
    }
}

#[cfg(test)]
mod ghash_fixed_tests {
    use super::*;
    #[test]
    fn fixed_row_matches_scalar_on_basis_and_random() {
        if !std::arch::is_x86_feature_detected!("vpclmulqdq") { return; }
        let mut x=0x0123456789abcdef_fedcba9876543210u128;
        let mut next=|| { x^=x<<13; x^=x>>7; x^=x<<17; B128(x) };
        let mut inputs:Vec<_>=(0..128).map(|i|B128(1u128<<i)).collect();
        inputs.extend([B128::ZERO,B128::ONE,B128(u128::MAX)]);
        inputs.extend((0..2050).map(|_| next()));
        let factors:Vec<_>=[B128::ZERO,B128::ONE,B128(u128::MAX)]
            .into_iter().chain((0..48).map(|_|next())).collect();
        for f in factors {
            let mut out=vec![B128::ZERO;inputs.len()];
            unsafe { scale_b128_row(&inputs,f,out.as_mut_ptr()); }
            for (i,(&a,&b)) in inputs.iter().zip(&out).enumerate() {
                assert_eq!(b,a*f,"input {i}, factor {f:?}");
            }
        }
    }
}

// =============================================================================================
// the expansion on several threads
// =============================================================================================

/// [`eq_expand_b128_into`] on up to `threads` threads, into the `2^r.len()` words at `out`.
///
/// The serial doubling appends the newest coordinate in the lowest bit of the index, so entry
/// `j` of the table is `high[j >> h] * low[j & (2^h - 1)]` with `high` the expansion of the
/// first `n - h` coordinates and `low` that of the last `h`: the two small tables are built
/// serially and the product is cut over the threads along `j >> h`, each writing its own
/// contiguous run. Below 16 coordinates the serial expansion is used as it is.
///
/// # Safety
/// `out` must be writable for `2^r.len()` words.
pub unsafe fn eq_expand_b128_parallel(r: &[B128], out: *mut B128, threads: usize) {
    let n = r.len();
    if n < 16 || threads <= 1 {
        let t = eq_expand_b128(r);
        core::ptr::copy_nonoverlapping(t.as_ptr(), out, t.len());
        return;
    }
    let h = n / 2;
    let high = eq_expand_b128(&r[..n - h]);
    let low = eq_expand_b128(&r[n - h..]);
    let runs = crate::par::ranges(high.len(), threads, 1);
    let out = out as usize;
    std::thread::scope(|scope| {
        for run in runs {
            let (high, low) = (&high, &low);
            scope.spawn(move || {
                let out = out as *mut B128;
                for jh in run {
                    let hv = high[jh];
                    let base = unsafe { out.add(jh << h) };
                    for (jl, &lv) in low.iter().enumerate() {
                        unsafe { base.add(jl).write(hv * lv) };
                    }
                }
            });
        }
    });
}

/// [`eq_expand_b128_parallel`] and [`partial_evals_parallel`] in one pass: each thread writes
/// its stretch of the expansion (one `2^h`-word run per high index, 64 KB at `h = 12`) and
/// feeds it to the partials while it is still in cache, so the table is written once and never
/// read back from memory for the partials. The expansion at `out` is the full table on return.
/// Below 16 coordinates the two serial steps are run one after the other.
///
/// # Safety
/// `out` must be writable for `2^r.len()` words, which is `pi0.len()`.
pub unsafe fn eq_expand_and_partials(
    r: &[B128],
    pi0: &[B128],
    out: *mut B128,
    threads: usize,
) -> Vec<B128> {
    let n = r.len();
    assert_eq!(pi0.len(), 1 << n, "one word per eq entry");
    if n < 16 || threads <= 1 {
        eq_expand_b128_parallel(r, out, 1);
        let eq = core::slice::from_raw_parts(out, 1 << n);
        return partial_evals(pi0, eq);
    }
    let h = n / 2;
    let high = eq_expand_b128(&r[..n - h]);
    let low = eq_expand_b128(&r[n - h..]);
    debug_assert_eq!(low.len() % TILE, 0, "a stretch is whole tiles");
    let runs = crate::par::ranges(high.len(), threads, 1);
    let (out, pi0_addr) = (out as usize, pi0.as_ptr() as usize);
    let parts: Vec<Vec<B128>> = crate::par::pool().map(runs.len(), |i| {
        let run = runs[i].clone();
        let out = out as *mut B128;
        let pi0 = pi0_addr as *const B128;
        let mut acc = PartialAcc::new();
        for jh in run {
            let hv = high[jh];
            let base = unsafe { out.add(jh << h) };
            unsafe { scale_b128_row(&low, hv, base) };
            unsafe { acc.add_tiles(pi0.add(jh << h), base, low.len() / TILE) };
        }
        acc.finish()
    });
    let mut v = vec![B128::ZERO; PACK];
    for part in parts {
        for (vk, &x) in v.iter_mut().zip(&part) {
            *vk = *vk + x;
        }
    }
    v
}


/// The first quarter of the expansion and the partials of each quarter of the words against
/// it: `out` receives `eq[j]` for `j < n / 4` (the expansion of coordinates `2 ..`, scaled by
/// `(1 + r_hi[0]) (1 + r_hi[1])`), and set `q` of the result is
/// `V_q[b] = sum_{j < n/4} eq[j] bit_b(pi0[j + q n/4])`.
///
/// Every other entry of the table is a scalar multiple of a first-quarter entry — the index's
/// two leading bits are coordinates 0 and 1, so `eq[j + q n/4] = eq[j] rho_q` with `rho_q` the
/// product of `c_k = r_hi[k] / (1 + r_hi[k])` over the set bits of `q` — so the 128 partials of
/// the whole table are `V_0 + c_1 V_1 + c_0 V_2 + c_0 c_1 V_3`, and the first two messages of the
/// switch are combinations of the four sets as well ([`LeanSwitch::with_quarters`]). The
/// expansion is written once at a quarter of the size and the words are read once.
///
/// Requires at least 18 coordinates and no coordinate equal to one; the caller takes the full
/// path otherwise.
///
/// # Safety
/// `out` must be writable for `2^(r.len() - 2)` words, which is `pi0.len() / 4`.
pub unsafe fn eq_expand_and_partials_quarters(
    r: &[B128], pi0: &[B128], out: *mut B128, threads: usize,
) -> [Vec<B128>; 4] {
    eq_expand_and_partials_blocks::<4>(r, pi0, out, threads)
}

/// The first eighth of eq and eight partial sets. The output must hold pi0.len()/8 words.
pub unsafe fn eq_expand_and_partials_eighths(
    r: &[B128], pi0: &[B128], out: *mut B128, threads: usize,
) -> [Vec<B128>; 8] {
    eq_expand_and_partials_blocks::<8>(r, pi0, out, threads)
}

/// First sixteenth of eq; out must hold pi0.len()/16 words.
pub unsafe fn eq_expand_and_partials_sixteenths(
    r: &[B128], pi0: &[B128], out: *mut B128, threads: usize,
) -> [Vec<B128>; 16] {
    eq_expand_and_partials_blocks::<16>(r, pi0, out, threads)
}

/// The sixteen prefix partial sets of [`eq_expand_and_partials_sixteenths`] with the first
/// block of the expansion returned as [`EqFactors`] instead of written out: each runner
/// expands a row into a buffer of its own, consumes it, and moves on.
///
/// Requires at least 18 coordinates; the coordinates must not be one for the factors to be
/// the switch's first block (the caller checks that before).
pub fn eq_expand_and_partials_factored(
    r: &[B128], pi0: &[B128], threads: usize,
) -> ([Vec<B128>; 16], EqFactors) {
    unsafe { partials_blocks::<16>(r, pi0, threads) }
}

/// [`eq_expand_and_partials_factored`] with `G` prefix blocks (a power of two, at most 256):
/// `G` partial sets, and the first `1 / G` of the expansion as factors.
pub fn eq_expand_and_partials_factored_blocks<const G: usize>(
    r: &[B128], pi0: &[B128], threads: usize,
) -> ([Vec<B128>; G], EqFactors) {
    assert!(G.is_power_of_two() && G <= 256);
    unsafe { partials_blocks::<G>(r, pi0, threads) }
}

unsafe fn eq_expand_and_partials_blocks<const G: usize>(
    r: &[B128], pi0: &[B128], out: *mut B128, threads: usize,
) -> [Vec<B128>; G] {
    let (sets, f) = partials_blocks::<G>(r, pi0, threads);
    for (jh, &hv) in f.high.iter().enumerate() {
        scale_b128_row(&f.low, hv, out.add(jh << f.h));
    }
    sets
}

unsafe fn partials_blocks<const G: usize>(
    r: &[B128], pi0: &[B128], threads: usize,
) -> ([Vec<B128>; G], EqFactors) {
    let n = r.len();
    assert!(n >= 18, "the prefix path wants a large table");
    assert_eq!(pi0.len(), 1 << n, "one word per eq entry");
    let prefix = G.trailing_zeros() as usize;
    assert!(G.is_power_of_two() && n >= prefix + 12);
    let block_len = 1usize << (n - prefix);
    let rest = &r[prefix..];
    let h = rest.len() / 2;
    let scale = r[..prefix].iter().fold(B128::ONE, |a, &ri| a * (B128::ONE + ri));
    let high: Vec<B128> = eq_expand_b128(&rest[..rest.len() - h])
        .into_iter()
        .map(|x| x * scale)
        .collect();
    let low = eq_expand_b128(&rest[rest.len() - h..]);
    debug_assert_eq!(low.len() % TILE, 0, "a stretch is whole tiles");
    debug_assert_eq!(high.len() * low.len(), block_len);
    let groups = low.len() / 8;
    // Rows are handed out a few at a time from a shared counter, so a runner that starts late
    // or shares its core does less of them instead of holding the others up.
    let rows = high.len();
    let chunk = (rows / (32 * threads.max(1))).clamp(1, 8);
    let chunks = rows.div_ceil(chunk);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let runners = threads.max(1).min(chunks);
    let pi0_addr = pi0.as_ptr() as usize;
    let parts: Vec<Vec<RawAcc>> = crate::par::pool().map(runners, |_| {
        let pi0 = pi0_addr as *const B128;
        let mut acc: Vec<RawAcc> = vec![[[_mm512_setzero_si512(); 16]; 2]; G];
        let mut scratch = RowScratch::new(groups);
        let mut row = vec![B128::ZERO; low.len()];
        loop {
            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if c >= chunks {
                break;
            }
            let end = ((c + 1) * chunk).min(rows);
            for jh in c * chunk..end {
                scale_b128_row(&low, high[jh], row.as_mut_ptr());
                let next_row = if jh + 1 < end { pi0.add((jh + 1) << h) } else { core::ptr::null() };
                partials_row(&mut acc, pi0.add(jh << h), block_len, row.as_ptr(), groups, &mut scratch, next_row);
            }
        }
        acc
    });
    let mut total: Vec<RawAcc> = vec![[[_mm512_setzero_si512(); 16]; 2]; G];
    for part in &parts {
        for (t, p) in total.iter_mut().zip(part) {
            for hh in 0..2 {
                for k in 0..16 {
                    t[hh][k] = _mm512_xor_si512(t[hh][k], p[hh][k]);
                }
            }
        }
    }
    let v = core::array::from_fn(|q| finish_raw(&total[q]));
    (v, EqFactors { high, low, h })
}

/// One partial set's accumulator in the layout of [`PartialAcc`]: `acc[h][k].qword[i].byte[t]
/// .bit[k']` sums `V[8k + 7 - k'][8(8h + i) + t]`.
type RawAcc = [[__m512i; 16]; 2];

/// `vpermt2b` indices taking eight words held in two vectors (words 0..4, then 4..8) to
/// `out[h].qword[l].byte[s] = byte 8h + l of word s`.
#[repr(C, align(64))]
struct GroupIdx([[u8; 64]; 2]);

static GROUP_IDX: GroupIdx = GroupIdx({
    let mut t = [[0u8; 64]; 2];
    let mut h = 0;
    while h < 2 {
        let mut l = 0;
        while l < 8 {
            let mut s = 0;
            while s < 8 {
                t[h][8 * l + s] = (if s >= 4 { 64 } else { 0 } | (16 * (s & 3) + 8 * h + l)) as u8;
                s += 1;
            }
            l += 1;
        }
        h += 1;
    }
    t
});

/// The eight words at `p` bit-sliced across the words: `out[h].qword[l].byte[j].bit[s]` is bit
/// `8(8h + l) + j` of word `7 - s`. Two byte permutes and two bit transposes.
#[inline(always)]
unsafe fn slice_group(p: *const B128, idx: &[__m512i; 2], ident: __m512i) -> [__m512i; 2] {
    let z0 = _mm512_loadu_si512(p as *const __m512i);
    let z1 = _mm512_loadu_si512(p.add(4) as *const __m512i);
    [
        affine(ident, _mm512_permutex2var_epi8(z0, idx[0], z1)),
        affine(ident, _mm512_permutex2var_epi8(z0, idx[1], z1)),
    ]
}

/// Per-runner buffers of [`partials_row`]: the row's `eq` and one block's words, sliced.
struct RowScratch {
    xs: Vec<[__m512i; 2]>,
    aps: Vec<[__m512i; 2]>,
}

impl RowScratch {
    fn new(groups: usize) -> Self {
        let z = unsafe { _mm512_setzero_si512() };
        RowScratch { xs: vec![[z; 2]; groups], aps: vec![[z; 2]; groups] }
    }
}

/// The partials of one row: for every block `q`, `acc[q] += sum_j eq[j] bit(words[q stride + j])`
/// over the `8 groups` words of the row. The row's `eq` is sliced once; per block the words are
/// sliced into a buffer whose qwords are the broadcast matrices, and each half of the block's
/// accumulator stays in registers across the row: 256 products per 64 words, no transposes
/// beyond the two permutes per eight words.
///
/// The words of the next block (or, after the last block, `next_row`'s first block when it is
/// not null) are prefetched during each block's products, which are long enough to cover the
/// memory latency: the words then arrive while the GFNI port is busy instead of after it.
#[inline(never)]
unsafe fn partials_row(
    acc: &mut [RawAcc], words: *const B128, stride: usize, eq: *const B128, groups: usize,
    scratch: &mut RowScratch, next_row: *const B128,
) {
    debug_assert!(groups % 2 == 0 && scratch.xs.len() >= groups);
    let ident = _mm512_set1_epi64(GF_IDENT);
    let idx = [
        _mm512_load_si512(GROUP_IDX.0[0].as_ptr() as *const __m512i),
        _mm512_load_si512(GROUP_IDX.0[1].as_ptr() as *const __m512i),
    ];
    let xs = scratch.xs.as_mut_ptr();
    for g in 0..groups {
        *xs.add(g) = slice_group(eq.add(8 * g), &idx, ident);
    }
    let aps = scratch.aps.as_mut_ptr();
    let blocks = acc.len();
    for (q, acc_q) in acc.iter_mut().enumerate() {
        let w = words.add(q * stride);
        for g in 0..groups {
            *aps.add(g) = slice_group(w.add(8 * g), &idx, ident);
        }
        let next = if q + 1 < blocks { w.add(stride) } else { next_row };
        for (h, acc_qh) in acc_q.iter_mut().enumerate() {
            let mut a = *acc_qh;
            let mut g = 0;
            while g < groups {
                if !next.is_null() {
                    // One group of the next block per pair of groups here: half the groups
                    // in each of the two passes.
                    let p = next.add(8 * (h * groups / 2 + g / 2)) as *const i8;
                    _mm_prefetch::<_MM_HINT_T1>(p);
                    _mm_prefetch::<_MM_HINT_T1>(p.add(64));
                }
                let x0 = (*xs.add(g))[h];
                let x1 = (*xs.add(g + 1))[h];
                let m0 = aps.add(g) as *const i64;
                let m1 = aps.add(g + 1) as *const i64;
                for k in 0..16 {
                    a[k] = xor3(
                        a[k],
                        affine(x0, _mm512_set1_epi64(*m0.add(k))),
                        affine(x1, _mm512_set1_epi64(*m1.add(k))),
                    );
                }
                g += 2;
            }
            *acc_qh = a;
        }
    }
}

/// [`PartialAcc::finish`] on a raw accumulator, with bit transposes instead of bit loops:
/// `affine(ident, a)` gives `qword[i].byte[j] = reverse(byte 8h + i of V[8k + 7 - j])`, a second
/// affine reverses the bits back, and one byte permute per four words gathers each word's
/// sixteen bytes from the two halves.
fn finish_raw(acc: &RawAcc) -> Vec<B128> {
    static IDX: GroupIdx = GroupIdx({
        // out_m.byte[16 w + b] = R[b >> 3].qword[b & 7].byte[7 - 4m - w]
        let mut t = [[0u8; 64]; 2];
        let mut m = 0;
        while m < 2 {
            let mut w = 0;
            while w < 4 {
                let mut b = 0;
                while b < 16 {
                    t[m][16 * w + b] = (((b >> 3) << 6) | ((b & 7) << 3) | (7 - 4 * m - w)) as u8;
                    b += 1;
                }
                w += 1;
            }
            m += 1;
        }
        t
    });
    let mut v = vec![B128::ZERO; PACK];
    unsafe {
        let ident = _mm512_set1_epi64(GF_IDENT);
        let idx = [
            _mm512_load_si512(IDX.0[0].as_ptr() as *const __m512i),
            _mm512_load_si512(IDX.0[1].as_ptr() as *const __m512i),
        ];
        for k in 0..16 {
            let r: [__m512i; 2] = core::array::from_fn(|h| affine(affine(ident, acc[h][k]), ident));
            for m in 0..2 {
                let o = _mm512_permutex2var_epi8(r[0], idx[m], r[1]);
                _mm512_storeu_si512(v.as_mut_ptr().add(8 * k + 4 * m) as *mut __m512i, o);
            }
        }
    }
    v
}

/// `c_k = r / (1 + r)`, the ratio between the two halves of the table at coordinate `k`.
pub fn half_ratio(r: B128) -> B128 {
    r * b128_inv(B128::ONE + r)
}

/// The 128 partials of the whole table from the four quarter sets.
pub fn partials_from_quarters(v: &[Vec<B128>; 4], r_hi: &[B128]) -> Vec<B128> {
    partials_from_blocks(v, r_hi)
}

pub fn partials_from_eighths(v: &[Vec<B128>; 8], r_hi: &[B128]) -> Vec<B128> {
    partials_from_blocks(v, r_hi)
}

pub fn partials_from_sixteenths(v: &[Vec<B128>; 16], r_hi: &[B128]) -> Vec<B128> {
    partials_from_blocks(v, r_hi)
}

/// The 128 partials of the whole table from any power-of-two number of prefix-block sets.
pub fn partials_from_sets(v: &[Vec<B128>], r_hi: &[B128]) -> Vec<B128> {
    assert!(v.len().is_power_of_two());
    partials_from_blocks(v, r_hi)
}

fn partials_from_blocks(v: &[Vec<B128>], r_hi: &[B128]) -> Vec<B128> {
    let prefix = v.len().trailing_zeros() as usize;
    let c: Vec<_> = r_hi[..prefix].iter().copied().map(half_ratio).collect();
    let scales: Vec<_> = (0..v.len()).map(|q| {
        (0..prefix).filter(|&k| q & (1 << (prefix - 1 - k)) != 0)
            .fold(B128::ONE, |a, k| a * c[k])
    }).collect();
    (0..PACK).map(|b| v.iter().zip(&scales)
        .fold(B128::ZERO, |a, (v, &c)| a + c * v[b])).collect()
}

// =============================================================================================
// the lean switch prover
// =============================================================================================
//
// [`SwitchProver`] materialises the two sumcheck polynomials in full — `a[j] = sum_k bit_k(eq[j])
// batch[k]` and `p[j] = lift(pi0[j])`, 24 bytes each, 2^l of each — and halves both round by
// round. Neither is needed in that form.
//
// `a` is a GF(2)-linear image of the `eq` word: `a_0[j] = L_0(eq[j])` with `L_0` the map
// `poly_from_eq` applies. The fold `a_{k+1}[j] = a_k[j] + r_k (a_k[j] + a_k[j + half])` keeps
// it one: bit `l - 1 - k` of the index is coordinate `k`, so for `j < half` the two halves
// differ by the constant `c_k = r_hi[k] / (1 + r_hi[k])`, `eq[j + half] = c_k eq[j]`, and
//
//     a_{k+1}[j] = L_k(eq[j]) + r_k L_k((1 + c_k) eq[j]) = L_{k+1}(eq[j]),
//     L_{k+1}(x) = L_k(x) + r_k L_k(d_k x),     d_k = 1 + c_k = (1 + r_hi[k])^-1,
//
// which is again linear in `x`. So the prover keeps `L_k` as the images of the 128 basis words
// and applies it to the *original* `eq` table, which it never rewrites, through the same GFNI
// tiles `poly_from_eq` uses; the message's `a_k[j] + a_k[j + half]` is the second map
// `L'_k = L_k o (x -> d_k x)` on the same tile. The identity needs `1 + r_hi[k] != 0`; a point
// with a coordinate equal to one is refused by [`LeanSwitch::new`] and the caller falls back.
//
// Without prefix partials, p_1 is folded from the words on the fly and p_2 is
// materialized. With 2^d partial sets, the first d messages use only those sets;
// p_d is then materialized directly as an unreduced weighted dot product of the
// original words, reducing once per output. The remaining folds are fused with
// the following message, using the low part of the same three plane buffers.
//
// The messages are the values [`SwitchProver`] sends: the same products, reassociated over the
// threads (XOR), so the transcript and the proof are unchanged.

use super::sumcheck::Poly;

/// `x^-1` in `B128` (`x^(2^128 - 2)`); `x` must be nonzero.
fn b128_inv(x: B128) -> B128 {
    let mut r = B128::ONE;
    let mut b = x;
    let mut e: u128 = u128::MAX - 1;
    while e != 0 {
        if e & 1 == 1 {
            r = r * b;
        }
        b = b * b;
        e >>= 1;
    }
    r
}

/// `L(x) = sum_k bit_k(x) images[k]`.
fn apply_images(images: &[F162], x: B128) -> F162 {
    assert_eq!(images.len(), PACK);
    let mut acc = F162::ZERO;
    let mut bits = x.0;
    while bits != 0 {
        let k = bits.trailing_zeros() as usize;
        acc += images[k];
        bits &= bits - 1;
    }
    acc
}

/// `L o (x -> d x)` as images: `L'(e_i) = L(d e_i)`.
fn compose_scale(images: &[F162], d: B128) -> Vec<F162> {
    // Apply the same tiled linear map used by the long passes to the 128 scaled
    // basis words. This avoids 128 unpredictable bit tests for every image.
    let words: [B128; PACK] = core::array::from_fn(|i| B128(1u128 << i) * d);
    let m = eq_matrices(images);
    let mut out = vec![F162::ZERO; PACK];
    unsafe {
        for t in 0..PACK / TILE {
            let mut mapped = [[_mm512_setzero_si512(); 8]; 3];
            map_tile(words.as_ptr().add(TILE * t), &m, &mut mapped);
            let lanes: [[[u64; 8]; 8]; 3] = core::mem::transmute(mapped);
            for g in 0..8 {
                for i in 0..8 {
                    out[TILE * t + 8 * g + i] = F162(core::array::from_fn(|l| lanes[l][g][i]));
                }
            }
        }
    }
    out
}

/// Where a pass reads `p_k` from: the words themselves in round 0, the words folded once by
/// the first challenge in round 1 (`p_1[j] = p_0[j] + r_0 (p_0[j] + p_0[j + half_0])`, never
/// stored), and the planes from round 2 on.
#[derive(Clone, Copy)]
enum Src {
    Words(*const B128),
    WordsFolded {
        words: *const B128,
        half: usize,
        r: F162,
    },
    Planes([*const u64; 3]),
}

#[inline(always)]
unsafe fn load_words(w: *const B128) -> [__m512i; 3] {
    let (lo, hi) = split_words(w);
    [lo, hi, _mm512_setzero_si512()]
}

#[inline(always)]
unsafe fn load_p(src: Src, j: usize) -> [__m512i; 3] {
    match src {
        Src::Words(w) => load_words(w.add(j)),
        Src::WordsFolded { words, half, r } => {
            fold8(bcast3(r), load_words(words.add(j)), load_words(words.add(j + half)))
        }
        Src::Planes(p) => [
            _mm512_loadu_si512(p[0].add(j) as *const __m512i),
            _mm512_loadu_si512(p[1].add(j) as *const __m512i),
            _mm512_loadu_si512(p[2].add(j) as *const __m512i),
        ],
    }
}

#[inline(always)]
unsafe fn store_p(dst: [*mut u64; 3], j: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_storeu_si512(dst[k].add(j) as *mut __m512i, v[k]);
    }
}

/// [`store_p`] with non-temporal stores: a round whose output is far larger than the cache is
/// read back from memory in the next round whatever happens, so the read-for-ownership of its
/// lines is pure traffic. `dst[k] + j` must be 64-byte aligned (the planes are 2 MB-aligned and
/// `j` is a multiple of 8).
#[inline(always)]
unsafe fn stream_p(dst: [*mut u64; 3], j: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_stream_si512(dst[k].add(j) as *mut __m512i, v[k]);
    }
}

/// Fold outputs of at least this many elements (per plane) are written non-temporally: at
/// 24 bytes per element this is 50 MB, past what stays in the cache between two rounds.
const STREAM_FROM: usize = 1 << 21;
/// Words ahead of the current tile that [`fold_message_range`] prefetches in each stream.
const FOLD_AHEAD: usize = 128;

unsafe fn get_p(src: Src, j: usize) -> F162 {
    match src {
        Src::Words(w) => F162::from_b128(*w.add(j)),
        Src::WordsFolded { words, half, r } => {
            let (p0, p1) = (F162::from_b128(*words.add(j)), F162::from_b128(*words.add(j + half)));
            p0 + r * (p0 + p1)
        }
        Src::Planes(p) => F162([*p[0].add(j), *p[1].add(j), *p[2].add(j)]),
    }
}

unsafe fn set_p(dst: [*mut u64; 3], j: usize, x: F162) {
    for k in 0..3 {
        *dst[k].add(j) = x.0[k];
    }
}

#[inline(always)]
unsafe fn xor3p(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    [
        _mm512_xor_si512(a[0], b[0]),
        _mm512_xor_si512(a[1], b[1]),
        _mm512_xor_si512(a[2], b[2]),
    ]
}

#[inline(always)]
unsafe fn bcast3(x: F162) -> [__m512i; 3] {
    [
        _mm512_set1_epi64(x.0[0] as i64),
        _mm512_set1_epi64(x.0[1] as i64),
        _mm512_set1_epi64(x.0[2] as i64),
    ]
}

/// `p0 + r (p0 + p1)` on eight elements.
#[inline(always)]
unsafe fn fold8(rb: [__m512i; 3], p0: [__m512i; 3], p1: [__m512i; 3]) -> [__m512i; 3] {
    xor3p(p0, super::f162::mul_soa8(rb, xor3p(p0, p1)))
}

/// The eight lanes of a finished accumulator summed into one element.
unsafe fn horiz8(acc: [__m512i; 12]) -> F162 {
    let r = super::f162::reduce_soa8(acc);
    let mut w = [[0u64; 8]; 3];
    for k in 0..3 {
        _mm512_storeu_si512(w[k].as_mut_ptr() as *mut __m512i, r[k]);
    }
    let mut out = F162::ZERO;
    for i in 0..8 {
        out += F162([w[0][i], w[1][i], w[2][i]]);
    }
    out
}

/// The map `m` ([`eq_matrices`]) applied to the 64 words at `eq`, word-sliced: `out[l][g]` is
/// limb `l` of words `8 g .. 8 g + 8` — the tile of [`poly_from_eq`], kept in registers.
#[inline(always)]
unsafe fn map_tile(eq: *const B128, m: &[[u64; 16]; 24], out: &mut [[__m512i; 8]; 3]) {
    let mut xe = [_mm512_setzero_si512(); 16];
    transpose_tile(eq as *const u8, &mut xe);
    let mut acc = [_mm512_setzero_si512(); 24];
    for (a, row) in acc.iter_mut().zip(m) {
        for k in (0..16).step_by(2) {
            *a = xor3(
                *a,
                affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
            );
        }
    }
    for (l, o) in out.iter_mut().enumerate() {
        *o = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
    }
}

/// [`map_tile`] for two maps of the same 64 words, transposing them once.
#[inline(always)]
unsafe fn map_tile2(
    eq: *const B128,
    m: &[[u64; 16]; 24],
    mp: &[[u64; 16]; 24],
    out: &mut [[__m512i; 8]; 3],
    outp: &mut [[__m512i; 8]; 3],
) {
    let mut xe = [_mm512_setzero_si512(); 16];
    transpose_tile(eq as *const u8, &mut xe);
    for (m, out) in [(m, out), (mp, outp)] {
        let mut acc = [_mm512_setzero_si512(); 24];
        for (a, row) in acc.iter_mut().zip(m) {
            for k in (0..16).step_by(2) {
                *a = xor3(
                    *a,
                    affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                    affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
                );
            }
        }
        for (l, o) in out.iter_mut().enumerate() {
            *o = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
        }
    }
}

/// The first block of the `eq` expansion in factored form, as
/// [`eq_expand_and_partials_factored`] leaves it: entry `j` is `high[j >> h] * low[j & (2^h - 1)]`
/// (the serial doubling puts the newest coordinates in the lowest bits of the index). The
/// switch reads `eq` a tile of 64 entries at a time, and a tile is one row of `low` scaled by
/// one `high` entry, so the table need not be stored: recomputing a tile costs a few dozen
/// carry-less products, writing and re-reading the table costs memory traffic and, for a fresh
/// buffer, page faults that stall every runner of the pass that first touches it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqFactors {
    high: Vec<B128>,
    low: Vec<B128>,
    h: usize,
}

impl EqFactors {
    /// The entries the factors describe.
    pub fn len(&self) -> usize {
        self.high.len() << self.h
    }

    pub fn is_empty(&self) -> bool {
        self.high.is_empty()
    }

    /// Entry `j`.
    pub fn get(&self, j: usize) -> B128 {
        self.high[j >> self.h] * self.low[j & ((1 << self.h) - 1)]
    }

    /// Every entry, in order.
    pub fn to_vec(&self) -> Vec<B128> {
        (0..self.len()).map(|j| self.get(j)).collect()
    }
}

/// Where a pass reads `eq` from: a stored table, or [`EqFactors`] (rows of at least a tile).
#[derive(Clone, Copy)]
enum EqSrc {
    Table(*const B128),
    Factored { high: *const B128, low: *const B128, h: usize },
}

// `EqSrc` points into buffers that outlive every pass and are only read.
unsafe impl Send for EqSrc {}
unsafe impl Sync for EqSrc {}

#[repr(C, align(64))]
struct EqTile([B128; TILE]);

impl EqSrc {
    /// The 64 entries from `j0` (a multiple of 64): in place, or computed into `buf`.
    #[inline(always)]
    unsafe fn tile(self, j0: usize, buf: &mut EqTile) -> *const B128 {
        match self {
            EqSrc::Table(p) => p.add(j0),
            EqSrc::Factored { high, low, h } => {
                let row = core::slice::from_raw_parts(low.add(j0 & ((1 << h) - 1)), TILE);
                scale_b128_row(row, *high.add(j0 >> h), buf.0.as_mut_ptr());
                buf.0.as_ptr()
            }
        }
    }

    unsafe fn at(self, j: usize) -> B128 {
        match self {
            EqSrc::Table(p) => *p.add(j),
            EqSrc::Factored { high, low, h } => *high.add(j >> h) * *low.add(j & ((1 << h) - 1)),
        }
    }
}

/// The message of a round over `j` in `range`, `half` the round's half: `e0 = sum L(eq[j])
/// p[j]`, `einf = sum L'(eq[j]) (p[j] + p[j + half])`.
unsafe fn message_range(
    eq: EqSrc,
    src: Src,
    half: usize,
    ml: &[[u64; 16]; 24],
    mlp: &[[u64; 16]; 24],
    range: core::ops::Range<usize>,
) -> [F162; 2] {
    let mut acc0 = [_mm512_setzero_si512(); 12];
    let mut acci = [_mm512_setzero_si512(); 12];
    let mut av = [[_mm512_setzero_si512(); 8]; 3];
    let mut apv = [[_mm512_setzero_si512(); 8]; 3];
    let mut buf = EqTile([B128::ZERO; TILE]);
    for j0 in range.step_by(TILE) {
        map_tile2(eq.tile(j0, &mut buf), ml, mlp, &mut av, &mut apv);
        for g in 0..8 {
            let j = j0 + 8 * g;
            let p0 = load_p(src, j);
            let p1 = load_p(src, j + half);
            super::f162::mac_soa8(&mut acc0, [av[0][g], av[1][g], av[2][g]], p0);
            super::f162::mac_soa8(&mut acci, [apv[0][g], apv[1][g], apv[2][g]], xor3p(p0, p1));
        }
    }
    [horiz8(acc0), horiz8(acci)]
}

/// Bind all prefix coordinates directly. Polynomial products are XOR-accumulated
/// before one field reduction per output, instead of reducing each tree node.
///
/// The blocks are `len` words apart, a multiple of the L2's set span, so the words of every
/// block at one position share a cache set: more than [`BIND_WAYS`] streams at once evict each
/// other's prefetched lines. More blocks are taken [`BIND_WAYS`] at a time over a stretch of
/// [`BIND_STRETCH`] outputs, the later sweeps adding their reduced sums to the stored ones
/// (reduction is linear, so the values are the same).
/// Words ahead of the current eight that [`materialize_prefix_range`] prefetches in each stream.
const MATERIALIZE_AHEAD: usize = 64;

unsafe fn materialize_prefix_range(
    words: *const B128, len: usize, weights: &[[__m512i; 3]],
    dst: [*mut u64; 3], range: core::ops::Range<usize>,
) {
    let mut j0 = range.start;
    while j0 < range.end {
        let j1 = (j0 + BIND_STRETCH).min(range.end);
        for (s, ws) in weights.chunks(BIND_WAYS).enumerate() {
            let q0 = s * BIND_WAYS;
            for j in (j0..j1).step_by(8) {
                let mut acc = [_mm512_setzero_si512(); 12];
                for (i, &weight) in ws.iter().enumerate() {
                    // The streams' lines 1 KB ahead: sixteen read streams per runner, and the
                    // hardware prefetcher does not run far enough ahead of the clmul work on
                    // all of them (measured 9.4 -> 8.1 ms for the round; 0.5 KB and 1.5 KB
                    // ahead are both slower, 32 KB much slower). A prefetch past the end of the
                    // table is harmless.
                    let a = words.wrapping_add(j + MATERIALIZE_AHEAD + (q0 + i) * len) as *const i8;
                    _mm_prefetch::<_MM_HINT_T0>(a);
                    _mm_prefetch::<_MM_HINT_T0>(a.wrapping_add(64));
                    super::f162::mac_soa8_word(&mut acc, weight, load_words(words.add(j + (q0 + i) * len)));
                }
                let mut v = super::f162::reduce_soa8(acc);
                if s > 0 {
                    v = xor3p(v, load_p(Src::Planes(dst.map(|p| p as *const u64)), j));
                }
                store_p(dst, j, v);
            }
        }
        j0 = j1;
    }
}

/// Blocks one sweep of the binding pass reads at once (see [`materialize_prefix_range`]). Few
/// streams with long runs each (64 KB per block and sweep) suit the memory best, which pays for
/// the extra reductions: 32 blocks over 2^19 outputs measured 5.0 ms at 2 x 4096, 5.1 ms at
/// 4 x 4096, 6.3 ms at 16 x 512 and 16.5 ms in one sweep (pages faulted in beforehand).
const BIND_WAYS: usize = 2;
/// Outputs per sweep of the binding pass.
const BIND_STRETCH: usize = 4096;

/// Rounds over fewer elements than this run on the calling thread: below it, a dispatch costs
/// more than the round's work. The switch's rounds follow each other within microseconds, so
/// the pool's workers are still spinning, not parked: splitting the rounds from 2^10 elements
/// on measured the switch ~0.4 ms faster than from 2^15. (Lower under test, so that the small
/// instances there take the split path too.)
const PAR_FROM: usize = if cfg!(test) { 1 << 8 } else { 1 << 10 };

/// `0 .. n` cut for the pool, or whole when `n` is below [`PAR_FROM`].
fn round_ranges(n: usize, threads: usize) -> Vec<core::ops::Range<usize>> {
    if n < PAR_FROM {
        vec![0..n]
    } else {
        crate::par::ranges(n, threads, TILE)
    }
}

/// One fused step over `j` in `range` (a sub-range of `0 .. hh`): the fold of round `k`,
/// `p_{k+1}[i] = p_k[i] + r (p_k[i] + p_k[i + half])` for `i = j` and `i = j + hh`, written to
/// `dst`, and the message of round `k + 1` (half `hh`) over the elements just written.
unsafe fn fold_message_range(
    eq: EqSrc,
    src: Src,
    dst: [*mut u64; 3],
    half: usize,
    hh: usize,
    r: F162,
    ml: &[[u64; 16]; 24],
    mlp: &[[u64; 16]; 24],
    range: core::ops::Range<usize>,
) -> [F162; 2] {
    let rb = bcast3(r);
    let mut acc0 = [_mm512_setzero_si512(); 12];
    let mut acci = [_mm512_setzero_si512(); 12];
    let mut av = [[_mm512_setzero_si512(); 8]; 3];
    let mut apv = [[_mm512_setzero_si512(); 8]; 3];
    let mut buf = EqTile([B128::ZERO; TILE]);
    let stream = hh >= STREAM_FROM;
    for j0 in range.step_by(TILE) {
        // The four source rows of the next tiles, all three planes, a tile ahead of the
        // hardware prefetcher; a prefetch past a plane's end is harmless.
        if let Src::Planes(p) = src {
            for k in 0..3 {
                for o in [j0, j0 + half, j0 + hh, j0 + hh + half] {
                    let a = p[k].wrapping_add(o + FOLD_AHEAD) as *const i8;
                    for l in 0..(TILE * 8 / 64) {
                        _mm_prefetch::<_MM_HINT_T0>(a.wrapping_add(64 * l));
                    }
                }
            }
        }
        map_tile2(eq.tile(j0, &mut buf), ml, mlp, &mut av, &mut apv);
        for g in 0..8 {
            let j = j0 + 8 * g;
            let q0 = fold8(rb, load_p(src, j), load_p(src, j + half));
            let q1 = fold8(rb, load_p(src, j + hh), load_p(src, j + hh + half));
            if stream {
                stream_p(dst, j, q0);
                stream_p(dst, j + hh, q1);
            } else {
                store_p(dst, j, q0);
                store_p(dst, j + hh, q1);
            }
            super::f162::mac_soa8(&mut acc0, [av[0][g], av[1][g], av[2][g]], q0);
            super::f162::mac_soa8(&mut acci, [apv[0][g], apv[1][g], apv[2][g]], xor3p(q0, q1));
        }
    }
    if stream {
        _mm_sfence();
    }
    [horiz8(acc0), horiz8(acci)]
}

/// The `eq` a [`LeanSwitch`] reads: a stored table, or the factors of one.
enum EqView<'a> {
    Table(&'a [B128]),
    Factored(EqFactors),
}

impl EqView<'_> {
    fn len(&self) -> usize {
        match self {
            EqView::Table(t) => t.len(),
            EqView::Factored(f) => f.len(),
        }
    }

    fn src(&self) -> EqSrc {
        match self {
            EqView::Table(t) => EqSrc::Table(t.as_ptr()),
            EqView::Factored(f) => EqSrc::Factored { high: f.high.as_ptr(), low: f.low.as_ptr(), h: f.h },
        }
    }
}

/// The most prefix coordinates the partial sets of a [`LeanSwitch`] may cover.
pub const MAX_PREFIX: usize = 8;

/// The switch prover that keeps `a` as a linear map and `p` in one half-size buffer (see the
/// section comment). Drives the same transcript as [`SwitchProver`]: `msg`, `fold`, `msg`, ...
/// `final_eval`.
pub struct LeanSwitch<'a> {
    pi0: &'a [B128],
    eq: EqView<'a>,
    r_hi: &'a [B128],
    /// `L_k`, as the images of the 128 basis words.
    l: Vec<F162>,
    /// `L'_k = L_k o (x -> d_k x)`.
    lp: Vec<F162>,
    /// Materialized p after the prefix rounds, word-sliced.
    p: [HugeBuf<u64>; 3],
    /// Challenges defining the virtual prefix folds.
    prefix_r: [F162; MAX_PREFIX],
    /// Prefix-block partial sets: four, eight or sixteen blocks supply the first
    /// two, three or four messages, respectively; eq holds the first block only.
    blocks: Option<Vec<Vec<B128>>>,
    /// The current round's half, once `msg` has been called.
    half: usize,
    round: usize,
    pending: Option<[F162; 2]>,
    threads: usize,
}

impl<'a> LeanSwitch<'a> {
    /// Capacity covering the fallback path: p_2 has n/4 elements. Prefix-block
    /// paths use less of these buffers; every path needs the final element.
    pub fn plane_len(n: usize) -> usize {
        (n / 4).max(1)
    }

    /// `None` when some coordinate of `r_hi` is one, which the caller treats with
    /// [`SwitchProver`]. `planes` are the three buffers `p` lives in, at least
    /// [`plane_len`](Self::plane_len) elements each; they may be prefaulted ahead of time.
    pub fn new(
        pi0: &'a [B128],
        eq: &'a [B128],
        r_hi: &'a [B128],
        batch: &[F162],
        threads: usize,
        planes: [HugeBuf<u64>; 3],
    ) -> Option<Self> {
        Self::build(pi0, EqView::Table(eq), r_hi, batch, threads, planes, None)
    }

    /// The prover over the first quarter of the expansion and the four quarter partial sets
    /// ([`eq_expand_and_partials_quarters`]): rounds 0 and 1 are combinations of the sets, the
    /// fold of round 1 is the first pass over the words.
    pub fn with_quarters(
        pi0: &'a [B128],
        eq_quarter: &'a [B128],
        r_hi: &'a [B128],
        batch: &[F162],
        threads: usize,
        planes: [HugeBuf<u64>; 3],
        blocks: [Vec<B128>; 4],
    ) -> Option<Self> {
        assert!(r_hi.len() >= 3, "the quarter path wants rounds past the second");
        Self::build(pi0, EqView::Table(eq_quarter), r_hi, batch, threads, planes, Some(blocks.into()))
    }

    /// Eight blocks provide the first three messages without storing p1 or p2.
    pub fn with_eighths(
        pi0: &'a [B128], eq: &'a [B128], r_hi: &'a [B128], batch: &[F162],
        threads: usize, planes: [HugeBuf<u64>; 3], sets: [Vec<B128>; 8],
    ) -> Option<Self> {
        assert!(r_hi.len() >= 4);
        Self::build(pi0, EqView::Table(eq), r_hi, batch, threads, planes, Some(sets.into()))
    }

    /// Sixteen partial sets provide the first four messages without storing p1..p3.
    pub fn with_sixteenths(
        pi0: &'a [B128], eq: &'a [B128], r_hi: &'a [B128], batch: &[F162],
        threads: usize, planes: [HugeBuf<u64>; 3], sets: [Vec<B128>; 16],
    ) -> Option<Self> {
        assert!(r_hi.len() >= 5);
        Self::build(pi0, EqView::Table(eq), r_hi, batch, threads, planes, Some(sets.into()))
    }

    /// [`with_sixteenths`](Self::with_sixteenths) over the first block in the factored form
    /// [`eq_expand_and_partials_factored`] returns: the table is never stored.
    pub fn with_sixteenths_factored(
        pi0: &'a [B128], eq: EqFactors, r_hi: &'a [B128], batch: &[F162],
        threads: usize, planes: [HugeBuf<u64>; 3], sets: [Vec<B128>; 16],
    ) -> Option<Self> {
        assert!(r_hi.len() >= 5);
        assert!(eq.low.len() % TILE == 0, "a tile is within one row of the factors");
        Self::build(pi0, EqView::Factored(eq), r_hi, batch, threads, planes, Some(sets.into()))
    }

    /// [`with_sixteenths_factored`](Self::with_sixteenths_factored) for any power-of-two
    /// number of prefix sets from 2 to [`MAX_PREFIX`] coordinates' worth: the first
    /// `log2(sets.len())` messages come from the sets.
    pub fn with_sets_factored(
        pi0: &'a [B128], eq: EqFactors, r_hi: &'a [B128], batch: &[F162],
        threads: usize, planes: [HugeBuf<u64>; 3], sets: Vec<Vec<B128>>,
    ) -> Option<Self> {
        let prefix = sets.len().trailing_zeros() as usize;
        assert!(sets.len().is_power_of_two() && (1..=MAX_PREFIX).contains(&prefix));
        assert!(r_hi.len() > prefix);
        assert!(eq.low.len() % TILE == 0, "a tile is within one row of the factors");
        Self::build(pi0, EqView::Factored(eq), r_hi, batch, threads, planes, Some(sets))
    }

    fn build(
        pi0: &'a [B128],
        eq: EqView<'a>,
        r_hi: &'a [B128],
        batch: &[F162],
        threads: usize,
        planes: [HugeBuf<u64>; 3],
        blocks: Option<Vec<Vec<B128>>>,
    ) -> Option<Self> {
        let n = pi0.len();
        assert!(n >= 2 && n.is_power_of_two() && n == 1 << r_hi.len(), "2^l words");
        match &blocks {
            None => assert_eq!(eq.len(), n, "one eq entry per word"),
            Some(v) => {
                assert_eq!(eq.len(), n / v.len(), "the first block of the expansion");
                assert!(v.iter().all(|s| s.len() == PACK), "128 partials per block");
            }
        }
        assert_eq!(batch.len(), PACK, "one batch weight per bit");
        for plane in &planes {
            assert!(plane.len() >= Self::plane_len(n), "a plane is too short");
        }
        if r_hi.iter().any(|&r| r == B128::ONE) {
            return None;
        }
        let l = batch.to_vec();
        let lp = compose_scale(&l, b128_inv(B128::ONE + r_hi[0]));
        Some(LeanSwitch {
            pi0,
            eq,
            r_hi,
            l,
            lp,
            p: planes,
            prefix_r: [F162::ZERO; MAX_PREFIX],
            blocks,
            half: n,
            round: 0,
            pending: None,
            threads: threads.max(1),
        })
    }

    /// Message k before materialization: bind the first k word-group coordinates
    /// with F162 weights, combine the remaining groups using their GHASH eq ratios,
    /// then apply L_k (for e0) or L'_k (for einf) to the resulting bit partials.
    fn message_from_blocks(&self) -> [F162; 2] {
        let sets = self.blocks.as_ref().expect("partial sets");
        let prefix = sets.len().trailing_zeros() as usize;
        let k = self.round;
        let suffix = prefix - k - 1;
        let ratios: Vec<_> = self.r_hi[k + 1..prefix].iter().copied().map(half_ratio).collect();
        let scales: Vec<_> = (0..1usize << suffix).map(|q| {
            (0..suffix).filter(|&i| q & (1 << (suffix - 1 - i)) != 0)
                .fold(B128::ONE, |a, i| a * ratios[i])
        }).collect();
        let rs = self.prefix_r;
        let tables = [psi_table(&self.l), psi_table(&self.lp)];
        let mut msg = [F162::ZERO; 2];
        for z in 0..1usize << k {
            let weight = (0..k).fold(F162::ONE, |a, i| {
                a * if z & (1 << (k - 1 - i)) != 0 { rs[i] } else { F162::ONE + rs[i] }
            });
            let mut terms = [F162::ZERO; 2];
            for b in 0..PACK {
                let mut sums = [B128::ZERO; 2];
                for (s, &scale) in scales.iter().enumerate() {
                    let group = (z << (prefix - k)) | s;
                    sums[0] = sums[0] + scale * sets[group][b];
                    sums[1] = sums[1] + scale * (sets[group][b] + sets[group | (1 << suffix)][b]);
                }
                let mut monomial = F162::ZERO;
                monomial.0[b / 64] = 1 << (b % 64);
                for t in 0..2 { terms[t] += psi(&tables[t], sums[t]) * monomial; }
            }
            for t in 0..2 { msg[t] += weight * terms[t]; }
        }
        msg
    }

    /// The buffers for a fresh switch over `n` words.
    pub fn planes(n: usize) -> [HugeBuf<u64>; 3] {
        let len = Self::plane_len(n);
        [HugeBuf::new(len), HugeBuf::new(len), HugeBuf::new(len)]
    }

    fn src(&self) -> Src {
        match self.round {
            0 => Src::Words(self.pi0.as_ptr()),
            1 => Src::WordsFolded {
                words: self.pi0.as_ptr(),
                half: self.pi0.len() / 2,
                r: self.prefix_r[0],
            },
            _ => Src::Planes([self.p[0].as_ptr(), self.p[1].as_ptr(), self.p[2].as_ptr()]),
        }
    }

    /// The message over `src` at `half`: `e0 = sum_{j < half} L(eq[j]) p[j]`,
    /// `einf = sum L'(eq[j]) (p[j] + p[j + half])`, with the maps the struct holds.
    fn message(&self, src: Src, half: usize) -> [F162; 2] {
        if half < TILE {
            let mut e0 = F162::ZERO;
            let mut einf = F162::ZERO;
            for j in 0..half {
                let e = unsafe { self.eq.src().at(j) };
                let (a, ap) = (apply_images(&self.l, e), apply_images(&self.lp, e));
                let (p0, p1) = unsafe { (get_p(src, j), get_p(src, j + half)) };
                e0 += a * p0;
                einf += ap * (p0 + p1);
            }
            return [e0, einf];
        }
        let ml = eq_matrices(&self.l);
        let mlp = eq_matrices(&self.lp);
        let runs = round_ranges(half, self.threads);
        let eq = self.eq.src();
        let parts: Vec<[F162; 2]> = crate::par::pool().map(runs.len(), |i| unsafe {
            message_range(eq, src, half, &ml, &mlp, runs[i].clone())
        });
        let mut m = [F162::ZERO; 2];
        for part in parts {
            m[0] += part[0];
            m[1] += part[1];
        }
        m
    }

    fn dst(&mut self) -> [*mut u64; 3] {
        [
            self.p[0].as_mut_ptr(),
            self.p[1].as_mut_ptr(),
            self.p[2].as_mut_ptr(),
        ]
    }

    /// The round's two message values.
    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        if let Some(m) = self.pending.take() {
            return m;
        }
        if self.blocks.as_ref().is_some_and(|v| self.round < v.len().trailing_zeros() as usize) {
            return self.message_from_blocks();
        }
        debug_assert_eq!(self.round, 0);
        self.message(self.src(), self.half)
    }

    /// Bind the round's challenge: fold `p`, advance the map, and — when a round follows —
    /// compute its message in the same pass. Prefix-block messages defer the first
    /// writes until all prefix challenges are known, then bind the words directly.
    pub fn fold(&mut self, r: F162) {
        let half = self.half;
        let hh = half / 2;
        let last = self.round + 1 == self.r_hi.len();
        let src = self.src();
        let dst = self.dst();
        // L_{k+1} = L_k + r L'_k, then L'_{k+1} from the next coordinate.
        for i in 0..PACK {
            self.l[i] = self.l[i] + r * self.lp[i];
        }
        if !last {
            self.lp = compose_scale(&self.l, b128_inv(B128::ONE + self.r_hi[self.round + 1]));
        }
        if last {
            debug_assert_eq!(half, 1);
            let (p0, p1) = unsafe { (get_p(src, 0), get_p(src, 1)) };
            unsafe { set_p(dst, 0, p0 + r * (p0 + p1)) };
            self.round += 1;
            return;
        }
        if self.round == 0 {
            self.prefix_r[0] = r;
            self.round = 1;
            if self.blocks.is_none() {
                let m = self.message(self.src(), hh);
                self.pending = Some(m);
            }
            return;
        }
        if self.blocks.as_ref().is_some_and(|v| self.round + 1 < v.len().trailing_zeros() as usize) {
            self.prefix_r[self.round] = r;
            self.round += 1;
            return;
        }
        if self.blocks.as_ref().is_some_and(|v| self.round + 1 == v.len().trailing_zeros() as usize) {
            self.prefix_r[self.round] = r;
            let weights = eq_expand_f162(&self.prefix_r[..self.round + 1]);
            if half < 8 {
                for j in 0..half {
                    let value = weights.iter().enumerate().fold(F162::ZERO, |a, (q, &w)|
                        a + w * F162::from_b128(self.pi0[j + q * half]));
                    unsafe { set_p(dst, j, value) };
                }
                self.pending = Some(self.message(Src::Planes(dst.map(|p| p as *const u64)), hh));
                self.round += 1;
                return;
            }
            let weights: Vec<_> = weights.into_iter().map(|w| unsafe { bcast3(w) }).collect();
            let words = self.pi0.as_ptr() as usize;
            let dst = dst.map(|p| p as usize);
            // Handed out in chunks from a shared counter, like the partials' rows: a runner
            // that starts late or shares its core takes fewer of them.
            let chunk = (half / (16 * self.threads)).clamp(8, 1 << 12) & !7;
            let chunks = half.div_ceil(chunk);
            let next = std::sync::atomic::AtomicUsize::new(0);
            // Consecutive chunks are a quarter of the output apart, so that the runners write
            // to different huge pages and a page's first-touch fault holds up one or two of
            // them, not all.
            let ways = if chunks % 4 == 0 { 4 } else { 1 };
            crate::par::pool().run(self.threads.min(chunks), |_| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= chunks {
                    break;
                }
                let c = (k % ways) * (chunks / ways) + k / ways;
                unsafe {
                    materialize_prefix_range(words as *const B128, half, &weights,
                        dst.map(|p| p as *mut u64), c * chunk..((c + 1) * chunk).min(half));
                }
            });
            self.pending = Some(self.message(Src::Planes(dst.map(|p| p as *const u64)), hh));
            self.round += 1;
            return;
        }
        if hh < TILE {
            let mut e0 = F162::ZERO;
            let mut einf = F162::ZERO;
            for j in 0..hh {
                let (p0, p1) = unsafe { (get_p(src, j), get_p(src, j + half)) };
                let q0 = p0 + r * (p0 + p1);
                let (p2, p3) = unsafe { (get_p(src, j + hh), get_p(src, j + hh + half)) };
                let q1 = p2 + r * (p2 + p3);
                unsafe {
                    set_p(dst, j, q0);
                    set_p(dst, j + hh, q1);
                }
                let e = unsafe { self.eq.src().at(j) };
                let (a, ap) = (apply_images(&self.l, e), apply_images(&self.lp, e));
                e0 += a * q0;
                einf += ap * (q0 + q1);
            }
            self.pending = Some([e0, einf]);
            self.round += 1;
            return;
        }
        let ml = eq_matrices(&self.l);
        let mlp = eq_matrices(&self.lp);
        let runs = round_ranges(hh, self.threads);
        let eq = self.eq.src();
        let dst = dst.map(|p| p as usize);
        let parts: Vec<[F162; 2]> = crate::par::pool().map(runs.len(), |i| unsafe {
            fold_message_range(
                eq,
                src,
                dst.map(|p| p as *mut u64),
                half,
                hh,
                r,
                &ml,
                &mlp,
                runs[i].clone(),
            )
        });
        let mut m = [F162::ZERO; 2];
        for part in parts {
            m[0] += part[0];
            m[1] += part[1];
        }
        self.pending = Some(m);
        self.round += 1;
    }

    /// `p` at the end of the sumcheck: the value the commitment opens to. Every last fold
    /// writes it to the planes' first element.
    pub fn final_eval(&self) -> F162 {
        assert_eq!(self.round, self.r_hi.len(), "the sumcheck has not finished");
        F162([
            unsafe { *self.p[0].as_ptr() },
            unsafe { *self.p[1].as_ptr() },
            unsafe { *self.p[2].as_ptr() },
        ])
    }
}

// `Src` holds raw pointers into buffers that outlive every pass and are only written through
// `dst` on disjoint ranges.
unsafe impl Send for Src {}
unsafe impl Sync for Src {}

/// [`Poly`] is referenced for the fallback path's type; keep the import live.
#[allow(dead_code)]
fn _poly_type_marker(_: &Poly) {}

#[cfg(test)]
mod lean_tests {
    use super::*;

    fn next(x: &mut u128) -> u128 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (*x >> 64)
    }

    /// The lean prover sends the messages `SwitchProver` sends and ends on the same value, at
    /// lengths that exercise the tiled, the scalar and the mixed paths, single- and
    /// multi-threaded.
    #[test]
    fn lean_switch_matches_switch_prover() {
        let mut x = 0x0123_4567_89AB_CDEF_1122_3344_5566_7788u128;
        for l in [1usize, 3, 6, 7, 8, 10, 17] {
            for threads in [1usize, 3] {
                let n = 1 << l;
                let pi0: Vec<B128> = (0..n).map(|_| B128(next(&mut x))).collect();
                let r_hi: Vec<B128> = (0..l).map(|_| B128(next(&mut x))).collect();
                let batch: Vec<F162> = (0..PACK)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let r: Vec<F162> = (0..l)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let eq = eq_expand_b128(&r_hi);
                let mut eq_par = vec![B128::ZERO; n];
                unsafe { eq_expand_b128_parallel(&r_hi, eq_par.as_mut_ptr(), threads) };
                assert_eq!(eq, eq_par, "l {l}");
                assert_eq!(partial_evals(&pi0, &eq), partial_evals_parallel(&pi0, &eq, threads));
                let mut eq_fused = vec![B128::ZERO; n];
                let fused = unsafe { eq_expand_and_partials(&r_hi, &pi0, eq_fused.as_mut_ptr(), threads) };
                assert_eq!(eq, eq_fused, "fused expansion, l {l}");
                assert_eq!(partial_evals(&pi0, &eq), fused, "fused partials, l {l}");

                let mut reference = SwitchProver::new(&pi0, &eq, &batch);
                let mut lean =
                    LeanSwitch::new(&pi0, &eq, &r_hi, &batch, threads, LeanSwitch::planes(n)).unwrap();
                for (round, &ri) in r.iter().enumerate() {
                    let want = reference.msg();
                    let got = lean.msg();
                    assert_eq!(want, got, "l {l} round {round}");
                    reference.fold(ri);
                    lean.fold(ri);
                }
                assert_eq!(reference.final_eval(), lean.final_eval(), "l {l}");
            }
        }
    }

    #[test]
    fn quarter_path_matches_switch_prover() {
        let mut x = 0x7777_1234_ABCD_0001u128;
        for l in [18usize, 19] {
            for threads in [1usize, 3] {
                let n = 1 << l;
                let pi0: Vec<B128> = (0..n).map(|_| B128(next(&mut x))).collect();
                let r_hi: Vec<B128> = (0..l).map(|_| B128(next(&mut x))).collect();
                let batch: Vec<F162> = (0..PACK)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let r: Vec<F162> = (0..l)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let eq = eq_expand_b128(&r_hi);
                let mut quarter = vec![B128::ZERO; n / 4];
                let sets = unsafe { eq_expand_and_partials_quarters(&r_hi, &pi0, quarter.as_mut_ptr(), threads) };
                assert_eq!(quarter[..], eq[..n / 4], "the first quarter, l {l}");
                assert_eq!(partials_from_quarters(&sets, &r_hi), partial_evals(&pi0, &eq), "partials, l {l}");
                let mut reference = SwitchProver::new(&pi0, &eq, &batch);
                let mut lean =
                    LeanSwitch::with_quarters(&pi0, &quarter, &r_hi, &batch, threads, LeanSwitch::planes(n), sets).unwrap();
                for (round, &ri) in r.iter().enumerate() {
                    assert_eq!(reference.msg(), lean.msg(), "l {l} round {round}");
                    reference.fold(ri);
                    lean.fold(ri);
                }
                assert_eq!(reference.final_eval(), lean.final_eval(), "l {l}");
            }
        }
    }

    #[test]
    fn eighth_path_matches_every_reference_message() {
        let mut x=0x66778899aabbccdd1122334455667788u128;
        for l in [18usize,19] {
            let n=1usize<<l;
            let words:Vec<_>=(0..n).map(|_|B128(next(&mut x))).collect();
            let batch:Vec<_>=(0..PACK).map(|_|F162([next(&mut x) as u64,next(&mut x) as u64,next(&mut x) as u64 & ((1<<34)-1)])).collect();
            for case in 0..6 {
                let mut hi:Vec<_>=(0..l).map(|_|B128(next(&mut x))).collect();
                if (1..=3).contains(&case) {hi[case-1]=B128::ZERO;}
                let r:Vec<_>=(0..l).map(|_|match case {
                    4=>F162::ZERO,5=>F162::ONE,
                    _=>F162([next(&mut x) as u64,next(&mut x) as u64,next(&mut x) as u64 & ((1<<34)-1)])
                }).collect();
                let full=eq_expand_b128(&hi);
                for workers in [1usize,3,8] {
                    let sentinel=B128(0xdeadbeefcafebabe);
                    let mut prefix=vec![sentinel;n/4];
                    let sets=unsafe {eq_expand_and_partials_eighths(&hi,&words,prefix.as_mut_ptr(),workers)};
                    assert_eq!(&prefix[..n/8],&full[..n/8]);
                    assert!(prefix[n/8..].iter().all(|&w|w==sentinel));
                    assert_eq!(partials_from_eighths(&sets,&hi),partial_evals(&words,&full));
                    let mut reference=SwitchProver::new(&words,&full,&batch);
                    let mut lean=LeanSwitch::with_eighths(&words,&prefix[..n/8],&hi,&batch,workers,LeanSwitch::planes(n),sets).unwrap();
                    for (round,&challenge) in r.iter().enumerate() {
                        assert_eq!(reference.msg(),lean.msg(),"l {l} case {case} workers {workers} round {round}");
                        reference.fold(challenge);lean.fold(challenge);
                    }
                    assert_eq!(reference.final_eval(),lean.final_eval());
                }
            }
            let mut hi=vec![B128::ZERO;l];
            let prefix=vec![B128::ZERO;n/8];
            for i in 0..l {
                hi[i]=B128::ONE;
                assert!(LeanSwitch::with_eighths(&words,&prefix,&hi,&batch,3,LeanSwitch::planes(n),core::array::from_fn(|_|vec![B128::ZERO;PACK])).is_none());
                hi[i]=B128::ZERO;
            }
        }
    }

    #[test]
    fn sixteenth_path_matches_switch_prover() {
        let mut x = 0x7777_1234_ABCD_0001u128;
        for l in [18usize, 19] {
            for threads in [1usize, 3] {
                let n = 1 << l;
                let pi0: Vec<B128> = (0..n).map(|_| B128(next(&mut x))).collect();
                let r_hi: Vec<B128> = (0..l).map(|_| B128(next(&mut x))).collect();
                let batch: Vec<F162> = (0..PACK)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let r: Vec<F162> = (0..l)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let eq = eq_expand_b128(&r_hi);
                let mut quarter = vec![B128::ZERO; n / 16];
                let sets = unsafe { eq_expand_and_partials_sixteenths(&r_hi, &pi0, quarter.as_mut_ptr(), threads) };
                assert_eq!(quarter[..], eq[..n / 16], "the first block, l {l}");
                assert_eq!(partials_from_sixteenths(&sets, &r_hi), partial_evals(&pi0, &eq), "partials, l {l}");
                let mut reference = SwitchProver::new(&pi0, &eq, &batch);
                let mut lean =
                    LeanSwitch::with_sixteenths(&pi0, &quarter, &r_hi, &batch, threads, LeanSwitch::planes(n), sets).unwrap();
                for (round, &ri) in r.iter().enumerate() {
                    assert_eq!(reference.msg(), lean.msg(), "l {l} round {round}");
                    reference.fold(ri);
                    lean.fold(ri);
                }
                assert_eq!(reference.final_eval(), lean.final_eval(), "l {l}");
            }
        }
    }

    #[test]
    fn factored_path_matches_switch_prover() {
        let mut x = 0x0A0B_0C0D_1234_5678_9ABC_DEF0_1357_2468u128;
        for l in [18usize, 19] {
            for threads in [1usize, 3, 8] {
                let n = 1 << l;
                let pi0: Vec<B128> = (0..n).map(|_| B128(next(&mut x))).collect();
                let r_hi: Vec<B128> = (0..l).map(|_| B128(next(&mut x))).collect();
                let batch: Vec<F162> = (0..PACK)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let r: Vec<F162> = (0..l)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let eq = eq_expand_b128(&r_hi);
                let (sets, factors) = eq_expand_and_partials_factored(&r_hi, &pi0, threads);
                assert_eq!(factors.len(), n / 16);
                assert_eq!(factors.to_vec()[..], eq[..n / 16], "the first block, l {l}");
                assert_eq!(partials_from_sixteenths(&sets, &r_hi), partial_evals(&pi0, &eq), "partials, l {l}");
                let mut reference = SwitchProver::new(&pi0, &eq, &batch);
                let mut lean = LeanSwitch::with_sixteenths_factored(&pi0, factors, &r_hi, &batch, threads,
                    LeanSwitch::planes(n), sets).unwrap();
                for (round, &ri) in r.iter().enumerate() {
                    assert_eq!(reference.msg(), lean.msg(), "l {l} threads {threads} round {round}");
                    reference.fold(ri);
                    lean.fold(ri);
                }
                assert_eq!(reference.final_eval(), lean.final_eval(), "l {l}");
            }
        }
    }

    /// More prefix sets than one sweep of the binding pass reads (32 and 64 blocks), and fewer.
    #[test]
    fn factored_sets_path_matches_switch_prover() {
        let mut x = 0x5151_2626_3737_4848_9A9A_0B0B_1C1C_2D2Du128;
        fn run<const G: usize>(x: &mut u128, l: usize, threads: usize) {
            let n = 1 << l;
            let pi0: Vec<B128> = (0..n).map(|_| B128(next(x))).collect();
            let r_hi: Vec<B128> = (0..l).map(|_| B128(next(x))).collect();
            let batch: Vec<F162> = (0..PACK)
                .map(|_| F162([next(x) as u64, next(x) as u64, (next(x) as u64) & ((1 << 34) - 1)]))
                .collect();
            let r: Vec<F162> = (0..l)
                .map(|_| F162([next(x) as u64, next(x) as u64, (next(x) as u64) & ((1 << 34) - 1)]))
                .collect();
            let eq = eq_expand_b128(&r_hi);
            let (sets, factors) = eq_expand_and_partials_factored_blocks::<G>(&r_hi, &pi0, threads);
            assert_eq!(factors.len(), n / G);
            assert_eq!(factors.to_vec()[..], eq[..n / G], "the first block, l {l} G {G}");
            assert_eq!(partials_from_sets(&sets, &r_hi), partial_evals(&pi0, &eq), "partials, l {l} G {G}");
            let mut reference = SwitchProver::new(&pi0, &eq, &batch);
            let mut lean = LeanSwitch::with_sets_factored(&pi0, factors, &r_hi, &batch, threads,
                LeanSwitch::planes(n), sets.into()).unwrap();
            for (round, &ri) in r.iter().enumerate() {
                assert_eq!(reference.msg(), lean.msg(), "l {l} G {G} threads {threads} round {round}");
                reference.fold(ri);
                lean.fold(ri);
            }
            assert_eq!(reference.final_eval(), lean.final_eval(), "l {l} G {G}");
        }
        for threads in [1usize, 3, 8] {
            run::<32>(&mut x, 18, threads);
            run::<64>(&mut x, 19, threads);
            run::<8>(&mut x, 18, threads);
        }
    }

    #[test]
    fn prefix_materialization_small_and_degenerate_challenges() {
        let mut x = 0x938e_1212_dcf3_0033u128;
        for l in [5usize, 6, 7, 8, 10] {
            let n = 1 << l;
            let words: Vec<_> = (0..n).map(|_| B128(next(&mut x))).collect();
            let mut hi: Vec<_> = (0..l).map(|_| B128(next(&mut x))).collect();
            hi[0] = B128::ZERO;
            let eq = eq_expand_b128(&hi);
            let sets: [Vec<B128>; 16] = core::array::from_fn(|q|
                partial_evals(&words[q * n / 16..(q + 1) * n / 16], &eq[..n / 16]));
            let batch: Vec<_> = (0..PACK).map(|_| F162([next(&mut x) as u64,
                next(&mut x) as u64, next(&mut x) as u64 & ((1 << 34) - 1)])).collect();
            for first in [F162::ZERO, F162::ONE] {
                let mut reference = SwitchProver::new(&words, &eq, &batch);
                let mut sw = LeanSwitch::with_sixteenths(&words, &eq[..n / 16], &hi,
                    &batch, 3, LeanSwitch::planes(n), sets.clone()).unwrap();
                for k in 0..l {
                    assert_eq!(sw.msg(), reference.msg(), "l {l}, round {k}");
                    let r = if k == 0 { first } else if k == 1 { F162::ONE + first }
                        else { F162([next(&mut x) as u64, next(&mut x) as u64,
                            next(&mut x) as u64 & ((1 << 34) - 1)]) };
                    sw.fold(r); reference.fold(r);
                }
                assert_eq!(sw.final_eval(), reference.final_eval());
            }
        }
    }

    #[test]
    fn inverse_and_scaled_maps() {
        let mut x = 0x9E3779B97F4A7C15u128;
        for _ in 0..8 {
            let v = B128(next(&mut x) | 1);
            assert_eq!(v * b128_inv(v), B128::ONE);
        }
        let images: Vec<F162> = (0..PACK)
            .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
            .collect();
        for d in [B128::ZERO, B128::ONE, B128(u128::MAX), B128(next(&mut x))] {
            let expected: Vec<_> = (0..PACK)
                .map(|i| apply_images(&images, B128(1u128 << i) * d))
                .collect();
            assert_eq!(compose_scale(&images, d), expected, "scaled basis {d:?}");
        }
        let d = B128(next(&mut x));
        let scaled = compose_scale(&images, d);
        for _ in 0..8 {
            let w = B128(next(&mut x));
            assert_eq!(apply_images(&scaled, w), apply_images(&images, w * d));
        }
        let r = B128(next(&mut x));
        assert_eq!((B128::ONE + r) * b128_inv(B128::ONE + r), B128::ONE);
    }
}
