//! The folding step of `Pi_fold`, on top of the commitment, entirely in the 648-slot NTT domain
//! of `R_648`.
//!
//! # The algebra
//!
//! A witness of `r` chunks `W_0, .., W_{r-1}`, each `len_ring` elements of
//! `R_648 = Z_q[X]/(X^648 - X^324 + 1)`, is committed under one key `A` to the `r` elements
//! `C_j = sum_i A_i W_{j,i}`. A challenge `c_j` is a short signed element of the subring
//! `R_162 = Z_q[Z]/Phi_243(Z)`, embedded into `R_648` as `c_j(-X^4)` (coefficient of `X^{4m}` is
//! `(-1)^m c_{j,m}`, everything else zero — the embedding of "The lift is a ring extension of
//! degree 4"). The folded witness and its commitment are
//!
//! ```text
//!     v = sum_j c_j W_j   (len_ring elements),        A v = sum_j c_j C_j,
//! ```
//! the second identity by `R_648`-linearity of `A`. Both sides are slot-wise in the NTT domain:
//! `NTT(v)[u] = sum_j NTT(c_j)[u] NTT(W_j)[u]`, one scalar per challenge per slot. Multiplication
//! by a subring element acts on the four `R_162` components of an element alike, so the whole
//! fold is a single length-`r` inner product per slot with no ring multiplication anywhere.
//!
//! # What it costs
//!
//! [`CommitmentKey::commit_into_aux`](crate::ring::CommitmentKey::commit_into_aux) already left
//! `NTT_3889(W)` in memory, so the fold never transforms the witness again: it reads that
//! transform once, which is the DRAM floor of the step. Every chunk contributes one `vpmaddwd`
//! per slot vector into an accumulator of 82 944 B per batch position, and [`G`] positions at a
//! time take all the chunks, so the accumulator stays in L2 whatever `len_ring` is (5.3 MB for
//! all 64 positions of `len_ring = 2048`, swept once per chunk pair, was the L3 traffic of the
//! step: 100.8 -> 68.1 ms on the i7-11850H, at the DRAM floor). The small transform's own lazy
//! reduction (`|W| <= 7.5 q`) survives untouched into the products, and each group's
//! accumulator is folded back exactly, `x = l + (h + c) R mod q`, every [`FOLD_PERIOD`] chunks.
//!
//! Only the base limb's transform is kept, and the prover stops there: `v` is inverted back to
//! coefficients modulo the base prime, where it becomes a genuine small-integer vector. The
//! verifier is the one that transforms it forward again modulo every limb (`ntt::gen_small`,
//! `ntt::gen_large` above `2^14`, or `ntt::gen_quad` for a quadratic-slot one) and
//! recomputes `A v`.
//!
//! # The base limb is any of the seven
//!
//! Nothing above depends on the tree the base limb's `R_648` has. The transform the commitment
//! keeps is 648 rows either way — the 648 linear slots of a splitting prime, or the two
//! coefficients of each of the 324 quadratic leaves — and the *challenge* is what decides whether
//! the slot-wise product is a scalar one:
//!
//! `c` enters `R_648` as `c(-X^4)`, a polynomial in `X^4`. Modulo a quadratic leaf `X^2 - psi'^u`
//! that is `X^4 = psi'^{2u}`, a constant, so the leaf image of a challenge is the *scalar*
//! `c(-theta^v)` and its `X`-coefficient row vanishes. Multiplying a witness leaf `w_0 + w_1 X`
//! by it is `c w_0 + c w_1 X`: the same one `vpmaddwd` per slot vector as for a splitting base,
//! once the scalar is written into both rows of the leaf ([`duplicate_leaf_scalars`]). The
//! degree-2 product with its bilinear rank 3 is the *commitment's* problem, where `A` is uniform
//! ([`crate::simd::commit`]'s `MacQ`), not the fold's.
//!
//! What does change with the base is the accumulation bound: `|W|` is the binary kernel's
//! declared output bound for that prime — 6.96 q for 2917, 7.50 for 3889, 3.17 for 4861, 2.29 for
//! 9721, 1.94 for 12637, 1.79 and 1.58 for the two above `2^14` — and `|c|` is `(q-1)/2`, so
//! [`fold_period`] is per prime and runs from 64 chunks down to 4. The inverse transform back to
//! coefficients is that tree's ([`intt_gen_batch32`], `ntt::gen_large`'s, or
//! `ntt::gen_quad`'s).
use crate::challenge::ShortChallenge;
use crate::key::AuxData;
use crate::limb::{dispatch_limb, is_quad};
use crate::params::{quadratic_slots, N, QS, QS_LARGE, QS_QUAD, QUAD_CLASS_SLOT, QUAD_SLOTS};
use crate::ring::{Batch32, Representation, RingElement, BASE_PRIME, N162, PRIMES, SLOT_648};
use crate::simd::commit as cm;
use crate::simd::slots as sl;
use crate::simd::ntt::bin_large as vl;
use crate::simd::ntt::gen_small::{self as vg, intt_gen_batch32, ntt_gen_batch32};
use crate::simd::ntt::gen_large as vgl;
use crate::simd::ntt::gen_quad::{self as vgq, intt_quad_gen_batch32, ntt_quad_gen_batch32};
use core::arch::x86_64::*;
use std::sync::OnceLock;

/// The default base limb, the prime the witness transform is kept in unless [`crate::Params`]
/// names another.
pub const Q1: u16 = BASE_PRIME;
/// The second prime of the default limb list, reached — like every additional limb — by an
/// inverse transform of `v` and a forward one on `len_ring / 32` batches.
pub const Q2: u16 = PRIMES[1];

// =============================================================================================
// the accumulation bound
// =============================================================================================

/// Bound on one lane of the kept transform: the base kernel's declared output bound, whichever
/// tree it runs ([`cm::w_bound`] for a splitting prime, [`cm::w_bound_quad`] for a quadratic
/// one).
pub const fn witness_bound(q: u16) -> i64 {
    if quadratic_slots(q) {
        cm::w_bound_quad(q)
    } else {
        cm::w_bound(q)
    }
}

/// What one chunk adds to one accumulator lane: `|W| <= witness_bound(q)` times `|c| <= (q-1)/2`
/// (a fully reduced centered challenge slot, [`cm::a_bound`]) — one product, unlike the
/// commitment's four, because a lane of this accumulator carries one ring element rather than
/// four.
pub const fn fold_per_chunk(q: u16) -> i64 {
    witness_bound(q) * cm::a_bound(q)
}

/// Chunks accumulated between two fold-backs of a base limb's accumulator.
///
/// `|acc| <= acc_after_reduce + P * fold_per_chunk` must fit `i32`; for `q = 3889` that is
/// `2^15 (1 + 3312) + P * 29167 * 1944 = 108 560 384 + P * 56 700 648`, so `P = 32` (1 923 013 120)
/// fits and `P = 64` does not. The chunks are accumulated in pairs, so the period must be even.
pub const fn fold_period(q: u16) -> usize {
    cm::period_for(q, fold_per_chunk(q))
}

/// The default base's period, and the constant the module comment quotes.
pub const FOLD_PERIOD: usize = fold_period(Q1);

const fn fits(q: u16, p: usize) -> bool {
    cm::acc_after_reduce(q) + (p as i64) * fold_per_chunk(q) <= i32::MAX as i64
}
const _: () = assert!(FOLD_PERIOD == 32);
const _: () = assert!(fits(Q1, FOLD_PERIOD));
const _: () = assert!(!fits(Q1, 2 * FOLD_PERIOD));
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(fits(QS_QUAD[i], fold_period(QS_QUAD[i])) && fold_period(QS_QUAD[i]) % 2 == 0);
        i += 1;
    }
    let mut i = 0;
    while i < 2 {
        assert!(fits(QS[i], fold_period(QS[i])) && fold_period(QS[i]) % 2 == 0);
        assert!(fits(QS_LARGE[i], fold_period(QS_LARGE[i])) && fold_period(QS_LARGE[i]) % 2 == 0);
        i += 1;
    }
};

/// Batches of `A v` between two fold-backs of a splitting limb's accumulator: both operands are
/// centered, so a lane grows by `4 ((q-1)/2)^2` per batch. 128 for 3889 and 16 for 9721, where
/// the whole product used to run unreduced over the eight batches of the basic shape; 4 for the
/// two primes above `2^14`, which is what forces the fold-back to exist at all.
pub const fn av_period(q: u16) -> usize {
    cm::period_for(q, 4 * cm::a_bound(q) * cm::a_bound(q))
}

const fn av_fits(q: u16) -> bool {
    cm::acc_after_reduce(q) + (av_period(q) as i64) * 4 * cm::a_bound(q) * cm::a_bound(q)
        <= i32::MAX as i64
}
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(av_fits(PRIMES[i]) && av_period(PRIMES[i]) >= 1);
        assert!(av_fits(QS_LARGE[i]) && av_period(QS_LARGE[i]) >= 1);
        i += 1;
    }
};

// =============================================================================================
// small vector helpers
// =============================================================================================

/// `floor(2^43 / q)`, the Barrett magic of [`barrett_u31`].
const fn barrett_magic(q: u16) -> u64 {
    (1u64 << 43) / q as u64
}

/// The multiple of `q` added before [`barrett_u31`] to make a folded-back lane non-negative:
/// the smallest one at least [`cm::acc_after_reduce`], so the sum is in `[0, 2 shift_up]`.
const fn shift_up(q: u16) -> i32 {
    let a = cm::acc_after_reduce(q);
    let k = (a + q as i64 - 1) / q as i64;
    (k * q as i64) as i32
}

/// What [`barrett_u31`] needs of a base limb: the shifted lane is a non-negative i32, and the
/// 32x32 product against the magic stays inside the 64-bit lane it is formed in. `2 shift_up` is
/// `2^15 (1 + 2^16 mod q)` rounded up to a multiple of q, so it is largest for 17497
/// (855 million, `2^29.7`) and 9721 (473 million); the products are `2^59` at worst.
const fn drain_fits(q: u16) -> bool {
    let p = 2 * shift_up(q) as u64;
    p < (1u64 << 31) && p * barrett_magic(q) < (1u64 << 63)
}
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(drain_fits(QS_QUAD[i]));
        i += 1;
    }
    let mut i = 0;
    while i < 2 {
        assert!(drain_fits(QS[i]) && drain_fits(QS_LARGE[i]));
        i += 1;
    }
};

/// `p mod q` for `0 <= p < 2^31` with `p * barrett_magic(q) < 2^63` ([`drain_fits`]), 16 lanes at
/// a time: the quotient estimate is short by at most `p / 2^43 < 1`, so one conditional subtract
/// finishes it.
#[inline(always)]
unsafe fn barrett_u31<const Q: u16>(p: __m512i) -> __m512i {
    let q = _mm512_set1_epi32(Q as i32);
    let mag = _mm512_set1_epi64(barrett_magic(Q) as i64);
    let lo = _mm512_set1_epi64(0xFFFF_FFFFu32 as i64);
    let he = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_and_si512(p, lo), mag));
    let ho = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(p), mag));
    let t = _mm512_or_si512(he, _mm512_slli_epi64::<32>(ho));
    let r = _mm512_sub_epi32(p, _mm512_mullo_epi32(t, q));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, q))
}

/// Declared output bound of the generic-input kernel of `q` — `ntt::gen_small` for a splitting
/// prime, `ntt::gen_quad` for a quadratic-slot one.
pub const fn gen_bound(q: u16) -> i32 {
    if q == 3889 {
        13231
    } else if q == 9721 {
        20652
    } else if vl::is_large(q) {
        vgl::output_bound(q)
    } else {
        vgq::output_bound(q)
    }
}
const _: () = assert!(gen_bound(3889) == vg::Tw::<3889>::OUTPUT_BOUND);
const _: () = assert!(gen_bound(9721) == vg::Tw::<9721>::OUTPUT_BOUND);

/// The shift [`center_epi16`] uses: the smallest power of two `K` with `K q >= gen_bound(q)`
/// (4 for 3889, 9721 and 4861, 8 for 2917, 2 for 12637, and 1 for 17497 and 19441, whose generic
/// kernel reduces its own output for exactly this reason: at `q = 19441` even `K = 2` puts
/// `K q + gen_bound` outside a u16 lane).
pub const fn center_k(q: u16) -> u32 {
    let mut k = 1u32;
    while (k * q as u32) < gen_bound(q) as u32 {
        k *= 2;
    }
    k
}

/// `K q + gen_bound(q)` must stay inside a u16 lane, which is what makes the unsigned trick work.
const fn center_fits(q: u16) -> bool {
    center_k(q) * q as u32 + gen_bound(q) as u32 <= 65535
}
const _: () = assert!(center_fits(3889) && center_fits(9721));
const _: () = assert!(center_fits(2917) && center_fits(4861) && center_fits(12637));
const _: () = assert!(center_fits(QS_LARGE[0]) && center_fits(QS_LARGE[1]));
const _: () = assert!(center_k(QS_LARGE[0]) == 1 && center_k(QS_LARGE[1]) == 1);

/// `x mod q` centered into `[-(q-1)/2, (q-1)/2]` for 32 i16 lanes with `|x| <= K q`: shift by
/// `K q` into `[0, 2 K q) < 2^16`, `log2 K + 1` unsigned conditional subtracts, then the
/// centering subtract.
#[inline(always)]
unsafe fn center_epi16<const Q: u16>(x: __m512i) -> __m512i {
    let q = Q as u32;
    let mut k = center_k(Q);
    let mut v = _mm512_add_epi16(x, _mm512_set1_epi16((k * q) as i16));
    while k >= 1 {
        let s = _mm512_sub_epi16(v, _mm512_set1_epi16((k * q) as i16));
        v = _mm512_min_epu16(v, s);
        k /= 2;
    }
    let hi = _mm512_cmpgt_epu16_mask(v, _mm512_set1_epi16(((q - 1) / 2) as i16));
    _mm512_mask_sub_epi16(v, hi, v, _mm512_set1_epi16(q as i16))
}

/// Every slot of a batch fully reduced and centered (`|v| <= (q-1)/2`); the input must satisfy
/// `|v| <= center_k(Q) * q`, which every generic kernel's output bound does.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn center_batch<const Q: u16>(b: &mut Batch32) {
    for j in 0..N {
        let p = b.v[j].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(p, center_epi16::<Q>(_mm512_load_si512(p as *const __m512i)));
    }
}

// =============================================================================================
// the challenges
// =============================================================================================

/// The `r` challenges transformed, packed so that the scalar pair `(c_{2i}[u], c_{2i+1}[u])` is
/// one dword — a `vpbroadcastd` memory operand, and exactly the operand `vpmaddwd` wants against
/// two interleaved witness rows.
pub(crate) struct ChallengeNtt {
    /// `pair[i][u]` = `c_{2i}[u] as u16 | (c_{2i+1}[u] as u16) << 16`.
    pair: Vec<[u32; N]>,
}

/// The `r` challenges embedded as `c(-X^4)` into as many `Batch32` of coefficients as they need.
fn embed(challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let nb = challenges.len().div_ceil(32);
    let mut out: Vec<Batch32> = (0..nb)
        .map(|_| Batch32::zero(Representation::Coefficients))
        .collect();
    for (j, c) in challenges.iter().enumerate() {
        for i in 0..c.weight {
            let m = c.positions[i] as usize;
            let co = 1 - 2 * ((c.signs >> i) & 1) as i16;
            out[j / 32].v[4 * m][j % 32] = if m % 2 == 0 { co } else { -co };
        }
    }
    out
}

/// Which generic-input kernel a splitting prime runs, as an associated const so the choice is
/// made before the branches are emitted.
struct Gen<const Q: u16>;

impl<const Q: u16> Gen<Q> {
    const LARGE: bool = vl::is_large(Q);
}

/// `NTT(b)` for a splitting prime, fully reduced and centered: `ntt::gen_small` below `2^14`,
/// `ntt::gen_large` above it.
///
/// # Safety
/// AVX-512 F/BW/VL/VBMI; `b` holds coefficients with `|x| <= q`.
#[inline(always)]
unsafe fn forward_split<const Q: u16>(b: &mut Batch32) {
    if Gen::<Q>::LARGE {
        vgl::ntt_gen_batch32::<Q>(b);
    } else {
        ntt_gen_batch32::<Q>(b);
    }
    center_batch::<Q>(b);
}

/// Transform the embedded challenges modulo `Q` and fully reduce them to centered slots, which is
/// what the accumulation bound of [`FOLD_PERIOD`] assumes.
pub(crate) fn challenge_ntt<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = slots(challenges);
    unsafe {
        for b in bs.iter_mut() {
            forward_split::<Q>(b);
        }
    }
    pack(&bs, challenges.len())
}

/// Each leaf's scalar written into both of the leaf's rows.
///
/// `c(-X^4)` is a polynomial in `X^4`, and `X^4 = psi'^{2u}` in the leaf `X^2 - psi'^u`, so a
/// challenge's leaf image is the scalar `c(-theta^v)`: row `2j+1` of its transform is zero and
/// row `2j` carries the scalar. Copying it up makes the leaf product `c (w_0 + w_1 X)` the same
/// row-wise multiply a splitting limb's slot product is.
fn duplicate_leaf_scalars(b: &mut Batch32) {
    for j in 0..QUAD_SLOTS {
        debug_assert!(
            b.v[2 * j + 1] == [0i16; 32],
            "a challenge leaf is not a scalar"
        );
        b.v[2 * j + 1] = b.v[2 * j];
    }
}

/// The challenge transform a quadratic-slot *base* limb folds against.
pub(crate) fn challenge_ntt_quad_base<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = slots(challenges);
    unsafe {
        for b in bs.iter_mut() {
            ntt_quad_gen_batch32::<Q>(b);
            center_batch::<Q>(b);
        }
    }
    bs.iter_mut().for_each(duplicate_leaf_scalars);
    pack(&bs, challenges.len())
}

pub(crate) fn challenge_batches(q: u16, challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let mut bs = slots(challenges);
    forward_limb(q, &mut bs);
    bs
}

fn slots(challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let r = challenges.len();
    assert!(
        r >= 2 && r % 2 == 0,
        "the fold pairs the chunks: r must be even"
    );
    embed(challenges)
}

/// The transformed batches read out per challenge and packed into the dword pairs the
/// accumulation wants.
fn pack(bs: &[Batch32], r: usize) -> ChallengeNtt {
    let mut pair = vec![[0u32; N]; r / 2];
    for i in 0..r / 2 {
        for u in 0..N {
            let lo = bs[(2 * i) / 32].v[u][(2 * i) % 32] as u16;
            let hi = bs[(2 * i + 1) / 32].v[u][(2 * i + 1) % 32] as u16;
            pair[i][u] = lo as u32 | ((hi as u32) << 16);
        }
    }
    ChallengeNtt { pair }
}

// =============================================================================================
// the accumulator
// =============================================================================================

/// One 64-byte aligned accumulator vector.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AccVec([i32; 16]);

/// `acc[(b * 648 + u) * 2 + h]`: batch position `b` inside a chunk, slot `u`, half `h`.
///
/// Half 0 is `vpunpcklwd` of the two witness rows, half 1 is `vpunpckhwd`, so lane `t` of half
/// `h` carries ring element [`lane_of`]`(h, t)` of the chunk. The permutation is undone once, when
/// the accumulator is read out.
const fn lane_of(h: usize, t: usize) -> usize {
    8 * (t / 4) + 4 * h + t % 4
}

/// The `vpermi2d` indices that undo [`lane_of`]: `NAT[k]` gathers lanes `16k .. 16k+16` of the
/// natural element order out of (half 0, half 1).
const NAT: [[i32; 16]; 2] = {
    let mut nat = [[0i32; 16]; 2];
    let mut h = 0;
    while h < 2 {
        let mut t = 0;
        while t < 16 {
            let p = lane_of(h, t);
            nat[p / 16][p % 16] = (16 * h + t) as i32;
            t += 1;
        }
        h += 1;
    }
    nat
};

/// Two consecutive chunks into the accumulator of one batch position: 648 slot vectors, one
/// `vpunpck` and one `vpmaddwd` per half.
///
/// The two witness rows and the accumulator are three plain forward streams, which is all the
/// hardware prefetcher needs: one `prefetcht1` per slot aimed one step ahead (as
/// `simd::commit` does for `A`) measured 4.96 ms against 4.26 ms with none — unlike the
/// commitment, this loop has no compute to hide the extra fill-buffer pressure behind.
///
/// # Safety
/// `w0`, `w1` and `acc` cover 648 vectors (`acc` 648 pairs).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_pair(w0: *const i16, w1: *const i16, cp: *const u32, acc: *mut i32) {
    for u in 0..N {
        let a = _mm512_load_si512(w0.add(32 * u) as *const __m512i);
        let b = _mm512_load_si512(w1.add(32 * u) as *const __m512i);
        let c = _mm512_set1_epi32(*cp.add(u) as i32);
        let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c);
        let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c);
        let d = acc.add(32 * u);
        _mm512_store_si512(
            d as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d as *const __m512i), lo),
        );
        _mm512_store_si512(
            d.add(16) as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d.add(16) as *const __m512i), hi),
        );
    }
}

/// The exact fold-back of [`crate::simd::commit`] over one batch position's accumulator.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn fold_back<const Q: u16>(acc: *mut i32, vecs: usize) {
    for j in 0..vecs {
        let p = acc.add(16 * j);
        _mm512_store_si512(
            p as *mut __m512i,
            cm::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)),
        );
    }
}

/// The accumulator of one batch position, folded back a last time, reduced to `[0, q)` and
/// written out in the natural element order as centered slots of `out`.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn drain<const Q: u16>(acc: *const i32, out: &mut Batch32) {
    let up = _mm512_set1_epi32(shift_up(Q));
    let half = _mm512_set1_epi32((Q as i32 - 1) / 2);
    let q = _mm512_set1_epi32(Q as i32);
    let i0 = _mm512_loadu_si512(NAT[0].as_ptr() as *const __m512i);
    let i1 = _mm512_loadu_si512(NAT[1].as_ptr() as *const __m512i);
    for u in 0..N {
        let p = acc.add(32 * u);
        let lo = cm::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i));
        let hi = cm::reduce_vec::<Q>(_mm512_load_si512(p.add(16) as *const __m512i));
        let lo = barrett_u31::<Q>(_mm512_add_epi32(lo, up));
        let hi = barrett_u31::<Q>(_mm512_add_epi32(hi, up));
        let n0 = _mm512_permutex2var_epi32(lo, i0, hi);
        let n1 = _mm512_permutex2var_epi32(lo, i1, hi);
        let c0 = _mm512_mask_sub_epi32(n0, _mm512_cmpgt_epi32_mask(n0, half), n0, q);
        let c1 = _mm512_mask_sub_epi32(n1, _mm512_cmpgt_epi32_mask(n1, half), n1, q);
        let d = out.v[u].as_mut_ptr();
        _mm256_storeu_si256(d as *mut __m256i, _mm512_cvtepi32_epi16(c0));
        _mm256_storeu_si256(d.add(16) as *mut __m256i, _mm512_cvtepi32_epi16(c1));
    }
}

/// Batch positions accumulated together: 324 KB of accumulator, and `2 G` witness streams.
const G: usize = 4;

/// `v_ntt = sum_j c_j o W_j` modulo the base prime `Q`, centered, one `Batch32` per batch
/// position.
fn accumulate<const Q: u16>(w: &AuxData, ch: &ChallengeNtt, bpc: usize) -> Vec<Batch32> {
    let pairs = ch.pair.len();
    let g = G.min(bpc);
    let starts: Vec<usize> = (0..bpc).step_by(g).collect();
    let mut out: Vec<Batch32> = (0..bpc)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    // Each `b0` group streams a disjoint set of batch positions and writes a disjoint slice of
    // `out`. The address is a usize so the worker is `Send`.
    let out_addr = out.as_mut_ptr() as usize;
    let stride = std::mem::size_of::<Batch32>();
    std::thread::scope(|scope| {
        for b0 in starts {
            let w = w;
            let ch = ch;
            scope.spawn(move || {
                let mut acc = vec![AccVec([0i32; 16]); g * N * 2];
                let base = acc.as_mut_ptr() as *mut i32;
                unsafe {
                    for i in 0..pairs {
                        for b in 0..g {
                            let row = |j: usize| {
                                w.batch(j * bpc + b0 + b).v.as_ptr() as *const i16
                            };
                            accumulate_pair(
                                row(2 * i),
                                row(2 * i + 1),
                                ch.pair[i].as_ptr(),
                                base.add(32 * N * b),
                            );
                        }
                        if (2 * (i + 1)) % fold_period(Q) == 0 {
                            fold_back::<Q>(base, g * N * 2);
                        }
                    }
                    for b in 0..g {
                        let dst = &mut *((out_addr + (b0 + b) * stride) as *mut Batch32);
                        drain::<Q>(base.add(32 * N * b), dst);
                    }
                }
            });
        }
    });
    out
}

// =============================================================================================
// A v
// =============================================================================================

/// `y[u] = sum_i A_i[u] v_i[u] mod q`, the commitment of the folded witness, on the commitment's
/// own packed accumulator: `|v| <= (q-1)/2` and `|A| <= (q-1)/2`, folded back every
/// [`av_period`] batches.
pub(crate) fn a_times_v<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    unsafe {
        for b in 0..a.len() {
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_batch::<false>(v[b].v.as_ptr() as *const i16, ar, ar as *const i8, ap);
            if (b + 1) % av_period(Q) == 0 {
                cm::reduce_acc::<Q>(ap);
            }
        }
    }
    cm::finish::<Q>(&acc)
}

/// The same for a quadratic-slot limb, on the quadratic accumulator of
/// [`crate::simd::commit`]: three sums per leaf, combined into the leaf's two rows at the end.
/// `|v| <= (q-1)/2` and `|A| <= (q-1)/2`, so the fold-back periods are the wider [`av_period`]
/// ones rather than the commitment's.
pub(crate) fn a_times_v_quad<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::QuadAcc::zero();
    let (p01, p2) = (
        acc.p01.as_mut_ptr() as *mut i32,
        acc.p2.as_mut_ptr() as *mut i32,
    );
    unsafe {
        for b in 0..a.len() {
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_quad_batch::<Q, false>(
                v[b].v.as_ptr() as *const i16,
                ar,
                ar as *const i8,
                p01,
                p2,
            );
            if (b + 1) % av_period_quad(Q) == 0 {
                cm::reduce_quad_acc::<Q>(&mut acc);
            }
        }
    }
    cm::finish_quad::<Q>(&acc)
}

/// Batches of `A v` between two fold-backs of a quadratic limb's accumulators: both operands are
/// centered (`(q-1)/2`), so the widest lane grows by `16 ((q-1)/2)^2` per batch (the Karatsuba
/// `P_2`) or `8 ((q-1)/2)^2` (the schoolbook one).
pub const fn av_period_quad(q: u16) -> usize {
    let per = if cm::karatsuba(q) {
        16 * cm::a_bound(q) * cm::a_bound(q)
    } else {
        8 * cm::a_bound(q) * cm::a_bound(q)
    };
    cm::period_for(q, per)
}
const _: () = assert!(av_period_quad(2917) >= 1 && av_period_quad(4861) >= 1);
const _: () = assert!(av_period_quad(12637) >= 1);

// =============================================================================================
// the fold
// =============================================================================================

/// The inverse transform of a splitting base limb: `ntt::gen_small` below `2^14`,
/// `ntt::gen_large` above it, chosen before the branches are emitted.
///
/// # Safety
/// AVX-512 F/BW/VL/VBMI; `b` is a centered transform.
#[inline(always)]
unsafe fn inverse_split<const Q: u16>(b: &mut Batch32) {
    if Gen::<Q>::LARGE {
        vgl::intt_gen_batch32::<Q>(b);
    } else {
        intt_gen_batch32::<Q>(b);
    }
}

/// Run `f` on each batch from several threads. Batches are disjoint; the pointer is a `usize`
/// because `*mut Batch32` is not `Send`.
fn parallel_batches(vb: &mut [Batch32], f: impl Fn(&mut Batch32) + Sync) {
    let n = vb.len();
    if n == 0 {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, 8)
        .min(n);
    let addr = vb.as_mut_ptr() as usize;
    let stride = std::mem::size_of::<Batch32>();
    crate::par::pool().run(threads, |t| {
        let mut i = t;
        while i < n {
            unsafe {
                f(&mut *((addr + i * stride) as *mut Batch32));
            }
            i += threads;
        }
    });
}

/// The fold over a splitting base limb: transform the challenges, accumulate, invert.
fn fold_split<const Q: u16>(
    aux: &AuxData,
    challenges: &[ShortChallenge],
    bpc: usize,
) -> Vec<Batch32> {
    let ch = challenge_ntt::<Q>(challenges);
    let mut vb = accumulate::<Q>(aux, &ch, bpc);
    parallel_batches(&mut vb, |b| unsafe { inverse_split::<Q>(b) });
    vb
}

/// The same over a quadratic-slot base limb.
fn fold_quad<const Q: u16>(
    aux: &AuxData,
    challenges: &[ShortChallenge],
    bpc: usize,
) -> Vec<Batch32> {
    let ch = challenge_ntt_quad_base::<Q>(challenges);
    let mut vb = accumulate::<Q>(aux, &ch, bpc);
    parallel_batches(&mut vb, |b| unsafe { intt_quad_gen_batch32::<Q>(b) });
    vb
}

/// `v = sum_j c_j W_j`, in coefficient form modulo the base prime, centered — the amortised
/// witness.
///
/// Stage (a) embeds the challenges as `c(-X^4)` and transforms them modulo the base limb, (b) is
/// the slot-wise inner product over the kept witness (the 85 MB stream), (c) is the base limb's
/// inverse transform, whose output is already fully reduced and centered, so `v` is the true
/// integer vector: a coefficient is a sum of `r w` signed 0/1 terms, standard deviation
/// `sqrt(r w / 2)`, two orders below `q/2` for the narrowest base there is (2917, `q/2 = 1458.5`).
pub(crate) fn fold_witness(
    aux: &AuxData,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), aux.chunks(), "one challenge per chunk");
    assert_eq!(
        bpc,
        aux.batches_per_chunk(),
        "the key and the chunks disagree"
    );
    let vb = dispatch_limb!(
        q,
        split |Q| fold_split::<Q>(aux, challenges, bpc),
        quad |Q| fold_quad::<Q>(aux, challenges, bpc),
    );
    (0..32 * bpc).map(|i| vb[i / 32].get(i % 32)).collect()
}

const QUAD_COMPONENT_SLOT: [[u16; N162]; 4] = {
    let mut m = [[0u16; N162]; 4];
    let mut s = 0;
    while s < N162 {
        m[0][s] = 2 * QUAD_CLASS_SLOT[0][s];
        m[1][s] = 2 * QUAD_CLASS_SLOT[0][s] + 1;
        m[2][s] = 2 * QUAD_CLASS_SLOT[1][s];
        m[3][s] = 2 * QUAD_CLASS_SLOT[1][s] + 1;
        s += 1;
    }
    m
};

const _: () = {
    let mut seen = [false; N];
    let mut t = 0;
    while t < 4 {
        let mut s = 0;
        while s < N162 {
            let u = QUAD_COMPONENT_SLOT[t][s] as usize;
            assert!(!seen[u]);
            seen[u] = true;
            s += 1;
        }
        t += 1;
    }
};

pub(crate) fn component_slots(quad: bool) -> &'static [[u16; N162]; 4] {
    if quad {
        &QUAD_COMPONENT_SLOT
    } else {
        &SLOT_648
    }
}

/// `NTT(v)` for one limb, in place on centered coefficient batches, fully reduced and centered.
pub(crate) fn forward_limb(q: u16, bs: &mut [Batch32]) {
    unsafe {
        for b in bs.iter_mut() {
            b.representation = Representation::Coefficients;
            dispatch_limb!(
                q,
                split |Q| forward_split::<Q>(b),
                quad |Q| {
                    ntt_quad_gen_batch32::<Q>(b);
                    center_batch::<Q>(b);
                },
            )
        }
    }
}

/// `A v` for one limb, dispatched on its prime.
pub(crate) fn a_times_v_limb(q: u16, a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    dispatch_limb!(
        q,
        split |Q| a_times_v::<Q>(a, v),
        quad |Q| a_times_v_quad::<Q>(a, v),
    )
}

fn a_times_v_fwd<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    let mut w = Box::new(Batch32::zero(Representation::Coefficients));
    unsafe {
        for b in 0..a.len() {
            w.v.copy_from_slice(&v[b].v);
            w.representation = Representation::Coefficients;
            forward_split::<Q>(&mut w);
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_batch::<false>(w.v.as_ptr() as *const i16, ar, ar as *const i8, ap);
            if (b + 1) % av_period(Q) == 0 {
                cm::reduce_acc::<Q>(ap);
            }
        }
    }
    cm::finish::<Q>(&acc)
}

fn a_times_v_fwd_quad<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    assert_eq!(a.len(), v.len());
    let mut acc = cm::QuadAcc::zero();
    let (p01, p2) = (
        acc.p01.as_mut_ptr() as *mut i32,
        acc.p2.as_mut_ptr() as *mut i32,
    );
    let mut w = Box::new(Batch32::zero(Representation::Coefficients));
    unsafe {
        for b in 0..a.len() {
            w.v.copy_from_slice(&v[b].v);
            w.representation = Representation::Coefficients;
            ntt_quad_gen_batch32::<Q>(&mut w);
            center_batch::<Q>(&mut w);
            let ar = a[b].v.as_ptr() as *const i16;
            cm::mac_quad_batch::<Q, false>(
                w.v.as_ptr() as *const i16,
                ar,
                ar as *const i8,
                p01,
                p2,
            );
            if (b + 1) % av_period_quad(Q) == 0 {
                cm::reduce_quad_acc::<Q>(&mut acc);
            }
        }
    }
    cm::finish_quad::<Q>(&acc)
}

pub(crate) fn a_times_v_forward(q: u16, a: &[Batch32], v: &[Batch32]) -> [u32; N] {
    dispatch_limb!(
        q,
        split |Q| a_times_v_fwd::<Q>(a, v),
        quad |Q| a_times_v_fwd_quad::<Q>(a, v),
    )
}

static SLOT_TABLES: [OnceLock<Box<sl::SlotTable>>; crate::limb::LIMBS] =
    [const { OnceLock::new() }; crate::limb::LIMBS];

fn slot_prime_index(q: u16) -> usize {
    crate::limb::PRIMES
        .iter()
        .position(|&p| p == q)
        .unwrap_or_else(|| unreachable!("no limb with q = {q}"))
}

fn build_slot_table(q: u16) -> Box<sl::SlotTable> {
    let units: Vec<ShortChallenge> = (0..N162)
        .map(|p| {
            let mut c = ShortChallenge::zero();
            c.positions[0] = p as u8;
            c.weight = 1;
            c
        })
        .collect();
    let quad = is_quad(q);
    let mut bs = challenge_batches(q, &units);
    if quad {
        bs.iter_mut().for_each(duplicate_leaf_scalars);
    }
    let map = component_slots(quad);
    let mut t = sl::SlotTable::zero();
    for p in 0..N162 {
        for s in 0..N162 {
            let x = bs[p / 32].v[map[0][s] as usize][p % 32];
            t.rows[2 * p][s] = x;
            t.rows[2 * p + 1][s] = -x;
        }
    }
    t
}

fn slot_table(q: u16) -> &'static sl::SlotTable {
    SLOT_TABLES[slot_prime_index(q)].get_or_init(|| build_slot_table(q))
}

fn fold_columns_limb<const Q: u16>(
    challenges: &[ShortChallenge],
    columns: &[[*const i16; 4]],
    out: &mut [[i16; N162]; 4],
) {
    let table = slot_table(Q);
    let mut acc = [sl::SlotAcc::zero(); 4];
    let mut ch = sl::Ch::zero();
    let period = sl::slot_period(Q);
    unsafe {
        for (j, c) in challenges.iter().enumerate() {
            sl::challenge_slots::<Q>(table, c, &mut ch);
            for (t, a) in acc.iter_mut().enumerate() {
                sl::mac_slots(a, &ch, columns[j][t]);
            }
            if (j + 1) % period == 0 {
                for a in acc.iter_mut() {
                    sl::reduce_slots::<Q>(a);
                }
            }
        }
        for (t, a) in acc.iter().enumerate() {
            sl::finish_slots::<Q>(a, &mut out[t]);
        }
    }
}

pub(crate) fn fold_columns_slots(
    q: u16,
    challenges: &[ShortChallenge],
    columns: &[[*const i16; 4]],
    out: &mut [[i16; N162]; 4],
) {
    dispatch_limb!(q, |Q| fold_columns_limb::<Q>(challenges, columns, out))
}

// =============================================================================================
// the streaming fold: the transform regenerated from the words instead of kept
// =============================================================================================
//
// At the challenge's shape the kept base-limb transform is `columns * bpc` batches of 41472
// bytes — 5.4 GB for 2^24 words — written once with non-temporal stores, faulted in, read once
// here and freed at exit. One binary transform of a batch is a few hundred cycles per ring
// element, so regenerating it from the words costs less than streaming it, and the prover's
// footprint drops to the words themselves.
//
// The accumulation is [`accumulate`]'s exactly: the same `accumulate_pair` over the same
// lazily reduced kernel output (`ntt_bin_batch32` writes the values `MacKeep` streamed out), the
// same fold-back cadence, the same `drain` and inverse transform, so the fold is bit for bit the
// one the kept transform gives. Each worker owns a run of batch positions and regenerates the
// `(column, position)` transforms it needs, so every transform is computed exactly once.

use crate::fields::scalar::B128;
use crate::simd::ntt::bin_asm as vb;
use crate::simd::ntt::bin_quad as vq;
use crate::simd::transpose_f162::{slice_b128_into, BinaryIndex32};

/// The binary transform of a base limb's tree, as a type so that a splitting prime only ever
/// instantiates the splitting kernels and a quadratic-slot prime only the quadratic one (the
/// kernels' tables are built at compile time for their own class of prime).
trait WordTransform {
    /// `NTT_q(w)` of the 32 binary ring elements `idx` indexes, lazily reduced, in `out`.
    ///
    /// # Safety
    /// AVX-512 F/BW/VL/VBMI/VBMI2/VNNI and GFNI.
    unsafe fn transform(idx: &BinaryIndex32, out: &mut Batch32);
}

struct SplitTransform<const Q: u16>;
struct QuadTransform<const Q: u16>;

impl<const Q: u16> WordTransform for SplitTransform<Q> {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
    unsafe fn transform(idx: &BinaryIndex32, out: &mut Batch32) {
        if Gen::<Q>::LARGE {
            vl::ntt_bin_batch32::<Q>(idx, out);
        } else {
            vb::ntt_bin_batch32::<Q>(idx, out);
        }
    }
}

impl<const Q: u16> WordTransform for QuadTransform<Q> {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
    unsafe fn transform(idx: &BinaryIndex32, out: &mut Batch32) {
        vq::ntt_quad_bin_batch32::<Q>(idx, out);
    }
}

/// `NTT_q(w)` of the 32 binary ring elements formed by `words`, lazily reduced, in `out`.
///
/// # Safety
/// AVX-512 F/BW/VL/VBMI/VBMI2/VNNI and GFNI.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn transform_words<T: WordTransform>(
    words: &[B128; 128],
    idx: &mut BinaryIndex32,
    out: &mut Batch32,
) {
    slice_b128_into(words, idx);
    T::transform(idx, out);
}

/// [`accumulate`] with the witness transform regenerated from `words`: `v_ntt` modulo `Q`,
/// centered, one `Batch32` per batch position. `words` is `columns` columns of `128 * bpc`
/// words each.
fn accumulate_streaming<const Q: u16, T: WordTransform>(
    words: &[B128],
    columns: usize,
    ch: &ChallengeNtt,
    bpc: usize,
    workers: usize,
) -> Vec<Batch32> {
    let pairs = ch.pair.len();
    assert_eq!(2 * pairs, columns, "one challenge per column, paired");
    let column_words = 128 * bpc;
    assert_eq!(words.len(), columns * column_words, "the words are not whole columns");
    let g = G.min(bpc);
    let groups: Vec<usize> = (0..bpc).step_by(g).collect();
    let workers = workers.clamp(1, groups.len());
    let mut out: Vec<Batch32> = (0..bpc)
        .map(|_| Batch32::zero(Representation::Ntt))
        .collect();
    let out_addr = out.as_mut_ptr() as usize;
    let stride = std::mem::size_of::<Batch32>();
    let next = std::sync::atomic::AtomicUsize::new(0);
    crate::par::pool().run(workers, |_| {
        let mut acc = vec![AccVec([0i32; 16]); g * N * 2];
        let base = acc.as_mut_ptr() as *mut i32;
        let mut idx = BinaryIndex32::zero();
        let mut w0 = Box::new(Batch32::zero(Representation::Ntt));
        let mut w1 = Box::new(Batch32::zero(Representation::Ntt));
        loop {
            let gi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if gi >= groups.len() {
                break;
            }
            let b0 = groups[gi];
            let gb = g.min(bpc - b0);
            for v in acc.iter_mut() {
                *v = AccVec([0i32; 16]);
            }
            unsafe {
                for i in 0..pairs {
                    for b in 0..gb {
                        let at = |j: usize| {
                            let start = j * column_words + 128 * (b0 + b);
                            &*(words.as_ptr().add(start) as *const [B128; 128])
                        };
                        transform_words::<T>(at(2 * i), &mut idx, &mut w0);
                        transform_words::<T>(at(2 * i + 1), &mut idx, &mut w1);
                        accumulate_pair(
                            w0.v.as_ptr() as *const i16,
                            w1.v.as_ptr() as *const i16,
                            ch.pair[i].as_ptr(),
                            base.add(32 * N * b),
                        );
                    }
                    if (2 * (i + 1)) % fold_period(Q) == 0 {
                        fold_back::<Q>(base, gb * N * 2);
                    }
                }
                for b in 0..gb {
                    let dst = &mut *((out_addr + (b0 + b) * stride) as *mut Batch32);
                    drain::<Q>(base.add(32 * N * b), dst);
                }
            }
        }
    });
    out
}

fn fold_split_streaming<const Q: u16>(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    workers: usize,
) -> Vec<Batch32> {
    let ch = challenge_ntt::<Q>(challenges);
    let mut vb = accumulate_streaming::<Q, SplitTransform<Q>>(words, columns, &ch, bpc, workers);
    parallel_batches(&mut vb, |b| unsafe { inverse_split::<Q>(b) });
    vb
}

fn fold_quad_streaming<const Q: u16>(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    workers: usize,
) -> Vec<Batch32> {
    let ch = challenge_ntt_quad_base::<Q>(challenges);
    let mut vb = accumulate_streaming::<Q, QuadTransform<Q>>(words, columns, &ch, bpc, workers);
    parallel_batches(&mut vb, |b| unsafe { intt_quad_gen_batch32::<Q>(b) });
    vb
}

/// [`fold_witness`] from the `B128` words the witness was lifted from, with the base-limb
/// transform regenerated on the fly rather than read from an [`AuxData`]: the same `v`, in
/// coefficient form modulo the base prime `q`, centered. `words` is `columns` columns of
/// `128 * bpc` words; `workers` bounds the threads.
pub fn fold_witness_streaming(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    workers: usize,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc, "the words are not whole columns");
    let vb = dispatch_limb!(
        q,
        split |Q| fold_split_streaming::<Q>(words, columns, challenges, bpc, workers),
        quad |Q| fold_quad_streaming::<Q>(words, columns, challenges, bpc, workers),
    );
    (0..32 * bpc).map(|i| vb[i / 32].get(i % 32)).collect()
}

// =============================================================================================
// the fold from the words with bit-sliced counters
// =============================================================================================
//
// [`crate::fold_bits`] computes `v` without any transform: a few vector operations per
// challenge term on 512-lane bit planes of the words. This is the driver: the GFNI transpose of
// the library feeds the planes, the blocks go out over the pool, and two runtime checks guard
// the result — the GFNI transposer is compared with the portable one on one block (and
// replaced by it if they differ), and one batch position of the fold is recomputed through the
// transform path by the caller (`fold_witness_positions`) and compared.

use crate::fold_bits::{self as fb, Challenge, Lanes, Transposer};
use crate::simd::transpose_f162::transpose8x8_q;
use crate::fields::bitmat::{transpose_tile, GF_IDENT};

/// The library's GFNI tile transpose, continued to bit planes: per 64-word tile the byte
/// transpose, the 8x8 bit transpose of every qword (`gf2p8affine` by the identity), and an 8x8
/// byte transpose, which leaves qword `t` of vector `k` holding bit `8 k + t` of the tile's
/// 64 words in `word_at` order; a qword transpose across the eight tiles then assembles each
/// plane's 512 lanes.
///
/// Which plane index a given (`k`, `t`) lands on depends on the affine's bit convention, so the
/// driver derives the plane permutation at run time against the portable transposer.
pub struct Gfni {
    /// `plane[q]` is produced at raw index `perm[q]`.
    perm: [u8; fb::WORD_BITS],
    /// The word lane `l` carries.
    map: fn(usize) -> usize,
}

impl Gfni {
    /// A transposer validated against the portable one on `sample` (512 words), under each
    /// candidate lane order: `None` when no lane order and permutation of its raw planes
    /// reproduces the portable planes.
    pub fn validated(sample: &[u128]) -> Option<Gfni> {
        assert!(sample.len() >= fb::LANES);
        let mut raw = [fb::V::zero(); fb::WORD_BITS];
        unsafe { gfni_raw_planes(sample.as_ptr(), &mut raw) };
        let raw_words: Vec<[u64; 8]> = raw.iter().map(|v| v.words()).collect();
        let candidates: [fn(usize) -> usize; 2] = [fb::lane_word_reversed8, fb::lane_word];
        'maps: for map in candidates {
            let mut want = [fb::V::zero(); fb::WORD_BITS];
            unsafe { fb::portable_planes(sample.as_ptr(), &mut want, map) };
            let mut perm = [0u8; fb::WORD_BITS];
            let mut used = [false; fb::WORD_BITS];
            for (q, w) in want.iter().enumerate() {
                let w = w.words();
                let Some(at) = (0..fb::WORD_BITS).find(|&i| !used[i] && raw_words[i] == w) else {
                    continue 'maps;
                };
                used[at] = true;
                perm[q] = at as u8;
            }
            return Some(Gfni { perm, map });
        }
        None
    }
}

/// The raw planes of the GFNI path, before the permutation.
///
/// # Safety
/// AVX-512 F/BW/VBMI and GFNI; `words` addresses 512 readable `u128`.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn gfni_raw_planes(words: *const u128, out: &mut [fb::V; fb::WORD_BITS]) {
    // Byte (8 t + i) of the output takes byte (8 i + t) of the input: an 8x8 byte transpose.
    let idx: [u8; 64] = core::array::from_fn(|o| (8 * (o % 8) + o / 8) as u8);
    let t8 = _mm512_loadu_si512(idx.as_ptr() as *const __m512i);
    let ident = _mm512_set1_epi64(GF_IDENT);
    let mut u = [[_mm512_setzero_si512(); 16]; 8];
    let mut xp = [_mm512_setzero_si512(); 16];
    for tile in 0..8 {
        transpose_tile(words.add(64 * tile) as *const u8, &mut xp);
        for k in 0..16 {
            let bits = _mm512_gf2p8affine_epi64_epi8::<0>(ident, xp[k]);
            u[tile][k] = _mm512_permutexvar_epi8(t8, bits);
        }
    }
    for k in 0..16 {
        let o = transpose8x8_q(core::array::from_fn(|tile| u[tile][k]));
        for t in 0..8 {
            out[8 * k + t] = fb::V(o[t]);
        }
    }
}

impl Transposer for Gfni {
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; fb::WORD_BITS]) {
        let mut raw = [fb::V::zero(); fb::WORD_BITS];
        gfni_raw_planes(words, &mut raw);
        for q in 0..fb::WORD_BITS {
            out[q] = L::from_words(raw[self.perm[q] as usize].words());
        }
    }
    fn lane_word(&self, l: usize) -> usize {
        (self.map)(l)
    }
}

/// `v = sum_j c_j W_j` from the words with the bit-sliced counters, in coefficient form,
/// centered: the same elements [`fold_witness_streaming`] returns. `words` is `columns`
/// columns of `column_words` words (a multiple of 512). Blocks of 128 elements go out over the
/// pool.
pub fn fold_witness_bits(
    words: &[B128],
    columns: usize,
    column_words: usize,
    challenges: &[ShortChallenge],
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(column_words % fb::LANES == 0 && words.len() == columns * column_words);
    let cs: Vec<Challenge> = challenges
        .iter()
        .map(|c| Challenge {
            positions: c.positions,
            signs: c.signs,
            weight: c.weight,
        })
        .collect();
    let raw = words.as_ptr() as usize;
    let elements = column_words / 4;
    let blocks = elements / fb::BLOCK_ELEMENTS;
    let reach = fb::Reach::of(&cs);
    let mut out: Vec<[i16; N]> = vec![[0i16; N]; elements];
    let out_addr = out.as_mut_ptr() as usize;
    let sample = &words[..fb::LANES];
    let gfni = Gfni::validated(unsafe { core::slice::from_raw_parts(sample.as_ptr() as *const u128, fb::LANES) });
    FOLD_USED_GFNI.store(gfni.is_some(), std::sync::atomic::Ordering::Relaxed);
    let run_block = |block: usize| {
        let chunk = unsafe {
            core::slice::from_raw_parts_mut(
                (out_addr as *mut [i16; N]).add(block * fb::BLOCK_ELEMENTS),
                fb::BLOCK_ELEMENTS,
            )
        };
        let raw = raw as *const u128;
        unsafe {
            match &gfni {
                Some(t) => fb::fold_block::<fb::V, Gfni>(raw, columns, column_words, &cs, &reach, block, t, chunk),
                None => fb::fold_block::<fb::V, fb::Portable>(raw, columns, column_words, &cs, &reach, block, &fb::Portable, chunk),
            }
        }
    };
    crate::par::pool().run(blocks, run_block);
    out.into_iter()
        .map(|v| RingElement {
            v,
            representation: Representation::Coefficients,
        })
        .collect()
}

/// Whether the last [`fold_witness_bits`] ran on the GFNI transposer (else the portable one),
/// for the profile line.
pub static FOLD_USED_GFNI: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`fold_witness_streaming`] restricted to the batch positions `positions` (32 ring elements
/// each): element `32 b + i` of the fold for `b` in `positions`, in order — the transform path's
/// answer, for checking another path against.
pub fn fold_witness_positions(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    positions: core::ops::Range<usize>,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc && positions.end <= bpc);
    let vb = dispatch_limb!(
        q,
        split |Q| {
            let ch = challenge_ntt::<Q>(challenges);
            let mut vb = accumulate_streaming_positions::<Q, SplitTransform<Q>>(words, columns, &ch, bpc, positions.clone());
            parallel_batches(&mut vb, |b| unsafe { inverse_split::<Q>(b) });
            vb
        },
        quad |Q| {
            let ch = challenge_ntt_quad_base::<Q>(challenges);
            let mut vb = accumulate_streaming_positions::<Q, QuadTransform<Q>>(words, columns, &ch, bpc, positions.clone());
            parallel_batches(&mut vb, |b| unsafe { intt_quad_gen_batch32::<Q>(b) });
            vb
        },
    );
    (0..32 * vb.len()).map(|i| vb[i / 32].get(i % 32)).collect()
}

/// [`accumulate_streaming`] over the batch positions `positions` only, one position per task.
fn accumulate_streaming_positions<const Q: u16, T: WordTransform>(
    words: &[B128],
    columns: usize,
    ch: &ChallengeNtt,
    bpc: usize,
    positions: core::ops::Range<usize>,
) -> Vec<Batch32> {
    let pairs = ch.pair.len();
    assert_eq!(2 * pairs, columns, "one challenge per column, paired");
    let column_words = 128 * bpc;
    let n = positions.len();
    let mut out: Vec<Batch32> = (0..n).map(|_| Batch32::zero(Representation::Ntt)).collect();
    let out_addr = out.as_mut_ptr() as usize;
    let stride = std::mem::size_of::<Batch32>();
    let first = positions.start;
    crate::par::pool().run(n, |i| {
        let b = first + i;
        let mut acc = vec![AccVec([0i32; 16]); N * 2];
        let base = acc.as_mut_ptr() as *mut i32;
        let mut idx = BinaryIndex32::zero();
        let mut w0 = Box::new(Batch32::zero(Representation::Ntt));
        let mut w1 = Box::new(Batch32::zero(Representation::Ntt));
        unsafe {
            for p in 0..pairs {
                let at = |j: usize| {
                    let start = j * column_words + 128 * b;
                    &*(words.as_ptr().add(start) as *const [B128; 128])
                };
                transform_words::<T>(at(2 * p), &mut idx, &mut w0);
                transform_words::<T>(at(2 * p + 1), &mut idx, &mut w1);
                accumulate_pair(
                    w0.v.as_ptr() as *const i16,
                    w1.v.as_ptr() as *const i16,
                    ch.pair[p].as_ptr(),
                    base,
                );
                if (2 * (p + 1)) % fold_period(Q) == 0 {
                    fold_back::<Q>(base, N * 2);
                }
            }
            let dst = &mut *((out_addr + i * stride) as *mut Batch32);
            drain::<Q>(base, dst);
        }
    });
    out
}

/// Exact integer fold directly from the binary words, using 512-lane bit-sliced counters.
/// Shapes without whole 512-word blocks or with more than 32767 challenge terms retain the
/// NTT implementation. For honest challenge instances the exact coefficients equal its
/// centered residues; this path performs no modular reconstruction.
pub fn fold_witness_bitsliced(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    workers: usize,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc, "the words are not whole columns");
    if bpc % 4 != 0 || !bitsliced::fits(challenges) {
        return fold_witness_streaming(words, columns, challenges, bpc, q, workers);
    }
    bitsliced::fold_words(words, challenges, 128 * bpc, workers)
}

// Exact fold of the binary witness, independent of commitment-modulus arithmetic.
mod bitsliced {
    //! Integer convolution in Z[Y]/(Y^243 + 1), followed by reduction modulo
    //! Y^162 - Y^81 + 1. Since (Y^81 + 1)(Y^162 - Y^81 + 1) = Y^243 + 1,
    //! this gives exactly the original ring product, with no probabilistic reconstruction.
    //!
    //! A negative bit contribution is accumulated as its complement, with a scalar bias removed
    //! at the end: -b = (1-b)-1. The complement is never stored: a slot pair's negative terms
    //! are scheduled apart from its positive ones, and their leaf full adders take the carry of
    //! the complemented inputs (the sum of two complemented inputs is the sum of the direct
    //! ones). Every negative entry, guard and padding reads included, adds one to the bias of
    //! both its slots. The counter's low six bits use a 64-input carry-save tree (two ternlogs
    //! per full adder); the carry into bit six ripples through the remaining nine bits once per
    //! 64 inputs. The 15-bit counter cannot overflow when the total challenge weight plus the
    //! padding ([`fits`]) is at most 32767.
    use super::{B128, Representation, RingElement, ShortChallenge};
    use crate::fields::bitmat;
    use crate::simd::transpose_f162::transpose8x8_q;
    use core::arch::x86_64::*;

    const LANES: usize = 512;
    const SLOTS: usize = 243;
    const BITS: usize = 15;
    /// Columns per group: 16 * 129 planes, 132 KB per worker (measured on the full instance:
    /// 19.8 ms at 16 columns, 19.9 at 12 and 20, 20.0 at 24, 20.6 at 31, 21.6 at 63). The
    /// planes of a group (`GUARD + COL_GROUP * STRIDE`) must keep the qword offsets in u16.
    const COL_GROUP: usize = 16;
    /// Planes per column: the 128 direct planes and a zero guard, which is also the next
    /// column's leading guard. Every term's 128-plane run is flanked by zero planes, so a window
    /// of two consecutive slots reads a zero wherever the term does not reach, with no edge
    /// cases in the schedule.
    const STRIDE: usize = 129;
    /// Plane index of column `j`'s first direct plane; `GUARD + j * STRIDE - 1` is its leading
    /// zero guard, and planes 0 and 1 (the padding pair of a short chunk) are both zero.
    const GUARD: usize = 2;
    /// Slot pairs `(2q, 2q + 1)`; the odd slot of the last pair (243) is a dummy.
    const PAIRS: usize = SLOTS.div_ceil(2);
    /// Entries a group's schedule may add to a slot pair beyond one per term: padding of its
    /// positive and of its negative entries to multiples of 16.
    const PAD_PER_GROUP: usize = 2 * 15;
    // The offsets use x86's scale-8 addressing directly, avoiding a shift per input.
    const _: () = assert!((GUARD + COL_GROUP * STRIDE) * 8 <= u16::MAX as usize);

    /// Whether the counters cannot overflow: a slot's counter adds at most one entry per
    /// challenge term and the padding of every group.
    pub(super) fn fits(challenges: &[ShortChallenge]) -> bool {
        let weight: usize = challenges.iter().map(|c| c.weight).sum();
        weight + PAD_PER_GROUP * challenges.len().div_ceil(COL_GROUP) <= i16::MAX as usize
    }

    struct Group {
        first: usize,
        columns: usize,
        /// `terms[offsets[q] .. mid[q]]` and `terms[mid[q] .. offsets[q + 1]]` are the
        /// positive and the negative entries of slot pair `q`, each padded to a multiple of 16
        /// with offset 0 (planes 0 and 1 are zero): whole 64-input chunks, then at most one 32-
        /// and one 16-input tree.
        offsets: [usize; PAIRS + 1],
        mid: [usize; PAIRS],
        terms: Vec<u16>,
    }

    /// The schedule of column group `g` (columns `cs`), and per slot pair the number of its
    /// negative entries (padding included): each adds one to the bias of both slots.
    fn plan_group<const GROUP: usize>(g: usize, cs: &[ShortChallenge]) -> (Group, [i32; PAIRS]) {
        // A term's run `[m, end)` of slots meets slot pair `q` when `2q <= end - 1` and
        // `2q + 1 >= m`, i.e. `q` in `[m / 2, (end - 1) / 2]`; the wrapped run `[0, wrap)`
        // meets pairs `[0, (wrap - 1) / 2]`, with the opposite sign (`Y^243 = -1`).
        // `count_diff[negative]`: the entries of each sign per pair, as range differences.
        let mut count_diff = [[0i32; PAIRS + 1]; 2];
        let mut shifts = Vec::with_capacity(cs.len() * 32);
        for (j, c) in cs.iter().enumerate() {
            assert!(c.weight <= c.positions.len());
            for t in 0..c.weight {
                let m = c.positions[t] as usize;
                assert!(m < 162);
                let negative = ((c.signs >> t) & 1 != 0) ^ (m % 2 != 0);
                shifts.push((j as u16, m as u8, negative));
                let end = (m + 128).min(SLOTS);
                count_diff[negative as usize][m / 2] += 1;
                count_diff[negative as usize][(end - 1) / 2 + 1] -= 1;
                let wrap = (m + 128).saturating_sub(SLOTS);
                if wrap != 0 {
                    count_diff[!negative as usize][0] += 1;
                    count_diff[!negative as usize][(wrap - 1) / 2 + 1] -= 1;
                }
            }
        }
        let mut group = Group {
            first: g * GROUP,
            columns: cs.len(),
            offsets: [0; PAIRS + 1],
            mid: [0; PAIRS],
            terms: Vec::new(),
        };
        let mut negatives = [0i32; PAIRS];
        let mut count = [0i32; 2];
        for q in 0..PAIRS {
            for (c, diff) in count.iter_mut().zip(&count_diff) {
                *c += diff[q];
                debug_assert!(*c >= 0);
            }
            let (pos, neg) = ((count[0] as usize).div_ceil(16) * 16, (count[1] as usize).div_ceil(16) * 16);
            group.mid[q] = group.offsets[q] + pos;
            group.offsets[q + 1] = group.mid[q] + neg;
            negatives[q] = neg as i32;
        }
        // Offset 0 is the padding pair (planes 0 and 1 are zero), not a column's plane.
        group.terms.resize(group.offsets[PAIRS], 0);
        let mut at: [[usize; PAIRS]; 2] = [
            core::array::from_fn(|q| group.offsets[q]),
            core::array::from_fn(|q| group.mid[q]),
        ];
        for (j, m, negative) in shifts {
            let m = m as usize;
            let end = (m + 128).min(SLOTS);
            let column = GUARD + j as usize * STRIDE;
            // The plane of slot `2q` for the run starting at slot `m`: `column + 2q - m`, which
            // is the guard before the run when `2q = m - 1`; the odd slot reads one plane
            // further, the guard after the run when `2q + 1 = end`.
            let at_sign = &mut at[negative as usize];
            for q in m / 2..=(end - 1) / 2 {
                // The range-difference pass reserved one entry for this exact term in q.
                debug_assert!(at_sign[q] < if negative { group.offsets[q + 1] } else { group.mid[q] });
                let plane = (column + 2 * q).wrapping_sub(m);
                unsafe { *group.terms.as_mut_ptr().add(at_sign[q]) = (8 * plane) as u16; }
                at_sign[q] += 1;
            }
            let wrap = (m + 128).saturating_sub(SLOTS);
            if wrap != 0 {
                let at_sign = &mut at[!negative as usize];
                let run = column + (end - m);
                for q in 0..=(wrap - 1) / 2 {
                    debug_assert!(at_sign[q] < if !negative { group.offsets[q + 1] } else { group.mid[q] });
                    let plane = run + 2 * q;
                    unsafe { *group.terms.as_mut_ptr().add(at_sign[q]) = (8 * plane) as u16; }
                    at_sign[q] += 1;
                }
            }
        }
        (group, negatives)
    }

    fn plan<const GROUP: usize>(challenges: &[ShortChallenge], workers: usize) -> (Vec<Group>, [i32; SLOTS]) {
        let jobs = challenges.len().div_ceil(GROUP);
        let group = |g: usize| {
            let first = g * GROUP;
            let last = (first + GROUP).min(challenges.len());
            plan_group::<GROUP>(g, &challenges[first..last])
        };
        // The schedule is independent for each public column group; the pool's parked
        // workers take the groups, and the merge keeps column order.
        let built: Vec<(Group, [i32; PAIRS])> = if jobs < 4 || workers <= 1 {
            (0..jobs).map(group).collect()
        } else {
            crate::par::pool().map(jobs, group)
        };
        let mut groups = Vec::with_capacity(jobs);
        let mut negatives = [0i32; PAIRS];
        for (group, n) in built {
            groups.push(group);
            for (dst, x) in negatives.iter_mut().zip(n) { *dst += x; }
        }
        let bias = core::array::from_fn(|s| negatives[s / 2]);
        (groups, bias)
    }

    #[repr(C, align(64))]
    struct Aligned<T>(T);

    const fn swap_index(mask: usize, value: usize) -> [u8; 64] {
        let mut out = [0; 64];
        let mut p = 0;
        while p < 64 {
            out[p] = (64 * ((p & mask != 0) as usize) + (p & !mask) + value) as u8;
            p += 1;
        }
        out
    }

    static SWAPS: Aligned<[[[u8; 64]; 2]; 4]> = Aligned([
        [swap_index(8, 0), swap_index(8, 8)],
        [swap_index(4, 0), swap_index(4, 4)],
        [swap_index(2, 0), swap_index(2, 2)],
        [swap_index(1, 0), swap_index(1, 1)],
    ]);

    /// Exactly bitmat's byte transpose, specialized to four fixed rounds. Both the vector-slot
    /// bit and the byte-position bit exchanged in round r are 3-r. Constant pair indices let
    /// the compiler keep all sixteen vectors in registers instead of indexing a stack array.
    #[inline(always)]
    unsafe fn transpose_tile(src: *const B128, z: &mut [__m512i; 16]) {
        for i in 0..16 {
            z[i] = _mm512_loadu_si512((src as *const u8).add(64 * i) as *const __m512i);
        }
        macro_rules! pair {
            ($a:literal, $b:literal, $i0:ident, $i1:ident) => {{
                let (a, b) = (z[$a], z[$b]);
                z[$a] = _mm512_permutex2var_epi8(a, $i0, b);
                z[$b] = _mm512_permutex2var_epi8(a, $i1, b);
            }};
        }
        let i0 = _mm512_load_si512(SWAPS.0[0][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[0][1].as_ptr() as *const __m512i);
        pair!(0, 8, i0, i1); pair!(1, 9, i0, i1); pair!(2, 10, i0, i1); pair!(3, 11, i0, i1);
        pair!(4, 12, i0, i1); pair!(5, 13, i0, i1); pair!(6, 14, i0, i1); pair!(7, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[1][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[1][1].as_ptr() as *const __m512i);
        pair!(0, 4, i0, i1); pair!(1, 5, i0, i1); pair!(2, 6, i0, i1); pair!(3, 7, i0, i1);
        pair!(8, 12, i0, i1); pair!(9, 13, i0, i1); pair!(10, 14, i0, i1); pair!(11, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[2][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[2][1].as_ptr() as *const __m512i);
        pair!(0, 2, i0, i1); pair!(1, 3, i0, i1); pair!(4, 6, i0, i1); pair!(5, 7, i0, i1);
        pair!(8, 10, i0, i1); pair!(9, 11, i0, i1); pair!(12, 14, i0, i1); pair!(13, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[3][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[3][1].as_ptr() as *const __m512i);
        pair!(0, 1, i0, i1); pair!(2, 3, i0, i1); pair!(4, 5, i0, i1); pair!(6, 7, i0, i1);
        pair!(8, 9, i0, i1); pair!(10, 11, i0, i1); pair!(12, 13, i0, i1); pair!(14, 15, i0, i1);
    }

    // Byte transpose inside each vector, and its inverse with reversed GFNI matrix rows.
    const PERM: Aligned<[u8; 64]> = Aligned({
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = (8 * (i % 8) + i / 8) as u8;
            i += 1;
        }
        a
    });
    const BACK: Aligned<[u8; 64]> = Aligned({
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = (8 * (7 - i % 8) + i / 8) as u8;
            i += 1;
        }
        a
    });

    /// Lane l corresponds to this source word. GFNI reverses the eight matrix rows; the byte
    /// transpose uses bitmat's public word permutation. No permutation is needed in the hot loop;
    /// [`write_block`] relies on its shape (lane `64 g + 16 k + r` is word `64 g + 4 (r ^ 7) + k`).
    #[cfg_attr(not(test), allow(dead_code))]
    fn word_of(l: usize) -> usize {
        64 * (l / 64) + bitmat::word_at((l % 64 & !7) + 7 - l % 8)
    }

    /// The 128 planes of a block of 512 words to `out`; the guard planes around them are the
    /// caller's.
    ///
    /// The 8 KB the next call slices (`next`, any readable address) are prefetched a tile at a
    /// time: the blocks of a group are 256 KB apart, two fresh pages each, which the hardware
    /// prefetchers only start on after the first misses, and the table comes from DRAM.
    #[inline(always)]
    unsafe fn slice(words: *const B128, out: *mut __m512i, next: *const B128) {
        let zero = _mm512_setzero_si512();
        let mut tiles = [[zero; 16]; 8];
        for t in 0..8 {
            for i in 0..16 {
                _mm_prefetch::<{ _MM_HINT_T0 }>((next as *const i8).add(1024 * t + 64 * i));
            }
            transpose_tile(words.add(64 * t), &mut tiles[t]);
        }
        let perm = _mm512_load_si512(PERM.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(bitmat::GF_IDENT);
        for b in 0..16 {
            let rows = core::array::from_fn(|t| {
                _mm512_permutexvar_epi8(
                    perm,
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, tiles[t][b]),
                )
            });
            let planes = transpose8x8_q(rows);
            for p in 0..8 {
                _mm512_store_si512(out.add(8 * b + p), planes[p]);
            }
        }
    }

    /// Three equally weighted bit planes -> sum and double-weight carry.
    #[inline(always)]
    unsafe fn csa(sum: &mut __m512i, a: __m512i, b: __m512i) -> __m512i {
        // `vpternlogq` overwrites its first operand, so taking both outputs from the three
        // originals costs a register copy per full adder. Instead the sum is formed in place
        // and the carry is read off `(a, sum', b)`: with `sum = sum' ^ a ^ b`, the majority is
        // `a` when `a == b` and `!sum'` otherwise, truth table 0xb2. The carry can then land on
        // `a` or `b`, both dead afterwards.
        *sum = _mm512_ternarylogic_epi64::<0x96>(*sum, a, b);
        _mm512_ternarylogic_epi64::<0xb2>(a, *sum, b)
    }

    /// [`csa`] of `sum` and the complements of `a` and `b`: the sum is the same (the two
    /// complements cancel), and the carry `maj(sum, !a, !b)` is `!a` when `a == b` and the
    /// complement of the new sum otherwise, truth table 0x17 on `(a, sum', b)`.
    #[inline(always)]
    unsafe fn csa_neg(sum: &mut __m512i, a: __m512i, b: __m512i) -> __m512i {
        *sum = _mm512_ternarylogic_epi64::<0x96>(*sum, a, b);
        _mm512_ternarylogic_epi64::<0x17>(a, *sum, b)
    }

    // Isolate the counter's register allocation from the transpose's many vector constants.
    #[inline(never)]
    unsafe fn accumulate(planes: *const __m512i, terms: &[u16], acc: *mut __m512i) {
        let planes = planes as *const u64;
        let mut b0 = _mm512_load_si512(acc);
        let mut b1 = _mm512_load_si512(acc.add(1));
        let mut b2 = _mm512_load_si512(acc.add(2));
        let mut b3 = _mm512_load_si512(acc.add(3));
        let mut b4 = _mm512_load_si512(acc.add(4));
        let mut b5 = _mm512_load_si512(acc.add(5));
        let mut b6 = _mm512_load_si512(acc.add(6));
        let mut b7 = _mm512_load_si512(acc.add(7));
        let mut b8 = _mm512_load_si512(acc.add(8));
        let mut b9 = _mm512_load_si512(acc.add(9));
        let mut b10 = _mm512_load_si512(acc.add(10));
        let mut b11 = _mm512_load_si512(acc.add(11));
        let mut b12 = _mm512_load_si512(acc.add(12));
        let mut b13 = _mm512_load_si512(acc.add(13));
        let mut b14 = _mm512_load_si512(acc.add(14));
        for input in terms.chunks_exact(64) {
            let t = input.as_ptr();
            macro_rules! two {
                ($off:expr) => {{
                    let a = _mm512_load_si512(planes.add(*t.add($off) as usize) as *const __m512i);
                    let b = _mm512_load_si512(planes.add(*t.add($off + 1) as usize) as *const __m512i);
                    csa(&mut b0, a, b)
                }};
            }
            macro_rules! four {
                ($off:expr) => {{ let a = two!($off); let b = two!($off + 2); csa(&mut b1, a, b) }};
            }
            macro_rules! eight {
                ($off:expr) => {{ let a = four!($off); let b = four!($off + 4); csa(&mut b2, a, b) }};
            }
            macro_rules! sixteen {
                ($off:expr) => {{ let a = eight!($off); let b = eight!($off + 8); csa(&mut b3, a, b) }};
            }
            macro_rules! thirty_two {
                ($off:expr) => {{ let a = sixteen!($off); let b = sixteen!($off + 16); csa(&mut b4, a, b) }};
            }
            let a = thirty_two!(0);
            let b = thirty_two!(32);
            let mut carry = csa(&mut b5, a, b);
            macro_rules! ripple {
                ($bit:ident) => {{
                    let old = $bit;
                    $bit = _mm512_xor_si512(old, carry);
                    carry = _mm512_and_si512(old, carry);
                }};
            }
            ripple!(b6); ripple!(b7); ripple!(b8); ripple!(b9);
            ripple!(b10); ripple!(b11); ripple!(b12); ripple!(b13);
            b14 = _mm512_xor_si512(b14, carry);
        }
        _mm512_store_si512(acc, b0);
        _mm512_store_si512(acc.add(1), b1);
        _mm512_store_si512(acc.add(2), b2);
        _mm512_store_si512(acc.add(3), b3);
        _mm512_store_si512(acc.add(4), b4);
        _mm512_store_si512(acc.add(5), b5);
        _mm512_store_si512(acc.add(6), b6);
        _mm512_store_si512(acc.add(7), b7);
        _mm512_store_si512(acc.add(8), b8);
        _mm512_store_si512(acc.add(9), b9);
        _mm512_store_si512(acc.add(10), b10);
        _mm512_store_si512(acc.add(11), b11);
        _mm512_store_si512(acc.add(12), b12);
        _mm512_store_si512(acc.add(13), b13);
        _mm512_store_si512(acc.add(14), b14);
    }

    /// Two consecutive slots at once: every scheduled offset is the plane of the even slot, and
    /// the odd slot's plane is the next one, so one scalar offset serves two vector loads. Each
    /// slot keeps the six low counter bits of a 64-input tree in registers, plus a two-bit
    /// counter of the trees' weight-64 carries; every third chunk that counter is added into
    /// the nine upper counter bits in memory. The `pos` entries add their planes, the `neg`
    /// entries their complements (leaf full adders by [`csa_neg`]).
    #[inline(never)]
    unsafe fn accumulate_pair(planes: *const __m512i, pos: &[u16], neg: &[u16], acc: *mut __m512i) {
        let planes = planes as *const u64;
        let (even, odd) = (acc, acc.add(BITS));
        let mut s0: [__m512i; 6] = core::array::from_fn(|b| _mm512_load_si512(even.add(b)));
        let mut s1: [__m512i; 6] = core::array::from_fn(|b| _mm512_load_si512(odd.add(b)));
        // The weight-64 carries of up to three chunks, in registers at weights 64 and 128;
        // every third chunk they are added into the nine upper counter bits in memory.
        let zero = _mm512_setzero_si512();
        let (mut h0, mut h1) = ([zero; 2], [zero; 2]);
        let mut pending = 0usize;
        macro_rules! flush {
            ($slot:ident, $h:ident) => {{
                let p = $slot.add(6);
                let m6 = _mm512_load_si512(p);
                _mm512_store_si512(p, _mm512_xor_si512(m6, $h[0]));
                let mut carry = _mm512_and_si512(m6, $h[0]);
                let p = $slot.add(7);
                let m7 = _mm512_load_si512(p);
                let x = _mm512_ternarylogic_epi64::<0x96>(m7, $h[1], carry);
                _mm512_store_si512(p, x);
                carry = _mm512_ternarylogic_epi64::<0xb2>($h[1], x, carry);
                for b in 8..BITS - 1 {
                    let p = $slot.add(b);
                    let old = _mm512_load_si512(p);
                    _mm512_store_si512(p, _mm512_xor_si512(old, carry));
                    carry = _mm512_and_si512(old, carry);
                }
                let p = $slot.add(BITS - 1);
                _mm512_store_si512(p, _mm512_xor_si512(_mm512_load_si512(p), carry));
                $h = [zero; 2];
            }};
        }
        let mut t: *const u16 = pos.as_ptr();
        macro_rules! two {
            ($off:expr, $leaf:ident) => {{
                let pa = planes.add(*t.add($off) as usize);
                let pb = planes.add(*t.add($off + 1) as usize);
                let a0 = _mm512_load_si512(pa as *const __m512i);
                let b0 = _mm512_load_si512(pb as *const __m512i);
                let a1 = _mm512_load_si512(pa.add(8) as *const __m512i);
                let b1 = _mm512_load_si512(pb.add(8) as *const __m512i);
                ($leaf(&mut s0[0], a0, b0), $leaf(&mut s1[0], a1, b1))
            }};
        }
        macro_rules! level {
            ($bit:literal, $lower:ident, $off:expr, $leaf:ident) => {{
                let (x0, x1) = $lower!($off, $leaf);
                let (y0, y1) = $lower!($off + (1 << $bit), $leaf);
                (csa(&mut s0[$bit], x0, y0), csa(&mut s1[$bit], x1, y1))
            }};
        }
        macro_rules! four { ($off:expr, $leaf:ident) => { level!(1, two, $off, $leaf) }; }
        macro_rules! eight { ($off:expr, $leaf:ident) => { level!(2, four, $off, $leaf) }; }
        macro_rules! sixteen { ($off:expr, $leaf:ident) => { level!(3, eight, $off, $leaf) }; }
        macro_rules! thirty_two { ($off:expr, $leaf:ident) => { level!(4, sixteen, $off, $leaf) }; }
        // Two-bit register counters of the weight-64 carries: three fit before a flush.
        macro_rules! carry64 {
            ($c0:expr, $c1:expr) => {{
                let (c0, c1) = ($c0, $c1);
                let up = _mm512_and_si512(h0[0], c0);
                h0[0] = _mm512_xor_si512(h0[0], c0);
                h0[1] = _mm512_xor_si512(h0[1], up);
                let up = _mm512_and_si512(h1[0], c1);
                h1[0] = _mm512_xor_si512(h1[0], c1);
                h1[1] = _mm512_xor_si512(h1[1], up);
                pending += 1;
                if pending == 3 {
                    flush!(even, h0);
                    flush!(odd, h1);
                    pending = 0;
                }
            }};
        }
        // A carry of weight `2^$bit` from a partial tree, through half adders at that level and
        // above: what leaves level 5 is a weight-64 carry like a whole chunk's.
        macro_rules! half_adders {
            ($c0:expr, $c1:expr, $($bit:literal),+) => {{
                let (mut c0, mut c1) = ($c0, $c1);
                $(
                    let (u0, u1) = (_mm512_and_si512(s0[$bit], c0), _mm512_and_si512(s1[$bit], c1));
                    s0[$bit] = _mm512_xor_si512(s0[$bit], c0);
                    s1[$bit] = _mm512_xor_si512(s1[$bit], c1);
                    (c0, c1) = (u0, u1);
                )+
                carry64!(c0, c1);
            }};
        }
        macro_rules! entries {
            ($terms:expr, $leaf:ident) => {{
                let terms: &[u16] = $terms;
                for input in terms.chunks_exact(64) {
                    t = input.as_ptr();
                    let (c0, c1) = level!(5, thirty_two, 0, $leaf);
                    carry64!(c0, c1);
                }
                // The schedule pads each sign of a slot pair to a multiple of 16: a last 16, 32
                // or 48 inputs go through a 32-input and/or a 16-input tree.
                let rest = terms.chunks_exact(64).remainder();
                debug_assert!(rest.len() % 16 == 0);
                t = rest.as_ptr();
                if rest.len() >= 32 {
                    let (c0, c1) = thirty_two!(0, $leaf);
                    half_adders!(c0, c1, 5);
                }
                if rest.len() % 32 == 16 {
                    let (c0, c1) = sixteen!(rest.len() - 16, $leaf);
                    half_adders!(c0, c1, 4, 5);
                }
            }};
        }
        entries!(pos, csa);
        entries!(neg, csa_neg);
        if pending != 0 {
            flush!(even, h0);
            flush!(odd, h1);
        }
        for b in 0..6 {
            _mm512_store_si512(even.add(b), s0[b]);
            _mm512_store_si512(odd.add(b), s1[b]);
        }
    }

    /// Inverse bit transpose to 512 ordinary 16-bit lane counts.
    #[inline(always)]
    unsafe fn unslice(acc: *const __m512i, out: &mut [i16; LANES]) {
        let zero = _mm512_setzero_si512();
        let lo = transpose8x8_q(core::array::from_fn(|b| _mm512_load_si512(acc.add(b))));
        let hi = transpose8x8_q(core::array::from_fn(|b| {
            if b == 7 { zero } else { _mm512_load_si512(acc.add(8 + b)) }
        }));
        let perm = _mm512_load_si512(BACK.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(bitmat::GF_IDENT);
        let first = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let second = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        for t in 0..8 {
            let l = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, lo[t]));
            let h = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, hi[t]));
            let a = _mm512_unpacklo_epi8(l, h);
            let b = _mm512_unpackhi_epi8(l, h);
            let dst = out.as_mut_ptr().add(64 * t) as *mut __m512i;
            _mm512_storeu_si512(dst, _mm512_permutex2var_epi64(a, first, b));
            _mm512_storeu_si512(dst.add(1), _mm512_permutex2var_epi64(a, second, b));
        }
    }

    /// Coefficient slots of `R_162`: slot `s` of the output takes counter `s` minus counter
    /// `162 + s` below 81 and counter `s` plus counter `s + 81` from 81 on.
    const OUT_SLOTS: usize = 162;

    /// `vpermt2w` indices interleaving a 64-lane group's four components: `[word_of]` puts
    /// component `k` of element `16 g + (r ^ 7)` on lane `64 g + 16 k + r`, so half `h` of the
    /// result holds, in qword `j`, the four components of element `16 g + 8 h + 7 - j`.
    static INTERLEAVE: Aligned<[[u16; 32]; 2]> = Aligned({
        let mut a = [[0u16; 32]; 2];
        let mut h = 0;
        while h < 2 {
            let mut p = 0;
            while p < 32 {
                a[h][p] = (16 * (p % 4) + 8 * h + p / 4) as u16;
                p += 1;
            }
            h += 1;
        }
        a
    });

    /// A block's 243 counters, reduced modulo `Y^162 - Y^81 + 1` and with the bias removed, as
    /// its 128 ring elements at `output` (uninitialised memory, every field written).
    ///
    /// Per output slot and 64-lane group the four components interleave into one qword per
    /// element, and eight consecutive slots transpose into one 64-byte store per element. Each
    /// counter minus its bias is exact in an i16 lane (both are at most 32767); the combination
    /// is formed both wrapping and saturating, and any difference is the overflow the scalar
    /// reference reports.
    ///
    /// # Safety
    /// `acc` holds the 243 slots' counters; `output` addresses 128 writable elements.
    #[inline(never)]
    unsafe fn write_block(
        acc: *const __m512i,
        counts: &mut [[i16; LANES]],
        bias: &[i32; SLOTS],
        output: *mut RingElement,
    ) {
        debug_assert!(counts.len() >= SLOTS);
        for s in 0..SLOTS {
            unslice(acc.add(s * BITS), &mut counts[s]);
        }
        write_counts(counts, bias, output);
    }

    /// The output stage of [`write_block`] from counts already unsliced (and possibly summed
    /// over several parts of the column groups).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
    unsafe fn write_counts(
        counts: &[[i16; LANES]],
        bias: &[i32; SLOTS],
        output: *mut RingElement,
    ) {
        for e in 0..LANES / 4 {
            core::ptr::addr_of_mut!((*output.add(e)).representation).write(Representation::Coefficients);
        }
        let idx = [
            _mm512_load_si512(INTERLEAVE.0[0].as_ptr() as *const __m512i),
            _mm512_load_si512(INTERLEAVE.0[1].as_ptr() as *const __m512i),
        ];
        let mut overflow = 0u32;
        let zero = _mm512_setzero_si512();
        for s0 in (0..OUT_SLOTS).step_by(8) {
            let n = (OUT_SLOTS - s0).min(8);
            let mask = ((1u32 << n) - 1) as __mmask8;
            for g in 0..LANES / 64 {
                let mut rows = [[zero; 8]; 2];
                for ds in 0..8 {
                    // Past the last slot the window repeats it; those qwords are not stored.
                    let s = (s0 + ds).min(OUT_SLOTS - 1);
                    let (t, below) = if s < 81 { (162 + s, true) } else { (s + 81, false) };
                    let (bs, bt) = (_mm512_set1_epi16(bias[s] as i16), _mm512_set1_epi16(bias[t] as i16));
                    let mut v = [zero; 2];
                    for half in 0..2 {
                        let at = 64 * g + 32 * half;
                        let x = _mm512_sub_epi16(_mm512_loadu_si512(counts[s].as_ptr().add(at) as *const __m512i), bs);
                        let c = _mm512_loadu_si512(counts[t].as_ptr().add(at) as *const __m512i);
                        let y = if below { _mm512_sub_epi16(bt, c) } else { _mm512_sub_epi16(c, bt) };
                        let wrap = _mm512_add_epi16(x, y);
                        overflow |= _mm512_cmpneq_epi16_mask(wrap, _mm512_adds_epi16(x, y));
                        v[half] = wrap;
                    }
                    rows[0][ds] = _mm512_permutex2var_epi16(v[0], idx[0], v[1]);
                    rows[1][ds] = _mm512_permutex2var_epi16(v[0], idx[1], v[1]);
                }
                for h in 0..2 {
                    let t = transpose8x8_q(rows[h]);
                    for j in 0..8 {
                        let e = 16 * g + 8 * h + 7 - j;
                        let dst = (core::ptr::addr_of_mut!((*output.add(e)).v) as *mut i16).add(4 * s0);
                        if n == 8 {
                            _mm512_storeu_si512(dst as *mut __m512i, t[j]);
                        } else {
                            _mm512_mask_storeu_epi64(dst as *mut i64, mask, t[j]);
                        }
                    }
                }
            }
        }
        assert!(overflow == 0, "integer fold overflow");
    }

    pub(super) fn fold_words(
        words: &[B128],
        challenges: &[ShortChallenge],
        column_words: usize,
        workers: usize,
    ) -> Vec<RingElement> {
        fold_config::<COL_GROUP>(words, challenges, column_words, workers)
    }

    fn fold_config<const GROUP: usize>(
        words: &[B128],
        challenges: &[ShortChallenge],
        column_words: usize,
        workers: usize,
    ) -> Vec<RingElement> {
        assert!(GROUP > 0 && (GUARD + GROUP * STRIDE) * 8 <= u16::MAX as usize);
        #[cfg(test)]
        let profile = std::env::var_os("LABINIUS_BITFOLD_PROFILE").is_some();
        #[cfg(test)]
        let plan_start = std::time::Instant::now();
        let (groups, bias) = plan::<GROUP>(challenges, workers);
        #[cfg(test)]
        if profile {
            eprintln!("bitfold plan_us={}", plan_start.elapsed().as_micros());
        }
        let blocks = column_words / LANES;
        let len = column_words / 4;
        // Counters and bias are at most the total weight (checked by the caller), which is
        // what lets the output stage subtract them in i16 lanes.
        assert!(bias.iter().all(|&b| (0..=i16::MAX as i32).contains(&b)));
        // Tasks are (block, part of the column groups), handed out dynamically so a busy or
        // shared core does fewer of them; the second part of a block to finish adds its counts
        // to the first's (counts are additive and stay within the total weight) and writes.
        // Parts of the column groups a block is split over. A part is one task's
        // sweep over the table: the fewer columns a task touches at once, the
        // fewer distinct 4 KB pages it has in flight, and the second-level TLB
        // has 2048 entries. More parts cost a merge of the parts' counters.
        const PARTS: usize = 4;
        let parts = match std::env::var("LABINIUS_FOLD_PARTS") {
            Ok(v) => v.parse::<usize>().unwrap_or(PARTS),
            Err(_) => PARTS,
        }
        .max(1)
        .min(groups.len().max(1));
        let cuts: Vec<usize> = (0..=parts).map(|i| groups.len() * i / parts).collect();
        let tasks = blocks * parts;
        let pending: Vec<std::sync::Mutex<Vec<Vec<[i16; LANES]>>>> =
            (0..blocks).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut out: Vec<RingElement> = Vec::with_capacity(len);
        let out_addr = out.as_mut_ptr() as usize;
        let (groups, bias, cuts, pending, next) = (&groups, &bias, &cuts, &pending, &next);
        crate::par::pool().run(workers.clamp(1, tasks), |_| unsafe {
            let zero = _mm512_setzero_si512();
            // Zero everywhere the columns do not write: the guards and the padding pair.
            let mut planes = vec![zero; GUARD + GROUP * STRIDE];
            let mut acc = vec![zero; 2 * PAIRS * BITS];
            // One counts buffer per worker, for the case where a task owns a whole
            // block: `unslice` writes every lane, so the buffer never has to be
            // cleared, and the task writes straight out of it.
            let mut counts = if parts == 1 { vec![[0i16; LANES]; SLOTS] } else { Vec::new() };
            loop {
                let task = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if task >= tasks {
                    break;
                }
                let (block, part) = (task / parts, task % parts);
                acc.fill(zero);
                let range = cuts[part]..cuts[part + 1];
                for gi in range.clone() {
                    let group = &groups[gi];
                    for j in 0..group.columns {
                        let src = words.as_ptr().add((group.first + j) * column_words + block * LANES);
                        // What the next call slices: this group's next column, else the next
                        // group's first, else this block's start (harmless).
                        let next_src = if j + 1 < group.columns {
                            src.add(column_words)
                        } else if gi + 1 < range.end {
                            words.as_ptr().add(groups[gi + 1].first * column_words + block * LANES)
                        } else {
                            src
                        };
                        slice(src, planes.as_mut_ptr().add(GUARD + j * STRIDE), next_src);
                    }
                    for q in 0..PAIRS {
                        accumulate_pair(
                            planes.as_ptr(),
                            &group.terms[group.offsets[q]..group.mid[q]],
                            &group.terms[group.mid[q]..group.offsets[q + 1]],
                            acc.as_mut_ptr().add(2 * q * BITS),
                        );
                    }
                }
                if parts == 1 {
                    // The task owns the whole block: it reads the counters into
                    // its own buffer and writes them straight out.
                    for s in 0..SLOTS {
                        unslice(acc.as_ptr().add(s * BITS), &mut counts[s]);
                    }
                    let output = (out_addr as *mut RingElement).add(block * (LANES / 4));
                    write_counts(&counts, bias, output);
                    continue;
                }
                let mut own = vec![[0i16; LANES]; SLOTS];
                for s in 0..SLOTS {
                    unslice(acc.as_ptr().add(s * BITS), &mut own[s]);
                }
                let mut done = {
                    let mut v = pending[block].lock().unwrap();
                    v.push(own);
                    if v.len() < parts {
                        continue;
                    }
                    core::mem::take(&mut *v)
                };
                let mut total = done.pop().unwrap();
                for other in &done {
                    for (t, o) in total.iter_mut().zip(other) {
                        for (a, b) in t.iter_mut().zip(o) {
                            *a += *b;
                        }
                    }
                }
                let output = (out_addr as *mut RingElement).add(block * (LANES / 4));
                write_counts(&total, bias, output);
            }
        });
        // Every task wrote its whole range.
        unsafe { out.set_len(len) };
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::challenge::{sample_short_challenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};

        fn table(n: usize) -> Vec<B128> {
            let mut h = blake3::Hasher::new_keyed(&[81; 32]);
            h.update(b"labinius-fast/binary-table/v1");
            let mut out = vec![B128(0); n];
            let bytes = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, 16 * n) };
            h.finalize_xof().fill(bytes);
            out
        }

        fn challenges(n: usize) -> Vec<ShortChallenge> {
            let mut t = Transcript::new(b"labinius/test/bitfold");
            (0..n).map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0).collect()
        }

        #[test]
        fn bitfold_transpose_and_counter() {
            let words = table(LANES);
            unsafe {
                let zero = _mm512_setzero_si512();
                let mut planes = vec![zero; STRIDE];
                slice(words.as_ptr(), planes.as_mut_ptr(), words.as_ptr());
                for p in 0..128 {
                    let bits: [u64; 8] = core::mem::transmute(planes[p]);
                    for lane in 0..LANES {
                        let expected = (words[word_of(lane)].0 >> p) as u64 & 1;
                        assert_eq!(bits[lane / 64] >> (lane % 64) & 1, expected, "plane {p}, lane {lane}");
                    }
                }
                assert!((128..STRIDE).all(|p| core::mem::transmute::<_, [u64; 8]>(planes[p]) == [0u64; 8]));
                let mut acc = vec![zero; BITS];
                let terms: Vec<u16> = (0..128u16).cycle().take(32704).collect();
                let offsets: Vec<u16> = terms.iter().map(|p| 8 * p).collect();
                accumulate(planes.as_ptr(), &offsets, acc.as_mut_ptr());
                let mut counts = [0i16; LANES];
                unslice(acc.as_ptr(), &mut counts);
                for lane in 0..LANES {
                    let expected: usize = terms.iter().map(|&p| (words[word_of(lane)].0 >> p & 1) as usize).sum();
                    assert_eq!(counts[lane] as usize, expected, "lane {lane}");
                }
                // Exercise the very top of the declared counter range, including zero padding.
                acc.fill(zero);
                planes[0] = zero;
                planes[1] = _mm512_set1_epi32(-1);
                let mut terms = vec![8; 32768];
                terms[32767] = 0;
                accumulate(planes.as_ptr(), &terms, acc.as_mut_ptr());
                unslice(acc.as_ptr(), &mut counts);
                assert_eq!(counts, [32767; LANES]);
            }
        }

        /// The slot-pair kernel on schedules of every length class the plan pads to (whole
        /// chunks plus a 16-, 32- or 48-input tail), in pieces as a block's groups call it, and
        /// up to the top of the counter range.
        #[test]
        fn bitfold_pair_kernel_partial_chunks() {
            let mut x = 0x0123_4567_89AB_CDEFu64;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            unsafe {
                let zero = _mm512_setzero_si512();
                let mut planes = vec![zero; 66];
                for p in planes.iter_mut().skip(2) {
                    let w: [u64; 8] = core::array::from_fn(|_| next() & next());
                    *p = core::mem::transmute(w);
                }
                let bits: Vec<[u64; 8]> = planes.iter().map(|p| core::mem::transmute(*p)).collect();
                let bit = |p: usize, l: usize| (bits[p][l / 64] >> (l % 64) & 1) as usize;
                // Each call takes `n` positive entries and the next length's worth of negative
                // ones (`lens` cycled), so both signs meet every length class, alone and mixed.
                for lens in [vec![16usize], vec![32], vec![48], vec![80, 112, 144], vec![64 * 7 + 48; 9], vec![16; 40], vec![16000, 752, 15232], vec![0, 64]] {
                    let mut acc = vec![zero; 2 * BITS];
                    let mut want = [[0usize; LANES]; 2];
                    for (i, &n) in lens.iter().enumerate() {
                        let m = lens[(i + 1) % lens.len()];
                        let mut terms = |n: usize| -> Vec<u16> {
                            (0..n).map(|i| {
                                // a sprinkling of padding entries (offset 0: two zero planes)
                                if i % 13 == 5 { 0 } else { (8 * (2 + next() % 63)) as u16 }
                            }).collect()
                        };
                        let (pos, neg) = (terms(n), terms(m));
                        for (list, complement) in [(&pos, 0), (&neg, 1)] {
                            for &o in list {
                                let p = o as usize / 8;
                                for l in 0..LANES {
                                    want[0][l] += bit(p, l) ^ complement;
                                    want[1][l] += bit(p + 1, l) ^ complement;
                                }
                            }
                        }
                        accumulate_pair(planes.as_ptr(), &pos, &neg, acc.as_mut_ptr());
                    }
                    for slot in 0..2 {
                        let mut got = [0i16; LANES];
                        unslice(acc.as_ptr().add(slot * BITS), &mut got);
                        for l in 0..LANES {
                            assert_eq!(got[l] as usize, want[slot][l] & 0x7fff, "lens {lens:?} slot {slot} lane {l}");
                        }
                    }
                }
            }
        }

        /// The vector output stage against the scalar reduction it replaced, on random
        /// counters (the full 15-bit range) and biases chosen so that every value fits, and its
        /// overflow report.
        #[test]
        fn bitfold_write_block_matches_scalar() {
            for l in 0..LANES {
                let (g, k, r) = (l / 64, l % 64 / 16, l % 16);
                assert_eq!(word_of(l), 64 * g + 4 * (r ^ 7) + k, "lane {l}");
            }
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            unsafe {
                let zero = _mm512_setzero_si512();
                for round in 0..3 {
                    let mut acc = vec![zero; SLOTS * BITS + BITS];
                    for v in acc.iter_mut() {
                        let w: [u64; 8] = core::array::from_fn(|_| next());
                        *v = core::mem::transmute(w);
                    }
                    let mut counts = vec![[0i16; LANES]; SLOTS];
                    for s in 0..SLOTS {
                        unslice(acc.as_ptr().add(s * BITS), &mut counts[s]);
                    }
                    // Slots s and their partner t: bias such that the reference fits i16.
                    let mut bias = [0i32; SLOTS];
                    for s in 0..SLOTS {
                        bias[s] = (next() % 32768) as i32;
                    }
                    if round > 0 {
                        for s in 0..81 {
                            // x - z = (a - b_s) - (c - b_t): keep it in range by matching biases
                            // to the counters' means.
                            bias[s] = 16384;
                            bias[162 + s] = 16384;
                            bias[81 + s] = 16384 + (next() % 8) as i32;
                        }
                    }
                    let fits = (0..81).all(|i| {
                        (0..LANES).all(|l| {
                            let (a, b, c) = (counts[i][l] as i32, counts[81 + i][l] as i32, counts[162 + i][l] as i32);
                            let lo = (a - bias[i]) - (c - bias[162 + i]);
                            let hi = (b - bias[81 + i]) + (c - bias[162 + i]);
                            i16::try_from(lo).is_ok() && i16::try_from(hi).is_ok()
                        })
                    });
                    let mut want = vec![RingElement::zero(Representation::Coefficients); LANES / 4];
                    if fits {
                        for i in 0..81 {
                            for l in 0..LANES {
                                let w = word_of(l);
                                let (a, b, c) = (counts[i][l] as i32, counts[81 + i][l] as i32, counts[162 + i][l] as i32);
                                want[w / 4].v[4 * i + w % 4] = ((a - bias[i]) - (c - bias[162 + i])) as i16;
                                want[w / 4].v[4 * (81 + i) + w % 4] = ((b - bias[81 + i]) + (c - bias[162 + i])) as i16;
                            }
                        }
                    }
                    let mut got: Vec<RingElement> = Vec::with_capacity(LANES / 4);
                    let mut scratch = vec![[0i16; LANES]; SLOTS];
                    let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
                        write_block(acc.as_ptr(), &mut scratch, &bias, got.as_mut_ptr());
                    }));
                    assert_eq!(result.is_ok(), fits, "round {round}: overflow report");
                    if fits {
                        got.set_len(LANES / 4);
                        assert_eq!(got, want, "round {round}");
                    }
                }
            }
        }

        // Independent schoolbook reduction, including the high terms requiring two reductions.
        fn scalar(words: &[B128], cs: &[ShortChallenge], column_words: usize) -> Vec<RingElement> {
            (0..column_words / 4).map(|r| {
                let mut raw = [0i32; 4 * (128 + 162)];
                for (j, c) in cs.iter().enumerate() {
                    for t in 0..c.weight {
                        let m = c.positions[t] as usize;
                        let sign = if ((c.signs >> t) & 1 != 0) ^ (m % 2 != 0) { -1 } else { 1 };
                        for k in 0..4 {
                            let w = words[j * column_words + 4 * r + k].0;
                            for p in 0..128 {
                                raw[4 * (p + m) + k] += sign * ((w >> p) & 1) as i32;
                            }
                        }
                    }
                }
                for d in (648..raw.len()).rev() {
                    let value = raw[d];
                    raw[d - 324] += value;
                    raw[d - 648] -= value;
                }
                let mut out = RingElement::zero(Representation::Coefficients);
                for (v, x) in out.v.iter_mut().zip(raw) { *v = i16::try_from(x).unwrap(); }
                out
            }).collect()
        }

        #[test]
        fn bitfold_matches_schoolbook_and_ntt() {
            let columns = (COL_GROUP + 2).next_multiple_of(2);
            let words = table(columns * LANES);
            let cs = challenges(columns);
            let exact = fold_words(&words, &cs, LANES, 2);
            assert_eq!(exact, scalar(&words, &cs, LANES));
            for q in [3889, 2917, 4861, 9721, 12637, 17497, 19441] {
                assert_eq!(exact, super::super::fold_witness_streaming(&words, columns, &cs, 4, q, 2), "q {q}");
            }
            // All degrees, both signs, all four X components, and zero/all-one input patterns.
            for sign in [0, 1] {
                let cs: Vec<_> = (0..162).map(|m| {
                    let mut c = ShortChallenge::zero();
                    c.positions[0] = m;
                    c.weight = 1;
                    c.signs = sign;
                    c
                }).collect();
                let words: Vec<_> = (0..cs.len() * LANES).map(|i| B128(if i % 7 == 0 { 0 } else { u128::MAX })).collect();
                assert_eq!(fold_words(&words, &cs, LANES, 2), scalar(&words, &cs, LANES));
            }
        }

        #[test]
        #[ignore = "full 2^24-word timing; run explicitly on the box"]
        fn bitfold_bench_full() {
            let words = table(1 << 24);
            let cs = challenges(1024);
            let workers = crate::par::threads();
            let mut first: Option<Vec<RingElement>> = None;
            let mut times = Vec::new();
            for _ in 0..7 {
                let start = std::time::Instant::now();
                let v = fold_words(&words, &cs, 1 << 14, workers);
                times.push(start.elapsed().as_micros());
                if let Some(f) = &first { assert_eq!(&v, f); } else { first = Some(v); }
            }
            eprintln!("bitfold bench workers={} us={:?}", workers, times);
        }

        #[test]
        #[ignore = "full 2^24-word challenge instance; run explicitly on the box"]
        fn bitfold_full_instance_matches_streaming() {
            let words = table(1 << 24);
            let cs = challenges(1024);
            let mut exact = None;
            for (group, workers) in [(16, 4), (31, 4), (31, 2), (16, 2)] {
                let start = std::time::Instant::now();
                let v = match group {
                    16 => fold_config::<16>(&words, &cs, 1 << 14, workers),
                    31 => fold_config::<31>(&words, &cs, 1 << 14, workers),
                    _ => unreachable!(),
                };
                eprintln!("bitfold group={} workers={} exact_us={}", group, workers, start.elapsed().as_micros());
                if let Some(previous) = &exact {
                    assert_eq!(&v, previous);
                } else {
                    exact = Some(v);
                }
            }
            let start = std::time::Instant::now();
            let old = super::super::fold_witness_streaming(&words, 1024, &cs, 128, 3889, 4);
            eprintln!("bitfold ntt_us={}", start.elapsed().as_micros());
            assert_eq!(exact.unwrap(), old);
        }
    }

}
