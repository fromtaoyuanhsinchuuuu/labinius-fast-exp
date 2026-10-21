//! The binary shadow of the scheme: everything the commitment and the fold leave to the field
//! `F = F162 = GF(2)[x]/(x^162 + x^81 + 1)`.
//!
//! # The algebra
//!
//! The witness is the `wdim x r` matrix `W` over `F` held as the flat `&[F162]` the commitment
//! takes: entry `(i, j)` is `witness[i + wdim * j]`, so a column is one chunk of the commitment.
//! `F` is exactly `R_162 mod 2` under the crate's plain lift — the coefficients of an `R_162`
//! element reduced mod 2 are the bits of an `F162`, and a binary challenge is its own reduction —
//! so the whole fold has a shadow over `F`.
//!
//! Both sides of that shadow are multilinear extensions in `nu = log2(wdim) + log2(r)` variables,
//! split as `p0` (the row index `i`, low variables) and `p1` (the column index `j`, high ones),
//! with `eq(p, b) = prod_k (p_k if b_k = 1 else 1 + p_k)`. Writing `B = eq(p0, .)` for the row
//! vector of the left variables, the statement, the prover's message and the two checks are
//!
//! ```text
//!     t   = sum_{i,j} eq(p1, j) eq(p0, i) W[i + wdim j]    the claimed evaluation
//!     u   = B W          (r entries of F)                  the row evaluation, sent by the prover
//!     t   = u^T eq(p1)                                     the verifier's claim check
//!     v   = W c          (over R_648, the fold)            c_j the challenges
//!     B v = u^T c        (over F)                          the binary check
//! ```
//!
//! The last line is the only new identity: `B (W c) = (B W) c` mod 2, linearity of the row
//! evaluation against the challenge vector. Its left side is read off the folded witness `v` the
//! fold already produced (component `k` of packed element `m` is the `F162` at index `4m + k`,
//! and a coefficient's parity is that element's bit), its right side is the `r`-term inner
//! product of `u` against the challenges reduced mod 2
//! ([`ShortChallenge::to_f162`](crate::challenge::ShortChallenge::to_f162)).
//!
//! # How it is computed
//!
//! Every step is one dot product over `F`. The kernels are [`fields::f162`](crate::fields::f162)' word-sliced AVX-512
//! ones — limb `k` of 8 consecutive elements in one `zmm`, `mac_soa8` accumulating the 12
//! unreduced `clmul` products of a block and a single `reduce_soa8` at the end of the whole
//! product, exactly as that crate's sumcheck round does. One operand (the `eq` table, or the
//! challenges) is word-sliced once up front; the other is a raw `&[F162]` run — the witness never
//! leaves its own layout, it is transposed 8 elements at a time by three `vpermi2q`/`vpermq`
//! pairs inside the loop ([`load_soa8`]).
use crate::challenge::ShortChallenge;
use crate::fields::f162 as bf;
use crate::fields::scalar::F162;
use crate::fields::sumcheck::Poly;
use crate::params::N;
use crate::ring::RingElement;
use core::arch::x86_64::*;

/// `eq(ps, b) = prod_k (ps_k if bit k of b is 1 else 1 + ps_k)`, all `2^ps.len()` of them, by
/// doubling: variable `k` is bit `k` of the index.
pub fn eq_table(ps: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &p in ps {
        let n = t.len();
        t.resize(2 * n, F162::ZERO);
        for b in 0..n {
            // x (1 + p) = x + x p: one product per entry.
            let x = t[b];
            let y = x * p;
            t[b + n] = y;
            t[b] = x + y;
        }
    }
    t
}

// =============================================================================================
// the dot product over F
// =============================================================================================

/// The `vpermi2q` / `vpermq` index vectors of [`load_soa8`]: 24 consecutive `u64` (8 `F162`) to
/// limb 0, 1, 2 of those 8 elements. Limb `k` of element `p` is `u64` number `3p + k`, which for
/// `k = 0` is `0, 3, 6, 9, 12, 15` inside the first two vectors and `2, 5` inside the third, and
/// so on.
struct Idx {
    a: [__m512i; 3],
    b: [__m512i; 3],
    m: [__mmask8; 3],
}

#[inline(always)]
unsafe fn idx() -> Idx {
    Idx {
        a: [
            _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0),
            _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0),
            _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0),
        ],
        b: [
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 2, 5),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 3, 6),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 4, 7),
        ],
        m: [0b1100_0000, 0b1110_0000, 0b1110_0000],
    }
}

/// 8 consecutive `F162` (192 contiguous bytes) read as the word-sliced `[limb0, limb1, limb2]`
/// the [`fields::f162`](crate::fields::f162) kernels take.
///
/// # Safety
/// `p` addresses 24 readable `u64`.
#[inline(always)]
unsafe fn load_soa8(ix: &Idx, p: *const u64) -> [__m512i; 3] {
    let v0 = _mm512_loadu_si512(p as *const __m512i);
    let v1 = _mm512_loadu_si512(p.add(8) as *const __m512i);
    let v2 = _mm512_loadu_si512(p.add(16) as *const __m512i);
    core::array::from_fn(|k| {
        _mm512_mask_permutexvar_epi64(
            _mm512_permutex2var_epi64(v0, ix.a[k], v1),
            ix.m[k],
            ix.b[k],
            v2,
        )
    })
}

/// The 8 lanes of a finished accumulator summed into one element.
#[inline(always)]
unsafe fn horizontal(acc: [__m512i; 12]) -> F162 {
    let r = bf::reduce_soa8(acc);
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

/// `sum_k a_k b_k` over `F`, `a` word-sliced and `b` a raw run of `n` `F162`, with the reduction
/// deferred to the end: 12 unreduced `clmul` products per 8 elements, one [`bf::reduce_soa8`].
///
/// # Safety
/// `b` addresses `n` `F162` and `a` holds at least `n` elements.
unsafe fn dot(a: &Poly, b: *const F162, n: usize) -> F162 {
    debug_assert!(a.n >= n);
    let ix = idx();
    let (mut acc, blocks) = ([_mm512_setzero_si512(); 12], n / 8);
    for t in 0..blocks {
        let x = [
            _mm512_loadu_si512(a.w[0].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[1].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[2].as_ptr().add(8 * t) as *const __m512i),
        ];
        bf::mac_soa8(&mut acc, x, load_soa8(&ix, (b as *const u64).add(24 * t)));
    }
    let mut out = if blocks > 0 {
        horizontal(acc)
    } else {
        F162::ZERO
    };
    for k in 8 * blocks..n {
        out += a.get(k) * *b.add(k);
    }
    out
}

/// [`dot`] on two plain slices.
fn dot_slices(a: &[F162], b: &[F162]) -> F162 {
    assert_eq!(a.len(), b.len(), "the two operands have different lengths");
    unsafe { dot(&Poly::from_scalars(a), b.as_ptr(), b.len()) }
}

// =============================================================================================
// the four steps
// =============================================================================================

/// `u_j = sum_i eq(p0, i) W[i + wdim j]`, `wdim = 2^p0.len()`: one `wdim`-term dot product per
/// column, each a single deferred-reduction accumulation over a contiguous run of the witness.
pub(crate) fn row_evaluate(witness: &[F162], p0: &[F162]) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        witness.len() % wdim,
        0,
        "the witness is not a whole number of columns of {wdim}"
    );
    let eq = Poly::from_scalars(&eq_table(p0));
    (0..witness.len() / wdim)
        .map(|j| unsafe { dot(&eq, witness.as_ptr().add(j * wdim), wdim) })
        .collect()
}

/// `u^T eq(p1)`, the claim a row evaluation `u` implies.
pub(crate) fn claim(u: &[F162], p1: &[F162]) -> F162 {
    assert_eq!(
        u.len(),
        1 << p1.len(),
        "the row evaluation is not 2^{} elements",
        p1.len()
    );
    dot_slices(&eq_table(p1), u)
}

/// `u^T c`, the binary side of the fold: `sum_j u_j (c_j mod 2)`.
pub(crate) fn fold_binary(u: &[F162], challenges: &[ShortChallenge]) -> F162 {
    assert_eq!(u.len(), challenges.len(), "one challenge per column");
    let c: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    dot_slices(&c, u)
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, and its bit `p` is the parity of
/// coefficient `4p + k` of that element.
pub(crate) fn components_mod_2(v: &[RingElement]) -> Vec<F162> {
    let mut out = vec![F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        unsafe { element_mod_2(&e.v, &mut out[4 * m..4 * m + 4]) };
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,bmi2")]
unsafe fn element_mod_2(e: &[i16; N], out: &mut [F162]) {
    let ones = _mm512_set1_epi16(1);
    let p = e.as_ptr();
    let mut w = [[0u64; 3]; 4];
    for t in 0..N / 64 {
        let lo = _mm512_test_epi16_mask(
            _mm512_loadu_si512(p.add(64 * t) as *const __m512i),
            ones,
        ) as u64;
        let hi = _mm512_test_epi16_mask(
            _mm512_loadu_si512(p.add(64 * t + 32) as *const __m512i),
            ones,
        ) as u64;
        let bits = lo | (hi << 32);
        for (k, wk) in w.iter_mut().enumerate() {
            wk[t >> 2] |= _pext_u64(bits, 0x1111_1111_1111_1111u64 << k) << (16 * (t & 3));
        }
    }
    let rest = N - 64 * (N / 64);
    let bits = _mm512_test_epi16_mask(
        _mm512_maskz_loadu_epi16((1u64 << rest) as u32 - 1, p.add(64 * (N / 64))),
        ones,
    ) as u64;
    for (k, wk) in w.iter_mut().enumerate() {
        let q = N / 64 * 16;
        wk[q >> 6] |= _pext_u64(bits, 0x1111_1111_1111_1111u64 << k) << (q & 63);
    }
    for (k, wk) in w.iter().enumerate() {
        out[k] = F162(*wk);
    }
}

/// The binary check `B v == u^T c` over `F`: `sum_i eq(p0, i) (v_i mod 2) == u_folded`.
pub(crate) fn binary_check(p0: &[F162], v: &[RingElement], u_folded: F162) -> bool {
    4 * v.len() == 1usize << p0.len() && dot_slices(&eq_table(p0), &components_mod_2(v)) == u_folded
}

// =============================================================================================
// the row evaluation straight from the words
// =============================================================================================

/// The eight `B128` at `x` as the word-sliced `[limb0, limb1, 0]` of their lifts.
///
/// # Safety
/// `x` addresses 8 readable `B128`.
#[inline(always)]
unsafe fn load_soa8_words(x: *const F162Word) -> [__m512i; 3] {
    let a = _mm512_loadu_si512(x as *const __m512i);
    let b = _mm512_loadu_si512(x.add(4) as *const __m512i);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    [
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
        _mm512_setzero_si512(),
    ]
}

type F162Word = crate::fields::scalar::B128;

/// [`dot`] against a run of `n` `B128` words read as their lifts.
///
/// # Safety
/// `b` addresses `n` words and `a` holds at least `n` elements.
unsafe fn dot_words(a: &Poly, b: *const F162Word, n: usize) -> F162 {
    debug_assert!(a.n >= n);
    let (mut acc, blocks) = ([_mm512_setzero_si512(); 12], n / 8);
    for t in 0..blocks {
        // The words' lines 1 KB ahead, requested explicitly: with eight runners streaming the
        // table the hardware prefetcher alone leaves the clmul work waiting (6.6 -> 5.9 ms for
        // the phase; 4 KB and 16 KB ahead are no better). A prefetch past the end is harmless.
        let ahead = b.wrapping_add(8 * t + 64) as *const i8;
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead);
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead.wrapping_add(64));
        let x = [
            _mm512_loadu_si512(a.w[0].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[1].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[2].as_ptr().add(8 * t) as *const __m512i),
        ];
        bf::mac_soa8_word(&mut acc, x, load_soa8_words(b.add(8 * t)));
    }
    let mut out = if blocks > 0 {
        horizontal(acc)
    } else {
        F162::ZERO
    };
    for k in 8 * blocks..n {
        out += a.get(k) * F162::from_b128(*b.add(k));
    }
    out
}

/// [`row_evaluate`] of the witness `Witness::lifted` would build from `words`, read in place
/// and one column per task over `workers` threads: `u_j = sum_i eq(p0, i) lift(words[i + wdim j])`.
/// `eq(p0, .)` straight into the word-sliced layout [`dot_words`] reads: entry `i` is
/// `eq(p0[..3], i & 7) eq(p0[3..], i >> 3)`, eight entries per [`bf::mul_soa8`] against the
/// eight low ones, cut over the pool. The field is exact, so the entries are [`eq_table`]'s.
fn eq_poly(p0: &[F162], workers: usize) -> Poly {
    const LO: usize = 3;
    let n = 1usize << p0.len();
    if p0.len() < 2 * LO {
        return Poly::from_scalars(&eq_table(p0));
    }
    let lo = eq_table(&p0[..LO]);
    let hi = eq_table(&p0[LO..]);
    let mut w: [Vec<u64>; 3] = core::array::from_fn(|_| Vec::with_capacity(n));
    let addrs = w.each_mut().map(|v| v.as_mut_ptr() as usize);
    let runs = crate::par::ranges(hi.len(), workers, 8);
    crate::par::pool().run(runs.len(), |r| unsafe {
        let lo_v: [__m512i; 3] = core::array::from_fn(|k| {
            let limbs: [u64; 8] = core::array::from_fn(|i| lo[i].0[k]);
            _mm512_loadu_si512(limbs.as_ptr() as *const __m512i)
        });
        for ih in runs[r].clone() {
            let h = hi[ih].0;
            let v = bf::mul_soa8(lo_v, [
                _mm512_set1_epi64(h[0] as i64),
                _mm512_set1_epi64(h[1] as i64),
                _mm512_set1_epi64(h[2] as i64),
            ]);
            for (k, &a) in addrs.iter().enumerate() {
                _mm512_storeu_si512((a as *mut u64).add(8 * ih) as *mut __m512i, v[k]);
            }
        }
    });
    // Every entry was written above.
    for v in &mut w {
        unsafe { v.set_len(n) };
    }
    Poly { w, n }
}

pub fn row_evaluate_words(words: &[F162Word], p0: &[F162], workers: usize) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        words.len() % wdim,
        0,
        "the words are not a whole number of columns of {wdim}"
    );
    let columns = words.len() / wdim;
    let eq = eq_poly(p0, workers);
    let mut out = vec![F162::ZERO; columns];
    // Columns handed out a few at a time from a shared counter: a runner that starts late or
    // shares its core with another process takes fewer of them instead of holding the phase up.
    let chunk = columns.div_ceil(32 * workers.max(1)).max(1);
    let chunks = columns.div_ceil(chunk);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out_addr = out.as_mut_ptr() as usize;
    crate::par::pool().run(workers.clamp(1, chunks), |_| loop {
        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if k >= chunks {
            break;
        }
        // Chunk `k` owns `out[k * chunk ..]`.
        for j in k * chunk..((k + 1) * chunk).min(columns) {
            let value = unsafe { dot_words(&eq, words.as_ptr().add(j * wdim), wdim) };
            unsafe { (out_addr as *mut F162).add(j).write(value) };
        }
    });
    out
}

#[cfg(test)]
mod word_tests {
    use super::*;

    #[test]
    fn word_rows_match_scalar_products() {
        let mut state = 0x123456789abcdefu64;
        let mut next = || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17; state
        };
        for log in [0usize, 1, 3, 6, 7, 10, 14] {
            let p: Vec<_> = (0..log).map(|_| F162([next(), next(), next() & ((1 << 34) - 1)])).collect();
            let words: Vec<_> = (0..(5 << log)).map(|_| crate::fields::scalar::B128((next() as u128) | ((next() as u128) << 64))).collect();
            let eq = eq_table(&p);
            let expected: Vec<_> = words.chunks_exact(1 << log).map(|column|
                eq.iter().zip(column).fold(F162::ZERO, |a, (&e, &w)| a + e * F162::from_b128(w))).collect();
            for workers in [1usize, 3, 8] {
                assert_eq!(row_evaluate_words(&words, &p, workers), expected, "log {log}, workers {workers}");
                let poly = eq_poly(&p, workers);
                assert_eq!((0..eq.len()).map(|i| poly.get(i)).collect::<Vec<_>>(), eq, "log {log}");
            }
        }
    }
}
