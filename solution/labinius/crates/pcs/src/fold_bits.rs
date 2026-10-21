//! The fold `v = sum_j c_j W_j` straight from the binary words, with bit-sliced counters.
//!
//! # The algebra, in the verifier's terms
//!
//! A challenge `c_j` is a weight-28 signed subring element with positions `p_t` and signs; it
//! sits in `R_648` as `c_j(-X^4)`, whose coefficient of `X^{4p}` is the sign of position `p`
//! times `(-1)^p`. Ring element `m` of column `j` is binary: coefficient `4q + k` is bit `q` of
//! word `4m + k` of the column, and a word has 128 bits, so `q < 128`. The verifier's reference
//! fold computes, per element `m`,
//!
//! ```text
//!     raw[c] = sum_j sum_t sign_{j,t} * bits_{j,m}[c - 4 p_{j,t}]     (0 <= c < 2 * 648 - 1)
//!     for c = 1295 down to 648: raw[c - 324] += raw[c]; raw[c - 648] -= raw[c]
//! ```
//!
//! Index `c = 4 e + k` keeps the component `k`: contributions to `(e, k)` come from bit
//! `e - p_t` of word `4m + k`, for every `(j, t)` with `0 <= e - p_t < 128`, and `e` ranges
//! over `0 .. 289`. So for a fixed *row* `e` and a fixed lane `(m, k)`, the raw value is a signed
//! count of bits, and the reduction is the same shift-and-add on rows `e`.
//!
//! # The computation
//!
//! Lanes are `(m, k)` pairs, 512 of them per block (128 ring elements, the 512 consecutive words
//! `4 m + k` of a column slice). Each column's 512 words are transposed into 128 bit planes of
//! 512 lanes ([`Transposer`]); then, row by row, every contributing plane is pushed into a
//! carry-save accumulator ([`Pending`]: a binary counter of vectors, one full adder per three
//! vectors of equal weight) and the accumulator is flushed into the row's 16-plane bit-sliced
//! counter once per batch of columns. A negative term `-b` enters as `!b`, since `-b = !b - 1`,
//! and the number of negative terms of the row is subtracted at the end. Counts stay below
//! `28 * 1024 < 2^15`.
//!
//! Everything is XOR, AND and majority on 512-bit vectors: about four operations per term
//! against the hundreds of cycles per ring element a transform costs. [`Lanes`] is the vector
//! type — `__m512i` on x86-64, `[u64; 8]` elsewhere, the same algorithm over both — so the whole
//! fold runs and is tested on any machine.
//!
//! The lane order inside a 64-word tile follows [`word_at`], the byte scramble of the library's
//! GFNI tile transpose, so that transpose can feed this directly; the portable transposer uses
//! the same order.

/// Degree of the ring.
pub const N: usize = 648;
/// Rows of the raw fold: `e = q + p`, `q < 128`, `p < 162`.
pub const ROWS: usize = 289;
/// Bits of a word, the nonzero coefficients of a component.
pub const WORD_BITS: usize = 128;
/// Coefficients of a component after reduction.
pub const Q162: usize = 162;
/// Lanes per block: 512 words, 128 ring elements.
pub const LANES: usize = 512;
/// Ring elements per block.
pub const BLOCK_ELEMENTS: usize = LANES / 4;
/// Planes of the counter: counts are below `2^15`.
pub const COUNTER_PLANES: usize = 16;
/// Columns whose planes are resident together: 16 x 128 planes x 64 bytes = 128 KB.
pub const BATCH: usize = 16;

/// A challenge in the sparse form the library samples: `weight` sorted positions below 162, a
/// sign bit per position (`1` = negative).
#[derive(Clone, Copy, Debug)]
pub struct Challenge {
    pub positions: [u8; 32],
    pub signs: u32,
    pub weight: usize,
}

/// The word at byte position `p` of a transposed 64-word tile (the library's `bitmat::word_at`).
#[inline]
pub fn word_at(p: usize) -> usize {
    (((p >> 5) & 1) << 1) | ((p >> 4) & 1) | ((p & 0xf) << 2)
}

/// The word, within a 512-word block, that lane `l` carries.
#[inline]
pub fn lane_word(l: usize) -> usize {
    64 * (l / 64) + word_at(l % 64)
}

// =============================================================================================
// the vector type
// =============================================================================================

/// 512 lanes of one bit.
pub trait Lanes: Copy {
    fn zero() -> Self;
    fn ones() -> Self;
    /// # Safety
    /// `p` addresses 8 readable `u64`.
    unsafe fn load(p: *const u64) -> Self;
    /// # Safety
    /// `p` addresses 8 writable `u64`.
    unsafe fn store(self, p: *mut u64);
    fn xor(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    /// `(a ^ b ^ c, majority(a, b, c))`: a full adder on every lane.
    fn csa(a: Self, b: Self, c: Self) -> (Self, Self);
    /// The lanes as eight `u64`.
    fn words(self) -> [u64; 8];
    fn from_words(w: [u64; 8]) -> Self;
    /// The 16-plane bit-sliced counter at `planes` (plane `w` at `planes + 8 w`) as 512 `i32`
    /// counts.
    ///
    /// # Safety
    /// `planes` addresses `8 * COUNTER_PLANES` readable `u64`, `out` 512 writable `i32`.
    unsafe fn read_counter(planes: *const u64, out: *mut i32);
}

/// [`Lanes::read_counter`] one lane at a time, for the portable type and the tests.
///
/// # Safety
/// As [`Lanes::read_counter`].
pub unsafe fn read_counter_scalar(planes: *const u64, out: *mut i32) {
    for l in 0..LANES {
        let (word, bit) = (l / 64, l % 64);
        let mut v = 0i32;
        for w in 0..COUNTER_PLANES {
            v |= (((*planes.add(8 * w + word) >> bit) & 1) as i32) << w;
        }
        *out.add(l) = v;
    }
}

#[cfg(target_arch = "x86_64")]
mod vec512 {
    use super::Lanes;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct V(pub __m512i);

    impl Lanes for V {
        #[inline(always)]
        fn zero() -> Self {
            V(unsafe { _mm512_setzero_si512() })
        }
        #[inline(always)]
        fn ones() -> Self {
            V(unsafe { _mm512_set1_epi64(-1) })
        }
        #[inline(always)]
        unsafe fn load(p: *const u64) -> Self {
            V(_mm512_loadu_si512(p as *const __m512i))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut u64) {
            _mm512_storeu_si512(p as *mut __m512i, self.0)
        }
        #[inline(always)]
        fn xor(self, o: Self) -> Self {
            V(unsafe { _mm512_xor_si512(self.0, o.0) })
        }
        #[inline(always)]
        fn and(self, o: Self) -> Self {
            V(unsafe { _mm512_and_si512(self.0, o.0) })
        }
        #[inline(always)]
        fn csa(a: Self, b: Self, c: Self) -> (Self, Self) {
            unsafe {
                (
                    V(_mm512_ternarylogic_epi64::<0x96>(a.0, b.0, c.0)),
                    V(_mm512_ternarylogic_epi64::<0xE8>(a.0, b.0, c.0)),
                )
            }
        }
        #[inline(always)]
        fn words(self) -> [u64; 8] {
            unsafe { core::mem::transmute(self.0) }
        }
        #[inline(always)]
        fn from_words(w: [u64; 8]) -> Self {
            V(unsafe { core::mem::transmute(w) })
        }
        unsafe fn read_counter(planes: *const u64, out: *mut i32) {
            // Sixteen lanes at a time: plane `w`'s bits for those lanes are a 16-bit mask, and
            // a masked broadcast of `1 << w` ORed over the planes rebuilds the counts.
            for g in 0..super::LANES / 16 {
                let (word, shift) = (g / 4, 16 * (g % 4));
                let mut acc = _mm512_setzero_si512();
                for w in 0..super::COUNTER_PLANES {
                    let bits = (*planes.add(8 * w + word) >> shift) as u16;
                    acc = _mm512_mask_or_epi32(acc, bits, acc, _mm512_set1_epi32(1 << w));
                }
                _mm512_storeu_si512(out.add(16 * g) as *mut __m512i, acc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use vec512::V;

#[cfg(not(target_arch = "x86_64"))]
mod vec512 {
    use super::Lanes;

    #[derive(Clone, Copy)]
    pub struct V(pub [u64; 8]);

    impl Lanes for V {
        #[inline(always)]
        fn zero() -> Self {
            V([0; 8])
        }
        #[inline(always)]
        fn ones() -> Self {
            V([!0; 8])
        }
        #[inline(always)]
        unsafe fn load(p: *const u64) -> Self {
            V(core::ptr::read_unaligned(p as *const [u64; 8]))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut u64) {
            core::ptr::write_unaligned(p as *mut [u64; 8], self.0)
        }
        #[inline(always)]
        fn xor(self, o: Self) -> Self {
            V(core::array::from_fn(|i| self.0[i] ^ o.0[i]))
        }
        #[inline(always)]
        fn and(self, o: Self) -> Self {
            V(core::array::from_fn(|i| self.0[i] & o.0[i]))
        }
        #[inline(always)]
        fn csa(a: Self, b: Self, c: Self) -> (Self, Self) {
            (
                V(core::array::from_fn(|i| a.0[i] ^ b.0[i] ^ c.0[i])),
                V(core::array::from_fn(|i| {
                    (a.0[i] & b.0[i]) | (a.0[i] & c.0[i]) | (b.0[i] & c.0[i])
                })),
            )
        }
        #[inline(always)]
        fn words(self) -> [u64; 8] {
            self.0
        }
        #[inline(always)]
        fn from_words(w: [u64; 8]) -> Self {
            V(w)
        }
        unsafe fn read_counter(planes: *const u64, out: *mut i32) {
            super::read_counter_scalar(planes, out)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub use vec512::V;

// =============================================================================================
// the transpose
// =============================================================================================

/// 512 words into 128 bit planes: `out[q]` holds bit `q` of the words, lane `l` carrying word
/// [`lane_word`]`(l)` of the block, or whatever [`Transposer::lane_word`] says.
pub trait Transposer {
    /// # Safety
    /// `words` addresses 512 readable `u128`.
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]);
    /// The word (within the block) lane `l` carries.
    fn lane_word(&self, l: usize) -> usize {
        lane_word(l)
    }
}

/// The portable transpose: eight 64-word tiles, each a 128 x 64 bit matrix transposed with the
/// classic swap network, lanes in [`word_at`] order.
pub struct Portable;

/// The same with any lane order: lane `l` carries word `map(l)` of the block.
pub struct PortableWith(pub fn(usize) -> usize);

/// # Safety
/// `words` addresses 512 readable `u128`.
pub unsafe fn portable_planes<L: Lanes>(
    words: *const u128,
    out: &mut [L; WORD_BITS],
    map: fn(usize) -> usize,
) {
    let mut planes = [[0u64; 8]; WORD_BITS];
    for tile in 0..8 {
        // Gather the tile's 64 words in lane order, as two 64 x 64 bit matrices.
        let mut lo = [0u64; 64];
        let mut hi = [0u64; 64];
        for p in 0..64 {
            let w = *words.add(map(64 * tile + p));
            lo[p] = w as u64;
            hi[p] = (w >> 64) as u64;
        }
        transpose64(&mut lo);
        transpose64(&mut hi);
        for q in 0..64 {
            planes[q][tile] = lo[q];
            planes[64 + q][tile] = hi[q];
        }
    }
    for (o, p) in out.iter_mut().zip(&planes) {
        *o = L::from_words(*p);
    }
}

impl Transposer for Portable {
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]) {
        portable_planes(words, out, lane_word)
    }
}

impl Transposer for PortableWith {
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]) {
        portable_planes(words, out, self.0)
    }
    fn lane_word(&self, l: usize) -> usize {
        (self.0)(l)
    }
}

/// The lane order the GFNI path produces: the `gf2p8affine` by the identity transposes each
/// qword's 8x8 bits with the byte index reversed, so within every group of eight lanes the
/// order of [`word_at`] runs backwards.
#[inline]
pub fn lane_word_reversed8(l: usize) -> usize {
    64 * (l / 64) + word_at(8 * ((l % 64) / 8) + 7 - (l % 8))
}

/// In-place transpose of a 64 x 64 bit matrix held as 64 rows: afterwards row `b` bit `a` is
/// the old row `a` bit `b`.
pub fn transpose64(m: &mut [u64; 64]) {
    let mut j = 32;
    let mut mask = 0x0000_0000_FFFF_FFFFu64;
    while j != 0 {
        let mut k = 0;
        while k < 64 {
            for i in k..k + j {
                let t = ((m[i] >> j) ^ m[i + j]) & mask;
                m[i] ^= t << j;
                m[i + j] ^= t;
            }
            k += 2 * j;
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

// =============================================================================================
// the accumulator
// =============================================================================================

/// A binary counter of vectors: at most two pending vectors per weight `2^w`; a third of the
/// same weight is combined with them by one full adder, whose carry moves up one weight. Used
/// for the few vectors a row visit leaves over, not for the inputs themselves.
struct Pending<L: Lanes> {
    slots: [[L; 2]; COUNTER_PLANES],
    count: [u8; COUNTER_PLANES],
}

impl<L: Lanes> Pending<L> {
    #[inline(always)]
    fn new() -> Self {
        Pending {
            slots: [[L::zero(); 2]; COUNTER_PLANES],
            count: [0; COUNTER_PLANES],
        }
    }

    /// Add `x` at weight `2^w`.
    #[inline(always)]
    fn push(&mut self, mut x: L, mut w: usize) {
        loop {
            match self.count[w] {
                0 => {
                    self.slots[w][0] = x;
                    self.count[w] = 1;
                    return;
                }
                1 => {
                    self.slots[w][1] = x;
                    self.count[w] = 2;
                    return;
                }
                _ => {
                    let (s, c) = L::csa(self.slots[w][0], self.slots[w][1], x);
                    self.slots[w][0] = s;
                    self.count[w] = 1;
                    x = c;
                    w += 1;
                    debug_assert!(w < COUNTER_PLANES, "the pending sum overflows the counter");
                }
            }
        }
    }

    /// Add everything pending into the 16-plane bit-sliced counter at `planes` (plane `w` at
    /// `planes + 8 w`): the pending vectors are first compressed to one digit per weight (a
    /// full adder per weight over the two slots and the carry from below), and that number is
    /// added to the counter with a ripple of full adders. The two carry chains are independent.
    ///
    /// # Safety
    /// `planes` addresses `8 * COUNTER_PLANES` writable `u64`.
    #[inline(always)]
    unsafe fn flush_into(&self, planes: *mut u64) {
        let mut carry_pending = L::zero();
        let mut carry_add = L::zero();
        for w in 0..COUNTER_PLANES {
            let (a, b) = (self.slots[w][0], self.slots[w][1]);
            let (digit, up) = match self.count[w] {
                0 => (carry_pending, L::zero()),
                1 => (a.xor(carry_pending), a.and(carry_pending)),
                _ => L::csa(a, b, carry_pending),
            };
            carry_pending = up;
            let p = planes.add(8 * w);
            let (sum, out) = L::csa(L::load(p), digit, carry_add);
            sum.store(p);
            carry_add = out;
        }
        // Counts stay below 2^15, so nothing is carried out of the top plane.
    }
}

/// The Harley–Seal state: one vector each of weight 1, 2, 4 and 8.
struct Hs<L: Lanes> {
    ones: L,
    twos: L,
    fours: L,
    eights: L,
}

impl<L: Lanes> Hs<L> {
    #[inline(always)]
    fn new() -> Self {
        Hs {
            ones: L::zero(),
            twos: L::zero(),
            fours: L::zero(),
            eights: L::zero(),
        }
    }

    /// Sixteen inputs through fifteen full adders: the weight-16 carry comes back, the rest
    /// stays in the state.
    #[inline(always)]
    fn add16(&mut self, x: &[L; 16]) -> L {
        let (ones, twos_a) = L::csa(self.ones, x[0], x[1]);
        let (ones, twos_b) = L::csa(ones, x[2], x[3]);
        let (twos, fours_a) = L::csa(self.twos, twos_a, twos_b);
        let (ones, twos_a) = L::csa(ones, x[4], x[5]);
        let (ones, twos_b) = L::csa(ones, x[6], x[7]);
        let (twos, fours_b) = L::csa(twos, twos_a, twos_b);
        let (fours, eights_a) = L::csa(self.fours, fours_a, fours_b);
        let (ones, twos_a) = L::csa(ones, x[8], x[9]);
        let (ones, twos_b) = L::csa(ones, x[10], x[11]);
        let (twos, fours_a) = L::csa(twos, twos_a, twos_b);
        let (ones, twos_a) = L::csa(ones, x[12], x[13]);
        let (ones, twos_b) = L::csa(ones, x[14], x[15]);
        let (twos, fours_b) = L::csa(twos, twos_a, twos_b);
        let (fours, eights_b) = L::csa(fours, fours_a, fours_b);
        let (eights, sixteens) = L::csa(self.eights, eights_a, eights_b);
        self.ones = ones;
        self.twos = twos;
        self.fours = fours;
        self.eights = eights;
        sixteens
    }
}

// =============================================================================================
// one block
// =============================================================================================

/// A column's challenge in the form the row loop reads: per term its position and its sign as a
/// mask (`!0` negative, `0` positive) — the embedding `c(-X^4)`: the sign bit times `(-1)^p`.
#[derive(Clone, Copy)]
struct Terms {
    position: [u8; 32],
    negative: [u8; 32],
}

impl Terms {
    fn of(c: &Challenge) -> Terms {
        let mut t = Terms {
            position: [0; 32],
            negative: [0; 32],
        };
        for i in 0..c.weight {
            let p = c.positions[i];
            assert!((p as usize) < Q162, "a position is below 162");
            if i > 0 {
                assert!(p > t.position[i - 1], "positions are sorted and distinct");
            }
            t.position[i] = p;
            t.negative[i] = (((c.signs >> i) & 1 == 1) ^ (p % 2 == 1)) as u8;
        }
        t
    }
}

/// For every column and row, the terms that reach the row: `t` in `lo .. hi` has
/// `0 <= e - p_t < 128`. Positions are sorted, so the terms form a run.
pub struct Reach {
    columns: usize,
    lo_hi: Vec<[u8; 2]>,
}

impl Reach {
    /// Positions are sorted, so with `e` increasing, `hi(e) = #{t : p_t <= e}` and
    /// `lo(e) = #{t : p_t + 128 <= e}` both advance monotonically: one sweep per column.
    pub fn of(challenges: &[Challenge]) -> Reach {
        let mut lo_hi = vec![[0u8; 2]; challenges.len() * ROWS];
        for (j, c) in challenges.iter().enumerate() {
            let (mut lo, mut hi) = (0usize, 0usize);
            for e in 0..ROWS {
                while hi < c.weight && c.positions[hi] as usize <= e {
                    hi += 1;
                }
                while lo < c.weight && c.positions[lo] as usize + WORD_BITS <= e {
                    lo += 1;
                }
                lo_hi[j * ROWS + e] = [lo.min(hi) as u8, hi as u8];
            }
        }
        Reach {
            columns: challenges.len(),
            lo_hi,
        }
    }

    /// The same table term by term, the oracle of the tests.
    #[cfg(test)]
    fn brute(challenges: &[Challenge]) -> Reach {
        let mut lo_hi = vec![[0u8; 2]; challenges.len() * ROWS];
        for (j, c) in challenges.iter().enumerate() {
            for e in 0..ROWS {
                let mut lo = c.weight;
                let mut hi = 0;
                for t in 0..c.weight {
                    let p = c.positions[t] as usize;
                    if p <= e && e - p < WORD_BITS {
                        lo = lo.min(t);
                        hi = t + 1;
                    }
                }
                lo_hi[j * ROWS + e] = [lo.min(hi) as u8, hi as u8];
            }
        }
        Reach {
            columns: challenges.len(),
            lo_hi,
        }
    }
    #[inline(always)]
    fn of_row(&self, j: usize, e: usize) -> (usize, usize) {
        let r = self.lo_hi[j * ROWS + e];
        (r[0] as usize, r[1] as usize)
    }
}

/// The 16-plane counters of the [`ROWS`] rows of one block, plane-major per row.
struct Counters {
    planes: Vec<u64>,
}

impl Counters {
    fn new() -> Counters {
        Counters {
            planes: vec![0u64; ROWS * COUNTER_PLANES * 8],
        }
    }
    #[inline(always)]
    fn row(&mut self, e: usize) -> *mut u64 {
        unsafe { self.planes.as_mut_ptr().add(e * COUNTER_PLANES * 8) }
    }
    /// The count of row `e`, lane `l`.
    #[allow(dead_code)]
    fn count(&self, e: usize, l: usize) -> u32 {
        let base = e * COUNTER_PLANES * 8;
        let (word, bit) = (l / 64, l % 64);
        let mut v = 0u32;
        for w in 0..COUNTER_PLANES {
            v |= (((self.planes[base + 8 * w + word] >> bit) & 1) as u32) << w;
        }
        v
    }
}

/// The fold of one block: ring elements `block * 128 ..` of the fold, `out[i]` being element
/// `block * 128 + i`. `words` is the whole witness, `columns` columns of `column_words` words;
/// `challenges` one per column, `reach` their [`Reach`].
///
/// # Safety
/// AVX-512 F on x86-64 (the vector type), and `words` holds `columns * column_words` words with
/// `column_words` a multiple of 512 and `block * 512 + 512 <= column_words`.
pub unsafe fn fold_block<L: Lanes, T: Transposer>(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    reach: &Reach,
    block: usize,
    transposer: &T,
    out: &mut [[i16; N]],
) {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert_eq!(reach.columns, columns, "the reach is of these challenges");
    assert_eq!(out.len(), BLOCK_ELEMENTS, "one block of elements");
    let terms: Vec<Terms> = challenges.iter().map(Terms::of).collect();
    let mut counters = Counters::new();
    let mut negatives = vec![0u32; ROWS];
    // Each column's planes and their complements, so a negative term reads `!b` straight from
    // memory; `base[i][t]` is the address of term `t`'s plane for row 0, and row `e` reads it
    // `e` planes further on (`e - p_t` is the plane index).
    let mut planes: Vec<[L; 2 * WORD_BITS]> = (0..BATCH).map(|_| [L::zero(); 2 * WORD_BITS]).collect();
    let mut base = [[core::ptr::null::<L>(); 32]; BATCH];
    let mut negs = [0u32; BATCH];
    let mut list = [core::ptr::null::<L>(); 2 * BATCH * 32];
    let zero = L::zero();
    let zeros = [L::zero(); 16];
    for batch_start in (0..columns).step_by(BATCH) {
        let batch = BATCH.min(columns - batch_start);
        for i in 0..batch {
            let column = batch_start + i;
            let plane: &mut [L; 2 * WORD_BITS] = &mut planes[i];
            let (direct, complement) = plane.split_at_mut(WORD_BITS);
            let direct: &mut [L; WORD_BITS] = direct.try_into().unwrap();
            transposer.planes(words.add(column * column_words + block * LANES), direct);
            for q in 0..WORD_BITS {
                complement[q] = direct[q].xor(L::ones());
            }
            let t = &terms[column];
            let first = plane.as_ptr();
            for k in 0..32 {
                // Plane `e - p` of the direct or complemented set: `first + (neg ? 128 : 0) - p`,
                // then `+ e` per row. Positions are below 162 < 2^8; the pointer may point below
                // the array for rows the term does not reach, and is never dereferenced there.
                let offset = t.negative[k] as usize * WORD_BITS;
                base[i][k] = first.wrapping_add(offset).wrapping_sub(t.position[k] as usize);
            }
            negs[i] = (0..32).map(|k| t.negative[k] as u32).sum::<u32>();
            let _ = negs;
        }
        for e in 0..ROWS {
            // The planes reaching row `e`, listed.
            let mut n = 0usize;
            let mut negative = 0u32;
            for i in 0..batch {
                let j = batch_start + i;
                let (lo, hi) = reach.of_row(j, e);
                let t = &terms[j];
                let b = &base[i];
                for k in lo..hi {
                    *list.get_unchecked_mut(n) = b.get_unchecked(k).wrapping_add(e);
                    negative += *t.negative.get_unchecked(k) as u32;
                    n += 1;
                }
            }
            let mut hs = Hs::<L>::new();
            let mut pending = Pending::<L>::new();
            let mut at = 0usize;
            while at + 16 <= n {
                let x: [L; 16] = core::array::from_fn(|i| **list.get_unchecked(at + i));
                pending.push(hs.add16(&x), 4);
                at += 16;
            }
            if at < n {
                let mut x = zeros;
                for i in at..n {
                    x[i - at] = **list.get_unchecked(i);
                }
                pending.push(hs.add16(&x), 4);
            }
            let _ = zero;
            pending.push(hs.ones, 0);
            pending.push(hs.twos, 1);
            pending.push(hs.fours, 2);
            pending.push(hs.eights, 3);
            pending.flush_into(counters.row(e));
            negatives[e] += negative;
        }
    }
    // The raw rows as integers, then the verifier's reduction, then the elements.
    let mut raw = vec![0i32; ROWS * LANES];
    for e in 0..ROWS {
        let n = negatives[e] as i32;
        let row = &mut raw[e * LANES..(e + 1) * LANES];
        L::read_counter(counters.row(e), row.as_mut_ptr());
        for x in row.iter_mut() {
            *x -= n;
        }
    }
    for e in (Q162..ROWS).rev() {
        for l in 0..LANES {
            let x = raw[e * LANES + l];
            raw[(e - 81) * LANES + l] += x;
            raw[(e - Q162) * LANES + l] -= x;
        }
    }
    for l in 0..LANES {
        let word = transposer.lane_word(l);
        let (m, k) = (word / 4, word % 4);
        for q in 0..Q162 {
            out[m][4 * q + k] = i16::try_from(raw[q * LANES + l])
                .expect("a fold coefficient does not fit the wire codec");
        }
    }
}

/// Every block of the fold, serially: `out[m]` for every ring element `m` of the witness's
/// `column_words / 4` elements per column.
///
/// # Safety
/// As [`fold_block`].
pub unsafe fn fold_all<L: Lanes, T: Transposer>(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    transposer: &T,
) -> Vec<[i16; N]> {
    assert!(column_words % LANES == 0, "a column is whole blocks");
    let elements = column_words / 4;
    let reach = Reach::of(challenges);
    let mut out = vec![[0i16; N]; elements];
    for (block, chunk) in out.chunks_mut(BLOCK_ELEMENTS).enumerate() {
        fold_block::<L, T>(words, columns, column_words, challenges, &reach, block, transposer, chunk);
    }
    out
}

// =============================================================================================
// the reference, for the tests and the runtime check
// =============================================================================================

/// The verifier's reference fold of one ring element: `v_m = sum_j c_j W_{j,m}` over the
/// integers, by shifted adds of the words' bits and the ring's reduction.
pub fn reference_element(
    words: &[u128],
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    m: usize,
) -> [i16; N] {
    let mut raw = vec![0i32; 2 * N - 1];
    let mut bits = [0i32; N];
    for (j, c) in challenges.iter().enumerate() {
        let packed = &words[j * column_words + 4 * m..][..4];
        for (k, x) in packed.iter().enumerate() {
            for p in 0..N / 4 {
                bits[4 * p + k] = if p < WORD_BITS { ((x >> p) & 1) as i32 } else { 0 };
            }
        }
        for t in 0..c.weight {
            let position = c.positions[t] as usize;
            let negative = ((c.signs >> t) & 1 == 1) ^ (position % 2 == 1);
            let window = &mut raw[4 * position..4 * position + N];
            if negative {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot -= b;
                }
            } else {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot += b;
                }
            }
        }
    }
    for k in (N..2 * N - 1).rev() {
        let value = raw[k];
        raw[k] = 0;
        raw[k - N / 2] += value;
        raw[k - N] -= value;
    }
    let mut element = [0i16; N];
    for (slot, &value) in element.iter_mut().zip(&raw[..N]) {
        *slot = i16::try_from(value).expect("a fold coefficient does not fit the wire codec");
    }
    let _ = columns;
    element
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(x: &mut u128) -> u128 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (*x >> 64)
    }

    fn challenge(x: &mut u128, weight: usize) -> Challenge {
        let mut taken = [false; Q162];
        let mut positions = Vec::new();
        while positions.len() < weight {
            let p = (next(x) % Q162 as u128) as u8;
            if !taken[p as usize] {
                taken[p as usize] = true;
                positions.push(p);
            }
        }
        positions.sort_unstable();
        let mut c = Challenge {
            positions: [0; 32],
            signs: next(x) as u32 & ((1u32 << weight) - 1),
            weight,
        };
        c.positions[..weight].copy_from_slice(&positions);
        c
    }

    #[test]
    fn transpose64_is_a_transpose() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let m: [u64; 64] = core::array::from_fn(|_| next(&mut x) as u64);
        let mut t = m;
        transpose64(&mut t);
        for a in 0..64 {
            for b in 0..64 {
                assert_eq!((t[b] >> a) & 1, (m[a] >> b) & 1, "({a}, {b})");
            }
        }
        transpose64(&mut t);
        assert_eq!(t, m);
    }

    #[test]
    fn portable_planes_hold_the_words_bits() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let words: Vec<u128> = (0..LANES).map(|_| next(&mut x)).collect();
        let mut planes = [V::zero(); WORD_BITS];
        unsafe { Portable.planes(words.as_ptr(), &mut planes) };
        for q in 0..WORD_BITS {
            let w = planes[q].words();
            for l in 0..LANES {
                let bit = (w[l / 64] >> (l % 64)) & 1;
                assert_eq!(bit, ((words[lane_word(l)] >> q) & 1) as u64, "q {q} l {l}");
            }
        }
    }

    #[test]
    fn pending_counts_exactly() {
        let mut x = 0x2545F4914F6CDD1Du128;
        for n in [0usize, 1, 2, 3, 5, 17, 96, 300, 1000] {
            let inputs: Vec<[u64; 8]> = (0..n).map(|_| core::array::from_fn(|_| next(&mut x) as u64)).collect();
            let mut planes = [0u64; COUNTER_PLANES * 8];
            // Pre-load the counter with something, to exercise the ripple.
            for (i, p) in planes.iter_mut().enumerate() {
                *p = if i < 8 { next(&mut x) as u64 } else { 0 };
            }
            let before: Vec<u32> = (0..512)
                .map(|l| (0..COUNTER_PLANES).map(|w| (((planes[8 * w + l / 64] >> (l % 64)) & 1) as u32) << w).sum())
                .collect();
            let mut pending = Pending::<V>::new();
            for inp in &inputs {
                pending.push(V::from_words(*inp), 0);
            }
            unsafe { pending.flush_into(planes.as_mut_ptr()) };
            for l in 0..512 {
                let want: u32 = before[l]
                    + inputs.iter().map(|inp| ((inp[l / 64] >> (l % 64)) & 1) as u32).sum::<u32>();
                let got: u32 = (0..COUNTER_PLANES)
                    .map(|w| (((planes[8 * w + l / 64] >> (l % 64)) & 1) as u32) << w)
                    .sum();
                assert_eq!(got, want, "n {n} lane {l}");
            }
        }
    }

    #[test]
    fn any_lane_order_gives_the_same_fold() {
        let mut x = 0x5151_2323_7777u128;
        let (columns, column_words) = (9usize, 512usize);
        let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
        let challenges: Vec<Challenge> = (0..columns).map(|_| challenge(&mut x, 28)).collect();
        let reach = Reach::of(&challenges);
        let mut a = vec![[0i16; N]; BLOCK_ELEMENTS];
        let mut b = vec![[0i16; N]; BLOCK_ELEMENTS];
        unsafe {
            fold_block::<V, Portable>(words.as_ptr(), columns, column_words, &challenges, &reach, 0, &Portable, &mut a);
            let t = PortableWith(lane_word_reversed8);
            fold_block::<V, PortableWith>(words.as_ptr(), columns, column_words, &challenges, &reach, 0, &t, &mut b);
        }
        assert_eq!(a, b);
        assert_eq!(a[3][..], reference_element(&words, columns, column_words, &challenges, 3)[..]);
    }

    #[test]
    fn reach_sweep_matches_brute_force() {
        let mut x = 0xA5A5_5A5Au128;
        let challenges: Vec<Challenge> = (0..40)
            .map(|j| challenge(&mut x, [0usize, 1, 2, 28, 7][j % 5]))
            .collect();
        // Both spell an empty run as some `lo >= hi`; compare the runs they denote.
        let canon = |r: &Reach| -> Vec<(usize, usize)> {
            r.lo_hi
                .iter()
                .map(|&[lo, hi]| if lo < hi { (lo as usize, hi as usize) } else { (0, 0) })
                .collect()
        };
        assert_eq!(canon(&Reach::of(&challenges)), canon(&Reach::brute(&challenges)));
    }

    #[test]
    fn counter_read_out_matches_the_scalar_one() {
        let mut x = 0xF00D_1234u128;
        let planes: Vec<u64> = (0..COUNTER_PLANES * 8).map(|_| next(&mut x) as u64).collect();
        let mut a = [0i32; LANES];
        let mut b = [0i32; LANES];
        unsafe {
            read_counter_scalar(planes.as_ptr(), a.as_mut_ptr());
            V::read_counter(planes.as_ptr(), b.as_mut_ptr());
        }
        assert_eq!(a[..], b[..]);
    }

    #[test]
    fn bit_sliced_fold_matches_the_reference() {
        let mut x = 0xDEAD_BEEF_CAFE_F00Du128;
        for (columns, column_words) in [(2usize, 512usize), (6, 1024), (17, 512), (40, 1536)] {
            let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
            let challenges: Vec<Challenge> = (0..columns)
                .map(|j| challenge(&mut x, if j % 5 == 0 { 28 } else { 1 + (j % 28) }))
                .collect();
            let got = unsafe { fold_all::<V, Portable>(words.as_ptr(), columns, column_words, &challenges, &Portable) };
            let elements = column_words / 4;
            assert_eq!(got.len(), elements);
            for m in 0..elements {
                let want = reference_element(&words, columns, column_words, &challenges, m);
                assert_eq!(got[m][..], want[..], "columns {columns} element {m}");
            }
        }
    }
}
