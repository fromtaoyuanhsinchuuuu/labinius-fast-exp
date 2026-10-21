//! Forward NTT for **binary** inputs in the vertical batch-of-32 layout, with levels 4, 5 and 6
//! hand-scheduled in one `asm!` block per 27-block (27 zmm data registers resident).
//! Generated; see the report for the schedule variants tried.
//! Agrees with `ntt::bin_small` modulo q: the s2 = 1 half of every 162-block reuses the s2 = 0
//! lookups (see the kernel), and for q = 9721 it also reduces at different levels and with a
//! different Barrett (see Bounds below).
//!
//! Levels 0, 1, 2 and the level-3 twiddles are all folded into 16-entry lookup tables indexed by
//! the 4-bit nibble (b_i, b_{i+162}, b_{i+324}, b_{i+486}) of each polynomial; levels 3..6 are radix-3
//! signed-Montgomery butterflies. The whole batch is done depth-first per 162-block: the table
//! lookups, level 3 and level 4 write a 10 KB stack block, levels 5 and 6 are fused (a degree-9
//! sub-ring is exactly three degree-3 sub-rings, so its nine values never leave registers) and go
//! straight to the output.
//!
//! Measured on the i7-11850H, one core, cache-resident input: 280.2 cycles per ring element
//! for q = 3889 and 303.8 for q = 9721 (697 / 752 instructions, 699 / 760 uops, 268 / 291
//! port-0, 246 / 277 port-5).
//!
//! ## Instruction selection (port facts measured on this core)
//!
//! `vpermw` zmm is 2 uops (p0 + p5), so a 16-entry i16 lookup done with `vpermw` would cost one
//! port-0 slot per lookup (1080 per batch, ~17% of the port-0 budget). The tables are therefore
//! stored **byte-split** (low halves at byte n, high halves at byte 16+n) and looked up
//! with `vpermb` (1 uop, p5 only) on the byte-index rows `(n, 16+n)` that
//! `transpose_f162::slice_f162_into` emits directly (`BinaryIndex32`), so the kernel has no
//! index-expansion prologue at all.
//! `vpbroadcastd zmm, m32` really is a free load (0 p0/p5 uops), so every twiddle is stored as a
//! duplicated u32.
//!
//! ## Bounds (|lane| as a multiple of q; `tests/ntt.rs::bin_asm`: `proved_bounds_9721`
//! propagates the worst case over all i16 lane values, the i32 shadow model replays the exact
//! schedule on the test inputs)
//!
//! Table entries are centered, |T| <= q/2. A twiddle multiplication `mont(a, w, w')` with
//! |w| <= q/2 satisfies |mont| <= |a| q / 2^17 + q/2 + 1 < 0.75 q for any i16 `a`; the radix-3
//! butterfly therefore adds at most 1.5 q to the (untwiddled) `a0` input.
//!
//! | after            | q = 3889 | q = 9721 |
//! |------------------|---------:|---------:|
//! | levels 0+1+2     |   1.00 q |   1.00 q |
//! | level 3          |   3.00 q |   3.00 q |
//! | level 4          |   4.50 q |   2.03 q |
//! | level 5          |   6.00 q |   3.33 q |
//! | level 6 (output) |   7.50 q |   2.30 q |
//!
//! 2^15 / q = 8.42 (3889) and 3.37 (9721), so **q = 3889 needs no Barrett at all**. For q = 9721
//! the un-twiddled `a0` input of level 4 gets the **lookup Barrett** ([`barrett_lut_i16`]) and
//! the one of level 6 the two-multiply `vpmulhrsw` Barrett ([`params::barrett_i16`]); level 5
//! is left unreduced. The lookup Barrett spends no multiply slot at all:
//!
//! ```text
//! vpsrlw  s, a, 11        p0    s = bits 11..15 of the lane = (a >> 11) & 31
//! vpermw  s, s, [wcorr]   p5    32-entry i16 table -k(s) q (vpermw reads index bits 0..4 only)
//! vpaddw  a, a, s         p05
//! ```
//!
//! (Sapphire Rapids: `vpermw` zmm is a single port-5 uop. The Tiger Lake schedule this kernel was
//! first tuned on used a 5-uop byte-split form, `vpmultishiftqb` + `vpandd` + `vpord` + `vpermb`
//! + `vpaddw`, because `vpermw` cost a port-0 slot there; it gives bit-identical results.)
//!
//! i.e. 3 uops against the two-multiply Barrett's 2 multiplies + 1 flexible. The quotient
//! window `(a >> 11) & 31` determines `a`
//! to within 2^11, so the nearest multiple of q leaves |r| <= q/2 + 2^10 = 0.579 q -- tighter
//! than the `vpmulhrsw` estimate's 0.809 q (both exhaustive over all i16), and that is exactly
//! what lets level 5 run unreduced: 3.323 q there is still inside the 3.371 q budget, while
//! 0.809 q at level 4 would have grown to 3.588 q and overflowed. Level 6 only has to keep the
//! output inside i16, so there the two-multiply Barrett is the cheaper of the two (4 uops
//! against 5, and its two port-0 slots are affordable again now that level 5 costs nothing).
//! Reductions per batch: 432 (216 at each of levels 4 and 6) instead of 648, of which only 216
//! touch the multiply port.
//!
//! Measured alternatives for q = 9721 (cache-resident cycles per ring element, 316.6 before):
//! lookup at 4 and 6 304.6, lookup at 4 only + `vpmulhrsw` at 6 **303.8**, lookup at all three
//! levels 316.3 (the level-5 reduction is what costs, not the reduction itself); folding the
//! `vpandd`/`vpord` pair into one `vpternlogd` needs a second constant register, and the only
//! one to free is q -- as a memory operand of the 1944 `vpmulhw m, q` per batch it costs more
//! than the uop it saves (306.0); splitting the five uops across the block (the `vpermb` and
//! `vpaddw` deferred to the end of the multiply block) 308.7, all five at the end 303.6.
//! `vpermw` on a 32-entry table would take the index straight from a `vpmulhrsw` quotient, but
//! it is 2 uops (p0 + p5) and needs a p0 uop to build the index: 2 port-0 again.
use crate::params::*;
use crate::simd::ntt::r3_twiddles;
pub use crate::simd::transpose_f162::BinaryIndex32;
use crate::ring::element::*;
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[((k * 2 + s2) * 3 + r) * 2 + ab]`: the 16 centered i16 values of
    /// `base_k(n) * zeta''_{2k+s2}^r * (ab == 1 ? zeta'_k : 1)`, **byte-split** so that a single
    /// `vpermb` (1 uop, port 5 - unlike `vpermw`, which is 2 uops and costs a port-0 slot on this
    /// core) does the lookup: byte n is the low half of entry n, byte 16+n the high half.
    lut: [[u8; 64]; 96],
    /// Full 512-bit constants used as memory operands / register constants by the asm tail:
    /// `[q, om, omp, bv, ms, corr, and, or, wcorr]` at byte offsets 0, 64, ... - `bv` is the
    /// `vpmulhrsw` Barrett constant of level 6, `wcorr` the 32-entry i16 `-k q` table of the
    /// level-4 lookup Barrett (`vpsrlw` + `vpermw`, see `barrett_lut_i16`); `ms`, `corr`, `and`
    /// and `or` belong to the older byte-split `vpmultishiftqb`/`vpermb` form of that lookup and
    /// are no longer read by the kernel.
    cv: [[i16; 32]; 9],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2), each i16
    /// duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw4: [[u32; 4]; 24],
    tw5: [[u32; 4]; 72],
    tw6: [[u32; 4]; 216],
    /// omega and its companion.
    om: [u32; 2],
    /// q and the vpmulhrsw Barrett constant.
    qd: u32,
    bv: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

const fn r3_pair<const Q: u16>(z: u16) -> [u32; 4] {
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let (a, b) = mont_pair::<Q>(z);
    let (c, d) = mont_pair::<Q>(z2);
    [a, b, c, d]
}

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = Params::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    let mut lut = [[0u8; 64]; 96];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        let mut s2 = 0;
        while s2 < 2 {
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            let mut r = 0;
            while r < 3 {
                let f = pow_mod(z3, r as u64, q);
                let mut ab = 0;
                while ab < 2 {
                    let extra = if ab == 0 { 1 } else { zp };
                    let mut n = 0;
                    while n < 16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let v = base * f % q * extra % q;
                        // the table holds the centred value itself; nothing is scaled into Montgomery form
                        let e = center(v, q) as u16;
                        let ti = ((k * 2 + s2) * 3 + r) * 2 + ab;
                        lut[ti][n] = e as u8;
                        lut[ti][16 + n] = (e >> 8) as u8;
                        n += 1;
                    }
                    ab += 1;
                }
                r += 1;
            }
            s2 += 1;
        }
        k += 1;
    }

    let tw4 = r3_twiddles!(24, r3_pair::<Q>, Params::<Q>::ZETA_L4);
    let tw5 = r3_twiddles!(72, r3_pair::<Q>, Params::<Q>::ZETA_L5);
    let tw6 = r3_twiddles!(216, r3_pair::<Q>, Params::<Q>::ZETA_L6);
    let (oa, ob) = mont_pair::<Q>(Params::<Q>::OMEGA);
    let mut cv = [[0i16; 32]; 9];
    let mut i = 0;
    while i < 32 {
        // the level-4 lookup Barrett's `vpermw` table: word s is -k(s) q for the window
        // s = a >> 11 (logical) = (a >> 11) & 31 (arithmetic) of the lane a.
        cv[8][i] = barrett_lut_corr(i, Q);
        cv[0][i] = Q as i16;
        cv[1][i] = Params::<Q>::to_mont(Params::<Q>::OMEGA);
        cv[2][i] = Params::<Q>::mont_pre(Params::<Q>::to_mont(Params::<Q>::OMEGA));
        cv[3][i] = barrett_v(Q);
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that
        // word, i.e. the 5-bit quotient window (a >> 11) & 31 plus 3 ignored/masked bits.
        cv[4][i] = ((16 * (i % 4) + 11) * 257) as i16;
        // the byte-split correction table: byte u (u < 32) is the low half of -k(u) q,
        // byte 32 + u its high half; the `vpermb` index is u for the low byte of every lane
        // and 32 + u for the high byte.
        let (u0, u1) = (2 * i, 2 * i + 1);
        let (b0, b1) = (lut_byte(u0, Q), lut_byte(u1, Q));
        cv[5][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[6][i] = 0x1f1f;
        cv[7][i] = 0x2000;
        i += 1;
    }
    Tables {
        lut,
        cv,
        tw4,
        tw5,
        tw6,
        om: [oa, ob],
        qd: dup(Q as i16),
        bv: dup(barrett_v(Q)),
    }
}

static T3889: Tables = build_tables::<3889>();
static T9721: Tables = build_tables::<9721>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    if Q == 3889 {
        &T3889
    } else {
        &T9721
    }
}

/// Quotient estimate of the lookup Barrett: the multiple of q subtracted from a value whose
/// 5-bit window `(a >> 11) & 31` is `s`. `a` is then known to lie in an interval of width 2^11
/// centred at `2048 * sgn + 1023.5`, and `k` is the nearest multiple of q to that centre, so
/// `|a - k q| <= q/2 + 2^10` for every i16 `a`.
const fn lut_k(s: usize, q: u16) -> i64 {
    let sgn = if s < 16 { s as i64 } else { s as i64 - 32 };
    let num = 4096 * sgn + 2047;
    let den = 2 * q as i64;
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// `-k(s) * q`, the i16 the byte-split table adds back for window `s`.
pub const fn barrett_lut_corr(s: usize, q: u16) -> i16 {
    (-lut_k(s, q) * q as i64) as i16
}

/// Byte `u` of the 64-byte `vpermb` table: the low halves of `-k(s) q` at u = s < 32, the high
/// halves at u = 32 + s.
const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

/// The kernel's level-4 reduction, lane-wise: `vpsrlw 11` (the 5-bit window `(a >> 11) & 31`,
/// as a logical shift of the lane), `vpermw` on the 32-entry i16 table of `-k q`, `vpaddw` -
/// **3 uops, not one multiply slot**, against the two-multiply Barrett's 2 multiplies
/// + 1 flexible. Exhaustively over all i16, `max |r| = 5625 = 0.579 q` for q = 9721, where
/// [`params::barrett_i16`] only reaches 0.809 q - which is what makes the level-5 reduction
/// droppable.
#[inline]
pub fn barrett_lut_i16(a: i16, q: u16) -> i16 {
    a.wrapping_add(barrett_lut_corr(((a >> 11) & 31) as usize, q))
}

/// Which of levels 4, 5 and 6 reduce their un-twiddled `a0` input with the lookup Barrett
/// ([`barrett_lut_i16`]) and which with the two-multiply one ([`params::barrett_i16`]),
/// for q = 9721; q = 3889 reduces nothing.
pub const LUT_BARRETT_LEVELS: [bool; 3] = [true, false, false];
pub const MUL_BARRETT_LEVELS: [bool; 3] = [false, false, true];

/// Does the 7.5 q growth of the un-Barretted schedule still fit in i16?
pub const fn needs_barrett(q: u16) -> bool {
    15 * (q as u32) >= 2 * 32768
}

/// Declared output bound: max |lane| of `ntt_bin_batch32`, as a multiple of q (numerator / 1000).
pub const fn output_bound_milli_q(q: u16) -> u32 {
    if needs_barrett(q) {
        2294
    } else {
        7500
    }
}

// ---------------------------------------------------------------------------------------------
// arithmetic helpers
// ---------------------------------------------------------------------------------------------

/// `vpbroadcastd zmm, m32` - one pure load uop. Written as `asm!` because LLVM otherwise
/// "recognises" the duplicated-u32 splat and rebuilds it with vpmovsxwd/vpmovdw/vinserti64x4.
#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// `vpmulhw`. stdarch's `_mm512_mulhi_epi16` is written as sext -> mul -> shr -> trunc; LLVM
/// folds the multiply but leaves a `vpmovsxwd`/`vpmovdw`/`vinserti64x4` round trip on operands it
/// cannot see through (broadcast constants), and rematerialises it inside the hot loops.
#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpmulhw {0}, {1}, {2}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// 3-uop signed Montgomery twiddle multiply: a * x mod q in (-q, q).
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm512_sub_epi16(hi, t)
}

struct C {
    q: __m512i,
    om: __m512i,
    omp: __m512i,
}

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 3).
#[inline(always)]
unsafe fn r3_folded(c: &C, a0: __m512i, t1: __m512i, t2: __m512i) -> (__m512i, __m512i, __m512i) {
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}

#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m512i {
    _mm512_load_si512(p.add(64 * j) as *const __m512i)
}
#[inline(always)]
unsafe fn st(p: *mut i16, j: usize, v: __m512i) {
    _mm512_store_si512(p.add(32 * j) as *mut __m512i, v);
}

/// Levels 4, 5 and 6 of one 27-block, entirely in registers (deep schedule).
macro_rules! mont {
    ($tw:literal, $off:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd zmm", $t, ", dword ptr [{", $tw, "}", $off, "+4]\n",
            "vpmullw zmm", $t, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{", $tw, "}", $off, "]\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm", $t, ", dword ptr [{", $tw, "}", $off, "+12]\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{", $tw, "}", $off, "+8]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $t, ", zmm", $a1, ", zmm", $a2, "\n",
            "vpmullw zmm30, zmm", $t, ", [{c}+128]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", [{c}+64]\n",
            "vpmulhw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $t, ", zmm", $t, ", zmm30\n",
        )
    };
}

macro_rules! bfly {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpaddw zmm30, zmm", $a1, ", zmm", $a2, "\n",
            "vpsubw zmm", $a2, ", zmm", $a0, ", zmm", $a2, "\n",
            "vpaddw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $a1, ", zmm", $a0, ", zmm", $a1, "\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
        )
    };
    ($a0:literal, $a1:literal, $a2:literal, $t:literal, $o0:literal, $o1:literal, $o2:literal) => {
        concat!(
            "vpaddw zmm30, zmm", $a1, ", zmm", $a2, "\n",
            "vpsubw zmm", $a2, ", zmm", $a0, ", zmm", $a2, "\n",
            "vpaddw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vmovdqa64 [{o}+", $o1, "], zmm", $a2, "\n",
            "vpsubw zmm", $a1, ", zmm", $a0, ", zmm", $a1, "\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vmovdqa64 [{o}+", $o2, "], zmm", $a1, "\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
            "vmovdqa64 [{o}+", $o0, "], zmm", $a0, "\n",
        )
    };
}

macro_rules! level4 {
    (plain $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            "vmovdqa64 zmm", $a0, ", [{i}+", $o0, "]\n",
            "vmovdqa64 zmm", $a1, ", [{i}+", $o1, "]\n",
            "vmovdqa64 zmm", $a2, ", [{i}+", $o2, "]\n",
            mont!("t4", "", $a1, $a2, $t),
        )
    };
    (lut $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            level4!(plain $a0, $a1, $a2, $o0, $o1, $o2, $t),
            // logical a >> 11 is the window (a >> 11) & 31 (bits 11..15 of the lane), and
            // vpermw only reads the low 5 bits of each word index anyway.
            "vpsrlw zmm30, zmm", $a0, ", 11\n",
            "vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
        )
    };
}

macro_rules! level6 {
    (plain $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        mont!("t6", $off, $a1, $a2, $t)
    };
    (barrett $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd zmm", $t, ", dword ptr [{t6}", $off, "+4]\n",
            "vpmullw zmm", $t, ", zmm", $a1, ", zmm", $t, "\n",
            "vpmulhrsw zmm30, zmm", $a0, ", [{c}+192]\n",
            "vpmullw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $a0, ", zmm", $a0, ", zmm30\n",
            "vpbroadcastd zmm30, dword ptr [{t6}", $off, "]\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm", $t, ", dword ptr [{t6}", $off, "+12]\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{t6}", $off, "+8]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $t, ", zmm", $a1, ", zmm", $a2, "\n",
            "vpmullw zmm30, zmm", $t, ", [{c}+128]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", [{c}+64]\n",
            "vpmulhw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $t, ", zmm", $t, ", zmm30\n",
        )
    };
}

macro_rules! tail27 {
    ($m4:ident, $m6:ident) => {
        tail27!($m4, $m6, bfly)
    };
    ($m4:ident, $m6:ident, $finish:ident) => {
        concat!(
            level4!($m4 0, 9, 18, 0, 576, 1152, 28),
            level4!($m4 3, 12, 21, 192, 768, 1344, 29),
            bfly!(0, 9, 18, 28),
            level4!($m4 6, 15, 24, 384, 960, 1536, 28),
            bfly!(3, 12, 21, 29),
            level4!($m4 1, 10, 19, 64, 640, 1216, 29),
            bfly!(6, 15, 24, 28),
            level4!($m4 4, 13, 22, 256, 832, 1408, 28),
            bfly!(1, 10, 19, 29),
            level4!($m4 7, 16, 25, 448, 1024, 1600, 29),
            bfly!(4, 13, 22, 28),
            level4!($m4 2, 11, 20, 128, 704, 1280, 28),
            bfly!(7, 16, 25, 29),
            level4!($m4 5, 14, 23, 320, 896, 1472, 29),
            bfly!(2, 11, 20, 28),
            level4!($m4 8, 17, 26, 512, 1088, 1664, 28),
            bfly!(5, 14, 23, 29),
            bfly!(8, 17, 26, 28),
            mont!("t5", "", 3, 6, 28),
            mont!("t5", "+16", 21, 24, 29),
            bfly!(0, 3, 6, 28),
            mont!("t5", "+32", 12, 15, 28),
            bfly!(18, 21, 24, 29),
            mont!("t5", "", 4, 7, 29),
            bfly!(9, 12, 15, 28),
            mont!("t5", "+16", 22, 25, 28),
            bfly!(1, 4, 7, 29),
            mont!("t5", "+32", 13, 16, 29),
            bfly!(19, 22, 25, 28),
            mont!("t5", "", 5, 8, 28),
            bfly!(10, 13, 16, 29),
            mont!("t5", "+16", 23, 26, 29),
            bfly!(2, 5, 8, 28),
            mont!("t5", "+32", 14, 17, 28),
            bfly!(20, 23, 26, 29),
            bfly!(11, 14, 17, 28),
            level6!($m6 "", 0, 1, 2, 28),
            level6!($m6 "+16", 6, 7, 8, 29),
            $finish!(0, 1, 2, 28, 0, 64, 128),
            level6!($m6 "+32", 3, 4, 5, 28),
            $finish!(6, 7, 8, 29, 192, 256, 320),
            level6!($m6 "+48", 18, 19, 20, 29),
            $finish!(3, 4, 5, 28, 384, 448, 512),
            level6!($m6 "+64", 24, 25, 26, 28),
            $finish!(18, 19, 20, 29, 576, 640, 704),
            level6!($m6 "+80", 21, 22, 23, 29),
            $finish!(24, 25, 26, 28, 768, 832, 896),
            level6!($m6 "+96", 9, 10, 11, 28),
            $finish!(21, 22, 23, 29, 960, 1024, 1088),
            level6!($m6 "+112", 15, 16, 17, 29),
            $finish!(9, 10, 11, 28, 1152, 1216, 1280),
            level6!($m6 "+128", 12, 13, 14, 28),
            $finish!(15, 16, 17, 29, 1344, 1408, 1472),
            $finish!(12, 13, 14, 28, 1536, 1600, 1664),
        )
    };
}

#[inline(always)]
unsafe fn tail27_p(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        concat!("vmovdqa64 zmm27, [{c}]\n", tail27!(plain, plain)),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        out("zmm4") _,
        out("zmm5") _,
        out("zmm6") _,
        out("zmm7") _,
        out("zmm8") _,
        out("zmm9") _,
        out("zmm10") _,
        out("zmm11") _,
        out("zmm12") _,
        out("zmm13") _,
        out("zmm14") _,
        out("zmm15") _,
        out("zmm16") _,
        out("zmm17") _,
        out("zmm18") _,
        out("zmm19") _,
        out("zmm20") _,
        out("zmm21") _,
        out("zmm22") _,
        out("zmm23") _,
        out("zmm24") _,
        out("zmm25") _,
        out("zmm26") _,
        out("zmm27") _,
        out("zmm28") _,
        out("zmm29") _,
        out("zmm30") _,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn tail27_b(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        concat!(
            "vmovdqa64 zmm27, [{c}]\n",
            tail27!(lut, barrett),
        ),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        out("zmm4") _,
        out("zmm5") _,
        out("zmm6") _,
        out("zmm7") _,
        out("zmm8") _,
        out("zmm9") _,
        out("zmm10") _,
        out("zmm11") _,
        out("zmm12") _,
        out("zmm13") _,
        out("zmm14") _,
        out("zmm15") _,
        out("zmm16") _,
        out("zmm17") _,
        out("zmm18") _,
        out("zmm19") _,
        out("zmm20") _,
        out("zmm21") _,
        out("zmm22") _,
        out("zmm23") _,
        out("zmm24") _,
        out("zmm25") _,
        out("zmm26") _,
        out("zmm27") _,
        out("zmm28") _,
        out("zmm29") _,
        out("zmm30") _,
        options(nostack)
    );
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);

/// The final butterfly without its row-major scratch stores. The values and register
/// assignments are exactly those of `bfly!`; the arranged sink consumes them immediately.
macro_rules! bfly_registers {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal,
     $o0:literal, $o1:literal, $o2:literal) => {
        bfly!($a0, $a1, $a2, $t)
    };
}

macro_rules! hybrid_store {
    (plain, $reg:literal, $off:literal) => {
        concat!("vmovdqa64 [{acc}+", $off, "], zmm", $reg, "\n")
    };
    (reduce, $reg:literal, $off:literal) => {
        concat!(
            "vpmaddwd zmm31, zmm", $reg, ", [{fold}]\n",
            "vptestmd k1, zmm", $reg, ", [{fold}+64]\n",
            "vpaddd zmm31{{k1}}, zmm31, [{fold}+128]\n",
            "vmovdqa64 [{acc}+", $off, "], zmm31\n",
        )
    };
}

/// Three 8x8 qword transposes and one 4x4 128-bit transpose, in the same output order
/// as hybrid_commit::tr8/tr4. Each two-source pair first writes the free register and
/// then overwrites its second input, so only zmm28 is needed as extra transpose storage.
/// zmm29/30 are the base/tail sums; zmm31 and k1 implement the existing centered fold-back.
macro_rules! hybrid_register_sink {
    ($mode:ident) => {
        concat!(
            "vpunpcklqdq zmm28, zmm0, zmm2\n",
            "vpunpckhqdq zmm2, zmm0, zmm2\n",
            "vpunpcklqdq zmm0, zmm1, zmm6\n",
            "vpunpckhqdq zmm6, zmm1, zmm6\n",
            "vpunpcklqdq zmm1, zmm8, zmm7\n",
            "vpunpckhqdq zmm7, zmm8, zmm7\n",
            "vpunpcklqdq zmm8, zmm3, zmm5\n",
            "vpunpckhqdq zmm5, zmm3, zmm5\n",
            "vshufi64x2 zmm3, zmm28, zmm0, 136\n",
            "vshufi64x2 zmm0, zmm28, zmm0, 221\n",
            "vshufi64x2 zmm28, zmm1, zmm8, 136\n",
            "vshufi64x2 zmm8, zmm1, zmm8, 221\n",
            "vshufi64x2 zmm1, zmm2, zmm6, 136\n",
            "vshufi64x2 zmm6, zmm2, zmm6, 221\n",
            "vshufi64x2 zmm2, zmm7, zmm5, 136\n",
            "vshufi64x2 zmm5, zmm7, zmm5, 221\n",
            "vshufi64x2 zmm7, zmm3, zmm28, 136\n",
            "vshufi64x2 zmm28, zmm3, zmm28, 221\n",
            "vshufi64x2 zmm3, zmm1, zmm2, 136\n",
            "vshufi64x2 zmm2, zmm1, zmm2, 221\n",
            "vshufi64x2 zmm1, zmm0, zmm8, 136\n",
            "vshufi64x2 zmm8, zmm0, zmm8, 221\n",
            "vshufi64x2 zmm0, zmm6, zmm5, 136\n",
            "vshufi64x2 zmm5, zmm6, zmm5, 221\n",
            "vmovdqa64 zmm29, [{acc}+0]\n",
            "vmovdqa64 zmm30, [{acc}+64]\n",
            "vpdpwssd zmm29, zmm7, zmmword ptr [{key}+0]\n",
            "vpdpwssd zmm30, zmm7, zmmword ptr [{key}+512]\n",
            "vpdpwssd zmm29, zmm3, zmmword ptr [{key}+64]\n",
            "vpdpwssd zmm30, zmm3, zmmword ptr [{key}+576]\n",
            "vpdpwssd zmm29, zmm1, zmmword ptr [{key}+128]\n",
            "vpdpwssd zmm30, zmm1, zmmword ptr [{key}+640]\n",
            "vpdpwssd zmm29, zmm0, zmmword ptr [{key}+192]\n",
            "vpdpwssd zmm30, zmm0, zmmword ptr [{key}+704]\n",
            "vpdpwssd zmm29, zmm28, zmmword ptr [{key}+256]\n",
            "vpdpwssd zmm30, zmm28, zmmword ptr [{key}+768]\n",
            "vpdpwssd zmm29, zmm2, zmmword ptr [{key}+320]\n",
            "vpdpwssd zmm30, zmm2, zmmword ptr [{key}+832]\n",
            "vpdpwssd zmm29, zmm8, zmmword ptr [{key}+384]\n",
            "vpdpwssd zmm30, zmm8, zmmword ptr [{key}+896]\n",
            "vpdpwssd zmm29, zmm5, zmmword ptr [{key}+448]\n",
            "vpdpwssd zmm30, zmm5, zmmword ptr [{key}+960]\n",
            hybrid_store!($mode, 29, 0),
            hybrid_store!($mode, 30, 64),
            "vpunpcklqdq zmm28, zmm4, zmm18\n",
            "vpunpckhqdq zmm18, zmm4, zmm18\n",
            "vpunpcklqdq zmm4, zmm20, zmm19\n",
            "vpunpckhqdq zmm19, zmm20, zmm19\n",
            "vpunpcklqdq zmm20, zmm24, zmm26\n",
            "vpunpckhqdq zmm26, zmm24, zmm26\n",
            "vpunpcklqdq zmm24, zmm25, zmm21\n",
            "vpunpckhqdq zmm21, zmm25, zmm21\n",
            "vshufi64x2 zmm25, zmm28, zmm4, 136\n",
            "vshufi64x2 zmm4, zmm28, zmm4, 221\n",
            "vshufi64x2 zmm28, zmm20, zmm24, 136\n",
            "vshufi64x2 zmm24, zmm20, zmm24, 221\n",
            "vshufi64x2 zmm20, zmm18, zmm19, 136\n",
            "vshufi64x2 zmm19, zmm18, zmm19, 221\n",
            "vshufi64x2 zmm18, zmm26, zmm21, 136\n",
            "vshufi64x2 zmm21, zmm26, zmm21, 221\n",
            "vshufi64x2 zmm26, zmm25, zmm28, 136\n",
            "vshufi64x2 zmm28, zmm25, zmm28, 221\n",
            "vshufi64x2 zmm25, zmm20, zmm18, 136\n",
            "vshufi64x2 zmm18, zmm20, zmm18, 221\n",
            "vshufi64x2 zmm20, zmm4, zmm24, 136\n",
            "vshufi64x2 zmm24, zmm4, zmm24, 221\n",
            "vshufi64x2 zmm4, zmm19, zmm21, 136\n",
            "vshufi64x2 zmm21, zmm19, zmm21, 221\n",
            "vmovdqa64 zmm29, [{acc}+128]\n",
            "vmovdqa64 zmm30, [{acc}+192]\n",
            "vpdpwssd zmm29, zmm26, zmmword ptr [{key}+1024]\n",
            "vpdpwssd zmm30, zmm26, zmmword ptr [{key}+1536]\n",
            "vpdpwssd zmm29, zmm25, zmmword ptr [{key}+1088]\n",
            "vpdpwssd zmm30, zmm25, zmmword ptr [{key}+1600]\n",
            "vpdpwssd zmm29, zmm20, zmmword ptr [{key}+1152]\n",
            "vpdpwssd zmm30, zmm20, zmmword ptr [{key}+1664]\n",
            "vpdpwssd zmm29, zmm4, zmmword ptr [{key}+1216]\n",
            "vpdpwssd zmm30, zmm4, zmmword ptr [{key}+1728]\n",
            "vpdpwssd zmm29, zmm28, zmmword ptr [{key}+1280]\n",
            "vpdpwssd zmm30, zmm28, zmmword ptr [{key}+1792]\n",
            "vpdpwssd zmm29, zmm18, zmmword ptr [{key}+1344]\n",
            "vpdpwssd zmm30, zmm18, zmmword ptr [{key}+1856]\n",
            "vpdpwssd zmm29, zmm24, zmmword ptr [{key}+1408]\n",
            "vpdpwssd zmm30, zmm24, zmmword ptr [{key}+1920]\n",
            "vpdpwssd zmm29, zmm21, zmmword ptr [{key}+1472]\n",
            "vpdpwssd zmm30, zmm21, zmmword ptr [{key}+1984]\n",
            hybrid_store!($mode, 29, 128),
            hybrid_store!($mode, 30, 192),
            "vpunpcklqdq zmm28, zmm23, zmm22\n",
            "vpunpckhqdq zmm22, zmm23, zmm22\n",
            "vpunpcklqdq zmm23, zmm9, zmm11\n",
            "vpunpckhqdq zmm11, zmm9, zmm11\n",
            "vpunpcklqdq zmm9, zmm10, zmm15\n",
            "vpunpckhqdq zmm15, zmm10, zmm15\n",
            "vpunpcklqdq zmm10, zmm17, zmm16\n",
            "vpunpckhqdq zmm16, zmm17, zmm16\n",
            "vshufi64x2 zmm17, zmm28, zmm23, 136\n",
            "vshufi64x2 zmm23, zmm28, zmm23, 221\n",
            "vshufi64x2 zmm28, zmm9, zmm10, 136\n",
            "vshufi64x2 zmm10, zmm9, zmm10, 221\n",
            "vshufi64x2 zmm9, zmm22, zmm11, 136\n",
            "vshufi64x2 zmm11, zmm22, zmm11, 221\n",
            "vshufi64x2 zmm22, zmm15, zmm16, 136\n",
            "vshufi64x2 zmm16, zmm15, zmm16, 221\n",
            "vshufi64x2 zmm15, zmm17, zmm28, 136\n",
            "vshufi64x2 zmm28, zmm17, zmm28, 221\n",
            "vshufi64x2 zmm17, zmm9, zmm22, 136\n",
            "vshufi64x2 zmm22, zmm9, zmm22, 221\n",
            "vshufi64x2 zmm9, zmm23, zmm10, 136\n",
            "vshufi64x2 zmm10, zmm23, zmm10, 221\n",
            "vshufi64x2 zmm23, zmm11, zmm16, 136\n",
            "vshufi64x2 zmm16, zmm11, zmm16, 221\n",
            "vmovdqa64 zmm29, [{acc}+256]\n",
            "vmovdqa64 zmm30, [{acc}+320]\n",
            "vpdpwssd zmm29, zmm15, zmmword ptr [{key}+2048]\n",
            "vpdpwssd zmm30, zmm15, zmmword ptr [{key}+2560]\n",
            "vpdpwssd zmm29, zmm17, zmmword ptr [{key}+2112]\n",
            "vpdpwssd zmm30, zmm17, zmmword ptr [{key}+2624]\n",
            "vpdpwssd zmm29, zmm9, zmmword ptr [{key}+2176]\n",
            "vpdpwssd zmm30, zmm9, zmmword ptr [{key}+2688]\n",
            "vpdpwssd zmm29, zmm23, zmmword ptr [{key}+2240]\n",
            "vpdpwssd zmm30, zmm23, zmmword ptr [{key}+2752]\n",
            "vpdpwssd zmm29, zmm28, zmmword ptr [{key}+2304]\n",
            "vpdpwssd zmm30, zmm28, zmmword ptr [{key}+2816]\n",
            "vpdpwssd zmm29, zmm22, zmmword ptr [{key}+2368]\n",
            "vpdpwssd zmm30, zmm22, zmmword ptr [{key}+2880]\n",
            "vpdpwssd zmm29, zmm10, zmmword ptr [{key}+2432]\n",
            "vpdpwssd zmm30, zmm10, zmmword ptr [{key}+2944]\n",
            "vpdpwssd zmm29, zmm16, zmmword ptr [{key}+2496]\n",
            "vpdpwssd zmm30, zmm16, zmmword ptr [{key}+3008]\n",
            hybrid_store!($mode, 29, 256),
            hybrid_store!($mode, 30, 320),
            "vshufi64x2 zmm0, zmm12, zmm14, 136\n",
            "vshufi64x2 zmm1, zmm12, zmm14, 221\n",
            "vshufi64x2 zmm2, zmm13, zmm13, 136\n",
            "vshufi64x2 zmm3, zmm13, zmm13, 221\n",
            "vshufi64x2 zmm4, zmm0, zmm2, 136\n",
            "vshufi64x2 zmm5, zmm1, zmm3, 136\n",
            "vshufi64x2 zmm6, zmm0, zmm2, 221\n",
            "vshufi64x2 zmm7, zmm1, zmm3, 221\n",
            "vmovdqa64 zmm29, [{acc}+384]\n",
            "vmovdqa64 zmm30, [{acc}+448]\n",
            "vpdpwssd zmm29, zmm4, zmmword ptr [{key}+3072]\n",
            "vpdpwssd zmm30, zmm4, zmmword ptr [{key}+3328]\n",
            "vpdpwssd zmm29, zmm5, zmmword ptr [{key}+3136]\n",
            "vpdpwssd zmm30, zmm5, zmmword ptr [{key}+3392]\n",
            "vpdpwssd zmm29, zmm6, zmmword ptr [{key}+3200]\n",
            "vpdpwssd zmm30, zmm6, zmmword ptr [{key}+3456]\n",
            "vpdpwssd zmm29, zmm7, zmmword ptr [{key}+3264]\n",
            "vpdpwssd zmm30, zmm7, zmmword ptr [{key}+3520]\n",
            hybrid_store!($mode, 29, 384),
            hybrid_store!($mode, 30, 448),
        )
    };
}

#[repr(C, align(64))]
struct HybridFold([[i32; 16]; 3]);

const fn hybrid_fold<const Q: u16>() -> HybridFold {
    let r = (65536 % Q as i32) - Q as i32;
    HybridFold([
        [1 | ((r as u16 as i32) << 16); 16],
        [0x8000; 16],
        [r; 16],
    ])
}

/// The arranged hybrid sink fused with the unchanged 27-slot NTT tail. No transformed
/// value is stored and reloaded between the butterfly and its multiply-accumulate.
#[inline(always)]
unsafe fn tail27_hybrid<const Q: u16, const RED: bool>(
    bp: *const i16, t4: *const u32, t5: *const u32, t6: *const u32,
    cv: *const i16, key: *const i16, acc: *mut i32,
) {
    macro_rules! run {
        ($m4:ident, $m6:ident, $red:ident) => {
            core::arch::asm!(
                concat!(
                    "vmovdqa64 zmm27, [{c}]\n",
                    tail27!($m4, $m6, bfly_registers),
                    hybrid_register_sink!($red),
                    "/* {fold} */\n",
                ),
                i = in(reg) bp, t4 = in(reg) t4, t5 = in(reg) t5, t6 = in(reg) t6,
                c = in(reg) cv, key = in(reg) key, acc = in(reg) acc,
                fold = in(reg) &const { hybrid_fold::<Q>() },
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                out("zmm3") _,
                out("zmm4") _,
                out("zmm5") _,
                out("zmm6") _,
                out("zmm7") _,
                out("zmm8") _,
                out("zmm9") _,
                out("zmm10") _,
                out("zmm11") _,
                out("zmm12") _,
                out("zmm13") _,
                out("zmm14") _,
                out("zmm15") _,
                out("zmm16") _,
                out("zmm17") _,
                out("zmm18") _,
                out("zmm19") _,
                out("zmm20") _,
                out("zmm21") _,
                out("zmm22") _,
                out("zmm23") _,
                out("zmm24") _,
                out("zmm25") _,
                out("zmm26") _,
                out("zmm27") _,
                out("zmm28") _,
                out("zmm29") _,
                out("zmm30") _,
                out("zmm31") _,
                out("k1") _,
                options(nostack)
            )
        };
    }
    if needs_barrett(Q) {
        if RED { run!(lut, barrett, reduce); } else { run!(lut, barrett, plain); }
    } else if RED { run!(plain, plain, reduce); } else { run!(plain, plain, plain); }
}


// ---------------------------------------------------------------------------------------------
// where the finished blocks go
// ---------------------------------------------------------------------------------------------

/// Consumer of the transform's output, one 27-slot block at a time.
///
/// The kernel finishes the 648 slots as 24 independent blocks of 27: block `blk` holds slots
/// `27 blk .. 27 blk + 27` and is produced by a single `asm!` block out of 27 resident zmm
/// registers. The sink says where those 27 vectors are stored ([`dst`](BlockSink::dst)) and is
/// handed them the instant they are ([`block`](BlockSink::block)), which lets a consumer read a
/// block while it is still in L1 rather than after the whole 41 KB batch has been written.
///
/// Both methods are monomorphised into the kernel, so a sink that does nothing costs nothing.
/// A sink whose `block` uses AVX-512 must carry a `#[target_feature]` attribute covering the
/// intrinsics it uses, or it will not be inlined into the kernel.
///
/// # Safety
/// `dst` must return a 64-byte aligned pointer to 1728 writable bytes (27 vectors); the kernel
/// writes them and then calls `block` with the same pointer.
pub trait BlockSink {
    /// The fused consumer's opaque assembly must stay in one loop body.
    const REGISTER_SINK: bool = false;
    /// Where block `blk` is to be stored.
    unsafe fn dst(&mut self, blk: usize) -> *mut i16;
    /// Called once the 27 vectors of block `blk` are stored at `dst`.
    unsafe fn block(&mut self, blk: usize, dst: *const i16);
    /// Finish a 27-slot tail and consume it. Ordinary sinks retain the row-major staging;
    /// a fused sink may override this while preserving the transform's arithmetic.
    /// The input and twiddle pointers have the same alignment and extent as tail27_p/b.
    #[inline(always)]
    unsafe fn tail(&mut self, blk: usize, bp: *const i16, t4: *const u32,
                   t5: *const u32, t6: *const u32, cv: *const i16, bar: bool) {
        let out = self.dst(blk);
        if bar { tail27_b(bp, out, t4, t5, t6, cv); }
        else { tail27_p(bp, out, t4, t5, t6, cv); }
        self.block(blk, out);
    }
}

/// The sink of the plain entry points: block `blk` goes to its own place in the output batch and
/// nothing further happens, so the kernel is exactly the loop it would be without the hook.
pub struct OutSink(pub *mut i16);

impl BlockSink for OutSink {
    #[inline(always)]
    unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
        self.0.add(32 * 27 * blk)
    }
    #[inline(always)]
    unsafe fn block(&mut self, _blk: usize, _dst: *const i16) {}
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn ntt_core<const Q: u16, S: BlockSink>(input: &BinaryIndex32, sink: &mut S) {
    let t = tables::<Q>();
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
    };
    let bar = needs_barrett(Q);

    // The caller already holds the `vpermb` byte-index rows; the kernel reads them straight.
    let ip: *const u8 = input.rows.as_ptr() as *const u8;
    let mut blk: core::mem::MaybeUninit<Blk> = core::mem::MaybeUninit::uninit();
    let bp = blk.as_mut_ptr() as *mut i16;

    for k in 0..4 {
        let lut = t.lut.as_ptr().add(12 * k) as *const u8;
        let l = |s2: usize, r: usize, ab: usize| -> __m512i {
            _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
        };
        let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
        let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
        let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));

        // levels 0+1+2 (table lookups) fused with level 3 (omega multiply only).
        // The s2 = 1 sibling's level-3 root is zeta6 times the s2 = 0 one (zeta6 = -omega^2), so
        // its twiddled inputs are zeta6^r (x_r - y_r) on the s2 = 0 lookups: negating r = 1 and
        // rotating the butterfly outputs by two absorbs the zeta6^r, and the sibling needs no
        // lookups of its own (6 `vpermb` per row instead of 10). Same residues, same bounds.
        for i in 0..27 {
            let n0 = ldb(ip, i);
            let n0h = ldb(ip, i + 81);
            let n1 = ldb(ip, i + 27);
            let n1h = ldb(ip, i + 108);
            let n2 = ldb(ip, i + 54);
            let n2h = ldb(ip, i + 135);

            let x = _mm512_permutexvar_epi8(n0, l000);
            let y = _mm512_permutexvar_epi8(n0h, l001);
            let a0 = _mm512_add_epi16(x, y);
            let b0 = _mm512_sub_epi16(x, y);

            let x = _mm512_permutexvar_epi8(n1, l010);
            let y = _mm512_permutexvar_epi8(n1h, l011);
            let a1 = _mm512_add_epi16(x, y);
            let b1 = _mm512_sub_epi16(y, x);

            let x = _mm512_permutexvar_epi8(n2, l020);
            let y = _mm512_permutexvar_epi8(n2h, l021);
            let a2 = _mm512_add_epi16(x, y);
            let b2 = _mm512_sub_epi16(x, y);

            let (u0, u1, u2) = r3_folded(&c, a0, a1, a2);
            let (v0, v1, v2) = r3_folded(&c, b0, b1, b2);
            st(bp, i, u0);
            st(bp, i + 27, u1);
            st(bp, i + 54, u2);
            st(bp, i + 81, v2);
            st(bp, i + 108, v0);
            st(bp, i + 135, v1);
        }

        // levels 4, 5 and 6: one register-resident asm block per 27-block, handed to the sink.
        let cv = t.cv.as_ptr() as *const i16;
        // LLVM assigns a very small cost to an opaque asm block. For the fused consumer it
        // consequently unrolls all six large tails, expanding the hot function from about
        // 6 KB to over 32 KB. Keep that loop's trip count opaque to the optimizer only:
        // black_box is the identity, so every call still executes exactly six blocks.
        let tail_blocks = if S::REGISTER_SINK { core::hint::black_box(6usize) } else { 6 };
        for j in 0..tail_blocks {
            let kk = 6 * k + j;
            let bpj = bp.add(32 * 27 * j);
            // k is 0..4 and black_box(6) is still 6, hence kk < 24. Pointer arithmetic
            // avoids introducing bounds checks just because the compiler cannot see that.
            let (t4, t5, t6) = (
                t.tw4.as_ptr().add(kk).cast::<u32>(),
                t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
            );
            sink.tail(kk, bpj, t4, t5, t6, cv, bar);
        }
    }
}

/// Forward NTT of 32 binary polynomials: `out.v[j][p] = a_p(psi^SLOT_EXP[j]) mod q`, lazily
/// reduced (see the bound table above).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32<const Q: u16>(input: &BinaryIndex32, out: &mut Batch32) {
    ntt_core::<Q, _>(input, &mut OutSink(out.v.as_mut_ptr() as *mut i16));
    out.representation = Representation::Ntt;
}

/// The same transform with the output handed to `sink` block by block instead of being written to
/// a `Batch32`, for consumers that want each 27-slot block while it is still in L1
/// ([`crate::simd::commit`]).
///
/// # Safety
/// See [`BlockSink`]: `sink.dst` must give 27 writable 64-byte aligned vectors per block.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32_sink<const Q: u16, S: BlockSink>(
    input: &BinaryIndex32,
    sink: &mut S,
) {
    ntt_core::<Q, S>(input, sink);
}

struct HybridSink<const Q: u16, const RED: bool> {
    key: *const i16,
    acc: *mut i32,
}

impl<const Q: u16, const RED: bool> BlockSink for HybridSink<Q, RED> {
    const REGISTER_SINK: bool = true;
    unsafe fn dst(&mut self, _: usize) -> *mut i16 { unreachable!("fused tail has no scratch") }
    unsafe fn block(&mut self, _: usize, _: *const i16) { unreachable!("fused tail consumes registers") }
    #[inline(always)]
    unsafe fn tail(&mut self, blk: usize, bp: *const i16, t4: *const u32,
                   t5: *const u32, t6: *const u32, cv: *const i16, _: bool) {
        tail27_hybrid::<Q, RED>(bp, t4, t5, t6, cv,
            self.key.add(32 * 56 * blk), self.acc.add(16 * 8 * blk));
    }
}

/// Binary NTT into the arranged two-stream hybrid accumulator, with no row-major output
/// scratch. The original NTT entry point remains available as a differential reference.
///
/// # Safety
/// AVX-512F/BW/VL/VBMI/VBMI2/VNNI and GFNI are required. `key` addresses 24*56 aligned
/// 64-byte key vectors in the arranged hybrid format; `acc` addresses 24*8 aligned i32
/// vectors. The caller must retain the existing accumulator bound and fold-back schedule.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32_hybrid<const Q: u16, const RED: bool>(
    input: &BinaryIndex32, key: *const i16, acc: *mut i32,
) {
    assert!(Q == 3889 || Q == 9721, "the hybrid commitment's primes");
    ntt_core::<Q, _>(input, &mut HybridSink::<Q, RED> { key, acc });
}
