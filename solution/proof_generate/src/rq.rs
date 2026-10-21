//! Arithmetic in `R_Q = Z_Q[X]/(X^648 - X^324 + 1)` for an arbitrary integer modulus.
//!
//! The library's kernels instantiate their modular arithmetic per prime at compile time, so they
//! only run over the seven primes they were built for. The scheme does not need that: the
//! commitment lives in `R_Q` for whatever `Q` the proof names, and a verifier that is not scored
//! can afford schoolbook `u128` lanes. This module is the whole of `R_Q` the checker uses — the
//! public matrix, the two ring products the opening check compares, and the wire form of a ring
//! element.
use labinius::challenge::ShortChallenge;
use labinius::fields::scalar::F162;

/// Degree of the ring. One ring element is 648 coefficients modulo `Q`.
pub const N: usize = 648;
/// The middle exponent of `X^648 - X^324 + 1`, which the reduction folds onto.
pub const SPLIT: usize = N / 2;

/// Largest modulus this arithmetic is bounded for.
///
/// A folded coefficient travels as an `i16`, so `|v| <= 2^15` and a product `a v` with
/// `a < 2^96` stays below `2^111`. One ring element contributes at most `N` such products to a
/// coefficient of the accumulator, and [`ACCUMULATE`] elements pass between two reductions, so
/// the accumulator stays below `64 * 648 * 2^111 < 2^127` and never leaves `i128`. The bound is
/// chosen above the `2^90.9` product of all seven of the library's primes, so every modulus the
/// prime list could reach is still accepted.
pub const MAX_MODULUS: u128 = 1 << 96;

/// Ring elements between two reductions of the accumulator in [`a_times_v`]; see
/// [`MAX_MODULUS`].
const ACCUMULATE: usize = 64;

/// The seed the public matrix is expanded from. Fixed, and in particular independent of
/// `LABINIUS_SEED`: `A` is public parameter data, not instance data, so a prover may precompute
/// it.
pub const MATRIX_SEED: [u8; 32] = [0x5a; 32];

/// Domain separator of the matrix expansion.
const MATRIX_DOMAIN: &[u8] = b"labinius-fast/ajtai-matrix/v1";

/// Bits one residue modulo `q` occupies: the bit length of `q - 1`.
pub fn modulus_bits(q: u128) -> u32 {
    assert!(q >= 2, "a modulus is at least 2");
    128 - (q - 1).leading_zeros()
}

/// Bytes one residue modulo `q` takes on the wire, `ceil(log2(q) / 8)`.
pub fn coefficient_bytes(q: u128) -> usize {
    modulus_bits(q).div_ceil(8) as usize
}

/// A byte stream of one element's expansion, buffered so that a five-byte draw is not a BLAKE3
/// call. Buffering is invisible to the output: the stream is consumed strictly in order and no
/// byte is ever skipped.
struct Expansion {
    reader: blake3::OutputReader,
    buffer: [u8; 1 << 13],
    position: usize,
}

impl Expansion {
    /// The expansion of element `index` of the matrix for `q` over `elements` elements,
    /// positioned at its first byte.
    fn new(q: u128, elements: usize, index: usize) -> Expansion {
        let mut hasher = blake3::Hasher::new_keyed(&MATRIX_SEED);
        hasher.update(MATRIX_DOMAIN);
        hasher.update(&q.to_le_bytes());
        hasher.update(&(elements as u64).to_le_bytes());
        hasher.update(&(index as u64).to_le_bytes());
        let mut expansion = Expansion {
            reader: hasher.finalize_xof(),
            buffer: [0u8; 1 << 13],
            position: 0,
        };
        expansion.reader.fill(&mut expansion.buffer);
        expansion
    }

    /// The next `width` bytes as a little-endian integer, `width <= 16`. Sixteen bytes are read
    /// and only `width` consumed, which is the same value once the caller masks to
    /// [`modulus_bits`]: the bytes above the field sit above that mask.
    fn draw(&mut self, width: usize) -> u128 {
        if self.position + 16 > self.buffer.len() {
            let rest = self.buffer.len() - self.position;
            self.buffer.copy_within(self.position.., 0);
            self.position = 0;
            self.reader.fill(&mut self.buffer[rest..]);
        }
        let bytes = &self.buffer[self.position..self.position + 16];
        self.position += width;
        u128::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// The public matrix `A`: `elements` elements of `R_Q`, flattened, element `i`'s coefficient `t`
/// at index `N i + t`, every entry in `[0, q)`.
///
/// Element `i`'s coefficients come from the BLAKE3 extendable output keyed by [`MATRIX_SEED`]
/// and updated with [`MATRIX_DOMAIN`], the sixteen little-endian bytes of `q`, the eight
/// little-endian bytes of `elements`, and the eight little-endian bytes of `i`. The coefficients
/// come out in index order; each reads [`coefficient_bytes`] bytes of that stream little-endian,
/// keeps the low [`modulus_bits`] of them, and is rejected and redrawn while the value is at
/// least `q`. Rejection costs under one extra draw per coefficient and leaves every coefficient
/// exactly uniform on `[0, q)`; a power-of-two modulus never rejects.
///
/// One stream per element rather than one for the whole matrix is what keeps the derivation
/// parallel on both sides: a rejection shifts the stream, so a single stream would have to be
/// walked in order, and the prover pays for this inside the timed region.
///
/// This is a definition in `R_Q`, which is what makes the opening check modulus-agnostic. The
/// library's own `PublicParameters::from_seed` draws `A` per prime directly in that prime's NTT
/// domain, which is not the reduction of any single mod-`Q` object, so a prover that commits
/// through the limb kernels derives its own representation from this one: reduce modulo each
/// prime and transform. `A` depends on the fixed seed and on `q` alone, never on
/// `LABINIUS_SEED`, so a prover may precompute it.
pub fn ajtai_matrix(q: u128, elements: usize) -> Vec<u128> {
    let width = coefficient_bytes(q);
    let bits = modulus_bits(q);
    let mask = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = elements.div_ceil(workers);
    let mut matrix = vec![0u128; elements * N];
    std::thread::scope(|scope| {
        for (block, base) in matrix.chunks_mut(per * N).zip((0..).step_by(per)) {
            scope.spawn(move || {
                for (i, element) in block.chunks_mut(N).enumerate() {
                    let mut expansion = Expansion::new(q, elements, base + i);
                    for slot in element.iter_mut() {
                        loop {
                            let candidate = expansion.draw(width) & mask;
                            if candidate < q {
                                *slot = candidate;
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    matrix
}

/// Fold a raw polynomial product of degree below `2 N - 1` into `R_Q`, using
/// `X^648 = X^324 - 1`, and reduce it to the 648 coefficients in `[0, q)`.
///
/// The high coefficients are taken most significant first, each reduced before it is folded
/// down, so index `k` below `N` receives at most two contributions of magnitude below `q` and
/// the caller's bound on `raw` only has to leave `2 q` of room.
pub fn fold_into_ring(raw: &mut [i128], q: u128) -> Vec<u128> {
    assert_eq!(raw.len(), 2 * N - 1, "a raw product is 2 N - 1 coefficients");
    let modulus = q as i128;
    for k in (N..2 * N - 1).rev() {
        let value = raw[k].rem_euclid(modulus);
        raw[k] = 0;
        raw[k - SPLIT] += value;
        raw[k - N] -= value;
    }
    raw[..N]
        .iter()
        .map(|&x| x.rem_euclid(modulus) as u128)
        .collect()
}

/// One product in `R_Q` of a residue vector by a signed one, for the tests that pin
/// [`fold_into_ring`] and the two accumulating products against the library's
/// `scalar::mul_mod_phi`. The caller keeps `N max|a| max|b|` below `2^127`.
#[cfg(test)]
pub fn mul(a: &[u128], b: &[i64], q: u128) -> Vec<u128> {
    assert!(a.len() == N && b.len() == N, "both factors are N coefficients");
    let mut raw = vec![0i128; 2 * N - 1];
    for (j, &c) in b.iter().enumerate() {
        for (i, &x) in a.iter().enumerate() {
            raw[i + j] += (x as i128) * (c as i128);
        }
    }
    fold_into_ring(&mut raw, q)
}

/// `A v = sum_i A_i v_i` in `R_Q`, schoolbook.
///
/// `a` is [`ajtai_matrix`] and `v` the folded witness in the same flattened order. The inner
/// loop is one 648-coefficient shifted multiply-add per nonzero coefficient of `v`; the
/// accumulator is reduced every [`ACCUMULATE`] ring elements, which is what keeps it inside
/// `i128` for every modulus below [`MAX_MODULUS`].
pub fn a_times_v(a: &[u128], v: &[i16], q: u128) -> Vec<u128> {
    assert_eq!(a.len(), v.len(), "the matrix and the fold differ in length");
    assert_eq!(a.len() % N, 0, "the matrix is a whole number of elements");
    let modulus = q as i128;
    let mut raw = vec![0i128; 2 * N - 1];
    for (e, (element, fold)) in a.chunks_exact(N).zip(v.chunks_exact(N)).enumerate() {
        for (j, &c) in fold.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let weight = c.unsigned_abs() as u128;
            let window = &mut raw[j..j + N];
            if c > 0 {
                for (slot, &x) in window.iter_mut().zip(element) {
                    *slot += (x * weight) as i128;
                }
            } else {
                for (slot, &x) in window.iter_mut().zip(element) {
                    *slot -= (x * weight) as i128;
                }
            }
        }
        if (e + 1) % ACCUMULATE == 0 {
            for slot in raw.iter_mut() {
                *slot = slot.rem_euclid(modulus);
            }
        }
    }
    fold_into_ring(&mut raw, q)
}

/// `sum_j c_j C_j` in `R_Q`, the folded commitment.
///
/// `columns` holds the `challenges.len()` commitment columns flattened, each `N` coefficients in
/// `[0, q)`. A challenge is a weight-28 signed element of the subring `R_162` and sits in
/// `R_648` as `c(-X^4)`: its coefficient of `X^{4 m}` is the sign of position `m` times
/// `(-1)^m`, and every other coefficient is zero. Each term is therefore one shifted add of 648
/// coefficients, and the whole combination is 28 of them per column.
pub fn challenge_combination(columns: &[u128], challenges: &[ShortChallenge], q: u128) -> Vec<u128> {
    assert_eq!(
        columns.len(),
        challenges.len() * N,
        "one column per challenge"
    );
    let modulus = q as i128;
    let mut raw = vec![0i128; 2 * N - 1];
    for (j, (column, challenge)) in columns.chunks_exact(N).zip(challenges).enumerate() {
        for t in 0..challenge.weight {
            let m = challenge.positions[t] as usize;
            let sign = (1 - 2 * ((challenge.signs >> t) & 1) as i32) * if m % 2 == 0 { 1 } else { -1 };
            let window = &mut raw[4 * m..4 * m + N];
            if sign > 0 {
                for (slot, &x) in window.iter_mut().zip(column) {
                    *slot += x as i128;
                }
            } else {
                for (slot, &x) in window.iter_mut().zip(column) {
                    *slot -= x as i128;
                }
            }
        }
        if (j + 1) % ACCUMULATE == 0 {
            for slot in raw.iter_mut() {
                *slot = slot.rem_euclid(modulus);
            }
        }
    }
    fold_into_ring(&mut raw, q)
}

/// `A W` in `R_Q` for one binary column of the witness.
///
/// Ring element `m` of a column is binary: its coefficient `4 p + k` is bit `p` of field element
/// `4 m + k`. The product is therefore one shifted add of `A_m`'s coefficients per set bit, with
/// no multiplication anywhere. The accumulator takes `elements * N` terms below `q` and
/// [`fold_into_ring`] wants `2 q` of room on top, which the assertion checks.
pub fn commit_binary_column(matrix: &[u128], column: &[F162], q: u128) -> Vec<u128> {
    let elements = column.len() / 4;
    assert_eq!(matrix.len(), elements * N, "the matrix does not match the column");
    assert!(
        (elements * N) as u128 + 2 <= (1u128 << 127) / q,
        "the accumulator would overflow at this modulus and column length"
    );
    let mut raw = vec![0i128; 2 * N - 1];
    for (m, packed) in column.chunks_exact(4).enumerate() {
        let element = &matrix[m * N..(m + 1) * N];
        for (k, x) in packed.iter().enumerate() {
            for p in 0..N / 4 {
                if (x.0[p >> 6] >> (p & 63)) & 1 == 0 {
                    continue;
                }
                let window = &mut raw[4 * p + k..4 * p + k + N];
                for (slot, &a) in window.iter_mut().zip(element) {
                    *slot += a as i128;
                }
            }
        }
    }
    fold_into_ring(&mut raw, q)
}

/// Ring elements as residues modulo `q`, [`coefficient_bytes`] little-endian bytes each, in
/// index order and with no header: the shape comes from the parameters the verifier holds.
pub fn pack_ring_elements(coefficients: &[u128], q: u128) -> Vec<u8> {
    let width = coefficient_bytes(q);
    let mut out = Vec::with_capacity(width * coefficients.len());
    for &x in coefficients {
        debug_assert!(x < q, "a coefficient is not reduced");
        out.extend_from_slice(&x.to_le_bytes()[..width]);
    }
    out
}

/// The inverse of [`pack_ring_elements`] for `count` coefficients, strict.
///
/// The length has to be exactly right and every coefficient has to be below `q`, which is what
/// makes the encoding canonical. The byte-equality check that used to make the commitment
/// canonical — the verifier recomputing it and comparing — is gone, so this is the only thing
/// standing between the transcript and a prover that re-encodes one commitment many ways until
/// the challenges it draws suit it.
pub fn unpack_ring_elements(bytes: &[u8], count: usize, q: u128) -> Result<Vec<u128>, String> {
    let width = coefficient_bytes(q);
    if bytes.len() != width * count {
        return Err("the commitment has the wrong length".into());
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(width) {
        let mut buffer = [0u8; 16];
        buffer[..width].copy_from_slice(chunk);
        let value = u128::from_le_bytes(buffer);
        if value >= q {
            return Err("a commitment coefficient is not reduced".into());
        }
        out.push(value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use labinius::scalar;

    /// A deterministic stream of test values, independent of the matrix expansion.
    fn values(label: &[u8], n: usize) -> Vec<u128> {
        let mut bytes = vec![0u8; 16 * n];
        let mut hasher = blake3::Hasher::new();
        hasher.update(label);
        hasher.finalize_xof().fill(&mut bytes);
        bytes
            .chunks_exact(16)
            .map(|c| u128::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// The generic product agrees with the library's scalar reference at a prime the reference
    /// supports. `mul_mod_phi` is schoolbook over `u64` in `[0, q)`, so this pins the reduction
    /// modulo `X^648 - X^324 + 1` and the coefficient order at once.
    #[test]
    fn product_matches_the_scalar_reference() {
        for q in [3889u16, 2917, 19441] {
            let a: Vec<u128> = values(b"rq/a", N).iter().map(|x| x % q as u128).collect();
            let b: Vec<u128> = values(b"rq/b", N).iter().map(|x| x % q as u128).collect();
            let mut left = [0u32; N];
            let mut right = [0u32; N];
            for i in 0..N {
                left[i] = a[i] as u32;
                right[i] = b[i] as u32;
            }
            let want = scalar::mul_mod_phi(&left, &right, q);
            let signed: Vec<i64> = b.iter().map(|&x| x as i64).collect();
            let got = mul(&a, &signed, q as u128);
            assert_eq!(got, want.iter().map(|&x| x as u128).collect::<Vec<_>>());
        }
    }

    /// The same, with the second factor centered: the fold is a signed vector, and the reference
    /// only takes non-negative residues.
    #[test]
    fn signed_product_matches_the_scalar_reference() {
        let q = 3889u16;
        let a: Vec<u128> = values(b"rq/a2", N).iter().map(|x| x % q as u128).collect();
        let b: Vec<i64> = values(b"rq/b2", N)
            .iter()
            .map(|x| (x % 601) as i64 - 300)
            .collect();
        let mut left = [0u32; N];
        let mut right = [0u32; N];
        for i in 0..N {
            left[i] = a[i] as u32;
            right[i] = b[i].rem_euclid(q as i64) as u32;
        }
        let want = scalar::mul_mod_phi(&left, &right, q);
        let got = mul(&a, &b, q as u128);
        assert_eq!(got, want.iter().map(|&x| x as u128).collect::<Vec<_>>());
    }

    /// `A v` is the sum of the single products, at a modulus large enough that the periodic
    /// reduction of the accumulator runs several times.
    #[test]
    fn matrix_product_is_the_sum_of_products() {
        let q = 55_144_219_393u128;
        let elements = 200;
        let a: Vec<u128> = values(b"rq/matrix", elements * N)
            .iter()
            .map(|x| x % q)
            .collect();
        let v: Vec<i16> = values(b"rq/fold", elements * N)
            .iter()
            .map(|x| (x % 65535) as i16)
            .collect();
        let mut want = vec![0u128; N];
        for (element, fold) in a.chunks_exact(N).zip(v.chunks_exact(N)) {
            let signed: Vec<i64> = fold.iter().map(|&x| x as i64).collect();
            for (slot, term) in want.iter_mut().zip(mul(element, &signed, q)) {
                *slot = (*slot + term) % q;
            }
        }
        assert_eq!(a_times_v(&a, &v, q), want);
    }

    /// The challenge combination is the sum of the single products by the embedded challenge.
    #[test]
    fn challenge_combination_is_the_sum_of_products() {
        let q = 68_719_476_736u128;
        let mut transcript = labinius::Transcript::new(b"rq/challenges");
        let challenges: Vec<ShortChallenge> = (0..8)
            .map(|_| {
                labinius::challenge::sample_short_challenge(
                    &mut transcript,
                    labinius::challenge::DEFAULT_WEIGHT,
                    labinius::challenge::DEFAULT_BOUND,
                )
                .0
            })
            .collect();
        let columns: Vec<u128> = values(b"rq/columns", challenges.len() * N)
            .iter()
            .map(|x| x % q)
            .collect();
        let mut want = vec![0u128; N];
        for (column, challenge) in columns.chunks_exact(N).zip(&challenges) {
            let mut dense = vec![0i64; N];
            for t in 0..challenge.weight {
                let m = challenge.positions[t] as usize;
                let sign = 1 - 2 * ((challenge.signs >> t) & 1) as i64;
                dense[4 * m] = if m % 2 == 0 { sign } else { -sign };
            }
            for (slot, term) in want.iter_mut().zip(mul(column, &dense, q)) {
                *slot = (*slot + term) % q;
            }
        }
        assert_eq!(challenge_combination(&columns, &challenges, q), want);
    }

    /// The wire form round-trips, and the unpacker refuses a short string, a long one and an
    /// unreduced coefficient.
    #[test]
    fn wire_form_round_trips_and_rejects() {
        let q = 55_144_219_393u128;
        let coefficients: Vec<u128> = values(b"rq/wire", 3 * N).iter().map(|x| x % q).collect();
        let bytes = pack_ring_elements(&coefficients, q);
        assert_eq!(bytes.len(), 5 * 3 * N);
        assert_eq!(
            unpack_ring_elements(&bytes, 3 * N, q).unwrap(),
            coefficients
        );
        assert!(unpack_ring_elements(&bytes[..bytes.len() - 1], 3 * N, q).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(unpack_ring_elements(&longer, 3 * N, q).is_err());
        let mut unreduced = bytes.clone();
        unreduced[..5].copy_from_slice(&(q + 1).to_le_bytes()[..5]);
        assert!(unpack_ring_elements(&unreduced, 3 * N, q).is_err());
    }

    /// A power-of-two modulus never rejects a draw, so an element's expansion is the plain byte
    /// stream cut into `coefficient_bytes` fields: the matrix is defined for moduli the limb
    /// kernels have no arithmetic for.
    #[test]
    fn power_of_two_matrix_is_the_raw_stream() {
        let q = 1u128 << 36;
        let matrix = ajtai_matrix(q, 3);
        for (i, element) in matrix.chunks_exact(N).enumerate() {
            let mut expansion = Expansion::new(q, 3, i);
            for &x in element {
                assert_eq!(x, expansion.draw(5) & ((1 << 36) - 1));
            }
        }
        assert!(matrix.iter().all(|&x| x < q));
    }

    /// The expansion of one element does not depend on how the matrix was split over workers,
    /// and a different modulus or element count gives a different matrix.
    #[test]
    fn the_matrix_is_element_by_element() {
        let q = 55_144_219_393u128;
        let wide = ajtai_matrix(q, 64);
        assert_eq!(ajtai_matrix(q, 64)[..N], wide[..N]);
        assert_ne!(ajtai_matrix(q, 65)[..N], wide[..N]);
        assert_ne!(ajtai_matrix(q + 2, 64)[..N], wide[..N]);
    }
}
