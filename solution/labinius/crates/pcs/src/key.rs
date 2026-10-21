//! The Ajtai commitment key and what a commitment leaves behind for the fold.
//!
//! * [`CommitmentKey`] holds the Ajtai matrix `A` for the base [`Modulus`] and for each other
//!   [`Modulus`] the key was built with, in the NTT domain, centered, in the layout the
//!   AVX-512 kernel streams.
//! * [`CommitmentKey::commit_into_aux`] maps a witness — a plain `&[F162]`, read as binary ring
//!   elements of `R_648 = Z_q[X]/(X^648 - X^324 + 1)` four `F162` at a time — to a
//!   [`crate::ring::VerticallyAlignedMatrix`] of
//!   [`crate::ring::PowerOfThreeRingElementWithLimbs`], and keeps the transform the fold consumes.
use crate::fields::scalar::F162;
use crate::params::N;
use crate::ring::{
    components_of, Batch32, Modulus, PowerOfThreeRingElementWithLimbs, Representation,
    VerticallyAlignedMatrix,
};
use crate::limb::dispatch_limb;
use crate::rng::Rng;
use crate::simd::commit as cm;

// =============================================================================================
// the key
// =============================================================================================

/// `F162` elements per `Batch32` of the key: 128 `F162` = 32 ring elements.
const F162_PER_BATCH: usize = 128;

/// The Ajtai matrix `A` — one row of uniform ring elements, in the NTT domain, centered, for
/// every limb of the key. Opaque: the layout is the vertical one the AVX-512 kernels stream
/// ([`crate::simd::commit`]).
///
/// Limb 0 is the base [`Modulus`]; limb `1 + i` is the `i`-th additional one.
/// A split limb's rows are the 648 tree-order slots, a quadratic limb's are the 648 rows of the
/// quadratic tree (the two coefficients of each of the 324 leaves); in both cases every row is
/// uniform in `[-(q-1)/2, (q-1)/2]`.
pub struct CommitmentKey {
    a: Vec<KeyRows>,
    base: Modulus,
    additional: Vec<Modulus>,
    len_f162: usize,
}

/// One limb's `A`: built at run time, or borrowed from storage that outlives the key (the
/// bytes of the batches embedded in a binary by a build script, say).
pub enum KeyRows {
    Owned(Vec<Batch32>),
    Static(&'static [Batch32]),
}

impl KeyRows {
    #[inline]
    fn as_slice(&self) -> &[Batch32] {
        match self {
            KeyRows::Owned(v) => v,
            KeyRows::Static(s) => s,
        }
    }
}

impl CommitmentKey {
    /// A uniformly random key for `len_f162` witness elements over `base` and `additional`,
    /// deterministically from `seed`. `len_f162` must be a multiple of 128 (= 32 ring elements,
    /// one `Batch32` per limb).
    ///
    /// Costs `(1 + additional.len()) * len_f162 / 128 * 41472` bytes: 85 MB per limb for
    /// `len_f162 = 2^18`.
    pub fn random(len_f162: usize, seed: u64, base: Modulus, additional: &[Modulus]) -> Self {
        assert!(
            len_f162 > 0 && len_f162 % F162_PER_BATCH == 0,
            "len_f162 must be a multiple of 128"
        );
        assert_eq!(core::mem::size_of::<F162>(), 24, "F162 is not 24 bytes");
        let mut seen = vec![base];
        for l in additional {
            assert!(!seen.contains(l), "the limb {l:?} is listed twice");
            seen.push(*l);
        }
        let nb = len_f162 / F162_PER_BATCH;
        let primes: Vec<u16> = core::iter::once(base.prime())
            .chain(additional.iter().map(|l| l.prime()))
            .collect();
        let a: Vec<Vec<Batch32>> = primes
            .iter()
            .enumerate()
            .map(|(k, &q)| {
                let half = ((q - 1) / 2) as i16;
                let mut rng = Rng::new(seed ^ (0x9E37_79B9_u64.wrapping_mul(k as u64 + 1)));
                (0..nb)
                    .map(|_| {
                        let mut b = Batch32::zero(Representation::Ntt);
                        for j in 0..N {
                            for p in 0..32 {
                                b.v[j][p] = rng.below(q as u32) as i16 - half;
                            }
                        }
                        b
                    })
                    .collect()
            })
            .collect();
        CommitmentKey {
            a: a.into_iter().map(KeyRows::Owned).collect(),
            base,
            additional: additional.to_vec(),
            len_f162,
        }
    }

    /// A key whose matrix is given by its coefficients modulo `modulus`, rather than drawn per
    /// limb directly in that limb's NTT domain.
    ///
    /// `coefficients` is the `len_f162 / 4` ring elements of `A` flattened, element `i`'s
    /// coefficient `t` at index `N i + t` and every entry in `[0, modulus)`. Every limb takes the
    /// centered reduction of the same coefficients and its own forward transform, so the key is
    /// the reduction of one object of `R_modulus` and anyone who knows `modulus` and the
    /// coefficients can rebuild it. [`CommitmentKey::random`] cannot be rebuilt that way: its
    /// limbs are drawn independently in their own NTT domains and are not the reduction of any
    /// single mod-`Q` matrix.
    ///
    /// Costs one pass of reductions over the coefficients plus one forward transform per limb.
    pub fn from_coefficients(
        len_f162: usize,
        coefficients: &[u128],
        modulus: u128,
        base: Modulus,
        additional: &[Modulus],
    ) -> Self {
        assert!(
            len_f162 > 0 && len_f162 % F162_PER_BATCH == 0,
            "len_f162 must be a multiple of 128"
        );
        assert_eq!(
            coefficients.len(),
            len_f162 / 4 * N,
            "the matrix is len_f162 / 4 ring elements"
        );
        let mut seen = vec![base];
        for l in additional {
            assert!(!seen.contains(l), "the limb {l:?} is listed twice");
            seen.push(*l);
        }
        let nb = len_f162 / F162_PER_BATCH;
        let primes: Vec<u16> = core::iter::once(base.prime())
            .chain(additional.iter().map(|l| l.prime()))
            .collect();
        // Every (limb, run of batches) is independent: the reduction reads its own elements and
        // the forward transform is per batch, so the whole key goes out over the cores.
        let threads = crate::par::threads();
        let runs = crate::par::ranges(nb, threads.div_ceil(primes.len()).max(2), 1);
        let a: Vec<Vec<Batch32>> = std::thread::scope(|scope| {
            let handles: Vec<Vec<_>> = primes
                .iter()
                .map(|&q| {
                    runs.iter()
                        .map(|run| {
                            let run = run.clone();
                            scope.spawn(move || {
                                let mut batches: Vec<Batch32> = (0..run.len())
                                    .map(|_| Batch32::zero(Representation::Coefficients))
                                    .collect();
                                let first = run.start * 32 * N;
                                let last = run.end * 32 * N;
                                dispatch_limb!(q, |Q| reduce_coefficients::<Q>(
                                    &coefficients[first..last],
                                    modulus,
                                    &mut batches
                                ));
                                crate::fold::forward_limb(q, &mut batches);
                                batches
                            })
                        })
                        .collect()
                })
                .collect();
            handles
                .into_iter()
                .map(|limb| {
                    let mut all = Vec::with_capacity(nb);
                    for h in limb {
                        all.extend(h.join().unwrap());
                    }
                    all
                })
                .collect()
        });
        CommitmentKey {
            a: a.into_iter().map(KeyRows::Owned).collect(),
            base,
            additional: additional.to_vec(),
            len_f162,
        }
    }

    /// A key whose per-limb rows already exist in storage that outlives it: `rows[k]` is limb
    /// `k`'s `len_f162 / 128` batches in the NTT domain, exactly what [`row`](Self::row) of the
    /// key [`from_coefficients`](Self::from_coefficients) builds returns. Nothing is copied or
    /// computed, so a key built once at compile time costs the process nothing.
    pub fn from_static_rows(
        len_f162: usize,
        base: Modulus,
        additional: &[Modulus],
        rows: &[&'static [Batch32]],
    ) -> Self {
        assert!(
            len_f162 > 0 && len_f162 % F162_PER_BATCH == 0,
            "len_f162 must be a multiple of 128"
        );
        assert_eq!(rows.len(), 1 + additional.len(), "one row set per limb");
        let mut seen = vec![base];
        for l in additional {
            assert!(!seen.contains(l), "the limb {l:?} is listed twice");
            seen.push(*l);
        }
        for r in rows {
            assert_eq!(r.len(), len_f162 / F162_PER_BATCH, "one batch per 128 elements");
            assert!(
                r.iter().all(|b| b.representation == Representation::Ntt),
                "a key is in the NTT domain"
            );
        }
        CommitmentKey {
            a: rows.iter().map(|r| KeyRows::Static(r)).collect(),
            base,
            additional: additional.to_vec(),
            len_f162,
        }
    }

    /// The raw 648-row commitments of `columns` columns, every limb, with no transform kept and
    /// a bounded pool of `workers` threads: `out[c * limbs + k]` is column `c` modulo limb `k`,
    /// in `[0, q)`, exactly what [`AuxData::commitment`] would hold after
    /// [`commit_into_aux`](Self::commit_into_aux).
    ///
    /// The columns are taken `group` (at most [`cm::GROUP`]) at a time against one pass over
    /// `A`, as `commit_into_aux` takes them, but the groups are handed out to a fixed number of
    /// workers instead of one thread each, so each worker's accumulators stay in its own L2 and
    /// the machine runs as many kernels as it has cores. With two workers per physical core a
    /// group of 4 (348 KB of accumulators each) leaves both inside one L2. Each column may be the lifted `F162`
    /// witness or the `B128` words it was lifted from ([`cm::BatchSource`]).
    pub fn commit_columns_rows<E: cm::BatchSource>(
        &self,
        columns: &[E],
        workers: usize,
        group: usize,
    ) -> Vec<[u32; N]> {
        let r = columns.len();
        assert!(r > 0, "at least one column");
        for c in columns {
            assert_eq!(c.elems(), self.len_f162, "a column is len_f162() elements");
        }
        let limbs = self.limb_list();
        let nl = limbs.len();
        let g = group.clamp(1, cm::GROUP).min(r);
        let groups = r.div_ceil(g);
        let workers = workers.clamp(1, groups);
        let mut out = vec![[0u32; N]; r * nl];
        let out_addr = out.as_mut_ptr() as usize;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let placement = crate::par::CorePlacement::for_workers(workers);
        crate::par::pool().run(workers, |worker| {
            let _affinity = placement.as_ref().and_then(|p| p.pin(worker));
            let mut st = cm::Scratch::with_group(&limbs, g);
            let mut raw = vec![[0u32; N]; g * nl];
            loop {
                let gi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if gi >= groups {
                    break;
                }
                let c0 = gi * g;
                let gc = g.min(r - c0);
                cm::commit_limbs_group(
                    &columns[c0..c0 + gc],
                    &limbs,
                    None,
                    &mut st,
                    &mut raw[..gc * nl],
                );
                // Disjoint slices of `out`: group `gi` owns columns `c0 .. c0 + gc`.
                unsafe {
                    let dst = (out_addr as *mut [u32; N]).add(c0 * nl);
                    core::ptr::copy_nonoverlapping(raw.as_ptr(), dst, gc * nl);
                }
            }
        });
        out
    }

    /// Length of the key in `F162` elements: the size of one chunk of witness it commits to.
    pub fn len_f162(&self) -> usize {
        self.len_f162
    }

    /// Length of the key in `R_648` ring elements (`len_f162 / 4`).
    pub fn len_ring(&self) -> usize {
        self.len_f162 / 4
    }

    /// Number of limbs, `1 + additional().len()`.
    pub fn limbs(&self) -> usize {
        self.a.len()
    }

    /// The additional limbs, in the order the commitment reports them.
    pub fn additional(&self) -> &[Modulus] {
        &self.additional
    }

    /// The base limb: limb 0, whose transform a commitment keeps and whose domain the fold runs
    /// in.
    pub fn base(&self) -> Modulus {
        self.base
    }

    /// The limb at index `k` (limb 0 is [`base`](Self::base)).
    pub fn limb(&self, k: usize) -> Modulus {
        if k == 0 {
            self.base
        } else {
            self.additional[k - 1]
        }
    }

    /// The prime of limb `k`.
    pub fn prime(&self, k: usize) -> u16 {
        self.limb(k).prime()
    }

    /// Does limb `k` use the quadratic-slot tree?
    pub fn is_quadratic(&self, k: usize) -> bool {
        self.limb(k).is_quadratic()
    }

    /// The matrix itself, for limb `k`, in the vertical layout [`crate::simd::commit`] streams:
    /// `row(k)[b].v[j][p]` is row `j` of `A_{32b + p}` modulo `prime(k)`, centered. Exposed so
    /// that a caller can drive the kernel directly, or check a commitment against
    /// [`crate::scalar`].
    pub fn row(&self, k: usize) -> &[Batch32] {
        self.a[k].as_slice()
    }

    /// The limbs in the form the kernel driver takes.
    pub(crate) fn limb_list(&self) -> Vec<cm::Limb<'_>> {
        (0..self.limbs())
            .map(|k| cm::Limb {
                q: self.prime(k),
                quad: self.is_quadratic(k),
                a: self.a[k].as_slice(),
            })
            .collect()
    }

    /// Bytes of `A` held, over all limbs.
    pub fn bytes(&self) -> usize {
        self.limbs() * (self.len_f162 / F162_PER_BATCH) * core::mem::size_of::<Batch32>()
    }

    /// The four `R_162` components of the raw commitments of one chunk, one entry per limb.
    fn components(&self, raw: &[[u32; N]]) -> [PowerOfThreeRingElementWithLimbs; 4] {
        let mut out: [PowerOfThreeRingElementWithLimbs; 4] =
            core::array::from_fn(|_| PowerOfThreeRingElementWithLimbs {
                limbs: Vec::with_capacity(self.limbs()),
            });
        for k in 0..self.limbs() {
            let d = components_of(self.prime(k), &raw[k]);
            for (c, e) in d.into_iter().enumerate() {
                out[c].limbs.push(e);
            }
        }
        out
    }

    /// Commit to `witness` in `r` chunks under the same key, keeping what the folding step
    /// ([`crate::fold`]) consumes in a buffer the caller already owns.
    ///
    /// `r` must be a power of two and `witness.len()` must be `r * self.len_f162()`. Chunk `c` is
    /// `witness[c * len .. (c + 1) * len]`; it is read as `len / 4` binary ring elements of
    /// `R_648` (`crate::f162`), sliced once, and transformed and multiplied into the inner product
    /// `y = sum_i A_i * NTT(w_i)` once per limb. The chunks are taken [`cm::GROUP`] at a time
    /// against one pass over `A` (`crate::simd::commit`). Each limb's `y` is then
    /// split into its four `R_162` components, which become column `c` of the returned 4 x `r`
    /// matrix. The transform of every ring element modulo the base limb is written out as it
    /// goes, by the same block sink that feeds the base multiplication and with non-temporal
    /// stores, so keeping it costs a memory stream rather than a second transform.
    ///
    /// The 85 MB an [`AuxData`] holds for a 2^16-element witness is one `mmap` and 20 736 first
    /// touches, ~20 ms of page faults the kernel charges to whoever writes the pages first — more
    /// than the commitment itself. A prover that folds repeatedly allocates one [`AuxData::new`]
    /// and reuses it, and then keeping the witness costs what it should: the non-temporal stores,
    /// which hide behind the transform.
    pub fn commit_into_aux(
        &self,
        witness: &[F162],
        r: usize,
        aux: &mut AuxData,
    ) -> VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs> {
        self.check_witness(witness, r);
        let bpc = self.a[0].as_slice().len();
        assert!(
            aux.chunks == r && aux.batches.len() == r * bpc && aux.raw.len() == self.limbs(),
            "the auxiliary buffer does not match this key and r"
        );
        for v in aux.raw.iter_mut() {
            v.clear();
        }
        let limbs = self.limb_list();
        let nl = limbs.len();
        let g = cm::GROUP.min(r);
        let len_f162 = self.len_f162;
        let starts: Vec<usize> = (0..r).step_by(g).collect();
        // Each group writes a disjoint slice of `aux.batches` and returns its raw commitments.
        // The address is a usize so the closure is `Send`; the slices do not overlap.
        let batch_addr = aux.batches.as_mut_ptr() as usize;
        let stride = std::mem::size_of::<Batch32>();
        let mut groups: Vec<(usize, Vec<[u32; N]>)> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(starts.len());
            for c0 in starts {
                let limbs = &limbs;
                handles.push(scope.spawn(move || {
                    let mut st = cm::Scratch::with_group(limbs, g);
                    let mut raw = vec![[0u32; N]; g * nl];
                    let mut chunks: [&[F162]; cm::GROUP] = [&[]; cm::GROUP];
                    for i in 0..g {
                        let c = c0 + i;
                        chunks[i] = &witness[c * len_f162..(c + 1) * len_f162];
                    }
                    let batches = unsafe {
                        std::slice::from_raw_parts_mut(
                            (batch_addr + (c0 * bpc) * stride) as *mut Batch32,
                            g * bpc,
                        )
                    };
                    cm::commit_limbs_group(
                        &chunks[..g],
                        limbs,
                        Some(batches),
                        &mut st,
                        &mut raw,
                    );
                    (c0, raw)
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        groups.sort_by_key(|(c0, _)| *c0);
        let mut data = Vec::with_capacity(4 * r);
        for (_, raw) in groups {
            for y in raw.chunks_exact(nl) {
                data.extend(self.components(y));
                for (k, v) in y.iter().enumerate() {
                    aux.raw[k].push(*v);
                }
            }
        }
        VerticallyAlignedMatrix::new(4, r, data)
    }

    fn check_witness(&self, witness: &[F162], r: usize) {
        assert!(r.is_power_of_two(), "r must be a power of two");
        assert_eq!(
            witness.len(),
            r * self.len_f162,
            "witness must be r * len_f162() elements ({} * {})",
            r,
            self.len_f162
        );
    }
}

/// The centered residues of `coefficients` modulo `Q`, in the vertical layout the forward
/// transform takes: `batches[i / 32].v[t][i % 32]` is coefficient `t` of element `i`.
///
/// `Q` is a const parameter so that the two reductions below are multiply-shifts rather than
/// divisions; a coefficient is split into its two 64-bit halves because a `u128` division by a
/// runtime modulus is a library call and this runs once per coefficient and limb.
fn reduce_coefficients<const Q: u16>(coefficients: &[u128], modulus: u128, out: &mut [Batch32]) {
    let q = Q as u64;
    let half = ((Q - 1) / 2) as i16;
    let shifted = ((1u128 << 64) % Q as u128) as u64;
    for (i, element) in coefficients.chunks_exact(N).enumerate() {
        let (b, p) = (i / 32, i % 32);
        for (t, &x) in element.iter().enumerate() {
            debug_assert!(x < modulus, "a matrix coefficient is not reduced");
            let value = (((x >> 64) as u64 % q) * shifted + x as u64 % q) % q;
            let value = value as i16;
            out[b].v[t][p] = if value > half { value - Q as i16 } else { value };
        }
    }
}

/// `n` `Batch32`s whose 41472 bytes each are never read before the kernel writes them (zeroing
/// 85 MB would cost 4 ms of the very DRAM traffic the non-temporal stores are there to avoid).
fn uninit_batches(n: usize) -> Vec<Batch32> {
    let mut v: Vec<Batch32> = Vec::with_capacity(n);
    unsafe {
        let p = v.as_mut_ptr();
        for i in 0..n {
            (*p.add(i)).representation = Representation::Ntt;
        }
        v.set_len(n);
    }
    v
}

// =============================================================================================
// the auxiliary data
// =============================================================================================

/// Everything a commitment leaves behind that the folding step ([`crate::fold`]) needs, and
/// nothing a caller has to look inside: the witness's transform modulo the base limb in the
/// layout the kernel produced it, and the raw 648-row commitments of the `r` chunks for every
/// limb of the key.
///
/// Produced by [`CommitmentKey::commit_into_aux`] as a by-product of the commitment itself. For
/// 2^16 ring elements it holds 2048 `Batch32` = 85 MB, whatever the limb list is: only the base
/// limb's transform is kept.
#[derive(Clone)]
pub struct AuxData {
    /// The transform modulo the base limb, `batches[b].v[u][p]` = row `u` of ring element
    /// `32 b + p`, lazily reduced (`|v| <= 7.5 q`, the binary kernel's declared output bound).
    pub(crate) batches: Vec<Batch32>,
    /// `raw[k][j]` = the commitment of chunk `j` for limb `k`, 648 rows in `[0, q)`.
    pub(crate) raw: Vec<Vec<[u32; N]>>,
    pub(crate) chunks: usize,
}

impl AuxData {
    /// An empty buffer for `r` chunks of `len_ring` ring elements each over `limbs` limbs, to be
    /// filled by [`CommitmentKey::commit_into_aux`]. The `Batch32`s are left uninitialised: the
    /// kernel writes every one of their 41472 bytes before anything reads them, and zeroing 85 MB
    /// would cost more than the commitment.
    pub fn new(len_ring: usize, r: usize, limbs: usize) -> AuxData {
        assert!(r > 0 && len_ring > 0 && len_ring % 32 == 0 && limbs > 0);
        AuxData {
            batches: uninit_batches(r * len_ring / 32),
            raw: (0..limbs).map(|_| Vec::with_capacity(r)).collect(),
            chunks: r,
        }
    }

    /// Number of chunks the witness was split into (`r`).
    pub fn chunks(&self) -> usize {
        self.chunks
    }
    /// Number of limbs the commitments were taken over.
    pub fn limbs(&self) -> usize {
        self.raw.len()
    }
    /// Bytes of witness transform held.
    pub fn bytes(&self) -> usize {
        self.batches.len() * core::mem::size_of::<Batch32>()
    }
    /// `Batch32`s per chunk: `len_ring / 32`, 8 for a 1024-`F162` key.
    pub fn batches_per_chunk(&self) -> usize {
        self.batches.len() / self.chunks
    }
    /// Batch `i` of the kept transform, `32 i` .. `32 i + 32` in witness order; chunk `j` is
    /// batches `j * batches_per_chunk()` onward. Read-only, for checking the fold against
    /// [`crate::scalar`].
    pub fn batch(&self, i: usize) -> &Batch32 {
        &self.batches[i]
    }
    /// The commitment of chunk `j` for limb `k`: 648 rows in `[0, q)`, before the four-way
    /// decomposition. This is the form the fold's consistency identity `A v = sum_j c_j C_j`
    /// lives in.
    pub fn commitment(&self, k: usize, j: usize) -> &[u32; N] {
        &self.raw[k][j]
    }
}
