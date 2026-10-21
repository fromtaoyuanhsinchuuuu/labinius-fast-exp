//! Packed-binary PCS opening, including the GHASH-to-F162 field switch.
//!
//! The default commitment modulus is 3889 * 9721 * 1024. Two witness NTTs feed
//! ordinary residues and an exact auxiliary computation of the 1024 residue.
//! Public matrix images and reconstruction bounds are computed at build time
//! from the fixed matrix seed and modulus. Other supported prime products use
//! the generic RNS implementation.
//!
//! The timed process reads the fresh table in place, commits without retaining
//! witness transforms, runs the eighth-table field switch, computes the exact
//! integer fold, and writes the checker-compatible proof. Transcript challenges
//! are derived using the chosen modulus and the resulting commitment.
use bincode::Options;
use labinius::challenge::{
    sample_short_challenges_parallel, ShortChallenge, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
#[cfg(test)]
use labinius::key::CommitmentKey;
use labinius::params::N;
use labinius::ring::Representation;
use labinius::scheme::*;
use labinius::{wire, Transcript};
use serde::{Deserialize, Serialize};

mod keygen;
mod modulus_choice;
mod parameters;
/// Byte-for-byte the checker's `rq.rs`, so that the two sides of the format can be diffed. A
/// release build of the prover only uses the matrix expansion and the packer; the rest is what
/// the tests check the proof against.
#[allow(dead_code)]
mod rq;
use parameters::{COLUMNS, COLUMN_LEN, COLUMN_LOG, LOG_PACKED, LOG_VARS, MODULUS, RING_ELEMENTS};

/// The key of the default modulus, precomputed by `build.rs` and embedded as the bytes of its
/// batches (empty when the build host could not run the kernels). 64-byte aligned so the
/// batches are read in place.
#[repr(C, align(64))]
struct Aligned64<T>(T);
const EMBEDDED_KEY_LEN: usize = include_bytes!(concat!(env!("OUT_DIR"), "/key.bin")).len();
static EMBEDDED_KEY: Aligned64<[u8; EMBEDDED_KEY_LEN]> =
    Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/key.bin")));

const LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct Proof {
    version: u32,
    modulus: u128,
    #[serde(serialize_with = "serialize_blob")]
    commitment: Vec<u8>,
    partials: Vec<u128>,
    rounds: Vec<[[u64; 3]; 2]>,
    opened: [u64; 3],
    #[serde(serialize_with = "serialize_blob")]
    row: Vec<u8>,
    #[serde(serialize_with = "serialize_blob")]
    folded: Vec<u8>,
}
/// A `Vec<u8>` as one byte string: under this codec's fixed-width encoding that is the same
/// `u64` length and the same bytes a sequence of `u8` gives, written as one block instead of
/// one byte at a time. The checker deserialises it as the `Vec<u8>` it always was.
fn serialize_blob<S: serde::Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(bytes)
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
use keygen::params;
fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}

/// Threads the parallel phases use: what the sandbox lets the process see.
fn threads() -> usize {
    labinius::par::threads()
}

// =============================================================================================
// the witness table, read in place
// =============================================================================================

#[cfg(target_os = "linux")]
mod sys {
    use core::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
    }
    pub const PROT_NONE: i32 = 0;
    pub const PROT_READ: i32 = 1;
    pub const PROT_WRITE: i32 = 2;
    pub const MAP_SHARED: i32 = 1;
    pub const MAP_FIXED: i32 = 0x10;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_ANONYMOUS: i32 = 0x20;
    pub const MAP_NORESERVE: i32 = 0x4000;
    pub const MADV_HUGEPAGE: i32 = 14;
    /// Linux 5.14+: fault the range in now, read-only, without blocking a later writer.
    pub const MADV_POPULATE_READ: i32 = 22;
    /// Linux 5.14+: fault the range in now, writable (for a shared file mapping: page-cache pages
    /// allocated and their blocks reserved, as a first write would).
    pub const MADV_POPULATE_WRITE: i32 = 23;
}

// =============================================================================================
// memory: large allocations on huge pages
// =============================================================================================

/// The process's allocator: glibc's for everything below [`huge::LARGE`], and for every larger
/// block a mapping of its own, 2 MB aligned, with transparent huge pages requested.
///
/// The prover's big buffers (the commitment rows, the commitment and fold outputs, the proof's
/// byte fields, the switch's tables) are written once and freed whole. On 4 KB pages each of
/// them costs one fault and one zeroed page per 4 KB when it is first written, and the same
/// number of page frees when it is unmapped — and whatever is still mapped at the end is unmapped
/// by the kernel in the exit path, after the proof is written, inside the measured time (about
/// 0.25 us per 4 KB page: 2.2 ms for the ~30 MB the baseline still held). A huge page is one fault,
/// one 2 MB clear and one free.
#[cfg(target_os = "linux")]
mod huge {
    use super::sys;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::alloc::{GlobalAlloc, Layout, System};

    const HUGE: usize = 1 << 21;
    /// Blocks at least this large get a mapping of their own.
    pub const LARGE: usize = 1 << 20;
    /// Blocks from this size up to [`LARGE`] (alignment at most this) come from huge-page chunks
    /// in quarter-octave classes, each class recycling its blocks through a free list (never
    /// returned to the kernel). glibc served them from per-thread arenas that it trimmed back to
    /// the kernel (`MADV_DONTNEED`) when they were freed, and the next allocation faulted the
    /// pages in again; larger per-dispatch buffers (the fold's ~840 KB per worker) were mmapped
    /// and unmapped each time. Blocks are carved with a 64-byte gap between them, not at
    /// power-of-two offsets: the commitment's sixty-four 6 KB accumulators at 8 KB-aligned
    /// addresses all fell into the same cache sets.
    const MEDIUM: usize = 1 << 16;
    const LISTS: usize = 4 * ((LARGE / MEDIUM).trailing_zeros() as usize + 1);
    /// Gap between consecutively carved blocks.
    const GAP: usize = 64;

    pub struct Alloc;

    #[inline]
    fn large(size: usize, align: usize) -> bool {
        size >= LARGE && align <= HUGE
    }

    #[inline]
    fn medium(size: usize, align: usize) -> bool {
        size >= MEDIUM && size < LARGE && align <= MEDIUM
    }

    /// A medium request's block size (`size` rounded up to a quarter of its octave) and the
    /// free list of that size.
    #[inline]
    fn class(size: usize) -> (usize, usize) {
        let k = (usize::BITS - 1 - (size - 1).leading_zeros()) as usize;
        let step = 1usize << (k - 2);
        let bytes = (size + step - 1) & !(step - 1);
        (bytes, 4 * (k - MEDIUM.trailing_zeros() as usize + 1) + (bytes >> (k - 2)) - 5)
    }

    struct SpinLock(AtomicBool);

    impl SpinLock {
        #[inline]
        fn lock(&self) {
            while self
                .0
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
            }
        }
        #[inline]
        fn unlock(&self) {
            self.0.store(false, Ordering::Release);
        }
    }

    /// A class's free blocks: an intrusive LIFO list (a free block's first word is the next).
    struct FreeList {
        lock: SpinLock,
        head: UnsafeCell<*mut u8>,
    }

    /// The chunk medium blocks are carved from.
    struct Carve {
        lock: SpinLock,
        at: UnsafeCell<usize>,
        end: UnsafeCell<usize>,
    }

    // Guarded by their locks.
    unsafe impl Sync for FreeList {}
    unsafe impl Sync for Carve {}

    static FREE: [FreeList; LISTS] = [const {
        FreeList {
            lock: SpinLock(AtomicBool::new(false)),
            head: UnsafeCell::new(core::ptr::null_mut()),
        }
    }; LISTS];
    static CARVE: Carve = Carve {
        lock: SpinLock(AtomicBool::new(false)),
        at: UnsafeCell::new(0),
        end: UnsafeCell::new(0),
    };

    /// A medium block for `size` bytes at `align`, and whether it is fresh (still zero).
    unsafe fn medium_alloc(size: usize, align: usize) -> (*mut u8, bool) {
        let (bytes, i) = class(size);
        let list = &FREE[i];
        list.lock.lock();
        let head = *list.head.get();
        if !head.is_null() && head as usize % align == 0 {
            *list.head.get() = *(head as *mut *mut u8);
            list.lock.unlock();
            return (head, false);
        }
        list.lock.unlock();
        let align = align.max(GAP);
        CARVE.lock.lock();
        let mut at = (*CARVE.at.get() + align - 1) & !(align - 1);
        if *CARVE.end.get() == 0 || at + bytes > *CARVE.end.get() {
            let chunk = map(HUGE) as usize;
            if chunk == 0 {
                CARVE.lock.unlock();
                return (core::ptr::null_mut(), false);
            }
            at = chunk;
            *CARVE.end.get() = chunk + HUGE;
        }
        *CARVE.at.get() = at + bytes + GAP;
        CARVE.lock.unlock();
        (at as *mut u8, true)
    }

    unsafe fn medium_free(p: *mut u8, size: usize) {
        let list = &FREE[class(size).1];
        list.lock.lock();
        *(p as *mut *mut u8) = *list.head.get();
        *list.head.get() = p;
        list.lock.unlock();
    }

    #[inline]
    fn span(size: usize) -> usize {
        (size + HUGE - 1) & !(HUGE - 1)
    }

    /// A fresh (zeroed) 2 MB-aligned mapping of `span(size)` bytes, or null.
    unsafe fn map(size: usize) -> *mut u8 {
        let len = span(size);
        let raw = sys::mmap(
            core::ptr::null_mut(),
            len + HUGE,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE,
            -1,
            0,
        );
        if raw as isize == -1 || raw.is_null() {
            return core::ptr::null_mut();
        }
        let raw = raw as usize;
        let start = (raw + HUGE - 1) & !(HUGE - 1);
        if start > raw {
            sys::munmap(raw as *mut _, start - raw);
        }
        let (end, raw_end) = (start + len, raw + len + HUGE);
        if raw_end > end {
            sys::munmap(end as *mut _, raw_end - end);
        }
        // Advisory: on a kernel without THP the block is ordinary pages.
        sys::madvise(start as *mut _, len, sys::MADV_HUGEPAGE);
        start as *mut u8
    }

    unsafe impl GlobalAlloc for Alloc {
        #[inline]
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if large(l.size(), l.align()) {
                map(l.size())
            } else if medium(l.size(), l.align()) {
                medium_alloc(l.size(), l.align()).0
            } else {
                System.alloc(l)
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            if large(l.size(), l.align()) {
                map(l.size())
            } else if medium(l.size(), l.align()) {
                let (p, fresh) = medium_alloc(l.size(), l.align());
                if !p.is_null() && !fresh {
                    core::ptr::write_bytes(p, 0, l.size());
                }
                p
            } else {
                System.alloc_zeroed(l)
            }
        }

        #[inline]
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if large(l.size(), l.align()) {
                sys::munmap(p as *mut _, span(l.size()));
            } else if medium(l.size(), l.align()) {
                medium_free(p, l.size())
            } else {
                System.dealloc(p, l)
            }
        }

        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            let (old_large, new_large) = (large(l.size(), l.align()), large(new_size, l.align()));
            let (old_medium, new_medium) = (medium(l.size(), l.align()), medium(new_size, l.align()));
            if !old_large && !new_large && !old_medium && !new_medium {
                return System.realloc(p, l, new_size);
            }
            if old_large && new_large && span(new_size) == span(l.size()) {
                return p;
            }
            if old_medium && new_medium && class(new_size).0 == class(l.size()).0 {
                return p;
            }
            let q = self.alloc(Layout::from_size_align_unchecked(new_size, l.align()));
            if !q.is_null() {
                core::ptr::copy_nonoverlapping(p, q, l.size().min(new_size));
                self.dealloc(p, l);
            }
            q
        }
    }
}

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: huge::Alloc = huge::Alloc;

/// Whether the witness table's mapping asks for huge pages.
fn table_huge_pages() -> bool {
    match std::env::var("LABINIUS_TABLE_HUGE") {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

/// The harness witness table: `(1 << LOG_VARS) / 8` BLAKE3 XOF bytes, each 16 little-endian
/// bytes one GHASH word. The timed prover does not recompute it, and does not copy it either: on
/// Linux the file is mapped read-only and the mapping is read as the `B128` words it is (`B128`
/// is `repr(transparent)` over its little-endian `u128`, and the host is little-endian). The
/// pages are faulted in by [`Table::populate`] on a thread of its own while the key is set up.
/// Elsewhere, or if the mapping fails, the file is read into a buffer.
enum Table {
    #[cfg(target_os = "linux")]
    Mapped {
        ptr: *const B128,
        bytes: usize,
    },
    Owned(Vec<B128>),
}

impl Table {
    const WORDS: usize = 1 << LOG_PACKED;
    const BYTES: usize = (1 << LOG_VARS) / 8;

    fn load(path: &str) -> Table {
        assert!(cfg!(target_endian = "little"), "the table is little-endian");
        assert_eq!(core::mem::size_of::<B128>(), 16, "B128 is 16 bytes");
        let file = std::fs::File::open(path).expect("open witness table");
        let len = file.metadata().expect("stat witness table").len();
        assert_eq!(len, Self::BYTES as u64, "witness table has the wrong length");
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let ptr = unsafe {
                sys::mmap(
                    core::ptr::null_mut(),
                    Self::BYTES,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize != -1 && !ptr.is_null() && (ptr as usize) % 16 == 0 {
                // The table is read in 8 KB chunks at a 256 KB stride by the fold and
                // in 2 KB chunks by the commitment: a page-table walk per chunk
                // otherwise, and the fold's sweep touches 2048 distinct 4 KB pages
                // at once, which is the whole second-level TLB. Ask for huge
                // pages; the kernel may refuse, and the fallback is the same
                // mapping it would have used.
                if table_huge_pages() {
                    unsafe { sys::madvise(ptr, Self::BYTES, sys::MADV_HUGEPAGE) };
                }
                return Table::Mapped {
                    ptr: ptr as *const B128,
                    bytes: Self::BYTES,
                };
            }
        }
        let mut words = vec![B128::ZERO; Self::WORDS];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, Self::BYTES)
        };
        let mut file = file;
        std::io::Read::read_exact(&mut file, bytes).expect("witness table has the wrong length");
        let mut extra = [0u8; 1];
        let extra_n = std::io::Read::read(&mut file, &mut extra).expect("read witness table");
        assert_eq!(extra_n, 0, "witness table has the wrong length");
        Table::Owned(words)
    }

    fn words(&self) -> &[B128] {
        match self {
            #[cfg(target_os = "linux")]
            Table::Mapped { ptr, .. } => unsafe { std::slice::from_raw_parts(*ptr, Self::WORDS) },
            Table::Owned(v) => v,
        }
    }

}

/// Fault the words' pages in, page-cache pages into page tables: `MADV_POPULATE_READ` where the
/// kernel has it (older kernels refuse it, harmlessly), one read per page otherwise (the
/// fault-around maps a run of pages per fault). A read-only touch, so it runs on another thread
/// while the key is built.
fn populate_words(words: &[B128], keys: &[u8], threads: usize) {
    let runs = labinius::par::ranges(words.len(), threads, 4096 / 16);
    let kruns = labinius::par::ranges(keys.len(), threads, 4096);
    labinius::par::pool().run(runs.len().max(kruns.len()), |i| {
        if let Some(r) = kruns.get(i) {
            populate_part(keys[r.clone()].as_ptr() as usize, r.len());
        }
        if let Some(r) = runs.get(i) {
            populate_part(words[r.clone()].as_ptr() as usize, r.len() * 16);
        }
    });
}

fn populate_part(addr: usize, bytes: usize) {
    #[cfg(target_os = "linux")]
    {
        // whole pages: the range need not start on one
        let start = addr & !4095;
        let done = unsafe {
            sys::madvise(start as *mut core::ffi::c_void, bytes + (addr - start), sys::MADV_POPULATE_READ)
        };
        if done == 0 {
            return;
        }
    }
    let mut off = 0;
    let mut sink = 0u8;
    while off < bytes {
        sink ^= unsafe { core::ptr::read_volatile((addr as *const u8).add(off)) };
        off += 4096;
    }
    std::hint::black_box(sink);
}

// A read-only mapping: any thread may read or unmap it.
#[cfg(target_os = "linux")]
unsafe impl Send for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Table::Mapped { ptr, bytes } = self {
            unsafe { sys::munmap(*ptr as *mut core::ffi::c_void, *bytes) };
        }
    }
}

#[cfg(test)]
fn trace(seed: &[u8; 32]) -> Vec<B128> {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"labinius-fast/binary-table/v1");
    let mut b = vec![0; (1 << LOG_VARS) / 8];
    h.finalize_xof().fill(&mut b);
    b.chunks_exact(16)
        .map(|v| B128(u128::from_le_bytes(v.try_into().unwrap())))
        .collect()
}
fn transcript(seed: &[u8; 32], modulus: u128, commitment: &[u8]) -> Transcript {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(seed);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_bytes(&modulus.to_le_bytes());
    // The 3.3 MB commitment hashed as blake3's subtrees on every core: the same transcript.
    t.absorb_bytes_parallel(commitment, threads());
    t
}
fn point(t: &mut Transcript) -> (Vec<B128>, Vec<B128>) {
    (
        t.sample_b128(b"r-lo", 7),
        t.sample_b128(b"r-hi", LOG_PACKED),
    )
}
fn claim(partials: &[B128], lo: &[B128]) -> B128 {
    partials
        .iter()
        .zip(cf::eq_expand_b128(lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e)
}
/// The checker's folded-witness encoding, which this must reproduce byte for byte: one
/// little-endian `i32` per coefficient, element by element. Written into a sized buffer, in
/// parallel over elements.
const FOLDED_BYTES: usize = 4;
fn pack_folded(folded: &FoldedWitness) -> Vec<u8> {
    let elements = folded.elements();
    assert!(
        elements
            .iter()
            .all(|e| e.representation == Representation::Coefficients),
        "a fold is in coefficient form"
    );
    let mut out = vec![0u8; FOLDED_BYTES * N * elements.len()];
    let runs = labinius::par::ranges(elements.len(), threads(), 1);
    let out_addr = out.as_mut_ptr() as usize;
    labinius::par::pool().run(runs.len(), |i| {
        let run = runs[i].clone();
        // Run `i` owns the bytes of elements `run`.
        let mine = unsafe {
            std::slice::from_raw_parts_mut(
                (out_addr as *mut u8).add(FOLDED_BYTES * N * run.start),
                FOLDED_BYTES * N * run.len(),
            )
        };
        for (slot, &c) in mine
            .chunks_exact_mut(FOLDED_BYTES)
            .zip(elements[run].iter().flat_map(|e| e.v.iter()))
        {
            slot.copy_from_slice(&(c as i32).to_le_bytes());
        }
    });
    out
}

/// `rq::pack_ring_elements`, byte for byte, into a sized buffer in parallel: `coefficient_bytes`
/// little-endian bytes per coefficient, no header.
#[cfg(test)]
fn pack_coefficients(coefficients: &[u128], modulus: u128) -> Vec<u8> {
    let width = rq::coefficient_bytes(modulus);
    let mut out = vec![0u8; width * coefficients.len()];
    let runs = labinius::par::ranges(coefficients.len(), threads(), 1);
    let out_addr = out.as_mut_ptr() as usize;
    labinius::par::pool().run(runs.len(), |i| {
        let run = runs[i].clone();
        // Run `i` owns the bytes of coefficients `run`.
        let mine = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut u8).add(width * run.start), width * run.len())
        };
        for (slot, &x) in mine.chunks_exact_mut(width).zip(&coefficients[run]) {
            debug_assert!(x < modulus, "a coefficient is not reduced");
            slot.copy_from_slice(&x.to_le_bytes()[..width]);
        }
    });
    out
}
#[cfg(test)]
fn unpack_folded(bytes: &[u8], elements: usize) -> Result<FoldedWitness, String> {
    if bytes.len() != FOLDED_BYTES * N * elements {
        return Err("folded witness has the wrong length".into());
    }
    let mut words = bytes.chunks_exact(FOLDED_BYTES);
    let mut fold = Vec::with_capacity(elements);
    for _ in 0..elements {
        let mut e = labinius::RingElement648::zero(Representation::Coefficients);
        for slot in e.v.iter_mut() {
            let word = i32::from_le_bytes(words.next().unwrap().try_into().unwrap());
            *slot = i16::try_from(word).map_err(|_| "coefficient out of range".to_string())?;
        }
        fold.push(e);
    }
    Ok(FoldedWitness::of(fold))
}
/// The `COLUMNS` folding challenges, derived the way the checker derives them: the row
/// evaluation absorbed, then one weight-28 signed subring element per column. The samples are
/// independent derivations of the transcript, so they are drawn on every core; the result is
/// the sequence `sample_short_challenge` would produce.
fn folding_challenges(t: &mut Transcript, row: &RowEvaluation) -> Vec<ShortChallenge> {
    row.absorb(t);
    sample_short_challenges_parallel(t, COLUMNS, DEFAULT_WEIGHT, DEFAULT_BOUND, threads())
}

/// Garner's lift of one coefficient from its residues modulo `primes` (each already in
/// `[0, q)`) to the integer modulo their product. The reductions are Barrett's with the
/// library's magic `floor(2^43 / q)`: exact for operands below `2^31`, which every product of
/// two residues of primes below `2^15` is, and no division.
struct Garner {
    primes: Vec<u64>,
    magic: Vec<u64>,
    inverses: Vec<Vec<u64>>,
}

impl Garner {
    fn of(primes: &[u16]) -> Garner {
        let limbs = primes.len();
        Garner {
            primes: primes.iter().map(|&q| q as u64).collect(),
            magic: primes.iter().map(|&q| (1u64 << 43) / q as u64).collect(),
            inverses: (0..limbs)
                .map(|k| {
                    (0..k)
                        .map(|m| {
                            let (q, p) = (primes[k] as u64, primes[m] as u64);
                            (1..q).find(|x| x * p % q == 1).expect("the primes are distinct")
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// `x mod q_k` for `x < 2^31`.
    #[inline(always)]
    fn reduce(&self, k: usize, x: u64) -> u64 {
        let q = self.primes[k];
        let r = x - ((x * self.magic[k]) >> 43) * q;
        if r >= q {
            r - q
        } else {
            r
        }
    }

    #[inline]
    fn lift(&self, residues: &[u64], mixed: &mut [u64]) -> u128 {
        let limbs = self.primes.len();
        for k in 0..limbs {
            let q = self.primes[k];
            let mut x = residues[k];
            for m in 0..k {
                // `mixed[m] < q_m < 2^15`, so `x + q - reduce(mixed[m]) < 2^16` and the
                // product with an inverse below `q` stays below `2^31`.
                x = self.reduce(k, (x + q - self.reduce(k, mixed[m])) * self.inverses[k][m]);
            }
            mixed[k] = x;
        }
        let mut value = 0u128;
        for k in (0..limbs).rev() {
            value = value * self.primes[k] as u128 + mixed[k] as u128;
        }
        value
    }
}

/// The commitment's columns as coefficients modulo `modulus`, which is what the proof carries,
/// from the kernel's raw rows: `rows[c * limbs + k]` is column `c`'s 648 NTT-domain rows
/// modulo limb `k`. Each limb's rows go through that limb's inverse transform
/// ([`labinius::bd::coefficients_from_rows`]) 32 columns at a time, and Garner's algorithm
/// lifts the residues to the single integer modulo their product; the lift is exact because
/// `modulus` is that product. The column blocks are independent and run on `workers` threads,
/// and each block packs directly into its own range of proof bytes.
fn commitment_bytes_rows(
    rows: &[[u32; N]],
    primes: &[u16],
    modulus: u128,
    columns: usize,
    workers: usize,
    buffer: Option<Vec<u8>>,
) -> Vec<u8> {
    let limbs = primes.len();
    let hybrid=modulus==modulus_choice::HYBRID_MODULUS;
    let stride=if hybrid {3}else{limbs};
    assert_eq!(rows.len(), columns * stride);
    let garner = Garner::of(primes);
    let width = rq::coefficient_bytes(modulus);
    assert!(!hybrid || width == 5);
    // Every byte is written below; a buffer of the right length faulted in earlier saves the
    // lift its page faults (two huge pages that every runner would otherwise queue on).
    let mut out = match buffer {
        Some(b) if b.len() == columns * N * width => b,
        _ => vec![0u8; columns * N * width],
    };
    let blocks = columns.div_ceil(32);
    let runs = labinius::par::ranges(blocks, workers, 1);
    let out_addr = out.as_mut_ptr() as usize;
    // One run per logical CPU, placed explicitly for this dispatch and released at its end:
    // this is the first dispatch after the commitment's per-core placement, and the runner
    // spread the scheduler produced from there measured the phase at twice its cost.
    let placement = labinius::par::CorePlacement::for_logical(runs.len());
    let wide_lift = hybrid && lift_hybrid_wide_available();
    labinius::par::pool().run(runs.len(), |r| {
        let _affinity = placement.as_ref().and_then(|p| p.pin(r));
        let mut residues: Vec<Vec<i16>> = (0..limbs).map(|_| vec![0i16; 32 * N]).collect();
        let mut mixed = vec![0u64; limbs];
        let mut at = vec![0u64; limbs];
        for blk in runs[r].clone() {
            let c0 = blk * 32;
            let cols = (columns - c0).min(32);
            for k in 0..limbs {
                let refs: Vec<&[u32; N]> = (0..cols).map(|c| &rows[(c0 + c) * stride + k]).collect();
                labinius::bd::coefficients_from_rows(primes[k], &refs, &mut residues[k][..cols * N]);
            }
            // Block `blk` owns these disjoint, fixed-width wire coefficients.
            let dst = unsafe {
                std::slice::from_raw_parts_mut((out_addr as *mut u8).add(c0 * N * width), cols * N * width)
            };
            if hybrid && wide_lift {
                for p in 0..cols {
                    let (at, end) = (p * N, (p + 1) * N);
                    // Column `p` owns its `5 N` bytes of `dst`.
                    unsafe {
                        lift_hybrid_column(
                            &residues[0][at..end],
                            &residues[1][at..end],
                            &rows[(c0 + p) * 3 + 2],
                            &mut dst[5 * at..5 * end],
                        )
                    };
                }
            } else if hybrid {
                for (i, word) in dst.chunks_exact_mut(5).enumerate() {
                    let value = lift_hybrid(residues[0][i], residues[1][i], rows[(c0 + i / N) * 3 + 2][i % N]);
                    debug_assert!((value as u128) < modulus);
                    word.copy_from_slice(&value.to_le_bytes()[..5]);
                }
            } else {
                for (i, word) in dst.chunks_exact_mut(width).enumerate() {
                    for k in 0..limbs {
                        at[k] = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
                    }
                    let value = garner.lift(&at, &mut mixed);
                    debug_assert!(value < modulus);
                    word.copy_from_slice(&value.to_le_bytes()[..width]);
                }
            }
        }
    });
    out
}

/// Whether [`lift_hybrid_column`] can run here.
fn lift_hybrid_wide_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Output byte `k < 40` of eight lifted coefficients is byte `k % 5` of 64-bit lane `k / 5`.
#[cfg(target_arch = "x86_64")]
static LIFT_BYTES: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut k = 0;
    while k < 40 {
        t[k] = ((k / 5) * 8 + k % 5) as u8;
        k += 1;
    }
    t
};

/// [`lift_hybrid`] of one column's `N` coefficients, eight per step in 64-bit lanes, written as
/// the wire's 5-byte little-endian words: the same arithmetic with `t = b - a` brought into
/// `[0, 9721)` instead of `b + 9721 - a`, and `x mod 9721` for `x = 3242 t < 2^25` by Barrett's
/// `q = floor(x ceil(2^45 / 9721) / 2^45)`, exact there (the error term is below
/// `x / 2^45 < 2^-20 < 1 / 9721`).
///
/// # Safety
/// The CPU must have AVX-512 F, BW and VBMI ([`lift_hybrid_wide_available`]); `r0`, `r1` must
/// hold `N` residues each and `out` must be `5 N` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn lift_hybrid_column(r0: &[i16], r1: &[i16], tail: &[u32; N], out: &mut [u8]) {
    use core::arch::x86_64::*;
    assert!(r0.len() == N && r1.len() == N && out.len() == 5 * N && N % 8 == 0);
    let q0 = _mm512_set1_epi64(3889);
    let q1 = _mm512_set1_epi64(9721);
    let inv = _mm512_set1_epi64(3242);
    let magic = _mm512_set1_epi64(3_619_418_999);
    let k153 = _mm512_set1_epi64(153);
    let low10 = _mm512_set1_epi64(1023);
    let product = _mm512_set1_epi64(3889 * 9721);
    let pick = _mm512_loadu_si512(LIFT_BYTES.as_ptr() as *const _);
    let keep: __mmask64 = (1u64 << 40) - 1;
    for j in 0..N / 8 {
        let a = _mm512_cvtepi16_epi64(_mm_loadu_si128(r0.as_ptr().add(8 * j) as *const __m128i));
        let a = _mm512_add_epi64(a, _mm512_and_si512(_mm512_srai_epi64::<63>(a), q0));
        let b = _mm512_cvtepi16_epi64(_mm_loadu_si128(r1.as_ptr().add(8 * j) as *const __m128i));
        let b = _mm512_add_epi64(b, _mm512_and_si512(_mm512_srai_epi64::<63>(b), q1));
        let t = _mm512_sub_epi64(b, a);
        let t = _mm512_add_epi64(t, _mm512_and_si512(_mm512_srai_epi64::<63>(t), q1));
        let x = _mm512_mul_epu32(t, inv);
        let q = _mm512_srli_epi64::<45>(_mm512_mul_epu32(x, magic));
        let m = _mm512_sub_epi64(x, _mm512_mul_epu32(q, q1));
        let value = _mm512_add_epi64(a, _mm512_mul_epu32(m, q0));
        let tl = _mm512_cvtepu32_epi64(_mm256_loadu_si256(tail.as_ptr().add(8 * j) as *const __m256i));
        let digit = _mm512_and_si512(_mm512_mul_epu32(_mm512_sub_epi64(tl, value), k153), low10);
        let lifted = _mm512_add_epi64(value, _mm512_mul_epu32(digit, product));
        _mm512_mask_storeu_epi8(
            out.as_mut_ptr().add(40 * j) as *mut i8,
            keep,
            _mm512_permutexvar_epi8(pick, lifted),
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn lift_hybrid_column(_: &[i16], _: &[i16], _: &[u32; N], _: &mut [u8]) {
    unreachable!("no wide lift off x86-64")
}

/// The hybrid modulus's lift with constant reductions: centered residues modulo 3889 and 9721
/// (3889^-1 mod 9721 = 3242), then the 1024 digit from the tail row's residue.
#[inline(always)]
fn lift_hybrid(r0: i16, r1: i16, tail: u32) -> u64 {
    let a = (if r0 < 0 { r0 as i32 + 3889 } else { r0 as i32 }) as u64;
    let b = (if r1 < 0 { r1 as i32 + 9721 } else { r1 as i32 }) as u64;
    let value = a + 3889 * (((b + 9721 - a) * 3242) % 9721);
    let digit = (tail as u64).wrapping_sub(value).wrapping_mul(153) & 1023;
    value + 3889 * 9721 * digit
}

/// The baseline's recombination through the library's commitment matrix, kept as the reference
/// the tests check [`commitment_bytes_rows`] against.
#[cfg(test)]
fn commitment_coefficients(commitment: &Commitment, primes: &[u16], modulus: u128) -> Vec<u128> {
    let residues = labinius::bd::column_coefficients(commitment.matrix(), primes);
    let garner = Garner::of(primes);
    let limbs = primes.len();
    let mut mixed = vec![0u64; limbs];
    let mut at = vec![0u64; limbs];
    let mut out = Vec::with_capacity(residues[0].len());
    for i in 0..residues[0].len() {
        for k in 0..limbs {
            at[k] = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
        }
        let value = garner.lift(&at, &mut mixed);
        debug_assert!(value < modulus);
        out.push(value);
    }
    out
}

/// The switch's rounds: the messages, the coordinates and the opened value, driven through
/// either prover.
trait SwitchRounds {
    fn msg(&mut self) -> [F162; 2];
    fn fold(&mut self, r: F162);
    fn final_eval(&self) -> F162;
}
impl SwitchRounds for cf::LeanSwitch<'_> {
    fn msg(&mut self) -> [F162; 2] {
        cf::LeanSwitch::msg(self)
    }
    fn fold(&mut self, r: F162) {
        cf::LeanSwitch::fold(self, r)
    }
    fn final_eval(&self) -> F162 {
        cf::LeanSwitch::final_eval(self)
    }
}
impl SwitchRounds for cf::SwitchProver {
    fn msg(&mut self) -> [F162; 2] {
        cf::SwitchProver::msg(self)
    }
    fn fold(&mut self, r: F162) {
        cf::SwitchProver::fold(self, r)
    }
    fn final_eval(&self) -> F162 {
        cf::SwitchProver::final_eval(self)
    }
}
fn switch_rounds(
    sw: &mut impl SwitchRounds,
    t: &mut Transcript,
) -> (Vec<[[u64; 3]; 2]>, Vec<F162>, F162) {
    let mut rounds = Vec::with_capacity(LOG_PACKED);
    let mut coords = Vec::with_capacity(LOG_PACKED);
    for _ in 0..LOG_PACKED {
        let msg = sw.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        sw.fold(r);
    }
    let opened = sw.final_eval();
    t.absorb_f162(&[opened]);
    (rounds, coords, opened)
}

/// Batch positions of the fold recomputed through the transform path and compared with the
/// bit-sliced fold: 32 ring elements each, out of 4096. Every challenge and every column enters
/// each element, so one position exercises the whole accumulation; a mismatch means a defect
/// in the bit-sliced path on this machine, and the transform path then produces the fold.
const FOLD_CHECK_POSITIONS: core::ops::Range<usize> = 0..1;

/// `v = sum_j c_j W_j`: the bit-sliced fold from the words, spot-checked against the transform
/// path on [`FOLD_CHECK_POSITIONS`], with the transform path as the fallback.
fn fold_checked(
    values: &[B128],
    challenges: &[ShortChallenge],
    base: u16,
    workers: usize,
) -> Vec<labinius::RingElement648> {
    let bpc = COLUMN_LEN / 128;
    let bits = labinius::fold::fold_witness_bits(values, COLUMNS, COLUMN_LEN, challenges);
    let check = labinius::fold::fold_witness_positions(
        values,
        COLUMNS,
        challenges,
        bpc,
        base,
        FOLD_CHECK_POSITIONS,
    );
    let first = 32 * FOLD_CHECK_POSITIONS.start;
    let agree = check
        .iter()
        .zip(&bits[first..first + check.len()])
        .all(|(a, b)| a.v == b.v);
    if agree {
        return bits;
    }
    eprintln!("labinius-fold-check mismatch: the bit-sliced fold disagrees with the transform path; folding through the transform");
    labinius::fold::fold_witness_streaming(values, COLUMNS, challenges, bpc, base, workers)
}

/// One commitment task per physical core, temporarily placed by the key's
/// worker entry point; eight columns share each pass over the public key.
const EMBEDDED_TAIL_LEN:usize=include_bytes!(concat!(env!("OUT_DIR"),"/tail-key.bin")).len();
static EMBEDDED_TAIL:Aligned64<[u8;EMBEDDED_TAIL_LEN]>=Aligned64(*include_bytes!(concat!(env!("OUT_DIR"),"/tail-key.bin")));
/// Both hybrid keys (base and tail, both primes) in the hybrid sink's arrangement, precomputed
/// by `build.rs` from the same public keys as the two images above (empty when the build host
/// could not run the kernels); read in place.
const EMBEDDED_HYBRID_LEN:usize=include_bytes!(concat!(env!("OUT_DIR"),"/hybrid-key.bin")).len();
static EMBEDDED_HYBRID:Aligned64<[u8;EMBEDDED_HYBRID_LEN]>=Aligned64(*include_bytes!(concat!(env!("OUT_DIR"),"/hybrid-key.bin")));
/// Prefix blocks of the partials pass: their sets supply the switch's first `log2` messages,
/// and the switch's binding pass then binds that many coordinates at once.
const PREFIX_SETS: usize = 32;
const COMMIT_GROUP: usize = 8;
const COMMIT_WORKERS: usize = 4;
/// Columns per worker of the hybrid commitment (12 KB of accumulators per column and prime). On
/// a host of two-thread cores the two threads of a core take the halves of each 32-column group,
/// so each batch of the arranged keys, 168 KB, is read from L3 once per 32 columns; 32 per
/// worker measured 1.8 ms slower (L2 pressure).
const HYBRID_GROUP: usize = 16;
/// The group size the commitment runs with: columns per worker, which sets how much
/// of the arranged key is reused from L2 and how large the accumulators get.
fn hybrid_group() -> usize {
    match std::env::var("LABINIUS_HYBRID_GROUP") {
        Ok(v) => v.parse::<usize>().unwrap_or(HYBRID_GROUP).max(1),
        Err(_) => HYBRID_GROUP,
    }
}


// =============================================================================================
// the proof file
// =============================================================================================

/// What [`prove`] produces: the proof's fields, the folded witness still as ring elements (its
/// wire form is written straight into the proof file).
struct ProofParts {
    modulus: u128,
    commitment: Vec<u8>,
    partials: Vec<u128>,
    rounds: Vec<[[u64; 3]; 2]>,
    opened: [u64; 3],
    row: Vec<u8>,
    folded: FoldedWitness,
}

impl ProofParts {
    fn into_proof(self) -> Proof {
        Proof {
            version: 1,
            modulus: self.modulus,
            folded: pack_folded(&self.folded),
            commitment: self.commitment,
            partials: self.partials,
            rounds: self.rounds,
            opened: self.opened,
            row: self.row,
        }
    }
}

/// Where the codec's encoding of a [`Proof`] puts each field, for a proof of this instance's shape
/// under `modulus`: fixed-width integers little-endian, each byte string and sequence after its
/// `u64` length, the `opened` array bare.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ProofLayout {
    commitment_bytes: usize,
    row_bytes: usize,
    folded_bytes: usize,
    partials: usize,
    rounds: usize,
}

impl ProofLayout {
    const HEAD: usize = 4 + 16 + 8;

    fn of_instance(modulus: u128) -> ProofLayout {
        ProofLayout {
            commitment_bytes: COLUMNS * N * rq::coefficient_bytes(modulus),
            row_bytes: (162 * COLUMNS).div_ceil(8),
            folded_bytes: FOLDED_BYTES * N * RING_ELEMENTS,
            partials: 1 << 7,
            rounds: LOG_PACKED,
        }
    }

    fn of_parts(parts: &ProofParts) -> ProofLayout {
        ProofLayout {
            commitment_bytes: parts.commitment.len(),
            row_bytes: parts.row.len(),
            folded_bytes: FOLDED_BYTES * N * parts.folded.elements().len(),
            partials: parts.partials.len(),
            rounds: parts.rounds.len(),
        }
    }

    /// The bytes between the commitment and the folded witness's bytes.
    fn middle(&self) -> usize {
        8 + 16 * self.partials + 8 + 48 * self.rounds + 24 + 8 + self.row_bytes + 8
    }

    fn commitment_at(&self) -> usize {
        Self::HEAD
    }

    fn middle_at(&self) -> usize {
        Self::HEAD + self.commitment_bytes
    }

    fn folded_at(&self) -> usize {
        self.middle_at() + self.middle()
    }

    fn len(&self) -> usize {
        self.folded_at() + self.folded_bytes
    }
}

/// The codec's encoding of `parts`, written into `out` (`layout.len()` bytes): the small
/// fields on this thread, the commitment and the folded witness (`i32` little-endian per
/// coefficient) in parallel on the pool.
///
/// # Safety
/// `out` must be valid for writes of `layout.len()` bytes, and `layout` must be
/// [`ProofLayout::of_parts`]`(parts)`.
unsafe fn encode_into(parts: &ProofParts, layout: &ProofLayout, out: *mut u8) {
    let put = |at: &mut usize, bytes: &[u8]| {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(*at), bytes.len());
        *at += bytes.len();
    };
    let mut at = 0;
    put(&mut at, &1u32.to_le_bytes());
    put(&mut at, &parts.modulus.to_le_bytes());
    put(&mut at, &(parts.commitment.len() as u64).to_le_bytes());
    debug_assert_eq!(at, layout.commitment_at());
    let mut at = layout.middle_at();
    let mut middle = Vec::with_capacity(layout.middle());
    middle.extend_from_slice(&(parts.partials.len() as u64).to_le_bytes());
    for v in &parts.partials {
        middle.extend_from_slice(&v.to_le_bytes());
    }
    middle.extend_from_slice(&(parts.rounds.len() as u64).to_le_bytes());
    for r in &parts.rounds {
        for x in r.iter().flatten() {
            middle.extend_from_slice(&x.to_le_bytes());
        }
    }
    for x in &parts.opened {
        middle.extend_from_slice(&x.to_le_bytes());
    }
    middle.extend_from_slice(&(parts.row.len() as u64).to_le_bytes());
    middle.extend_from_slice(&parts.row);
    middle.extend_from_slice(&(layout.folded_bytes as u64).to_le_bytes());
    debug_assert_eq!(middle.len(), layout.middle());
    put(&mut at, &middle);
    debug_assert_eq!(at, layout.folded_at());
    let elements = parts.folded.elements();
    assert!(
        elements
            .iter()
            .all(|e| e.representation == Representation::Coefficients),
        "a fold is in coefficient form"
    );
    // A few commitment slices and many folded runs, taken dynamically.
    let ctasks = 4;
    let cruns = labinius::par::ranges(parts.commitment.len(), ctasks, 4096);
    let fruns = labinius::par::ranges(elements.len(), 4 * threads(), 1);
    let (addr, folded_at) = (out as usize, layout.folded_at());
    labinius::par::pool().run(cruns.len() + fruns.len(), |i| {
        let out = addr as *mut u8;
        if i < cruns.len() {
            let r = cruns[i].clone();
            core::ptr::copy_nonoverlapping(
                parts.commitment.as_ptr().add(r.start),
                out.add(ProofLayout::HEAD + r.start),
                r.len(),
            );
            return;
        }
        let r = fruns[i - cruns.len()].clone();
        let mut dst = out.add(folded_at + FOLDED_BYTES * N * r.start);
        let mut wide = [0i32; N];
        for e in &elements[r] {
            for (w, &c) in wide.iter_mut().zip(e.v.iter()) {
                *w = (c as i32).to_le();
            }
            core::ptr::copy_nonoverlapping(wide.as_ptr() as *const u8, dst, FOLDED_BYTES * N);
            dst = dst.add(FOLDED_BYTES * N);
        }
    });
}

/// The proof file, created and mapped while the prover runs: the pool's idle workers create it
/// at its exact final length (blocks preallocated), map it at a 2 MB-aligned address with huge
/// pages requested, and fault its page-cache pages in (`MADV_POPULATE_WRITE`, 2 MB per idle
/// chunk, on the workers idle while the key is set up), so that at the end the proof is encoded
/// straight into mapped pages on every core. Written with `write`, the 14 MB proof cost ~3.5 ms
/// on one core after the prover's last phase (ext4 allocates, reserves and copies page by page
/// under the inode lock). With ext4's large folios the aligned mapping is mostly 2 MB page-cache
/// folios mapped by PMDs: faulting them in runs in parallel, and the exit drops a few PMDs
/// instead of 3400 dirty PTEs (~0.6 ms).
struct Output {
    state: std::sync::Arc<OutputState>,
    job: &'static labinius::par::IdleJob,
    populate: std::sync::OnceLock<&'static labinius::par::IdleJob>,
    /// The commitment's bytes, allocated at the start and faulted in by the setup's idle work.
    commitment: std::sync::Mutex<Option<Vec<u8>>>,
}

struct OutputState {
    path: String,
    layout: ProofLayout,
    map: std::sync::OnceLock<Option<Mapped>>,
    /// Faulting the pages in failed (a kernel without `MADV_POPULATE_WRITE`): the encoding
    /// faults them itself.
    populate_failed: std::sync::atomic::AtomicBool,
}

struct Mapped {
    _file: std::fs::File,
    addr: usize,
}

/// Bytes of the proof file each idle chunk faults in: one huge page.
const POPULATE_CHUNK: usize = 2 << 20;

impl OutputState {
    #[cfg(target_os = "linux")]
    fn create_and_map(&self) -> Option<Mapped> {
        use std::os::fd::AsRawFd;
        let len = self.layout.len();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .ok()?;
        // Allocated blocks spare the page faults ext4's delayed-allocation reservations (and
        // let them run in parallel); the length alone is enough where the filesystem cannot
        // preallocate.
        if unsafe { sys::fallocate(file.as_raw_fd(), 0, 0, len as i64) } != 0 {
            file.set_len(len as u64).ok()?;
        }
        // A 2 MB-aligned address, so that file offset 0 starts a huge page: reserve, then map
        // the file over the reservation.
        const HUGE: usize = 2 << 20;
        let span = len.div_ceil(HUGE) * HUGE;
        let reserved = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                span + HUGE,
                sys::PROT_NONE,
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if reserved as isize == -1 || reserved.is_null() {
            return None;
        }
        let at = (reserved as usize + HUGE - 1) & !(HUGE - 1);
        let ptr = unsafe {
            sys::mmap(
                at as *mut _,
                len,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED | sys::MAP_FIXED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 || ptr as usize != at {
            return None;
        }
        // Advisory: without large page-cache folios the mapping is 4 KB pages.
        unsafe { sys::madvise(ptr, len, sys::MADV_HUGEPAGE) };
        Some(Mapped {
            _file: file,
            addr: ptr as usize,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn create_and_map(&self) -> Option<Mapped> {
        None
    }

    fn mapped(&self) -> Option<&Mapped> {
        self.map.get_or_init(|| self.create_and_map()).as_ref()
    }

    /// Idle chunk `i`: create and map the file (`i == 0`), or fault in its `i - 1`-th piece.
    fn chunk(&self, i: usize) {
        use std::sync::atomic::Ordering;
        let Some(m) = self.mapped() else { return };
        if i == 0 || self.populate_failed.load(Ordering::Relaxed) {
            return;
        }
        let at = (i - 1) * POPULATE_CHUNK;
        let len = POPULATE_CHUNK.min(self.layout.len() - at);
        #[cfg(target_os = "linux")]
        if unsafe { sys::madvise((m.addr + at) as *mut _, len, sys::MADV_POPULATE_WRITE) } != 0 {
            self.populate_failed.store(true, Ordering::Relaxed);
        }
    }
}

impl Output {
    fn start(path: &str, modulus: u128) -> Output {
        let layout = ProofLayout::of_instance(modulus);
        let state = std::sync::Arc::new(OutputState {
            path: path.to_owned(),
            layout,
            map: std::sync::OnceLock::new(),
            populate_failed: std::sync::atomic::AtomicBool::new(false),
        });
        // Idle work for the workers while the key is set up: create and map the file, build
        // the challenge sampler's tables (needed ~290 ms in, by the folding challenges), read
        // the CPU topology the commitment's placement asks for, and fault in the commitment's
        // buffer, a huge page per chunk.
        let mut commitment = vec![0u8; layout.commitment_bytes];
        let (addr, bytes) = (commitment.as_mut_ptr() as usize, commitment.len());
        let pages = bytes.div_ceil(POPULATE_CHUNK);
        let worker_state = state.clone();
        let job = labinius::par::pool().idle(3 + pages, move |i| match i {
            0 => worker_state.chunk(0),
            1 => labinius::challenge::warm(),
            2 => labinius::par::warm_topology(),
            // The buffer is not touched by anyone else before `write` or `commitment_buffer`
            // quiesces this job; writing a zero where the fresh mapping already reads zero.
            _ => unsafe {
                let at = (i - 3) * POPULATE_CHUNK;
                core::ptr::write_volatile((addr + at) as *mut u8, 0);
                core::ptr::write_volatile((addr + (at + POPULATE_CHUNK).min(bytes) - 1) as *mut u8, 0);
            },
        });
        Output {
            state,
            job,
            populate: std::sync::OnceLock::new(),
            commitment: std::sync::Mutex::new(Some(commitment)),
        }
    }

    /// The commitment's buffer, once the setup's idle work is over (stopped if it is not).
    fn commitment_buffer(&self) -> Option<Vec<u8>> {
        labinius::par::pool().quiesce(self.job);
        self.commitment.lock().unwrap().take()
    }

    /// Fault the file's pages in on the idle workers, 2 MB per chunk. Called where the main
    /// thread is about to work alone (the transcript absorbing the commitment): a worker still
    /// in a chunk when a dispatch arrives delays it, and the commitment's coarse work units
    /// turn any delay at its start into a delay of its end.
    ///
    /// `first` are anonymous regions to fault in before the file, one chunk each (writable,
    /// and only ever populated: `MADV_POPULATE_WRITE` leaves what they hold as it is).
    fn populate(&self, first: Vec<(usize, usize)>) {
        let chunks = self.state.layout.len().div_ceil(POPULATE_CHUNK);
        let worker_state = self.state.clone();
        let job = labinius::par::pool().idle(first.len() + chunks, move |i| match first.get(i) {
            #[cfg(target_os = "linux")]
            Some(&(addr, len)) => unsafe {
                sys::madvise(addr as *mut _, len, sys::MADV_POPULATE_WRITE);
            },
            #[cfg(not(target_os = "linux"))]
            Some(_) => {}
            None => worker_state.chunk(i - first.len() + 1),
        });
        let _ = self.populate.set(job);
    }

    /// Write the proof: encoded into the mapping when it has the expected layout, else through
    /// the codec into a freshly truncated file.
    fn write(self, parts: ProofParts) {
        labinius::par::pool().quiesce(self.job);
        if let Some(job) = self.populate.get() {
            labinius::par::pool().quiesce(job);
        }
        let layout = ProofLayout::of_parts(&parts);
        if layout == self.state.layout {
            if let Some(m) = self.state.mapped() {
                unsafe { encode_into(&parts, &layout, m.addr as *mut u8) };
                // The pages are the file's; the exit unmaps them.
                return;
            }
        }
        #[cfg(target_os = "linux")]
        if let Some(Some(m)) = self.state.map.get() {
            unsafe { sys::munmap(m.addr as *mut _, self.state.layout.len()) };
        }
        use std::io::Write;
        let file = std::fs::File::create(&self.state.path).expect("create proof file");
        let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
        codec()
            .serialize_into(&mut out, &parts.into_proof())
            .expect("write proof");
        out.flush().expect("flush proof");
    }
}

fn prove(seed: &[u8; 32], modulus: u128, table_path: &str, output: Option<&Output>) -> ProofParts {
    let load_clock = std::time::Instant::now();
    let table = Table::load(table_path);
    let values = table.words();
    let load_time = load_clock.elapsed();
    let workers = threads();
    let clock = std::time::Instant::now();
    // The pool's workers are started here, once; every parallel phase runs on them.
    let _ = labinius::par::pool();
    // The switch's buffers: huge pages requested, faulted in by their first writers on every
    // core. (A thread touching them during the commitment ran on a hyperthread sibling of a
    // commitment worker and cost the commitment more than it saved the expansion.)
    let planes = cf::LeanSwitch::planes(1 << LOG_PACKED);
    let primes = modulus_choice::limbs(modulus).expect("invalid commitment modulus");
    let p = params(&primes);
    // The table's pages come in on every core; the key is the embedded one when the modulus
    // is the default and the build host could compute it, and is built otherwise.
    // ... and so do the embedded arranged keys' (otherwise a page-cache fault per 64 KB inside
    // the commitment, taken by all its workers at once as they reach the same batches: measured
    // 2 ms of commitment for 0.3-0.5 ms here).
    let image: &[u8] = if modulus == modulus_choice::HYBRID_MODULUS && modulus == MODULUS { &EMBEDDED_HYBRID.0 } else { &[] };
    populate_words(values, image, workers);
    let matrix_time = clock.elapsed();
    // The hybrid modulus commits against the embedded arranged keys when the build host made
    // them; otherwise the key (and the tail key) come from their images or are built.
    let arranged = if modulus == modulus_choice::HYBRID_MODULUS && modulus == MODULUS {
        labinius::simd::commit::hybrid::keys_from_image(&EMBEDDED_HYBRID.0, keygen::BATCHES)
    } else {
        None
    };
    let key = || match keygen::key_from_image(modulus, &EMBEDDED_KEY.0) {
        Some(key) => key,
        None => keygen::build_key(modulus),
    };
    let setup_time = clock.elapsed();
    let columns: Vec<&[B128]> = values.chunks_exact(COLUMN_LEN).collect();
    assert_eq!(columns.len(), COLUMNS, "the table is COLUMNS columns");
    let rows = if let Some(keys) = arranged {
        labinius::simd::commit::hybrid::commit_columns_rows_arranged(keys,&columns,workers,hybrid_group())
    } else if modulus==modulus_choice::HYBRID_MODULUS {
        let tail=keygen::key_from_image(modulus,&EMBEDDED_TAIL.0).unwrap_or_else(||keygen::build_tail_key(modulus));
        labinius::simd::commit::hybrid::commit_columns_rows(&key(),&tail,&columns,workers,HYBRID_GROUP)
    } else {key().commit_columns_rows(&columns, workers.min(COMMIT_WORKERS), COMMIT_GROUP)};
    let commit_time = clock.elapsed();
    let buffer = output.and_then(|o| o.commitment_buffer());
    let commitment = commitment_bytes_rows(&rows, &primes, modulus, COLUMNS, workers, buffer);
    let recombine_time = clock.elapsed();
    // Huge pages: a few page-table entries to drop. (A thread of its own to free it, as the
    // baseline did, is a ninth runnable thread beside eight busy ones.)
    drop(rows);
    if let Some(output) = output {
        // The pages of the switch's binding pass first: they are written ~10 ms from here,
        // and faulting them in there stalls the pass's runners.
        let bound = ((1usize << LOG_PACKED) / PREFIX_SETS * 8).min(planes[0].len() * 8);
        let first = planes.iter().flat_map(|p| {
            let (addr, _) = p.region();
            (0..bound).step_by(POPULATE_CHUNK).map(move |o| (addr + o, POPULATE_CHUNK.min(bound - o)))
        });
        output.populate(first.collect());
    }
    let mut t = transcript(seed, modulus, &commitment);
    let (lo, hi) = point(&mut t);
    // A coordinate equal to one (probability 2^-128 per coordinate) has no half ratio: the
    // baseline's prover over its own full workspace takes over for the whole switch.
    let degenerate = hi.iter().any(|&r| r == B128::ONE);
    let prefix_sets = if degenerate {
        None
    } else {
        Some(cf::eq_expand_and_partials_factored_blocks::<PREFIX_SETS>(&hi, values, workers))
    };
    let partials = match &prefix_sets {
        Some((sets, _)) => cf::partials_from_sets(sets, &hi),
        None => {
            let mut ws = cf::Workspace::new(LOG_PACKED, 1);
            ws.partial_evals(0, values, &hi)
        }
    };
    let value = claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let partials_time = clock.elapsed();
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let lean = prefix_sets.and_then(|(sets, eq)| {
        // The first block of the expansion, as factors: the switch recomputes its tiles.
        cf::LeanSwitch::with_sets_factored(values, eq, &hi, &batch, workers, planes, sets.into())
    });
    let (rounds, coords, opened) = match lean {
        Some(mut sw) => switch_rounds(&mut sw, &mut t),
        None => {
            // The baseline's prover, whose workspace holds its own full expansion.
            let mut ws = cf::Workspace::new(LOG_PACKED, 1);
            let again = ws.partial_evals(0, values, &hi);
            assert_eq!(again, partials);
            let mut sw = cf::SwitchProver::from_workspace(ws, values, &[F162::ONE], &batch);
            switch_rounds(&mut sw, &mut t)
        }
    };
    let switch_time = clock.elapsed();
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = RowEvaluation::of(labinius::eval::row_evaluate_words(values, point.p0(), workers));
    let row_time = clock.elapsed();
    let challenges = folding_challenges(&mut t, &row);
    let challenge_time = clock.elapsed();
    let folded = FoldedWitness::of(labinius::fold::fold_witness_bitsliced(
        values, COLUMNS, &challenges, COLUMN_LEN / 128, primes[0], workers,
    ));
    let fold_time = clock.elapsed();
    // The mapping is left to the exit, which tears it down without the TLB shootdowns an unmap
    // beside the pool's threads needs.
    std::mem::forget(table);
    let proof = ProofParts {
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded,
    };
    let encode_time = clock.elapsed();
    let _ = PROFILE.set(format!("labinius-profile load_us={} matrix_us={} setup_us={} commit_us={} recombine_us={} partials_us={} switch_us={} row_us={} challenges_us={} fold_us={} encode_us={} total_us={} threads={} fold_gfni={}\n",
        load_time.as_micros(), matrix_time.as_micros(), (setup_time-matrix_time).as_micros(),
        (commit_time-setup_time).as_micros(), (recombine_time-commit_time).as_micros(),
        (partials_time-recombine_time).as_micros(), (switch_time-partials_time).as_micros(),
        (row_time-switch_time).as_micros(), (challenge_time-row_time).as_micros(),
        (fold_time-challenge_time).as_micros(), (encode_time-fold_time).as_micros(),
        encode_time.as_micros(), workers,
        labinius::fold::FOLD_USED_GFNI.load(std::sync::atomic::Ordering::Relaxed)));
    proof
}

/// The run's `labinius-profile` line, written with the others at the end: stderr is unbuffered,
/// and a formatted line goes out as one `write` per piece.
static PROFILE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// CLOCK_MONOTONIC in nanoseconds: the clock an outside timer (Python's perf_counter on Linux)
/// reads, so process startup and teardown can be told apart from the prover's own phases.
fn monotonic_ns() -> u128 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    unsafe { clock_gettime(1, &mut ts) };
    ts.sec as u128 * 1_000_000_000 + ts.nsec as u128
}

/// Run every thread of this process under `SCHED_IDLE`: on a CPU where another runnable task
/// appears (a timer's wakeup, a kernel worker, the runner's agent) that task runs at once instead
/// of waiting for one of the prover's threads to be preempted. The prover still has every CPU
/// nothing else wants, which on an otherwise idle host is all of them. Threads created later
/// inherit the policy. Lowering one's own policy needs no privilege; a refusal changes nothing.
/// `LABINIUS_NORMAL_SCHED` (any value) keeps the default policy, for A/B measurements.
fn yield_to_everything_else() {
    #[repr(C)]
    struct SchedParam {
        priority: i32,
    }
    extern "C" {
        fn sched_setscheduler(pid: i32, policy: i32, param: *const SchedParam) -> i32;
    }
    const SCHED_IDLE: i32 = 5;
    unsafe { sched_setscheduler(0, SCHED_IDLE, &SchedParam { priority: 0 }) };
}

fn main() {
    let main_ns = monotonic_ns();
    if std::env::var_os("LABINIUS_NORMAL_SCHED").is_none() {
        yield_to_everything_else();
    }
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 5 || args.len() == 7,
        "usage: labinius-fast --prove FILE --table FILE [--modulus Q]"
    );
    assert_eq!(args[1], "--prove", "unknown command");
    assert_eq!(args[3], "--table", "witness table path is required");
    let seed = seed();
    let modulus = if args.len() == 7 {
        assert_eq!(args[5], "--modulus", "unknown prover option");
        modulus_choice::parse(&args[6]).expect("invalid commitment modulus")
    } else {
        MODULUS
    };
    let output_clock = std::time::Instant::now();
    let output = Output::start(&args[2], modulus);
    let proof = prove(&seed, modulus, &args[4], Some(&output));
    let prove_return = output_clock.elapsed();
    output.write(proof);
    let write_time = output_clock.elapsed() - prove_return;
    let report = format!(
        "{}labinius-output prove_return_us={} write_us={}\nlabinius-clock main_ns={} exit_ns={}\n",
        PROFILE.get().map_or("", |p| p.as_str()),
        prove_return.as_micros(),
        write_time.as_micros(),
        main_ns,
        monotonic_ns()
    );
    {
        use std::io::Write;
        let _ = std::io::stderr().write_all(report.as_bytes());
    }
    // The proof is on disk; nothing the destructors would do is observable.
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use labinius::eval::eq_table;
    use labinius::ring::Representation;

    fn checked_f162(x: [u64; 3]) -> F162 {
        assert_eq!(x[2] >> 34, 0, "noncanonical F162 element");
        F162(x)
    }
    fn dot_f162(a: &[F162], b: &[F162]) -> F162 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(F162::ZERO, |acc, (&x, &y)| acc + x * y)
    }
    /// `v mod 2` as field elements, in the witness's index order.
    fn components_mod_2(v: &[labinius::RingElement648]) -> Vec<F162> {
        let mut out = vec![F162::ZERO; 4 * v.len()];
        for (m, e) in v.iter().enumerate() {
            for p in 0..rq::N / 4 {
                for k in 0..4 {
                    if e.v[4 * p + k] & 1 != 0 {
                        out[4 * m + k].0[p >> 6] |= 1u64 << (p & 63);
                    }
                }
            }
        }
        out
    }
    /// The checker's acceptance condition, reimplemented here so that the round trip tests the
    /// proof this prover writes rather than the objects it holds. The ring arithmetic is the
    /// same [`rq`] the checker runs.
    fn verify(seed: &[u8; 32], proof: &Proof) {
        assert_eq!(proof.version, 1);
        assert_eq!(proof.partials.len(), 128);
        assert_eq!(proof.rounds.len(), LOG_PACKED);
        let modulus = proof.modulus;
        let values = trace(seed);
        let mut t = transcript(seed, modulus, &proof.commitment);
        let (lo, hi) = point(&mut t);
        let mut ws = cf::Workspace::new(LOG_PACKED, 1);
        let trusted = ws.partial_evals(0, &values, &hi);
        let value = claim(&trusted, &lo);
        let partials: Vec<_> = proof.partials.iter().copied().map(B128).collect();
        t.absorb_b128(&[value]);
        t.absorb_b128(&partials);
        let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
        let mut sw = cf::SwitchVerifier::start(&partials, value, &lo, &batch).unwrap();
        let mut coords = Vec::new();
        for raw in &proof.rounds {
            let msg = raw.map(checked_f162);
            t.absorb_f162(&msg);
            let r = t.sample_f162(b"switch-round", 1)[0];
            sw.round(msg, r);
            coords.push(r);
        }
        let opened = checked_f162(proof.opened);
        sw.finish(&hi, &coords, &batch, opened).unwrap();
        t.absorb_f162(&[opened]);
        let (high, low) = coords.split_at(COLUMN_LOG as usize);
        let p0: Vec<F162> = low.iter().rev().copied().collect();
        let p1: Vec<F162> = high.iter().rev().copied().collect();
        let row = wire::unpack_row_evaluation(&proof.row, COLUMNS).unwrap();
        let folded = unpack_folded(&proof.folded, RING_ELEMENTS).unwrap();
        let challenges = {
            row.absorb(&mut t);
            (0..COLUMNS)
                .map(|_| {
                    labinius::challenge::sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(dot_f162(&eq_table(&p1), row.values()), opened);
        let v = folded.elements();
        assert_eq!(v.len(), RING_ELEMENTS);
        let mut normsq = 0u64;
        let mut worst = 0u128;
        for e in v {
            assert_eq!(e.representation, Representation::Coefficients);
            for &c in &e.v {
                let magnitude = u64::from((c as i32).unsigned_abs());
                normsq += magnitude * magnitude;
                worst = worst.max(magnitude as u128);
            }
        }
        assert!(normsq <= 293_534_171_136);
        assert!(worst <= (modulus - 1) / 2);
        let columns =
            rq::unpack_ring_elements(&proof.commitment, COLUMNS * rq::N, modulus).unwrap();
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let coefficients: Vec<i16> = v.iter().flat_map(|e| e.v).collect();
        assert_eq!(
            rq::a_times_v(&matrix, &coefficients, modulus),
            rq::challenge_combination(&columns, &challenges, modulus),
            "the fold does not open the folded commitment"
        );
        let parities: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
        assert_eq!(
            dot_f162(&eq_table(&p0), &components_mod_2(v)),
            dot_f162(&parities, row.values())
        );
    }

    struct TableFile(std::path::PathBuf);
    impl Drop for TableFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    impl TableFile {
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    fn table_file(seed: &[u8; 32]) -> TableFile {
        let path = std::env::temp_dir().join(format!(
            "labinius-fast-table-{}-{}.bin",
            std::process::id(),
            seed.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        ));
        let mut bytes = Vec::new();
        for value in trace(seed) {
            bytes.extend_from_slice(&value.0.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        TableFile(path)
    }

    #[test]
    fn loaded_table_matches_trace() {
        let seed = [11u8; 32];
        let table = table_file(&seed);
        let loaded = Table::load(table.path());
        assert_eq!(loaded.words(), &trace(&seed)[..]);
    }

    #[test]
    fn baseline_prove_reads_witness_file() {
        let path = std::env::temp_dir().join(format!(
            "labinius-fast-short-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [1u8; 32]).unwrap();
        let result = std::panic::catch_unwind(|| {
            prove(&[1; 32], MODULUS, path.to_str().unwrap(), None);
        });
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "the prover accepted a truncated witness table");
    }

    #[test]
    fn round_trip_and_reject_wrong_instance() {
        let seed = [42; 32];
        let table = table_file(&seed);
        let proof = prove(&seed, MODULUS, table.path(), None).into_proof();
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        verify(&seed, &decoded);
        assert!(std::panic::catch_unwind(|| verify(&[43; 32], &decoded)).is_err());
    }

    #[test]
    fn solution_selected_modulus_round_trip() {
        let modulus = 9721u128 * 2917 * 4861;
        assert_ne!(modulus, MODULUS);
        let seed = [44; 32];
        let table = table_file(&seed);
        let proof = prove(&seed, modulus, table.path(), None).into_proof();
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        assert_eq!(decoded.modulus, modulus);
        verify(&[44; 32], &decoded);
    }

    #[test]
    fn packers_match_the_checker_format() {
        let mut x = 0x2545F4914F6CDD1Du128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let coefficients: Vec<u128> = (0..3 * rq::N + 5).map(|_| next() % MODULUS).collect();
        assert_eq!(
            pack_coefficients(&coefficients, MODULUS),
            rq::pack_ring_elements(&coefficients, MODULUS)
        );
        let elements: Vec<labinius::RingElement648> = (0..19)
            .map(|_| {
                let mut e = labinius::RingElement648::zero(Representation::Coefficients);
                for slot in e.v.iter_mut() {
                    *slot = (next() % 2001) as i16 - 1000;
                }
                e
            })
            .collect();
        let folded = FoldedWitness::of(elements);
        let mut want = Vec::new();
        for e in folded.elements() {
            for &c in &e.v {
                want.extend_from_slice(&(c as i32).to_le_bytes());
            }
        }
        assert_eq!(pack_folded(&folded), want);
    }

    /// The proof encoded in place, as the mapped proof file is written, is the codec's encoding.
    #[test]
    fn mapped_encoding_is_the_codec_encoding() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for (commitment_bytes, row_bytes, elements) in [(0usize, 0usize, 0usize), (5 * 648 * 3 + 1, 41, 19), (5 * 648 * 64, 20736, 300)] {
            let parts = ProofParts {
                modulus: MODULUS,
                commitment: (0..commitment_bytes).map(|_| next() as u8).collect(),
                partials: (0..128).map(|_| ((next() as u128) << 64) | next() as u128).collect(),
                rounds: (0..24).map(|_| [[next(), next(), next()], [next(), next(), next()]]).collect(),
                opened: [next(), next(), next()],
                row: (0..row_bytes).map(|_| next() as u8).collect(),
                folded: FoldedWitness::of(
                    (0..elements)
                        .map(|_| {
                            let mut e = labinius::RingElement648::zero(Representation::Coefficients);
                            for c in e.v.iter_mut() {
                                *c = (next() % 4001) as i16 - 2000;
                            }
                            e
                        })
                        .collect(),
                ),
            };
            let layout = ProofLayout::of_parts(&parts);
            let mut out = vec![0xA5u8; layout.len()];
            unsafe { encode_into(&parts, &layout, out.as_mut_ptr()) };
            assert_eq!(out, codec().serialize(&parts.into_proof()).unwrap());
        }
        assert_eq!(ProofLayout::of_instance(MODULUS).len(), 13_958_612, "the default proof's length");
    }

    /// The vectorised hybrid lift is the scalar one, word for word.
    #[test]
    fn wide_hybrid_lift_is_the_scalar_lift() {
        if !lift_hybrid_wide_available() {
            return;
        }
        let mut x = 0x0123_4567_89AB_CDEFu64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for round in 0..400 {
            let r0: Vec<i16> = (0..N).map(|_| (next() % 7777) as i16 - 3888).collect();
            let r1: Vec<i16> = (0..N).map(|_| (next() % 19441) as i16 - 9720).collect();
            let mut tail = [0u32; N];
            for (j, t) in tail.iter_mut().enumerate() {
                *t = match round % 3 {
                    0 => next() as u32,
                    1 => (next() % 1024) as u32,
                    _ => (j as u32).wrapping_mul(2654435761),
                };
            }
            let mut wide = vec![0u8; 5 * N];
            unsafe { lift_hybrid_column(&r0, &r1, &tail, &mut wide) };
            for i in 0..N {
                let want = lift_hybrid(r0[i], r1[i], tail[i]);
                assert!((want as u128) < modulus_choice::HYBRID_MODULUS);
                assert_eq!(&wide[5 * i..5 * i + 5], &want.to_le_bytes()[..5], "coefficient {i}");
            }
        }
        // The residue extremes.
        let r0: Vec<i16> = (0..N).map(|i| [-3888i16, 3888, 0, -1, 1][i % 5]).collect();
        let r1: Vec<i16> = (0..N).map(|i| [9720i16, -9720, 0, 1, -1, 4860][i % 6]).collect();
        let tail: [u32; N] = core::array::from_fn(|i| [0u32, 1023, u32::MAX, 512][i % 4]);
        let mut wide = vec![0u8; 5 * N];
        unsafe { lift_hybrid_column(&r0, &r1, &tail, &mut wide) };
        for i in 0..N {
            assert_eq!(&wide[5 * i..5 * i + 5], &lift_hybrid(r0[i], r1[i], tail[i]).to_le_bytes()[..5]);
        }
    }

    #[test]
    fn garner_lift_is_the_chinese_remainder() {
        let primes = modulus_choice::limbs(MODULUS).unwrap();
        let product:u128=primes.iter().map(|&q|q as u128).product();
        let garner = Garner::of(&primes);
        let mut x = 0x1234_5678u128;
        let mut mixed = vec![0u64; primes.len()];
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let value = x % product;
            let residues: Vec<u64> = primes.iter().map(|&q| (value % q as u128) as u64).collect();
            assert_eq!(garner.lift(&residues, &mut mixed), value);
        }
        let residues: Vec<u64> = primes.iter().map(|&q| q as u64 - 1).collect();
        assert_eq!(garner.lift(&residues, &mut mixed), product - 1);
    }

    #[test]
    fn bulk_byte_fields_serialize_as_byte_sequences() {
        #[derive(Serialize)]
        struct Plain<'a> {
            version: u32,
            modulus: u128,
            commitment: &'a Vec<u8>,
            partials: &'a Vec<u128>,
            rounds: &'a Vec<[[u64; 3]; 2]>,
            opened: [u64; 3],
            row: &'a Vec<u8>,
            folded: &'a Vec<u8>,
        }
        let proof = Proof {
            version: 1,
            modulus: MODULUS,
            commitment: (0..=255).collect(),
            partials: vec![0, u128::MAX, 17],
            rounds: vec![[[0, 1, 2], [u64::MAX, 4, 5]]],
            opened: [7, 8, 9],
            row: Vec::new(),
            folded: vec![3; 1025],
        };
        let plain = Plain {
            version: proof.version,
            modulus: proof.modulus,
            commitment: &proof.commitment,
            partials: &proof.partials,
            rounds: &proof.rounds,
            opened: proof.opened,
            row: &proof.row,
            folded: &proof.folded,
        };
        let bytes = codec().serialize(&proof).unwrap();
        assert_eq!(bytes, codec().serialize(&plain).unwrap());
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        assert_eq!(decoded.commitment, proof.commitment);
        assert_eq!(decoded.row, proof.row);
        assert_eq!(decoded.folded, proof.folded);
    }

    /// The embedded key is the key the prover would build, batch for batch, and the image
    /// round-trips through the in-place reader.
    #[test]
    fn embedded_key_matches_the_built_key() {
        let built = keygen::build_key(MODULUS);
        let image = keygen::key_image(&built);
        if EMBEDDED_KEY.0.is_empty() {
            // Built on a host without the kernels: only the image format is checked.
            assert_eq!(image.len(), built.limbs() * keygen::BATCHES * keygen::BATCH_BYTES);
            return;
        }
        assert_eq!(&EMBEDDED_KEY.0[..], &image[..], "the embedded image differs");
        assert!(keygen::key_from_image(55_144_219_393, &EMBEDDED_KEY.0).is_none(),
            "a different three-limb modulus must rebuild its matrix");
        let read = keygen::key_from_image(MODULUS, &EMBEDDED_KEY.0).expect("a valid image");
        for k in 0..built.limbs() {
            assert_eq!(read.prime(k), built.prime(k));
            for (a, b) in read.row(k).iter().zip(built.row(k)) {
                assert_eq!(a.v, b.v, "limb {k} differs");
                assert_eq!(a.representation, b.representation);
            }
        }
        assert!(keygen::key_from_image(9721u128 * 2917 * 4861, &EMBEDDED_KEY.0).is_none());
    }

    #[test]
    fn modulus_changes_transcript() {
        let mut a = transcript(&[0; 32], MODULUS, &[]);
        let mut b = transcript(&[0; 32], 9721u128 * 2917 * 4861, &[]);
        let mut x = [0; 32];
        let mut y = [0; 32];
        a.fill(b"test", &mut x);
        b.fill(b"test", &mut y);
        assert_ne!(x, y);
    }

    /// The limb path's commitment, recombined, is the schoolbook `A W_j` in `R_Q`.
    ///
    /// This is the one place where the derived limb form of `A` and the inverse of the
    /// commitment's per-limb NTT decomposition are checked against the definition in `R_Q`
    /// directly, with no transform anywhere in the reference. Two columns are enough: each is a
    /// 4096-term inner product of ring elements. Both the baseline's recombination through the
    /// commitment matrix and the direct one from the kernel's rows are checked, against each
    /// other and against the definition.
    #[test]
    fn commitment_is_the_schoolbook_product() {
        let modulus = 110_277_094_573; // Retained generic RNS path.
        let primes = modulus_choice::limbs(modulus).unwrap();
        let p = params(&primes);
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let key = CommitmentKey::from_coefficients(
            COLUMN_LEN,
            &matrix,
            modulus,
            p.base,
            &p.extra_moduli,
        );
        let pp = PublicParameters::with_key(p.clone(), rq::MATRIX_SEED, key);
        let values = trace(&[7; 32]);
        let w = Witness::lifted(&p, &values).unwrap();
        let (commitment, _) = Prover::cold(&pp).commit(&w);
        let coefficients = commitment_coefficients(&commitment, &primes, modulus);
        let columns: Vec<&[B128]> = values.chunks_exact(COLUMN_LEN).collect();
        let rows = pp.key().commit_columns_rows(&columns, 3, 4);
        let direct = commitment_bytes_rows(&rows, &primes, modulus, COLUMNS, 3, None);
        let direct = rq::unpack_ring_elements(&direct, COLUMNS * N, modulus).unwrap();
        assert_eq!(coefficients, direct, "the two recombinations differ");
        for j in [0usize, COLUMNS - 1] {
            let column = &w.elements()[j * COLUMN_LEN..(j + 1) * COLUMN_LEN];
            assert_eq!(
                coefficients[j * rq::N..(j + 1) * rq::N],
                rq::commit_binary_column(&matrix, column, modulus)[..],
                "column {j} does not match the schoolbook product"
            );
        }
    }

    /// The streaming fold from the words is the fold from the kept transform, and the row
    /// evaluation from the words is the one from the lifted witness.
    #[test]
    fn streaming_fold_and_row_match_the_library() {
        let modulus = MODULUS;
        let primes = modulus_choice::limbs(modulus).unwrap();
        let p = params(&primes);
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let key = CommitmentKey::from_coefficients(
            COLUMN_LEN,
            &matrix,
            modulus,
            p.base,
            &p.extra_moduli,
        );
        let pp = PublicParameters::with_key(p.clone(), rq::MATRIX_SEED, key);
        let values = trace(&[8; 32]);
        let w = Witness::lifted(&p, &values).unwrap();
        let mut prover = Prover::cold(&pp);
        let (_, opening) = prover.commit(&w);
        let mut t = Transcript::new(b"labinius-fast/test-fold");
        let coords = t.sample_f162(b"coords", LOG_PACKED);
        let point = EvaluationPoint::msb_first(&p, &coords);
        let row = w.row_evaluate(&point);
        let direct = labinius::eval::row_evaluate_words(&values, point.p0(), 3);
        assert_eq!(row.values(), &direct[..]);
        let challenges = folding_challenges(&mut t, &row);
        let folded = prover.fold(opening, &FoldingChallenges::of(challenges.clone()));
        let streamed = labinius::fold::fold_witness_streaming(
            &values,
            COLUMNS,
            &challenges,
            COLUMN_LEN / 128,
            primes[0],
            3,
        );
        assert_eq!(folded.elements(), &streamed[..]);
        let bits = labinius::fold::fold_witness_bits(&values, COLUMNS, COLUMN_LEN, &challenges);
        assert_eq!(folded.elements(), &bits[..], "the bit-sliced fold differs");
        let positions = labinius::fold::fold_witness_positions(
            &values,
            COLUMNS,
            &challenges,
            COLUMN_LEN / 128,
            primes[0],
            3..5,
        );
        assert_eq!(&folded.elements()[96..160], &positions[..]);
        assert_eq!(&fold_checked(&values, &challenges, primes[0], 3)[..], folded.elements());
    }
    /// The embedded arranged hybrid image is the arrangement of the keys the prover would build,
    /// and reads back in place.
    #[test]
    fn embedded_hybrid_image_matches_the_built_keys() {
        use labinius::simd::commit::hybrid;
        if MODULUS != modulus_choice::HYBRID_MODULUS { return; }
        let key = keygen::build_key(MODULUS);
        let tail = keygen::build_tail_key(MODULUS);
        let image = hybrid::arranged_image(&key, &tail);
        assert_eq!(image.len(), 2 * keygen::BATCHES * hybrid::KEY_BATCH_BYTES);
        if EMBEDDED_HYBRID.0.is_empty() { return; }
        assert!(EMBEDDED_HYBRID.0[..] == image[..], "the embedded arranged image differs");
        assert!(hybrid::keys_from_image(&EMBEDDED_HYBRID.0, keygen::BATCHES).is_some());
        assert!(hybrid::keys_from_image(&EMBEDDED_HYBRID.0, keygen::BATCHES - 1).is_none());
    }
    #[test]
    fn hybrid_commitment_matches_schoolbook_under_new_modulus() {
        let q=modulus_choice::HYBRID_MODULUS;let ring=128usize;let columns=4;
        let primes=modulus_choice::limbs(q).unwrap();let p=params(&primes);
        let matrix=rq::ajtai_matrix(q,ring);
        let key=CommitmentKey::from_coefficients(ring*4,&matrix,q,p.base,&p.extra_moduli);
        let tail=keygen::build_tail_key_from_matrix(&matrix);
        let mut t=Transcript::new(b"hybrid-commitment-regression");
        let values=t.sample_b128(b"binary",ring*4*columns);
        let cols:Vec<&[B128]>=values.chunks_exact(ring*4).collect();
        let rows=labinius::simd::commit::hybrid::commit_columns_rows(&key,&tail,&cols,3,4);
        let coefficients=commitment_bytes_rows(&rows,&primes,q,columns,3,None);
        let coefficients=rq::unpack_ring_elements(&coefficients,columns*N,q).unwrap();
        for c in 0..columns {
            let lifted:Vec<F162>=cols[c].iter().map(|&v|F162::from_b128(v)).collect();
            assert_eq!(&coefficients[c*N..(c+1)*N],&rq::commit_binary_column(&matrix,&lifted,q)[..]);
        }
    }

}
