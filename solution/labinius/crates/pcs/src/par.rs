//! The little the parallel paths share: how many threads to use and how to cut a range.

/// `available_parallelism`, or 1: asked once. (Each query reads `/proc/self/cgroup` and the
/// cgroup's `cpu.max` files up the hierarchy, a dozen syscalls, and several phases ask.)
pub fn threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// `n` items cut into at most `parts` contiguous ranges whose starts are multiples of `align`;
/// the last range takes the remainder. Empty ranges are not returned.
pub fn ranges(n: usize, parts: usize, align: usize) -> Vec<core::ops::Range<usize>> {
    let align = align.max(1);
    let parts = parts.max(1);
    let per = n.div_ceil(parts).div_ceil(align) * align;
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    while start < n {
        let end = (start + per).min(n);
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ranges;

    #[test]
    fn ranges_cover_once() {
        for (n, parts, align) in [(0, 4, 64), (1, 8, 64), (1000, 8, 64), (4096, 3, 32), (5, 8, 1)] {
            let r = ranges(n, parts, align);
            assert!(r.len() <= parts);
            let mut at = 0;
            for x in &r {
                assert_eq!(x.start, at);
                assert!(x.start % align == 0);
                assert!(x.end > x.start);
                at = x.end;
            }
            assert_eq!(at, n);
        }
    }
}

// =============================================================================================
// a persistent pool
// =============================================================================================

/// The process's pool, `threads()` wide, made on first use. Never dropped: its workers park
/// between dispatches and go away with the process.
pub fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool::new(threads()))
}
//
// `std::thread::scope` spawns a thread per task: a `clone`, a 2 MB stack mapping and its first
// faults, a join — tens of microseconds each, a few hundred times over a prover's phases. The
// pool below keeps `threads - 1` workers and runs the caller's closure on them and on the
// calling thread, which is the same shape as a scope with one spawn per task.
//
// Between dispatches a worker spins (`pause`) for [`WORKER_SPIN`] before it parks on a condvar.
// A parked worker's CPU goes idle, and on the c7i hosts the idle governor puts it in C6 (exit
// latency ~190 us): a dispatch that has to wake parked workers waits 0.1-0.3 ms for them, which
// is longer than many of the switch's per-round dispatches take. A spinning worker picks a task
// up within a cache-line transfer. Tasks are handed out from one claim word, and a dispatch is
// over when its tasks are done: a worker that arrives after every task is taken is not waited
// for.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long an idle worker spins for the next dispatch before it parks.
const WORKER_SPIN: Duration = Duration::from_millis(2);
/// How long the dispatching thread spins for the last tasks of its dispatch before it parks.
const MAIN_SPIN: Duration = Duration::from_millis(10);
/// Spins between two clock reads; every 16th read also yields the CPU, so that a spinning
/// thread holds it from another runnable thread for tens of microseconds, not a time slice.
const SPIN_CHECK: u32 = 64;
const SPIN_YIELD: u32 = 16 * SPIN_CHECK;

/// The claim word: `[epoch: 24][tasks: 20][next: 20]`. A dispatch publishes `(epoch, tasks, 0)`;
/// a runner takes task `next` with one `fetch_add`, which also tells it which dispatch the task
/// belongs to. Each runner overshoots `tasks` at most once per dispatch, so `next` stays below
/// `2^20` while `tasks + width` does.
const TASK_BITS: u32 = 20;
const TASK_MASK: u64 = (1 << TASK_BITS) - 1;
const EPOCH_SHIFT: u32 = 2 * TASK_BITS;
const EPOCH_MASK: u64 = (1 << (64 - EPOCH_SHIFT)) - 1;
/// Tasks per published dispatch; [`Pool::run`] cuts larger ones into rounds.
const MAX_TASKS: usize = 1 << 16;

#[inline]
fn epoch_of(w: u64) -> u64 {
    w >> EPOCH_SHIFT
}
#[inline]
fn tasks_of(w: u64) -> usize {
    ((w >> TASK_BITS) & TASK_MASK) as usize
}
#[inline]
fn next_of(w: u64) -> usize {
    (w & TASK_MASK) as usize
}

type Task = dyn Fn(usize) + Sync;

struct Shared {
    claim: AtomicU64,
    /// The current dispatch's closure. Written by the dispatcher before it publishes the claim
    /// word, read by a runner after its `fetch_add` returned an unfinished task of that dispatch:
    /// the dispatcher does not return, and so neither replaces nor frees it, before every task
    /// is done.
    job: UnsafeCell<Option<&'static Task>>,
    /// Tasks of the current dispatch that have finished.
    done: AtomicUsize,
    /// Workers parked on `wake`.
    sleepers: AtomicUsize,
    /// The dispatcher is parked on `finished`.
    waiting: AtomicBool,
    /// A dispatch is in progress (a nested or concurrent `run` executes inline).
    busy: AtomicBool,
    shutdown: AtomicBool,
    /// Background work for idle workers ([`Pool::idle`]), or null.
    idle: AtomicPtr<IdleJob>,
    /// Workers running one of `idle`'s chunks.
    idle_running: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
    finished: Condvar,
}

unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    /// Take and run tasks of whatever dispatch is current until none is left; the epoch it
    /// served.
    fn work(&self) -> u64 {
        loop {
            let w = self.claim.fetch_add(1, Ordering::AcqRel);
            let (n, tasks) = (next_of(w), tasks_of(w));
            if n >= tasks {
                return epoch_of(w);
            }
            // Task `n` of this dispatch is ours and not done, so its closure is in place.
            let f = unsafe { (*self.job.get()).unwrap_unchecked() };
            f(n);
            if self.done.fetch_add(1, Ordering::SeqCst) + 1 == tasks && self.waiting.load(Ordering::SeqCst) {
                let _g = self.lock.lock().unwrap();
                self.finished.notify_one();
            }
        }
    }
}

/// Background work for the pool's idle workers: `f(0)`, `f(1)`, ... `f(chunks - 1)`, handed out
/// in order, each run by whichever worker is waiting for the next dispatch (several at once when
/// several are idle). A dispatch that arrives meanwhile waits for the chunks in progress, so a
/// chunk should be short, or run before the first dispatch that matters.
pub struct IdleJob {
    f: Box<dyn Fn(usize) + Send + Sync>,
    chunks: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}

impl IdleJob {
    /// Chunks finished so far.
    pub fn done(&self) -> usize {
        self.done.load(Ordering::Acquire)
    }

    /// Whether every chunk has run.
    pub fn finished(&self) -> bool {
        self.done() == self.chunks
    }

    /// Stop handing out chunks; the chunks that will have run are `0 .. stop()`.
    pub fn stop(&self) {
        self.next.fetch_max(self.chunks, Ordering::SeqCst);
    }
}

/// A fixed set of worker threads that run the tasks of one dispatch at a time.
pub struct Pool {
    shared: &'static Shared,
    /// Runners per dispatch: the workers plus the caller.
    width: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    /// A pool of `width` runners: `width - 1` threads plus whoever calls [`run`](Self::run).
    pub fn new(width: usize) -> Pool {
        let width = width.max(1);
        // One per pool, alive as long as any of its workers: the process's pool is never
        // dropped, and a test pool's few hundred bytes are not worth reclaiming.
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            claim: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            waiting: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            idle: AtomicPtr::new(core::ptr::null_mut()),
            idle_running: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            finished: Condvar::new(),
        }));
        // Started by the caller: workers started by the first worker (the caller's setup then
        // runs while they come up) measured the commitment 10 ms slower.
        // Each worker is told the epoch the pool starts at, not left to read it when it first
        // runs: a worker whose start-up is delayed past the first dispatches (its stack mapping
        // waits for the mmap lock behind the table's population) would otherwise take an epoch
        // published meanwhile for one it has already served, skip that dispatch and park
        // through it — measured as a commitment run by seven runners, 12-26 ms slower, in about
        // a third of the runs.
        let start = epoch_of(shared.claim.load(Ordering::Acquire));
        let handles = (1..width)
            .map(|_| std::thread::spawn(move || worker(shared, start)))
            .collect();
        Pool {
            shared,
            width,
            handles,
        }
    }

    /// The pool's width: workers plus the caller.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Run `f(i)` for every `i` in `0 .. tasks`, each once, on the workers and the calling
    /// thread; returns once every task has finished. Tasks are handed out in order from a
    /// shared counter, so a slow task is not waited on by the others. A `run` from inside a
    /// task, or concurrent with another dispatch, runs its tasks inline.
    pub fn run<F: Fn(usize) + Sync>(&self, tasks: usize, f: F) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.width == 1 || self.shared.busy.swap(true, Ordering::Acquire) {
            for i in 0..tasks {
                f(i);
            }
            return;
        }
        let mut base = 0;
        while base < tasks {
            let n = (tasks - base).min(MAX_TASKS);
            if base == 0 && n == tasks {
                self.dispatch(n, &f);
            } else {
                self.dispatch(n, &|i: usize| f(base + i));
            }
            base += n;
        }
        self.shared.busy.store(false, Ordering::Release);
    }

    fn dispatch<'a>(&self, tasks: usize, f: &'a (dyn Fn(usize) + Sync + 'a)) {
        let s = self.shared;
        // `run` does not return before the dispatch's tasks are done, and no runner touches the
        // closure after its last task: the lifetime is only extended over this call.
        unsafe {
            *s.job.get() = Some(core::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Task>(f))
        };
        s.done.store(0, Ordering::Relaxed);
        let epoch = (epoch_of(s.claim.load(Ordering::Relaxed)) + 1) & EPOCH_MASK;
        s.claim.store(
            (epoch << EPOCH_SHIFT) | ((tasks as u64) << TASK_BITS),
            Ordering::SeqCst,
        );
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = s.lock.lock().unwrap();
            s.wake.notify_all();
        }
        s.work();
        if s.done.load(Ordering::Acquire) != tasks {
            let start = Instant::now();
            let mut spins = 0u32;
            while s.done.load(Ordering::Acquire) != tasks {
                core::hint::spin_loop();
                spins += 1;
                if spins % SPIN_YIELD == 0 {
                    std::thread::yield_now();
                }
                if spins % SPIN_CHECK == 0 && start.elapsed() > MAIN_SPIN {
                    s.waiting.store(true, Ordering::SeqCst);
                    let mut g = s.lock.lock().unwrap();
                    while s.done.load(Ordering::SeqCst) != tasks {
                        g = s.finished.wait(g).unwrap();
                    }
                    drop(g);
                    s.waiting.store(false, Ordering::Relaxed);
                    break;
                }
            }
        }
        unsafe { *s.job.get() = None };
    }

    /// Hand `chunks` calls of `f` to the workers' idle time ([`IdleJob`]): they run between
    /// dispatches, never on the calling thread, and replace any earlier job's remaining chunks.
    pub fn idle<F: Fn(usize) + Send + Sync + 'static>(&self, chunks: usize, f: F) -> &'static IdleJob {
        // Leaked: a job is a few words, a run makes one or two, and an idle worker may still
        // hold a reference to a replaced one.
        let job: &'static IdleJob = Box::leak(Box::new(IdleJob {
            f: Box::new(f),
            chunks,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        }));
        if let Some(old) = unsafe { self.shared.idle.swap(job as *const IdleJob as *mut IdleJob, Ordering::SeqCst).as_ref() } {
            old.stop();
        }
        if self.shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        job
    }

    /// Stop `job` (see [`IdleJob::stop`]) and wait for its chunks that are running, so that none
    /// runs after this returns.
    pub fn quiesce(&self, job: &IdleJob) {
        job.stop();
        let _ = self.shared.idle.compare_exchange(
            job as *const IdleJob as *mut IdleJob,
            core::ptr::null_mut(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        while self.shared.idle_running.load(Ordering::SeqCst) != 0 {
            core::hint::spin_loop();
        }
    }

    /// `f(i)` for every `i` in `0 .. n`, collected in order.
    pub fn map<T: Send, F: Fn(usize) -> T + Sync>(&self, n: usize, f: F) -> Vec<T> {
        let mut out: Vec<core::mem::MaybeUninit<T>> = (0..n).map(|_| core::mem::MaybeUninit::uninit()).collect();
        let addr = out.as_mut_ptr() as usize;
        self.run(n, |i| unsafe {
            (addr as *mut core::mem::MaybeUninit<T>).add(i).write(core::mem::MaybeUninit::new(f(i)));
        });
        let mut out = core::mem::ManuallyDrop::new(out);
        unsafe { Vec::from_raw_parts(out.as_mut_ptr() as *mut T, out.len(), out.capacity()) }
    }
}

/// A worker's loop. `seen` is the epoch the pool had when the worker was spawned: every later
/// dispatch is one it has not served, however late the thread itself starts.
fn worker(s: &'static Shared, mut seen: u64) {
    loop {
        // Wait for a dispatch with a task left to take: spin (doing idle work if there is any),
        // then park.
        let mut start = Instant::now();
        let mut spins = 0u32;
        loop {
            let w = s.claim.load(Ordering::Acquire);
            if epoch_of(w) != seen {
                if next_of(w) < tasks_of(w) {
                    break;
                }
                seen = epoch_of(w);
            }
            if s.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if idle_chunk(s) {
                start = Instant::now();
                continue;
            }
            core::hint::spin_loop();
            spins += 1;
            if spins % SPIN_YIELD == 0 {
                std::thread::yield_now();
            }
            if spins % SPIN_CHECK == 0 && start.elapsed() > WORKER_SPIN {
                s.sleepers.fetch_add(1, Ordering::SeqCst);
                let mut g = s.lock.lock().unwrap();
                while epoch_of(s.claim.load(Ordering::SeqCst)) == seen
                    && !s.shutdown.load(Ordering::SeqCst)
                    && s.idle.load(Ordering::SeqCst).is_null()
                {
                    g = s.wake.wait(g).unwrap();
                }
                drop(g);
                s.sleepers.fetch_sub(1, Ordering::SeqCst);
                // Back to the top: a new dispatch, idle work, or the shutdown.
                start = Instant::now();
            }
        }
        seen = s.work();
    }
}

/// Run one chunk of the idle job, if it has one left.
#[inline]
fn idle_chunk(s: &Shared) -> bool {
    let job = s.idle.load(Ordering::Acquire);
    if job.is_null() {
        return false;
    }
    // Counted before the chunk is claimed: `quiesce` stops the job, then waits for the count.
    s.idle_running.fetch_add(1, Ordering::SeqCst);
    // Jobs are never freed.
    let job = unsafe { &*job };
    let i = job.next.fetch_add(1, Ordering::SeqCst);
    let ran = i < job.chunks;
    if ran {
        (job.f)(i);
        job.done.fetch_add(1, Ordering::Release);
    }
    if !ran || i + 1 == job.chunks {
        let _ = s.idle.compare_exchange(
            job as *const IdleJob as *mut IdleJob,
            core::ptr::null_mut(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }
    s.idle_running.fetch_sub(1, Ordering::SeqCst);
    ran
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn every_task_runs_once_and_results_are_ordered() {
        let pool = Pool::new(5);
        for n in [0usize, 1, 3, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n {n}");
            let squares = pool.map(n, |i| i * i);
            assert_eq!(squares, (0..n).map(|i| i * i).collect::<Vec<_>>());
        }
        // Many dispatches in a row, with a worker-dependent amount of work.
        let total = AtomicUsize::new(0);
        for round in 0..200 {
            pool.run(16, |i| {
                let mut x = 0usize;
                for k in 0..(i * 100) {
                    x = x.wrapping_add(k ^ round);
                }
                total.fetch_add(x.min(1) + 1, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 200 * 16 + 200 * 15);
    }
}
/// Temporary per-task affinity, restored before a pool task returns.
#[cfg(target_os="linux")]
pub struct CorePlacement { cpus:Vec<usize> }
#[cfg(target_os="linux")]
pub struct AffinityGuard { old:[usize;16] }
#[cfg(target_os="linux")]
unsafe extern "C" {
    fn sched_getaffinity(pid:i32,size:usize,mask:*mut usize)->i32;
    fn sched_setaffinity(pid:i32,size:usize,mask:*const usize)->i32;
}
/// The first allowed logical CPU of each physical core, from `/proc/cpuinfo`: read once (the
/// kernel samples every CPU's frequency to print it), by [`warm_topology`] off the critical
/// path when the caller arranges it.
#[cfg(target_os="linux")]
fn physical_cores()->Option<&'static [usize]> {
    static FIRST:std::sync::OnceLock<Option<Vec<usize>>>=std::sync::OnceLock::new();
    FIRST.get_or_init(||core_cpus().map(|c|c.iter().map(|l|l[0]).collect())).as_deref()
}
/// The allowed logical CPUs of each physical core (cores in the order of their first CPU in
/// `/proc/cpuinfo`, each core's CPUs in that order): read once, like [`physical_cores`].
#[cfg(target_os="linux")]
fn core_cpus()->Option<&'static [Vec<usize>]> {
    static CORES:std::sync::OnceLock<Option<Vec<Vec<usize>>>>=std::sync::OnceLock::new();
    CORES.get_or_init(read_core_cpus).as_deref()
}
/// Read the CPU topology [`CorePlacement::for_workers`] needs now.
pub fn warm_topology() {
    #[cfg(target_os="linux")]
    let _=physical_cores();
}
#[cfg(target_os="linux")]
fn read_core_cpus()->Option<Vec<Vec<usize>>> {
        let mut allowed=[0usize;16];
        if unsafe {sched_getaffinity(0,std::mem::size_of_val(&allowed),allowed.as_mut_ptr())} !=0 {return None;}
        let info=std::fs::read_to_string("/proc/cpuinfo").ok()?;
        let mut seen=std::collections::BTreeMap::new();
        let mut cores:Vec<Vec<usize>>=Vec::new();
        for record in info.split("\n\n") {
            let mut cpu=None;let mut package=None;let mut core=None;
            for line in record.lines() {
                let Some((k,v))=line.split_once(':') else {continue};
                match k.trim() {
                    "processor"=>cpu=v.trim().parse::<usize>().ok(),
                    "physical id"=>package=Some(v.trim().to_owned()),
                    "core id"=>core=Some(v.trim().to_owned()),
                    _=>{}
                }
            }
            let (Some(cpu),Some(package),Some(core))=(cpu,package,core) else {continue};
            let bits=usize::BITS as usize;
            if cpu>=allowed.len()*bits || allowed[cpu/bits]&(1usize<<(cpu%bits))==0 {continue;}
            let next=cores.len();
            let i=*seen.entry((package,core)).or_insert(next);
            if i==next {cores.push(Vec::new());}
            cores[i].push(cpu);
        }
        Some(cores)
}
#[cfg(target_os="linux")]
impl CorePlacement {
    pub fn for_workers(workers:usize)->Option<Self> {
        let cpus=physical_cores()?;
        if workers==0 || cpus.len()<workers {return None;}
        Some(Self{cpus:cpus.to_vec()})
    }
    /// Tasks `2k` and `2k + 1` on the two hyperthreads of physical core `k`: for a dispatch of
    /// exactly twice as many tasks as there are cores, each core having exactly two allowed CPUs.
    pub fn for_sibling_pairs(tasks:usize)->Option<Self> {
        let cores=core_cpus()?;
        if cores.is_empty() || tasks!=2*cores.len() || cores.iter().any(|c|c.len()!=2) {return None;}
        Some(Self{cpus:cores.iter().flatten().copied().collect()})
    }
    /// One task per allowed logical CPU, in CPU order: for a dispatch of exactly `tasks` equal
    /// tasks on a pool as wide as the CPU set, so that no two of them share a CPU while another
    /// sits idle, whatever the scheduler's placement of the workers was when they were woken
    /// (after the commitment's per-core placement the eight runners are not spread evenly).
    pub fn for_logical(tasks:usize)->Option<Self> {
        let mut allowed=[0usize;16];
        if unsafe {sched_getaffinity(0,std::mem::size_of_val(&allowed),allowed.as_mut_ptr())} !=0 {return None;}
        let bits=usize::BITS as usize;
        let cpus:Vec<usize>=(0..allowed.len()*bits).filter(|&c| allowed[c/bits]&(1usize<<(c%bits))!=0).collect();
        if tasks==0 || cpus.len()<tasks {return None;}
        Some(Self{cpus})
    }
    pub fn pin(&self,worker:usize)->Option<AffinityGuard> {
        let cpu=*self.cpus.get(worker)?;
        let mut old=[0usize;16];
        if unsafe {sched_getaffinity(0,std::mem::size_of_val(&old),old.as_mut_ptr())} !=0 {return None;}
        let mut chosen=[0usize;16];
        chosen[cpu/usize::BITS as usize]=1usize<<(cpu%usize::BITS as usize);
        if unsafe {sched_setaffinity(0,std::mem::size_of_val(&chosen),chosen.as_ptr())} !=0 {return None;}
        Some(AffinityGuard{old})
    }
}
#[cfg(target_os="linux")]
impl Drop for AffinityGuard {
    fn drop(&mut self) {unsafe {sched_setaffinity(0,std::mem::size_of_val(&self.old),self.old.as_ptr());}}
}
#[cfg(not(target_os="linux"))]
pub struct CorePlacement;
#[cfg(not(target_os="linux"))]
impl CorePlacement {
    pub fn for_workers(_:usize)->Option<Self>{None}
    pub fn for_logical(_:usize)->Option<Self>{None}
    pub fn for_sibling_pairs(_:usize)->Option<Self>{None}
    pub fn pin(&self,_:usize)->Option<()>{None}
}

#[cfg(all(test,target_os="linux"))]
mod affinity_tests {
    use super::*;
    #[test]
    fn temporary_affinity_restores_the_original_mask() {
        let mut original=[0usize;16];
        assert_eq!(unsafe {sched_getaffinity(0,std::mem::size_of_val(&original),original.as_mut_ptr())},0);
        let Some(p)=CorePlacement::for_workers(1) else {return};
        {
            let Some(_guard)=p.pin(0) else {return};
            let mut pinned=[0usize;16];
            assert_eq!(unsafe {sched_getaffinity(0,std::mem::size_of_val(&pinned),pinned.as_mut_ptr())},0);
            assert_eq!(pinned.iter().map(|x|x.count_ones()).sum::<u32>(),1);
            assert!(pinned.iter().zip(original).all(|(a,b)|*a & b==*a));
        }
        let mut restored=[0usize;16];
        assert_eq!(unsafe {sched_getaffinity(0,std::mem::size_of_val(&restored),restored.as_mut_ptr())},0);
        assert_eq!(restored,original);
    }
}
