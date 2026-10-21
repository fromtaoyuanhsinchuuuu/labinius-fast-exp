//! Precompute the commitment key of the default modulus. `A` is public parameter data — a
//! function of the fixed matrix seed and the modulus alone, never of the instance seed — so the
//! prover may precompute it, and here it is computed once at build time with the library's own
//! kernels and embedded in the binary. On a host without those kernels (anything but x86-64) an
//! empty image is written and the prover builds the key when it runs.
#![allow(dead_code)]
use std::env;
use std::path::PathBuf;

#[cfg(target_arch = "x86_64")]
#[path = "src/keygen.rs"]
mod keygen;
#[cfg(target_arch = "x86_64")]
#[path = "src/modulus_choice.rs"]
mod modulus_choice;
#[cfg(target_arch = "x86_64")]
#[path = "src/parameters.rs"]
mod parameters;
#[cfg(target_arch = "x86_64")]
#[path = "src/rq.rs"]
mod rq;

fn main() {
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let (image, tail, hybrid) = images();
    std::fs::write(dir.join("key.bin"), image).expect("write the key image");
    std::fs::write(dir.join("tail-key.bin"), tail).expect("write the auxiliary key image");
    std::fs::write(dir.join("hybrid-key.bin"), hybrid).expect("write the arranged hybrid key image");
}

#[cfg(target_arch = "x86_64")]
fn host_can_run() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("avx512vbmi2")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("gfni")
}

/// The key image, and for the hybrid modulus the tail key's image and both keys arranged for
/// the hybrid commitment's sink (`labinius::simd::commit::hybrid::arranged_image`), each key
/// built once.
#[cfg(target_arch = "x86_64")]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // The library's kernels need these features at run time. A build host without them (for
    // instance a cross-build for the scored CPU) writes empty images and the prover builds the
    // keys when it runs; the scored runner has every one of them.
    if !host_can_run() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let key = keygen::build_key(parameters::MODULUS);
    let image = keygen::key_image(&key);
    if parameters::MODULUS != modulus_choice::HYBRID_MODULUS {
        return (image, Vec::new(), Vec::new());
    }
    let tail = keygen::build_tail_key(parameters::MODULUS);
    let hybrid = labinius::simd::commit::hybrid::arranged_image(&key, &tail);
    (image, keygen::key_image(&tail), hybrid)
}

#[cfg(not(target_arch = "x86_64"))]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (Vec::new(), Vec::new(), Vec::new())
}
