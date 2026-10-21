//! Fixed shape parameters must match the checker; solutions may change MODULUS.
//! The field modulus is x^128 + x^7 + x^2 + x + 1 (GHASH).
//! MODULUS describes the independent integer commitment modulus, not GHASH.
pub const LOG_VARS: usize = 31;
pub const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
pub const COLUMN_LOG: u32 = 10;
/// Columns of the witness, and so the number of commitment columns and folding challenges.
pub const COLUMNS: usize = 1 << COLUMN_LOG;
/// One column in `F162` elements.
pub const COLUMN_LEN: usize = 1 << (LOG_PACKED - COLUMN_LOG as usize);
/// Ring elements of one column: four `F162` make one element of `R_648`.
pub const RING_ELEMENTS: usize = COLUMN_LEN / 4;
/// Two NTT primes plus a 1024 residue. The latter reuses the witness transforms
/// through a certified exact-integer lift of the fixed public matrix.
pub const MODULUS:u128=38_712_288_256;
