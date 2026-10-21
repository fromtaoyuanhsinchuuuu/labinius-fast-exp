//! Commitment modulo 3889 * 9721 * 2^10 using only two witness NTTs.
//!
//! Both transforms feed two public-key streams: the ordinary residue and the centered
//! coefficient lift of A mod 1024. The latter is reconstructed exactly in certified
//! chunks, inverse transformed across columns, and accumulated mod 1024. Only public
//! parameters are precomputed. Every invocation consumes its complete fresh table.
//!
//! # The sink: transpose the transform, not the products
//!
//! A row of the transform is 32 i16 lanes, one per ring element of the batch, and the commitment
//! wants each row's lane sum against two keys. Folding each stream's products down to two lanes
//! per row costs a shuffle tree per stream (30 uops per 8 rows per stream). Instead every group of
//! 8 rows is transposed once at qword granularity (24 shuffles, shared by both streams): vector
//! `q` of the transposed group holds qword `q` (i16 lanes `4q .. 4q + 4`) of row `s` in its qword
//! slot `s`. Against keys stored in the same arrangement ([`KeyBatch`]), eight chained `vpdpwssd`
//! per stream leave every row's whole batch in the two i32 lanes of its slot of one accumulator
//! vector: 40 uops per 8 rows for both streams, the accumulate fused, and the same two-lanes-per-row
//! accumulator footprint as the folded form (6 KB per column-prime-stream), which matters more
//! than uops here: the L1 is shared with the transform's working set and its hyperthread sibling.
//! Rows 24..26 of a 27-row block are one group of four transposed at 128-bit granularity (four
//! lanes per row, a zero-key pad slot).
use super::*;
use crate::simd::ntt::gen_small;
const P0:u16=3889;
const P1:u16=9721;
const PRODUCT:i64=P0 as i64*P1 as i64;
pub const MODULUS:u128=PRODUCT as u128*1024;
pub const CHUNK_BATCHES:usize=7;

/// Arranged key vectors per 27-row block: three 8-row groups of eight qword vectors and one
/// 4-row group of four chunk vectors, each for A and then T.
const KBLK:usize=56;
/// Accumulator vectors per block: base and tail of the four groups.
const ABLK:usize=8;
/// One batch of one prime's two keys, arranged for the sink. Block `blk`, 8-row group `g < 3`:
/// vector `KBLK blk + 16 g + q` holds in its qword slot `s` the i16 lanes `4q .. 4q + 4` of row
/// `27 blk + 8 g + s` of A, vector `.. + 8 + q` the same of T. The 4-row group: vector
/// `KBLK blk + 48 + c` holds in its 128-bit slot `s` the lanes `8c .. 8c + 8` of row
/// `27 blk + 24 + s` of A (zero for `s = 3`), vector `.. + 4 + c` the same of T. 84 KB.
#[repr(C,align(64))]
pub struct KeyBatch {v:[[i16;32];24*KBLK]}
/// One column-prime's accumulators: vector `ABLK blk + 2 g` is the base stream of group `g` of
/// block `blk` (8-row groups: lanes `2s, 2s + 1` sum row `8 g + s`; the 4-row group: lanes
/// `4s .. 4s + 4` sum row `24 + s`), the next one its tail stream. 12 KB.
#[repr(C,align(64))]
struct GAcc {v:[[i32;16];24*ABLK]}

/// What one batch adds to one lane of an 8-row group's accumulator: eight qwords of two
/// products `W * K`, `|W| <= w_bound(q)` (the kernel's output bound) and `|K| <= a_bound(q)`
/// (both keys are stored centered). The 4-row group's lanes gain half of it.
const fn per_batch(q:u16)->i64 {16*w_bound(q)*a_bound(q)}
/// Centered `2^16 mod q` (-577 for 3889, -2511 for 9721): the weight [`reduce_vec_c`] gives the
/// high half, which leaves a lane at most `acc_after_reduce_c` (18.9M / 82.3M, against
/// [`reduce_vec`]'s 108.6M / 236.3M).
const fn r16c(q:u16)->i32 {(65536%q as i32)-q as i32}
const fn acc_after_reduce_c(q:u16)->i64 {32768*(1+r16c(q).unsigned_abs() as i64)}
/// Batches between two fold-backs of both streams: the largest power of two P with
/// `acc_after_reduce_c + P * per_batch <= i32::MAX` (2 for 3889, 1 for 9721, as for the folded
/// accumulator, whose lanes also gain sixteen products per batch). The tail stream restarts from
/// zero at every chunk, which only shortens its runs.
const fn period(q:u16)->usize {
 let mut p=1usize;
 while acc_after_reduce_c(q)+2*(p as i64)*per_batch(q)<=i32::MAX as i64 {p*=2;}
 p
}
/// A tail-stream row sum at the end of a chunk, formed from the raw lanes: at most `period - 1`
/// batches since the last fold-back, two lanes per row (8-row groups) or four lanes of half the
/// growth (the 4-row group).
const fn tail_row_bound(q:u16)->i64 {
 let rem=(period(q)-1) as i64;
 let max8=2*(acc_after_reduce_c(q)+rem*per_batch(q));
 let max4=4*(acc_after_reduce_c(q)+rem*(per_batch(q)/2));
 if max8>max4 {max8} else {max4}
}
/// [`red16`]'s bias: the largest multiple of q below 2^31.
const fn bias(q:u16)->i64 {((1i64<<31)/q as i64)*q as i64}
const _:()={
 let qs=[P0,P1];let mut i=0;
 while i<2 {
  let q=qs[i];
  assert!(acc_after_reduce_c(q)+(period(q) as i64)*per_batch(q)<=i32::MAX as i64);
  // finish_base folds every lane back (to at most acc_after_reduce) and sums at most four
  assert!(4*acc_after_reduce(q)<=i32::MAX as i64);
  // the tail finish sums raw lanes (no i32 overflow) and biases the sums into [0, 2^32)
  assert!(tail_row_bound(q)<=bias(q) && bias(q)<=i32::MAX as i64);
  i+=1;
 }
 assert!(period(P0)==2 && period(P1)==1);
 // red16's representatives (|r| < 2q) are inside the inverse transform's input bound
 assert!(2*(P0 as i32)<=gen_small::TwI::<P0>::IN_BOUND && 2*(P1 as i32)<=gen_small::TwI::<P1>::IN_BOUND);
};

/// [`reduce_vec`] with the centered weight [`r16c`]: `l + (h + c) r16c`, at most
/// `acc_after_reduce_c` (same three uops).
#[inline(always)]
unsafe fn reduce_vec_c<const Q:u16>(x:__m512i)->__m512i {
 let k=_mm512_set1_epi32(1|((r16c(Q) as u16 as i32)<<16));
 let r=_mm512_madd_epi16(x,k);
 let m=_mm512_test_epi32_mask(x,_mm512_set1_epi32(0x8000));
 _mm512_mask_add_epi32(r,m,r,_mm512_set1_epi32(r16c(Q)))
}
/// `q^-1 mod 2^16` (Newton).
const fn qinv16(q:u16)->u32 {
 let q=q as u32;let mut x=q;let mut i=0;
 while i<4 {x=x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)))&0xffff;i+=1;}
 x
}
/// [`red16`]'s word constants: `(W, W q^-1 mod 2^16)` per i32 lane, `W = 2^16 mod q` in the low
/// word and `2^32 mod q` in the high word.
const fn red16_w(q:u16)->(i32,i32) {
 let wl=(1u32<<16)%q as u32;let wh=((1u64<<32)%q as u64) as u32;
 let qi=qinv16(q);
 ((wl|(wh<<16)) as i32,(((wl*qi)&0xffff)|(((wh*qi)&0xffff)<<16)) as i32)
}
/// A representative of `x mod q` with `|r| < 2q`, lane-wise, for `|x| <= bias(q)`: `y = x + bias`
/// is `2^16 yh + yl` with both words unsigned, and an unsigned Montgomery product per word against
/// `2^16` (low) and `2^32` (high) mod q leaves each word in `(-q, q)` congruent to `yl` and
/// `2^16 yh`; `vpmaddwd` adds them. Six uops, none of them a shuffle.
#[inline(always)]
unsafe fn red16<const Q:u16>(x:__m512i)->__m512i {
 let (w,wp)=red16_w(Q);
 let y=_mm512_add_epi32(x,_mm512_set1_epi32(bias(Q) as i32));
 let m=_mm512_mullo_epi16(y,_mm512_set1_epi32(wp));
 let r=_mm512_sub_epi16(_mm512_mulhi_epu16(y,_mm512_set1_epi32(w)),_mm512_mulhi_epu16(m,_mm512_set1_epi16(Q as i16)));
 _mm512_madd_epi16(r,_mm512_set1_epi16(1))
}

/// `acc + x . [k + OFF]` pairwise: one `vpdpwssd` with a memory operand. Written as `asm!`
/// because LLVM splits `vpdpwssd` into `vpmaddwd` + `vpaddd` (twice the uops) whenever the
/// accumulator is on a dependency chain, which here it always is; the displacement is a `const`
/// operand so that the key address needs no `lea` of its own.
#[inline(always)]
unsafe fn dp<const OFF:usize>(acc:__m512i,x:__m512i,k:*const i16)->__m512i {
 let mut r=acc;
 core::arch::asm!("vpdpwssd {r}, {x}, zmmword ptr [{k} + {o}]",r=inout(zmm_reg) r,x=in(zmm_reg) x,k=in(reg) k,o=const OFF,
  options(pure,readonly,nostack,preserves_flags));
 r
}
/// The base and tail accumulations of a transposed group: vector `q` of `x` against arranged
/// key vectors `q` (base) and `T + q` (tail) of the group at `k`.
macro_rules! dp_group {
 ($sb:ident,$st:ident,$x:ident,$k:ident,$t:literal; $($q:literal),*) => {
  $( $sb=dp::<{64*$q}>($sb,$x[$q],$k); $st=dp::<{64*($t+$q)}>($st,$x[$q],$k); )*
 };
}
/// 8 x 8 qword transpose: qword slot `s` of output `q` is qword `q` of `r[s]` (24 shuffles).
#[inline(always)]
unsafe fn tr8(r:&[__m512i;8])->[__m512i;8] {
 // u: [r0.q(2i), r1.q(2i)] / [r0.q(2i+1), r1.q(2i+1)] in 128-bit lane i, and so on
 let u0=_mm512_unpacklo_epi64(r[0],r[1]);let u1=_mm512_unpackhi_epi64(r[0],r[1]);
 let u2=_mm512_unpacklo_epi64(r[2],r[3]);let u3=_mm512_unpackhi_epi64(r[2],r[3]);
 let u4=_mm512_unpacklo_epi64(r[4],r[5]);let u5=_mm512_unpackhi_epi64(r[4],r[5]);
 let u6=_mm512_unpacklo_epi64(r[6],r[7]);let u7=_mm512_unpackhi_epi64(r[6],r[7]);
 let w0=_mm512_shuffle_i64x2::<0x88>(u0,u2);let w1=_mm512_shuffle_i64x2::<0xDD>(u0,u2);
 let w2=_mm512_shuffle_i64x2::<0x88>(u4,u6);let w3=_mm512_shuffle_i64x2::<0xDD>(u4,u6);
 let w4=_mm512_shuffle_i64x2::<0x88>(u1,u3);let w5=_mm512_shuffle_i64x2::<0xDD>(u1,u3);
 let w6=_mm512_shuffle_i64x2::<0x88>(u5,u7);let w7=_mm512_shuffle_i64x2::<0xDD>(u5,u7);
 [_mm512_shuffle_i64x2::<0x88>(w0,w2),_mm512_shuffle_i64x2::<0x88>(w4,w6),
  _mm512_shuffle_i64x2::<0x88>(w1,w3),_mm512_shuffle_i64x2::<0x88>(w5,w7),
  _mm512_shuffle_i64x2::<0xDD>(w0,w2),_mm512_shuffle_i64x2::<0xDD>(w4,w6),
  _mm512_shuffle_i64x2::<0xDD>(w1,w3),_mm512_shuffle_i64x2::<0xDD>(w5,w7)]
}
/// 4 x 4 transpose of 128-bit chunks: slot `s` of output `c` is chunk `c` of `r[s]` (8 shuffles).
#[inline(always)]
unsafe fn tr4(r0:__m512i,r1:__m512i,r2:__m512i,r3:__m512i)->[__m512i;4] {
 let t0=_mm512_shuffle_i64x2::<0x88>(r0,r1);let t1=_mm512_shuffle_i64x2::<0xDD>(r0,r1);
 let t2=_mm512_shuffle_i64x2::<0x88>(r2,r3);let t3=_mm512_shuffle_i64x2::<0xDD>(r2,r3);
 [_mm512_shuffle_i64x2::<0x88>(t0,t2),_mm512_shuffle_i64x2::<0x88>(t1,t3),
  _mm512_shuffle_i64x2::<0xDD>(t0,t2),_mm512_shuffle_i64x2::<0xDD>(t1,t3)]
}

/// The multiply-accumulate sink. With `RED` (the batch that ends a fold-back period) every
/// accumulator vector is folded back ([`reduce_vec_c`]) on its way back to memory: the same values
/// a separate pass over the column's accumulators after the batch would leave, without that
/// pass's 192 loads and stores. (Software prefetching of the keys, of the next batch into L2 or
/// of the next block into L1, only measured slower: the hardware streamers already follow the
/// sequential key reads, and the extra requests compete for the fill buffers.)
struct TSink<const Q:u16,const RED:bool> {buf:*mut i16,k:*const i16,acc:*mut i32}
impl<const Q:u16,const RED:bool> TSink<Q,RED> {
 #[inline(always)]
 unsafe fn put(p:*mut __m512i,v:__m512i) {
  _mm512_store_si512(p,if RED {reduce_vec_c::<Q>(v)} else {v});
 }
}
impl<const Q:u16,const RED:bool> BlockSink for TSink<Q,RED> {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  let k=self.k.add(32*KBLK*blk);let acc=self.acc.add(16*ABLK*blk);
  for g in 0..3 {
   let mut r=[_mm512_setzero_si512();8];
   for s in 0..8 {r[s]=_mm512_load_si512(w.add(32*(8*g+s)) as *const __m512i);}
   let x=tr8(&r);
   let kg=k.add(32*16*g);
   let pb=acc.add(32*g) as *mut __m512i;let pt=acc.add(32*g+16) as *mut __m512i;
   let mut sb=_mm512_load_si512(pb);let mut st=_mm512_load_si512(pt);
   dp_group!(sb,st,x,kg,8; 0,1,2,3,4,5,6,7);
   Self::put(pb,sb);Self::put(pt,st);
  }
  let r24=_mm512_load_si512(w.add(32*24) as *const __m512i);
  let r25=_mm512_load_si512(w.add(32*25) as *const __m512i);
  let r26=_mm512_load_si512(w.add(32*26) as *const __m512i);
  let x=tr4(r24,r25,r26,r26);
  let kg=k.add(32*48);
  let pb=acc.add(32*3) as *mut __m512i;let pt=acc.add(32*3+16) as *mut __m512i;
  let mut sb=_mm512_load_si512(pb);let mut st=_mm512_load_si512(pt);
  dp_group!(sb,st,x,kg,4; 0,1,2,3);
  Self::put(pb,sb);Self::put(pt,st);
 }
}
/// One column's batch `b` modulo `Q` (`keys[2b]` for 3889, `keys[2b + 1]` for 9721), folded back
/// when it ends a period.
#[inline(always)]
unsafe fn sink_batch<const Q:u16>(idx:&BinaryIndex32,keys:&[KeyBatch],b:usize,acc:*mut i32) {
 let k=keys[2*b+if Q==P0 {0} else {1}].v.as_ptr() as *const i16;
 if (b+1)%period(Q)==0 {
  vb::ntt_bin_batch32_hybrid::<Q,true>(idx,k,acc);
 } else {
  vb::ntt_bin_batch32_hybrid::<Q,false>(idx,k,acc);
 }
}
const EVEN:[i32;16]=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
const ODD:[i32;16]=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31];
/// Lane `8v + i` is the sum of lanes `2i, 2i + 1` of `[a, b][v]`.
#[inline(always)]
unsafe fn hsum2(a:__m512i,b:__m512i)->__m512i {
 let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
 _mm512_add_epi32(_mm512_permutex2var_epi32(a,e,b),_mm512_permutex2var_epi32(a,o,b))
}
/// The two-lane partial sums of the 4-row group's slots packed into lanes `2s, 2s + 1`.
const PAIRS:[i32;16]=[0,1,4,5,8,9,12,13,0,0,0,0,0,0,0,0];
/// The base stream's sums of block `blk`, reduced to [0, q): rows `27 blk .. + 16` and, in
/// lanes 0..11, `27 blk + 16 .. + 27`. Every lane is folded back first, so no partial sum
/// exceeds `4 acc_after_reduce` (asserted above).
#[inline(always)]
unsafe fn block_sums<const Q:u16>(acc:*mut i32,blk:usize)->(__m512i,__m512i) {
 let mut v=[_mm512_setzero_si512();4];
 for g in 0..4 {
  v[g]=reduce_vec::<Q>(_mm512_load_si512(acc.add(16*(ABLK*blk+2*g)) as *const __m512i));
 }
 let lo=mod_q::<Q>(hsum2(v[0],v[1]));
 let v3=_mm512_add_epi32(v[3],_mm512_shuffle_epi32::<0x4E>(v[3]));
 let v3=_mm512_permutexvar_epi32(_mm512_loadu_si512(PAIRS.as_ptr() as *const __m512i),v3);
 let hi=mod_q::<Q>(hsum2(v[2],v3));
 (lo,hi)
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_base<const Q:u16>(acc:*mut i32)->[u32;N] {
 let mut y=[0u32;N];
 for blk in 0..24 {
  let (lo,hi)=block_sums::<Q>(acc,blk);
  let o=y.as_mut_ptr().add(27*blk) as *mut i32;
  _mm512_storeu_si512(o as *mut __m512i,lo);
  _mm512_mask_storeu_epi32(o.add(16),0x07ff,hi);
 }
 y
}
/// `hsum2`'s indices for the 4-row group: lanes 8..11 of the result sum lanes `4s, 4s + 1` of the
/// second operand (its slots' two-lane partial sums), lanes 12..15 repeat the zero pad slot.
const EVEN4:[i32;16]=[0,2,4,6,8,10,12,14,16,20,24,28,28,28,28,28];
const ODD4:[i32;16]=[1,3,5,7,9,11,13,15,17,21,25,29,29,29,29,29];
/// Word of [`tail_words`]' result holding row `r` of the block: `vpackssdw` leaves rows
/// `4L .. 4L + 4` and `16 + 4L .. 16 + 4L + 4` in words `8L .. 8L + 8` (128-bit lane `L`).
const fn row_word(r:usize)->usize {if r<16 {8*(r/4)+r%4} else {8*((r-16)/4)+4+(r-16)%4}}
/// The tail stream's 27 row sums of block `blk` as i16 representatives (`|r| < 2q`, see [`red16`])
/// in the words [`row_word`] names, the stream's accumulators cleared for the next chunk. The raw
/// lanes are summed without folding them back first ([`tail_row_bound`]).
#[inline(always)]
unsafe fn tail_words<const Q:u16>(acc:*mut i32,blk:usize)->__m512i {
 let p=acc.add(16*ABLK*blk+16) as *mut __m512i;
 let t0=_mm512_load_si512(p);let t1=_mm512_load_si512(p.add(2));
 let t2=_mm512_load_si512(p.add(4));let t3=_mm512_load_si512(p.add(6));
 for g in 0..4 {_mm512_store_si512(p.add(2*g),_mm512_setzero_si512());}
 let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
 let lo=_mm512_add_epi32(_mm512_permutex2var_epi32(t0,e,t1),_mm512_permutex2var_epi32(t0,o,t1));
 let t3=_mm512_add_epi32(t3,_mm512_shuffle_epi32::<0x4E>(t3));
 let e=_mm512_loadu_si512(EVEN4.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD4.as_ptr() as *const __m512i);
 let hi=_mm512_add_epi32(_mm512_permutex2var_epi32(t2,e,t3),_mm512_permutex2var_epi32(t2,o,t3));
 _mm512_packs_epi32(red16::<Q>(lo),red16::<Q>(hi))
}
/// `vpermt2q` indices pairing the 128-bit lanes of two vectors: `[a.0, b.0, a.1, b.1]` and
/// `[a.2, b.2, a.3, b.3]`.
const PAIR_LO:[i64;8]=[0,1,8,9,2,3,10,11];
const PAIR_HI:[i64;8]=[4,5,12,13,6,7,14,15];
/// Writes the blocks [`tail_words`] gave for 16 columns into lanes `l0 .. l0 + 16` of rows
/// `27 blk .. 27 blk + 27` of `pend`: an 8 x 8 word transpose inside each 128-bit lane for
/// columns 0..8 and 8..16 (24 unpacks each) leaves word `w` of lane `L` of eight columns in lane
/// `L` of one vector, and one `vpermt2q` per two lanes joins the two halves into a row's 32 bytes.
#[inline(always)]
unsafe fn put16(x:&[__m512i;16],pend:*mut [i16;32],blk:usize,l0:usize) {
 let mut c=[[_mm512_setzero_si512();8];2];
 for s in 0..2 {
  let x=&x[8*s..8*s+8];
  let a0=_mm512_unpacklo_epi16(x[0],x[1]);let a1=_mm512_unpackhi_epi16(x[0],x[1]);
  let a2=_mm512_unpacklo_epi16(x[2],x[3]);let a3=_mm512_unpackhi_epi16(x[2],x[3]);
  let a4=_mm512_unpacklo_epi16(x[4],x[5]);let a5=_mm512_unpackhi_epi16(x[4],x[5]);
  let a6=_mm512_unpacklo_epi16(x[6],x[7]);let a7=_mm512_unpackhi_epi16(x[6],x[7]);
  let b0=_mm512_unpacklo_epi32(a0,a2);let b1=_mm512_unpackhi_epi32(a0,a2);
  let b2=_mm512_unpacklo_epi32(a1,a3);let b3=_mm512_unpackhi_epi32(a1,a3);
  let b4=_mm512_unpacklo_epi32(a4,a6);let b5=_mm512_unpackhi_epi32(a4,a6);
  let b6=_mm512_unpacklo_epi32(a5,a7);let b7=_mm512_unpackhi_epi32(a5,a7);
  c[s]=[_mm512_unpacklo_epi64(b0,b4),_mm512_unpackhi_epi64(b0,b4),
   _mm512_unpacklo_epi64(b1,b5),_mm512_unpackhi_epi64(b1,b5),
   _mm512_unpacklo_epi64(b2,b6),_mm512_unpackhi_epi64(b2,b6),
   _mm512_unpacklo_epi64(b3,b7),_mm512_unpackhi_epi64(b3,b7)];
 }
 let il=_mm512_loadu_si512(PAIR_LO.as_ptr() as *const __m512i);
 let ih=_mm512_loadu_si512(PAIR_HI.as_ptr() as *const __m512i);
 let row=|r:usize| (*pend.add(27*blk+r)).as_mut_ptr().add(l0) as *mut __m256i;
 // word w of lane L is row row_word^-1: 4L + w (w < 4) or 16 + 4L + w - 4; rows >= 27 are junk
 for w in 0..8 {
  let r0=if w<4 {w} else {12+w};
  let d=_mm512_permutex2var_epi64(c[0][w],il,c[1][w]);
  _mm256_storeu_si256(row(r0),_mm512_castsi512_si256(d));
  _mm256_storeu_si256(row(r0+4),_mm512_extracti64x4_epi64::<1>(d));
  if r0+8<27 {
   let d=_mm512_permutex2var_epi64(c[0][w],ih,c[1][w]);
   _mm256_storeu_si256(row(r0+8),_mm512_castsi512_si256(d));
   if r0+12<27 {_mm256_storeu_si256(row(r0+12),_mm512_extracti64x4_epi64::<1>(d));}
  }
 }
}
/// The tail chunks of a group's `g` columns modulo `Q` (accumulators `acc[2c + k]`), column `c`
/// into lane `lane0 + c` of `pend` as representatives inside the inverse transform's input bound,
/// the accumulators cleared for the next chunk. Whole aligned halves of the lanes go through
/// [`put16`]; other group sizes (only tests use them) are scattered lane by lane.
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_tails<const Q:u16>(acc:*mut GAcc,k:usize,g:usize,pend:&mut Batch32,lane0:usize) {
 let col=|c:usize| (*acc.add(2*c+k)).v.as_mut_ptr() as *mut i32;
 let pv=pend.v.as_mut_ptr();
 if g%16==0 && lane0%16==0 {
  for h in 0..g/16 {
   for blk in 0..24 {
    let mut x=[_mm512_setzero_si512();16];
    for c in 0..16 {x[c]=tail_words::<Q>(col(16*h+c),blk);}
    put16(&x,pv,blk,lane0+16*h);
   }
  }
 } else {
  for c in 0..g {
   for blk in 0..24 {
    let mut t=[0i16;32];
    _mm512_storeu_si512(t.as_mut_ptr() as *mut __m512i,tail_words::<Q>(col(c),blk));
    for r in 0..27 {(*pv.add(27*blk+r))[lane0+c]=t[row_word(r)];}
   }
  }
 }
}
#[inline(always)]
unsafe fn canonical(x:__m512i,q:i16)->__m512i {
 _mm512_mask_add_epi16(x,_mm512_cmp_epi16_mask(x,_mm512_setzero_si512(),_MM_CMPINT_LT),x,_mm512_set1_epi16(q))
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn crt_accumulate(a:&Batch32,b:&Batch32,tail:&mut Batch32) {
 const INV:u16=inv_mod(P0 as u64,P1 as u64) as u16;
 const W:i16=Params::<P1>::to_mont(INV);
 const WP:i16=Params::<P1>::mont_pre(W);
 let qv=_mm512_set1_epi16(P1 as i16);
 for j in 0..648 {
  let av=canonical(_mm512_load_si512(a.v[j].as_ptr() as *const __m512i),P0 as i16);
  let bv=canonical(_mm512_load_si512(b.v[j].as_ptr() as *const __m512i),P1 as i16);
  let d=_mm512_sub_epi16(bv,av);
  let m=_mm512_mullo_epi16(d,_mm512_set1_epi16(WP));
  let h=_mm512_mulhi_epi16(d,_mm512_set1_epi16(W));
  let u=canonical(_mm512_sub_epi16(h,_mm512_mulhi_epi16(m,qv)),P1 as i16);
  let gt=_mm512_cmp_epi16_mask(u,_mm512_set1_epi16((P1/2) as i16),_MM_CMPINT_NLE);
  let eq=_mm512_cmp_epi16_mask(u,_mm512_set1_epi16((P1/2) as i16),_MM_CMPINT_EQ);
  let agt=_mm512_cmp_epi16_mask(av,_mm512_set1_epi16((P0/2) as i16),_MM_CMPINT_NLE);
  let mut low=_mm512_add_epi16(av,_mm512_mullo_epi16(u,_mm512_set1_epi16(P0 as i16)));
  low=_mm512_mask_sub_epi16(low,gt|(eq&agt),low,_mm512_set1_epi16((PRODUCT&65535) as u16 as i16));
  let old=_mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
  _mm512_store_si512(tail.v[j].as_mut_ptr() as *mut __m512i,_mm512_add_epi16(old,low));
 }
}
/// `n` all-zero `T` on the heap (every type it is used for is valid as zeroes).
fn zeroed_slice<T>(n:usize)->Box<[T]> {
 unsafe {
  let l=std::alloc::Layout::array::<T>(n).unwrap();
  let p=std::alloc::alloc_zeroed(l) as *mut T;
  if p.is_null() {std::alloc::handle_alloc_error(l)}
  Box::from_raw(core::ptr::slice_from_raw_parts_mut(p,n))
 }
}
/// A worker's scratch, allocated once and reused by every group it takes: the accumulators of up
/// to `gmax` columns (`acc[2c + k]`, column `c`, prime `k`), the two batches the finished tail
/// chunks are packed into for the inverse transform, the lane-wise CRT total and index rows.
/// The fused consumer needs no row-major transform-output block buffer.
struct Worker {acc:Box<[GAcc]>,pend:Box<[Batch32]>,total:Box<[Batch32]>,idx:Box<[BinaryIndex32]>}
impl Worker {
 fn new(gmax:usize)->Self {
  Worker{acc:zeroed_slice(2*gmax),pend:zeroed_slice(2),total:zeroed_slice(1),idx:zeroed_slice(1)}
 }
}
/// Zero lanes `from..32` of every row of `b`.
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn clear_lanes(b:&mut Batch32,from:usize) {
 let m=!((1u64<<from)-1) as u32;
 for j in 0..N {_mm512_mask_storeu_epi16(b.v[j].as_mut_ptr(),m,_mm512_setzero_si512());}
}
/// Whether the commitment prefetches the witness: the knob is read once per group.
fn prefetch_witness_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LABINIUS_PREFETCH_WITNESS") {
        Ok(v) => v != "0",
        Err(_) => true,
    })
}

/// Bring the first line of every page of `bytes` of the witness table into the cache: the
/// commitment reads it in 2 KB chunks whose neighbours are 256 KB away, a pattern the
/// hardware prefetchers do not follow across pages. Only the page starts are touched: the
/// L2 streamer picks up the rest of a page after the first miss, and prefetching every line
/// of every chunk costs more instructions than the misses it saves.
#[inline(always)]
fn prefetch_witness(ptr:*const crate::fields::scalar::B128,bytes:usize){
 #[cfg(target_arch="x86_64")]
 unsafe{
  use core::arch::x86_64::*;
  let base=ptr as *const i8;
  let mut off=0usize;
  while off<bytes{_mm_prefetch::<{_MM_HINT_T0}>(base.add(off));off+=4096;}
 }
}

/// One group of `g` columns: the batch is the outer loop, so each key batch is read from L3 once
/// per group and `g` times from L2. A finished tail chunk of column `c` goes to lane `s g + c`
/// of the pending pair, `s` the chunk's slot; the pair is inverse transformed and CRT-combined
/// into the lane-wise total when its `32 / g` slots are full and after the last chunk, and a
/// column's tail is the sum of its lanes of the total. Writes `out[3c .. 3c + 3]`.
/// `sync(b)` is called once batch `b`'s keys have been read for every column of the group.
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group<F:FnMut(usize)>(cols:&[&[B128]],keys:&[KeyBatch],nb:usize,ws:&mut Worker,out:*mut [u32;N],mut sync:F) {
 let g=cols.len();assert!(g>=1 && 2*g<=ws.acc.len() && keys.len()==2*nb);
 let slots=32/g;
 for a in ws.acc[..2*g].iter_mut() {core::ptr::write_bytes(a as *mut GAcc,0,1);}
 core::ptr::write_bytes(&mut ws.total[0] as *mut Batch32,0,1);
 let mut filled=0usize;
 for b in 0..nb {
  for c in 0..g {
   // The witness is the commitment's only DRAM traffic, and the hardware
   // prefetchers do not follow it: a batch's 2 KB chunk of a column sits 256 KB
   // from the same chunk of the neighbouring column, and advances by 2 KB only
   // after a whole group of columns has been transformed. Touch the chunks that
   // come next while this one is read and transformed.
   if prefetch_witness_on() {
    prefetch_witness(cols[c].as_ptr().add(128*b),2048);
    if b+1<nb {prefetch_witness(cols[c].as_ptr().add(128*(b+1)),2048);}
    if c+1<g {prefetch_witness(cols[c+1].as_ptr().add(128*b),2048);}
   }
   cols[c].slice_batch(b,&mut ws.idx[0]);
   let a0=ws.acc[2*c].v.as_mut_ptr() as *mut i32;let a1=ws.acc[2*c+1].v.as_mut_ptr() as *mut i32;
   sink_batch::<P0>(&ws.idx[0],keys,b,a0);
   sink_batch::<P1>(&ws.idx[0],keys,b,a1);
  }
   sync(b);
   if (b+1)%CHUNK_BATCHES==0 || b+1==nb {
    let (p0,p1)=ws.pend.split_at_mut(1);
    let acc=ws.acc.as_mut_ptr();
    finish_tails::<P0>(acc,0,g,&mut p0[0],filled*g);
    finish_tails::<P1>(acc,1,g,&mut p1[0],filled*g);
    filled+=1;
   if filled==slots || b+1==nb {
    // lanes of unfilled slots still hold the previous flush's coefficients
    if filled*g<32 {clear_lanes(&mut p0[0],filled*g);clear_lanes(&mut p1[0],filled*g);}
    p0[0].representation=Representation::Ntt;p1[0].representation=Representation::Ntt;
    gen_small::intt_gen_batch32::<P0>(&mut p0[0]);gen_small::intt_gen_batch32::<P1>(&mut p1[0]);
    crt_accumulate(&p0[0],&p1[0],&mut ws.total[0]);
    filled=0;
   }
  }
 }
 let total=&ws.total[0];
 for c in 0..g {
  out.add(3*c).write(finish_base::<P0>(ws.acc[2*c].v.as_mut_ptr() as *mut i32));
  out.add(3*c+1).write(finish_base::<P1>(ws.acc[2*c+1].v.as_mut_ptr() as *mut i32));
  out.add(3*c+2).write(core::array::from_fn(|j|{
   let mut s=0u16;
   for k in 0..slots {s=s.wrapping_add(total.v[j][k*g+c] as u16);}
   (s&1023) as u32
  }));
 }
}
/// One batch of one prime's base and tail keys in the sink's arrangement ([`KeyBatch`]).
fn arrange(a:&Batch32,t:&Batch32,out:&mut KeyBatch) {
 for blk in 0..24 {
  for (si,key) in [(0usize,a),(1,t)] {
   for g in 0..3 {for q in 0..8 {for s in 0..8 {
    out.v[KBLK*blk+16*g+8*si+q][4*s..4*s+4].copy_from_slice(&key.v[27*blk+8*g+s][4*q..4*q+4]);
   }}}
   for c in 0..4 {for s in 0..4 {
    let dst=&mut out.v[KBLK*blk+48+4*si+c][8*s..8*s+8];
    if s<3 {dst.copy_from_slice(&key.v[27*blk+24+s][8*c..8*c+8]);} else {dst.fill(0);}
   }}
  }
 }
}
fn check_keys(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey) {
 assert_eq!(base.limbs(),2);assert_eq!(tail.limbs(),2);
 assert_eq!((base.prime(0),base.prime(1)),(P0,P1));
 assert_eq!((tail.prime(0),tail.prime(1)),(P0,P1));
 assert_eq!(base.len_f162(),tail.len_f162());
}
/// Bytes of one arranged key batch.
pub const KEY_BATCH_BYTES:usize=core::mem::size_of::<KeyBatch>();
const _:()=assert!(KEY_BATCH_BYTES==24*KBLK*64 && core::mem::align_of::<KeyBatch>()==64);
/// Both keys of every batch in the sink's arrangement as bytes, `keys[2b]` (modulo 3889) and
/// `keys[2b + 1]` (modulo 9721) in order: the public-parameter image a build script embeds so
/// that the prover reads the arranged keys in place ([`keys_from_image`]).
pub fn arranged_image(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey)->Vec<u8> {
 check_keys(base,tail);
 let nb=base.row(0).len();
 let mut out=Vec::with_capacity(2*nb*KEY_BATCH_BYTES);
 let mut kb:Box<[KeyBatch]>=zeroed_slice(1);
 for i in 0..2*nb {
  arrange(&base.row(i%2)[i/2],&tail.row(i%2)[i/2],&mut kb[0]);
  for row in kb[0].v.iter() {for &x in row {out.extend_from_slice(&x.to_le_bytes());}}
 }
 out
}
/// The arranged keys of `nb` batches an [`arranged_image`] holds, read in place (`image` must be
/// 64-byte aligned; `None` for any other length).
pub fn keys_from_image(image:&'static [u8],nb:usize)->Option<&'static [KeyBatch]> {
 if !cfg!(target_endian="little") || image.len()!=2*nb*KEY_BATCH_BYTES || image.as_ptr() as usize%64!=0 {return None;}
 Some(unsafe{core::slice::from_raw_parts(image.as_ptr() as *const KeyBatch,2*nb)})
}
/// Both keys of every batch in the sink's arrangement: `keys[2b]` modulo 3889, `keys[2b + 1]`
/// modulo 9721.
pub fn arrange_keys(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey,workers:usize)->Box<[KeyBatch]> {
 check_keys(base,tail);
 let nb=base.row(0).len();
 let mut keys:Box<[KeyBatch]>=zeroed_slice(2*nb);
 let kp=keys.as_mut_ptr() as usize;
 let tasks=workers.clamp(1,2*nb);
 crate::par::pool().run(tasks,|w|{
  let mut i=w;
  while i<2*nb {
   unsafe{arrange(&base.row(i%2)[i/2],&tail.row(i%2)[i/2],&mut *(kp as *mut KeyBatch).add(i));}
   i+=tasks;
  }
 });
 keys
}
pub fn certify(raw:&[[i16;648]],chunk_batches:usize)->i64 {
 let mut worst=0i64;
 for chunk in raw.chunks(chunk_batches*32) {
  let mut pos=[0i64;648];let mut neg=[0i64;648];
  for a in chunk {
   let mut ap=[0i64;649];let mut an=ap;
   let mut sp=[0i64;325];let mut sn=sp;
   for j in 0..648 {ap[j+1]=ap[j]+(a[j] as i64).max(0);an[j+1]=an[j]+(-(a[j] as i64)).max(0);}
   for j in 0..324 {let v=a[j] as i64+a[j+324] as i64;sp[j+1]=sp[j]+v.max(0);sn[j+1]=sn[j]+(-v).max(0);}
   let range=|p:&[i64],n:&[i64],lo:usize,hi:usize|->(i64,i64) {
    if lo>hi {(0,0)}else{(p[hi+1]-p[lo],n[hi+1]-n[lo])}
   };
   for j in 0..648 {
    let mut parts=Vec::with_capacity(3);
    if j<324 {
     parts.push((range(&ap,&an,0,j),false));
     if j<=186 {
      parts.push((range(&sp,&sn,j+137,323),true));
      parts.push((range(&ap,&an,324,647),true));
     }else{parts.push((range(&ap,&an,j+137,647),true));}
    }else{
     parts.push((range(&sp,&sn,j.saturating_sub(511),323),false));
     parts.push((range(&ap,&an,324,j),false));
     if j<511 {parts.push((range(&ap,&an,j-187,323),false));}
    }
    for ((p,n),negative) in parts {if negative {pos[j]+=n;neg[j]+=p;}else{pos[j]+=p;neg[j]+=n;}}
   }
  }
  for j in 0..648 {worst=worst.max(pos[j]).max(neg[j]);}
 }
 assert!(2*worst<PRODUCT,"CRT capacity exceeded: {worst} vs {}",PRODUCT/2);
 worst
}


/// Commit B128 columns under two ordinary limbs and an auxiliary 1024 residue.
/// The tail key is the NTT of the centered coefficient lift of A mod 1024.
/// The public-key certificate establishes exact reconstruction for CHUNK_BATCHES.
/// Arranges the keys first; [`commit_columns_rows_arranged`] takes them arranged.
pub fn commit_columns_rows(
 base:&crate::key::CommitmentKey, tail:&crate::key::CommitmentKey,
 columns:&[&[B128]],workers:usize,group_size:usize,
)->Vec<[u32;N]> {
 check_keys(base,tail);
 for c in columns {assert_eq!(c.len(),base.len_f162());}
 commit_columns_rows_arranged(&arrange_keys(base,tail,workers),columns,workers,group_size)
}
/// [`commit_columns_rows`] against keys already in the sink's arrangement ([`arrange_keys`],
/// [`keys_from_image`]): `keys[2b]`, `keys[2b + 1]` for each batch of 128 words of a column.
pub fn commit_columns_rows_arranged(keys:&[KeyBatch],columns:&[&[B128]],workers:usize,group_size:usize)->Vec<[u32;N]> {
 assert!(!columns.is_empty(),"at least one witness column");
 let nb=keys.len()/2;
 assert!(nb>0 && keys.len()==2*nb,"two arranged key batches per batch");
 for c in columns {assert_eq!(c.len(),128*nb,"128 words per key batch");}
 let g=group_size.clamp(1,32).min(columns.len());
 // Uninitialised: every group writes all 3 rows of each of its columns below, and the groups
 // cover the columns. (`vec![[0u32;N];n]` is not a zeroed allocation for N > 16: it memsets
 // and faults in the 8 MB on this thread, ~3 ms before any worker starts.)
 let mut rows:Vec<[u32;N]>=Vec::with_capacity(columns.len()*3);let dst=rows.as_mut_ptr() as usize;
 let paired=if workers>=2 && workers%2==0 && columns.len()>g {crate::par::CorePlacement::for_sibling_pairs(workers)} else {None};
 if let Some(placement)=paired {
  commit_paired(keys,columns,workers,g,Some(&placement),dst);
 } else {
  let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
  let next=std::sync::atomic::AtomicUsize::new(0);
  let placement=crate::par::CorePlacement::for_workers(workers);
  crate::par::pool().run(workers,|worker|{
   let _guard=placement.as_ref().and_then(|p|p.pin(worker));
   let mut ws:Option<Worker>=None;
   loop {
    let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    if gi>=groups {break;}
    let c0=gi*g;let gc=g.min(columns.len()-c0);
    let ws=ws.get_or_insert_with(||Worker::new(g));
    unsafe{group(&columns[c0..c0+gc],&keys,nb,ws,(dst as *mut [u32;N]).add(3*c0),|_|{})};
   }
  });
 }
 // Every row was written: the groups partition the columns.
 unsafe{rows.set_len(columns.len()*3);}
 rows
}
/// A batch counter on a cache line of its own.
#[repr(align(128))]
struct Progress(std::sync::atomic::AtomicUsize);
/// How far (in batches) a worker may run ahead of its sibling before it waits.
const LEAD:usize=1;
/// How long a worker waits for its sibling before the pair stops waiting for each other.
const SIBLING_WAIT:std::time::Duration=std::time::Duration::from_micros(500);
/// The commitment with the two hyperthreads of a core sharing every key batch in their L2:
/// workers `2k` and `2k + 1` (placed on core `k`'s two CPUs) take the two halves of each group
/// of `2 g` columns the pair claims, and neither starts a batch more than [`LEAD`] batches
/// ahead of the other, so each batch of keys comes from L3 once per pair of groups rather than
/// once per group. Measured on a c7i host: the median of eight unplaced workers taking `g`-column
/// groups, without their runs 5-10 ms slower (a third to a half of them in some series; the
/// siblings of unplaced workers that start together already share most key batches, so halving
/// the L3 key reads gains little by itself). Holding the second sibling half a batch behind the
/// first measured 1 ms slower, and `g = 32` per sibling 1.8 ms slower (L2 pressure).
/// The pairing is only a schedule: the pair claims its groups through one slot per round,
/// whichever sibling comes first, so every column is committed whatever the pool's runners do;
/// a sibling that keeps the other waiting longer than [`SIBLING_WAIT`] (descheduled, or its
/// task not started) ends the waiting for the rest of the run.
fn commit_paired(keys:&[KeyBatch],columns:&[&[B128]],workers:usize,g:usize,placement:Option<&crate::par::CorePlacement>,dst:usize) {
 use std::sync::atomic::{AtomicBool,AtomicUsize,Ordering};
 let nb=keys.len()/2;
 let pairs=workers/2;let sg=2*g;
 let groups=columns.len().div_ceil(sg);
 let rounds=groups+1;
 let next=AtomicUsize::new(0);
 let slots:Vec<std::sync::OnceLock<usize>>=(0..pairs*rounds).map(|_|std::sync::OnceLock::new()).collect();
 let done:Vec<Progress>=(0..2*pairs).map(|_|Progress(AtomicUsize::new(0))).collect();
 let coupled:Vec<AtomicBool>=(0..pairs).map(|_|AtomicBool::new(true)).collect();
 crate::par::pool().run(2*pairs,|worker|{
  let _guard=placement.and_then(|p|p.pin(worker));
  let (pair,s)=(worker/2,worker%2);
  let mine=&done[worker].0;let other=&done[worker^1].0;let coupled=&coupled[pair];
  let mut ws:Option<Worker>=None;
  for r in 0..rounds {
   let gi=*slots[pair*rounds+r].get_or_init(||next.fetch_add(1,Ordering::Relaxed));
   if gi>=groups {break;}
   let c0=gi*sg;let gc=sg.min(columns.len()-c0);
   let h=gc.div_ceil(2);
   let (lo,n)=if s==0 {(c0,h)} else {(c0+h,gc-h)};
   let base=r*nb;
   if n==0 {mine.store(base+nb,Ordering::Release);continue;}
   let ws=ws.get_or_insert_with(||Worker::new(g));
   unsafe{group(&columns[lo..lo+n],keys,nb,ws,(dst as *mut [u32;N]).add(3*lo),|b|{
    let d=base+b+1;
    mine.store(d,Ordering::Release);
    if !coupled.load(Ordering::Relaxed) {return;}
    if other.load(Ordering::Acquire)+LEAD>=d {return;}
    let start=std::time::Instant::now();
    let mut spins=0u32;
    while other.load(Ordering::Acquire)+LEAD<d {
     core::hint::spin_loop();
     spins+=1;
     if spins%64==0 && start.elapsed()>SIBLING_WAIT {coupled.store(false,Ordering::Relaxed);return;}
    }
   })};
  }
  // nothing more to wait for
  mine.store(usize::MAX/2,Ordering::Release);
 });
}
#[cfg(test)]
mod tests {
 use super::*;
 /// The fused register consumer equals the original row-major scratch consumer at every
 /// accumulator lane, including centered fold-back. Arbitrary initial accumulators exercise
 /// the exact wrapping SIMD operations; centered keys and all-zero/all-one/random nibbles
 /// cover the production transform inputs and both primes.
 fn fused_register_sink_check<const Q:u16,const RED:bool>() {
  let mut state=0xa3e41d7b92c650f1u64 ^ Q as u64;
  let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;state};
  let mut key=zeroed_slice::<KeyBatch>(1);
  let mut got=zeroed_slice::<GAcc>(1);
  let mut want=zeroed_slice::<GAcc>(1);
  let mut scratch=Blk27([0;27*32]);
  let mut idx=BinaryIndex32::zero();
  for case in 0..24 {
   for row in &mut idx.rows {for lane in 0..32 {
    let n=match case {0=>0,1=>15,2=>(lane%16) as u8,_=>(next()&15) as u8};
    row[2*lane]=n;row[2*lane+1]=16+n;
   }}
   for v in &mut key[0].v {for x in v {
    *x=match case {0=>0,1=>Q as i16/2,2=>-(Q as i16/2),_=>(next()%Q as u64) as i16-Q as i16/2};
   }}
   for (a,b) in got[0].v.iter_mut().zip(&mut want[0].v) {for (a,b) in a.iter_mut().zip(b) {
    *a=match case {0=>0,1=>i32::MAX,2=>i32::MIN,_=>next() as i32};*b=*a;
   }}
   unsafe {
    let kp=key[0].v.as_ptr() as *const i16;
    vb::ntt_bin_batch32_sink::<Q,_>(&idx,&mut TSink::<Q,RED>{
     buf:scratch.0.as_mut_ptr(),k:kp,acc:want[0].v.as_mut_ptr() as *mut i32});
    vb::ntt_bin_batch32_hybrid::<Q,RED>(&idx,kp,got[0].v.as_mut_ptr() as *mut i32);
   }
   assert_eq!(got[0].v,want[0].v,"q={Q}, RED={RED}, case={case}");
  }
 }
 #[test]
 fn fused_register_sink_matches_staged_sink() {
  fused_register_sink_check::<P0,false>();fused_register_sink_check::<P0,true>();
  fused_register_sink_check::<P1,false>();fused_register_sink_check::<P1,true>();
 }
 #[test]
 fn centered_crt_boundaries() {
  unsafe{
   let mut a=Batch32::zero(Representation::Coefficients);let mut b=Batch32::zero(Representation::Coefficients);
   let mut out=Batch32::zero(Representation::Coefficients);
   let vals=[-PRODUCT/2,-PRODUCT/2+1,-9721,-3889,-1,0,1,3889,9721,PRODUCT/2-1,PRODUCT/2];
   for j in 0..648 {for p in 0..32 {
    let v=vals[(j*32+p)%vals.len()];
    a.v[j][p]=center(v.rem_euclid(P0 as i64) as u64,P0 as u64);
    b.v[j][p]=center(v.rem_euclid(P1 as i64) as u64,P1 as u64);
   }}
   crt_accumulate(&a,&b,&mut out);
   for j in 0..648 {for p in 0..32 {
    assert_eq!((out.v[j][p] as u16)&1023,vals[(j*32+p)%vals.len()].rem_euclid(1024) as u16);
   }}
  }
 }
 fn red16_check<const Q:u16>() {
  let b=bias(Q);
  let mut xs:Vec<i64>=vec![-b,-b+1,b,b-1,i32::MAX as i64,0,1,-1,-32768,32767,-32769,65535,65536,-65536];
  for k in -31i64..32 {xs.push(k*Q as i64+(k%7));xs.push((k<<26)+k);}
  let mut s=0x2545f4914f6cdd1du64;
  for _ in 0..20000 {s^=s<<13;s^=s>>7;s^=s<<17;xs.push((s%(2*b as u64+1)) as i64-b);}
  for ch in xs.chunks(16) {
   let mut v=[0i32;16];for (d,&x) in v.iter_mut().zip(ch) {*d=x as i32;}
   let mut r=[0i32;16];
   unsafe{_mm512_storeu_si512(r.as_mut_ptr() as *mut __m512i,red16::<Q>(_mm512_loadu_si512(v.as_ptr() as *const __m512i)));}
   for i in 0..16 {
    assert!((r[i] as i64).abs()<2*Q as i64,"red16 {} of {}: {}",Q,v[i],r[i]);
    assert_eq!((r[i] as i64-v[i] as i64).rem_euclid(Q as i64),0,"red16 {} of {}: {}",Q,v[i],r[i]);
   }
  }
 }
 #[test]
 fn red16_is_a_small_representative() {red16_check::<P0>();red16_check::<P1>();}
 #[test]
 fn public_key_certificate_matches_independent_monomials() {
  let mut x=0x38476591u64;
  let raw:Vec<[i16;648]>=(0..4).map(|_|core::array::from_fn(|_|{x^=x<<13;x^=x>>7;x^=x<<17;(x%1024) as i16-512})).collect();
  let mut pos=[0i64;648];let mut neg=[0i64;648];
  for a in &raw {for bit in 0..512 {
   let mut k=[0i64;1295];
   for j in 0..648 {k[j+bit]=a[j] as i64;}
   for j in (648..1295).rev() {let v=k[j];k[j-324]+=v;k[j-648]-=v;}
   for j in 0..648 {pos[j]+=k[j].max(0);neg[j]+=(-k[j]).max(0);}
  }}
  let expected=*pos.iter().chain(neg.iter()).max().unwrap();
  assert_eq!(certify(&raw,CHUNK_BATCHES),expected);
 }
 /// The transposed-group sink against the folded-lane implementation it replaced, bit for bit,
 /// on random centered keys and random columns: 128 batches (fold-backs at both periods,
 /// 19 tail chunks, several full and partial lane flushes), every group size from 1 to 32
 /// that the prover or the flush logic distinguishes, and a ragged last group.
 #[test]
 fn row_sink_matches_folded_reference() {
  use crate::fields::scalar::B128;
  use crate::key::CommitmentKey;
  use crate::ring::Modulus;
  let len=128*128;let ncols=37;
  let base=CommitmentKey::random(len,11,Modulus::Q3889_FS_S,&[Modulus::Q9721_FS_S]);
  let tail=CommitmentKey::random(len,12,Modulus::Q3889_FS_S,&[Modulus::Q9721_FS_S]);
  let mut x=0x9e3779b97f4a7c15u64;
  let words:Vec<B128>=(0..len*ncols).map(|_|{
   let mut w=[0u64;2];
   for v in w.iter_mut() {x^=x<<13;x^=x>>7;x^=x<<17;*v=x;}
   B128(((w[1] as u128)<<64)|w[0] as u128)
  }).collect();
  let cols:Vec<&[B128]>=words.chunks_exact(len).collect();
  let want=reference::commit_columns_rows(&base,&tail,&cols,3,16);
  for g in [1usize,2,3,4,5,6,8,16,32] {
   let got=commit_columns_rows(&base,&tail,&cols,3,g);
   assert!(got==want,"group size {g} differs from the reference");
  }
  // the sibling-pair schedule, without placement: ragged groups, empty halves, more pairs
  // than groups
  let keys=arrange_keys(&base,&tail,3);
  for (workers,g) in [(2usize,1usize),(2,16),(4,3),(4,16),(6,8),(8,16),(8,32)] {
   let mut rows:Vec<[u32;N]>=Vec::with_capacity(ncols*3);
   commit_paired(&keys,&cols,workers,g,None,rows.as_mut_ptr() as usize);
   unsafe{rows.set_len(ncols*3);}
   assert!(rows==want,"paired schedule ({workers} workers, {g} columns each) differs from the reference");
  }
 }
}
/// The previous sink (products folded to two lanes per row, 6 KB per column-prime-stream), kept
/// as the bit-exact reference of [`tests::row_sink_matches_folded_reference`].
#[cfg(test)]
mod reference {
 use super::*;
 const QUADS:usize=4;
 #[repr(C,align(64))]
 struct Acc4 {v:[[i32;16];24*QUADS]}
 impl Acc4 {
  fn zero()->Box<Self> {
   unsafe {
    let mut b=Box::<Self>::new_uninit();
    core::ptr::write_bytes(b.as_mut_ptr() as *mut u8,0,core::mem::size_of::<Self>());
    b.assume_init()
   }
  }
 }
 const fn period4(q:u16)->usize {period_for(q,4*acc_per_batch(q))}
 #[inline(always)]
 unsafe fn fold4(p0:__m512i,p1:__m512i,p2:__m512i,p3:__m512i)->__m512i {
  let u=pack2(p0,p1);let v=pack2(p2,p3);
  let y=_mm512_permutex2var_epi64(u,_mm512_set_epi64(13,12,7,6,9,8,3,2),v);
  _mm512_add_epi32(_mm512_mask_blend_epi64(0xCC,u,v),y)
 }
 #[inline(always)]
 unsafe fn fold8(p:[__m512i;8])->__m512i {
  let r0=fold4(p[0],p[1],p[2],p[3]);let r1=fold4(p[4],p[5],p[6],p[7]);
  _mm512_add_epi32(_mm512_mask_blend_epi64(0xAA,r0,r1),_mm512_alignr_epi8::<8>(r1,r0))
 }
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn reduce4<const Q:u16>(acc:*mut i32) {
  for j in 0..24*QUADS {
   let s=_mm512_load_si512(acc.add(16*j) as *const __m512i);
   _mm512_store_si512(acc.add(16*j) as *mut __m512i,reduce_vec::<Q>(s));
  }
 }
 #[inline(always)]
 unsafe fn m(k:*const i16,w:__m512i,j:usize)->__m512i {_mm512_madd_epi16(w,_mm512_load_si512(k.add(32*j) as *const __m512i))}
 #[inline(always)]
 unsafe fn rd<const Q:u16>(p:*const i32)->__m512i {reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i))}
 const EVEN:[i32;16]=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
 const ODD:[i32;16]=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31];
 #[inline(always)]
 unsafe fn hsum2(a:__m512i,b:__m512i)->__m512i {
  let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
  _mm512_add_epi32(_mm512_permutex2var_epi32(a,e,b),_mm512_permutex2var_epi32(a,o,b))
 }
 const SLOT_LANE:[i32;16]={
  let mut x=[0i32;16];let mut s=0;
  while s<16 {x[s]=(8*(s/8)+2*[0,2,1,3][s%4]+(s%8)/4) as i32;s+=1;}
  x
 };
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn finish4<const Q:u16>(acc:&Acc4)->[u32;N] {
  let idx=_mm512_loadu_si512(SLOT_LANE.as_ptr() as *const __m512i);
  let mut y=[0u32;N];
  let p=acc.v.as_ptr() as *const i32;
  for bl in 0..24 {
   let q=p.add(16*QUADS*bl);
   let lo=mod_q::<Q>(hsum2(rd::<Q>(q),rd::<Q>(q.add(16))));
   let hi=mod_q::<Q>(hsum2(rd::<Q>(q.add(32)),rd::<Q>(q.add(48))));
   let o=y.as_mut_ptr().add(27*bl) as *mut i32;
   _mm512_storeu_si512(o as *mut __m512i,_mm512_permutexvar_epi32(idx,lo));
   _mm512_mask_storeu_epi32(o.add(16),0x07ff,_mm512_permutexvar_epi32(idx,hi));
  }
  y
 }
 struct TwinSink {buf:*mut i16,a:*const i16,t:*const i16,base:*mut i32,tail:*mut i32}
 impl BlockSink for TwinSink {
  #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
  #[target_feature(enable="avx512f,avx512bw")]
  unsafe fn block(&mut self,blk:usize,w:*const i16) {
   let a=self.a.add(32*27*blk);let t=self.t.add(32*27*blk);
   for p in 0..QUADS {
    let j0=8*p;
    let j1=j0+1;
    let j2=j0+2;
    let j3=(j0+3).min(26);
    let j4=(j0+4).min(26);
    let j5=(j0+5).min(26);
    let j6=(j0+6).min(26);
    let j7=(j0+7).min(26);
    let w0=_mm512_load_si512(w.add(32*j0) as *const __m512i);
    let w1=_mm512_load_si512(w.add(32*j1) as *const __m512i);
    let w2=_mm512_load_si512(w.add(32*j2) as *const __m512i);
    let w3=_mm512_load_si512(w.add(32*j3) as *const __m512i);
    let w4=_mm512_load_si512(w.add(32*j4) as *const __m512i);
    let w5=_mm512_load_si512(w.add(32*j5) as *const __m512i);
    let w6=_mm512_load_si512(w.add(32*j6) as *const __m512i);
    let w7=_mm512_load_si512(w.add(32*j7) as *const __m512i);
    let bp=fold8([m(a,w0,j0),m(a,w1,j1),m(a,w2,j2),m(a,w3,j3),m(a,w4,j4),m(a,w5,j5),m(a,w6,j6),m(a,w7,j7)]);
    let tp=fold8([m(t,w0,j0),m(t,w1,j1),m(t,w2,j2),m(t,w3,j3),m(t,w4,j4),m(t,w5,j5),m(t,w6,j6),m(t,w7,j7)]);
    let ba=self.base.add(16*(QUADS*blk+p)) as *mut __m512i;
    let ta=self.tail.add(16*(QUADS*blk+p)) as *mut __m512i;
    _mm512_store_si512(ba,_mm512_add_epi32(_mm512_load_si512(ba),bp));
    _mm512_store_si512(ta,_mm512_add_epi32(_mm512_load_si512(ta),tp));
   }
  }
 }
 struct Out {base:Vec<[u32;N]>,tail:Vec<[u16;N]>}
 #[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
 unsafe fn group(cols:&[&[B128]],basekeys:&[&[Batch32]],tailkeys:&[&[Batch32]],nb:usize,chunk:usize)->Out {
  let g=cols.len();assert!(g<=32);
  let mut accs:Vec<Box<Acc4>>=(0..g*2).map(|_|Acc4::zero()).collect();
  let mut tails:Vec<Box<Acc4>>=(0..g*2).map(|_|Acc4::zero()).collect();
  let mut partial0=Batch32::zero(Representation::Ntt);let mut partial1=Batch32::zero(Representation::Ntt);
  let mut total=Batch32::zero(Representation::Coefficients);
  let mut idx=BinaryIndex32::zero();let mut buf=Blk27([0;27*32]);let bp=buf.0.as_mut_ptr();
  for b in 0..nb {
   for (c,col) in cols.iter().enumerate() {
    col.slice_batch(b,&mut idx);
    let p0=accs[2*c].v.as_mut_ptr() as *mut i32;let t0=tails[2*c].v.as_mut_ptr() as *mut i32;
    let p1=accs[2*c+1].v.as_mut_ptr() as *mut i32;let t1=tails[2*c+1].v.as_mut_ptr() as *mut i32;
    vb::ntt_bin_batch32_sink::<P0,_>(&idx,&mut TwinSink{buf:bp,a:basekeys[0][b].v.as_ptr() as *const i16,
     t:tailkeys[0][b].v.as_ptr() as *const i16,base:p0,tail:t0});
    vb::ntt_bin_batch32_sink::<P1,_>(&idx,&mut TwinSink{buf:bp,a:basekeys[1][b].v.as_ptr() as *const i16,
     t:tailkeys[1][b].v.as_ptr() as *const i16,base:p1,tail:t1});
    if (b+1)%period4(P0)==0 {reduce4::<P0>(p0);reduce4::<P0>(t0);}
    if (b+1)%period4(P1)==0 {reduce4::<P1>(p1);reduce4::<P1>(t1);}
   }
   if (b+1)%chunk==0 || b+1==nb {
    for c in 0..g {
     let a=finish4::<P0>(&tails[2*c]);let v=finish4::<P1>(&tails[2*c+1]);
     for j in 0..648 {partial0.v[j][c]=a[j] as i16;partial1.v[j][c]=v[j] as i16;}
     tails[2*c].v.fill([0;16]);tails[2*c+1].v.fill([0;16]);
    }
    partial0.representation=Representation::Ntt;partial1.representation=Representation::Ntt;
    gen_small::intt_gen_batch32::<P0>(&mut partial0);gen_small::intt_gen_batch32::<P1>(&mut partial1);
    crt_accumulate(&partial0,&partial1,&mut total);
   }
  }
  Out{
   base:(0..g).flat_map(|c|[finish4::<P0>(&accs[2*c]),finish4::<P1>(&accs[2*c+1])]).collect(),
   tail:(0..g).map(|c|core::array::from_fn(|j|(total.v[j][c] as u16)&1023)).collect(),
  }
 }
 pub fn commit_columns_rows(
  base:&crate::key::CommitmentKey, tail:&crate::key::CommitmentKey,
  columns:&[&[B128]],workers:usize,group_size:usize,
 )->Vec<[u32;N]> {
  let nb=base.row(0).len();let bk=[base.row(0),base.row(1)];let tk=[tail.row(0),tail.row(1)];
  let g=group_size.clamp(1,32).min(columns.len());
  let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
  let mut rows=vec![[0u32;N];columns.len()*3];let dst=rows.as_mut_ptr() as usize;
  let next=std::sync::atomic::AtomicUsize::new(0);
  crate::par::pool().run(workers,|_|{
   loop {
    let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    if gi>=groups {break;}
    let c0=gi*g;let gc=g.min(columns.len()-c0);
    let out=unsafe{group(&columns[c0..c0+gc],&bk,&tk,nb,CHUNK_BATCHES)};
    for c in 0..gc {unsafe{
     let p=(dst as *mut [u32;N]).add((c0+c)*3);
     p.write(out.base[2*c]);p.add(1).write(out.base[2*c+1]);
     p.add(2).write(core::array::from_fn(|j|out.tail[c][j] as u32));
    }}
   }
  });
  rows
 }
}
