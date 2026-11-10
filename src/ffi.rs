//! The C ABI (`extern "C"`, no binding library): what the engine's ctypes loader and, later, NIBUX call. Opaque handles;
//! bytes in as (pointer, length); a refusal is a return code or a null handle with its message written into the caller's
//! buffer. The caller keeps a grammar alive while matchers made from it live.
use crate::json::parse;
use crate::matcher::{Matcher, State};
use crate::schema::{compile, compile_description, Schema};
use crate::vocab::Vocab;
use std::slice;

pub struct MatcherHandle {
    schema: *const Schema,
    state: Option<State>,
}

fn with_matcher<R>(h: &mut MatcherHandle, f: impl FnOnce(&mut Matcher) -> R) -> R {
    // SAFETY: the caller keeps the grammar alive while the matcher lives (the ABI's contract)
    let schema = unsafe { &*h.schema };
    let state = h.state.take().expect("a matcher handle holds its state between calls");
    let mut m = Matcher::resume(schema, state).expect("the handle's state was made against the handle's grammar");
    let r = f(&mut m);
    h.state = Some(m.suspend());
    r
}

fn write_err(msg: &str, buf: *mut u8, cap: usize) {
    if buf.is_null() || cap == 0 { return; }
    let n = msg.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), buf, n);
        *buf.add(n) = 0;
    }
}

/// The vocabulary: `n` tokens whose bytes follow each other in `data`, `lens[i]` each (u32::MAX: a token no grammar emits).
/// Null when an end-of-sequence id lies past the `n` tokens (it would index past every mask: refused, never a panic
/// across the ABI).
/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_vocab_new(data: *const u8, lens: *const u32, n: u32, eos: *const u32, n_eos: u32) -> *mut Vocab {
    if unsafe { slice::from_raw_parts(eos, n_eos as usize) }.iter().any(|&e| e >= n) {
        return std::ptr::null_mut();
    }
    let lens = unsafe { slice::from_raw_parts(lens, n as usize) };
    let total: usize = lens.iter().filter(|&&l| l != u32::MAX).map(|&l| l as usize).sum();
    let data = unsafe { slice::from_raw_parts(data, total) };
    let mut toks = Vec::with_capacity(n as usize);
    let mut at = 0usize;
    for &l in lens {
        if l == u32::MAX { toks.push(None); } else { toks.push(Some(data[at..at + l as usize].to_vec())); at += l as usize; }
    }
    let eos = unsafe { slice::from_raw_parts(eos, n_eos as usize) }.to_vec();
    Box::into_raw(Box::new(Vocab::new(toks, eos)))
}

/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_vocab_free(v: *mut Vocab) {
    if !v.is_null() { unsafe { drop(Box::from_raw(v)); } }
}

/// A JSON schema's text compiled; null (and the refusal's message in `err`) when it cannot be enforced.
/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_grammar_json_schema(text: *const u8, len: usize, err: *mut u8, err_cap: usize) -> *mut Schema {
    let text = unsafe { slice::from_raw_parts(text, len) };
    let r = std::str::from_utf8(text).map_err(|_| "the schema is not UTF-8".to_string())
        .and_then(parse).and_then(|j| compile(&j));
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => { write_err(&e, err, err_cap); std::ptr::null_mut() }
    }
}

/// A grammar description's text (lit / schema / seq / choice / repeat / until / token, as JSON) compiled; null (and the
/// refusal's message in `err`) when it cannot be enforced.
/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_grammar_description(text: *const u8, len: usize, err: *mut u8, err_cap: usize) -> *mut Schema {
    let text = unsafe { slice::from_raw_parts(text, len) };
    let r = std::str::from_utf8(text).map_err(|_| "the description is not UTF-8".to_string())
        .and_then(parse).and_then(|j| compile_description(&j));
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => { write_err(&e, err, err_cap); std::ptr::null_mut() }
    }
}

/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_grammar_free(s: *mut Schema) {
    if !s.is_null() { unsafe { drop(Box::from_raw(s)); } }
}

/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_matcher_new(s: *const Schema) -> *mut MatcherHandle {
    let schema = unsafe { &*s };
    Box::into_raw(Box::new(MatcherHandle { schema: s, state: Some(Matcher::new(schema).suspend()) }))
}

/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_matcher_free(m: *mut MatcherHandle) {
    if !m.is_null() { unsafe { drop(Box::from_raw(m)); } }
}

/// The next step's mask into `out` (`words` int32 = ceil(vocab / 32)); -1 when `words` is not the vocabulary's.
/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_mask(v: *const Vocab, m: *mut MatcherHandle, out: *mut u32, words: usize) -> i32 {
    let v = unsafe { &*v };
    if words != v.words() { return -1; }
    let out = unsafe { slice::from_raw_parts_mut(out, words) };
    with_matcher(unsafe { &mut *m }, |mm| v.mask(mm, out));
    0
}

/// The token chosen; 0, or 1 when the grammar refuses it (the matcher then unchanged).
/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_advance(v: *const Vocab, m: *mut MatcherHandle, token: u32) -> i32 {
    let v = unsafe { &*v };
    with_matcher(unsafe { &mut *m }, |mm| if v.advance(mm, token).is_ok() { 0 } else { 1 })
}

/// # Safety
/// Every pointer is one this ABI returned and has not freed, or a caller's buffer of the stated length (the
/// module's contract above).
#[no_mangle]
pub unsafe extern "C" fn ng_accepting(m: *mut MatcherHandle) -> i32 {
    with_matcher(unsafe { &mut *m }, |mm| mm.accepting() as i32)
}
