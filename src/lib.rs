//! NeuroBrix's grammar engine for constrained decoding: a JSON Schema subset and grammar descriptions compiled to a
//! matcher, per-step token masks over a tokenizer's byte vocabulary. Std only; the README says how it is built.
pub mod json;
pub mod schema;
pub mod matcher;
pub mod vocab;
pub mod ffi;
