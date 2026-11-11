//! The matcher's contract seen from outside the crate: its frames are not reachable; what the vocabulary and the C ABI
//! need of it are questions and steps (NeuroBrix's bricks rule — nbxgrammar 0.1.1).
use nbxgrammar::json::parse;
use nbxgrammar::matcher::Matcher;
use nbxgrammar::schema::compile;

#[test]
fn a_state_set_aside_resumes_against_its_own_grammar_and_is_refused_by_name_against_another() {
    let a = compile(&parse(r#"{"type":"array","items":{"type":"integer"}}"#).unwrap()).unwrap();
    let b = compile(&parse(r#"{"type":"array","items":{"type":"integer"}}"#).unwrap()).unwrap();   // equal, not the same
    let mut m = Matcher::new(&a);
    assert!(m.feed_all(b"[1, 2"));
    let state = m.suspend();
    let mut again = Matcher::resume(&a, state).expect("its own grammar");
    assert!(again.feed_all(b"]") && again.accepting());
    let err = Matcher::resume(&b, again.suspend()).err().expect("another grammar is refused");
    assert!(err.contains("another grammar"), "{err}");
}

#[test]
fn a_plain_string_position_is_said_and_an_escape_or_a_partial_character_is_not() {
    let g = compile(&parse(r#"{"type":"string"}"#).unwrap()).unwrap();
    let at = |t: &[u8]| { let mut m = Matcher::new(&g); assert!(m.feed_all(t)); m.in_plain_string() };
    assert!(!at(b"") && at(b"\"") && at(b"\"ab") && !at(b"\"a\\") && !at(b"\"\\u00") && !at(&[b'"', 0xE6, 0x97]));
    assert!(!at(b"\"ab\""));
}

#[test]
fn a_finished_output_takes_nothing_more() {
    let g = compile(&parse(r#"{"type":"integer"}"#).unwrap()).unwrap();
    let mut m = Matcher::new(&g);
    assert!(m.feed_all(b"12") && m.accepting());
    m.finish();
    assert!(m.accepting() && !m.clone().feed(b'3') && m.token_steps().is_empty());
}
