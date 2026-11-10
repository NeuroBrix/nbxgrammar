//! Golden masks: the masks of fixed walks over fixed grammars, hashed, against hashes RECORDED on nbxgrammar 0.1.0
//! (520d8c1). The oracle (tests/oracle.rs) compares the mask with a brute force that steps the SAME matcher, so a
//! change of the matcher's stepping would move both together; these hashes do not move with it. A regroup of the
//! brick that changes no behaviour reproduces every one; a hash is re-recorded only by a change that says, in its
//! message, which outputs it changes.
use nbxgrammar::json::parse;
use nbxgrammar::matcher::Matcher;
use nbxgrammar::schema::{compile, compile_description, Schema};
use nbxgrammar::vocab::Vocab;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// 256 byte tokens, JSON and prose fragments, a partial UTF-8 character, 200 random fragments, three tokens without
/// bytes (two envelope markers and a reasoning end, ids OPEN, OPEN + 1, OPEN + 2), then the end of sequence.
fn tokens() -> Vec<Option<Vec<u8>>> {
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let mut toks: Vec<Option<Vec<u8>>> = (0u16..256).map(|b| Some(vec![b as u8])).collect();
    for s in ["{\"", "\":", "\": ", "\",", "\"}", "},", "}", "]", "[", "[\"", "\"]", " \"", "\"name", "name", "tags",
              "true", "false", "null", "nul", "12", "0", "-1", ".5", "e+", "1e3", "é", "日本", "🍎", "\\n", "\\\"",
              "\\u00e9", "a\"b", "\", \"", ", ", "\n  ", "   ", "\"a\"", "\"b\"", "\"n\": 1", "\"o\": {", "abc",
              "get_weather", "\"get_", "weather\"", "\"get_time\"", ", \"arguments\": ", "\n{\"name\": ", "\"city\": \"",
              "}\n", "</", "think", ">", "</think>", "<", "x<y", "\n\n", "</thi", "nk>\n\n{", "Okay, ", " the user"] {
        toks.push(Some(s.as_bytes().to_vec()));
    }
    toks.push(Some(vec![0xE6, 0x97]));
    for _ in 0..200 {
        let n = 1 + rng.below(5);
        let pool = b"{}[]\":, \nabnxyz01-.e\\<>/";
        toks.push(Some((0..n).map(|_| pool[rng.below(pool.len())]).collect()));
    }
    toks
}

fn fnv(h: &mut u64, x: u32) {
    for b in x.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); }
}

/// The hash of every mask, every chosen token and every acceptance bit of `walks` seeded walks of at most `steps`.
fn walk_hash(g: &Schema, v: &Vocab, n_tokens: u32, eos: u32, seed: u64, walks: usize, steps: usize) -> u64 {
    let mut rng = Rng(seed);
    let mut h = 0xcbf29ce484222325u64;
    for _ in 0..walks {
        let mut m = Matcher::new(g);
        for _ in 0..steps {
            let mut mask = vec![0u32; v.words()];
            v.mask(&m, &mut mask);
            mask.iter().for_each(|&w| fnv(&mut h, w));
            fnv(&mut h, m.accepting() as u32);
            let allowed: Vec<u32> = (0..n_tokens).filter(|&t| mask[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
            assert!(!allowed.is_empty(), "a dead end");
            let t = allowed[rng.below(allowed.len())];
            fnv(&mut h, t);
            v.advance(&mut m, t).unwrap();
            if t == eos { break; }
        }
    }
    h
}

const SCHEMAS: &[&str] = &[
    r#"{"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer"},"ok":{"type":"boolean"}},"required":["name","n"]}"#,
    r#"{"type":"array","items":{"enum":["a","b",12,1,null]},"minItems":1,"maxItems":4}"#,
    r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"x":{"type":["number","null"]},"o":{"type":"object","properties":{"a":{"const":true}},"required":["a"]}},"required":["tags"]}"#,
    r#"{"type":"number"}"#,
    r#"{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"integer"},"maxItems":2},{"type":"null"}]}"#,
    r##"{"type":"object","properties":{"p":{"$ref":"#/$defs/pt"},"q":{"type":"array","items":{"$ref":"#/$defs/pt"},"maxItems":2}},"required":["p"],"$defs":{"pt":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"number"}},"required":["x","y"]}}}"##,
];

fn descriptions(open: u32) -> Vec<String> {
    let call = |name: &str, params: &str| format!(r#"{{"lit": "\"{name}\"", "then": {{"seq": [{{"lit": ", \"arguments\": "}}, {{"schema": {params}}}, {{"lit": "}}\n"}}, {{"token": {}}}]}}}}"#, open + 1);
    vec![
        format!(r#"{{"repeat": {{"seq": [{{"token": {open}}}, {{"lit": "\n{{\"name\": "}}, {{"choice": [{}, {}]}}]}}, "min": 1, "max": 2, "sep": "\n"}}"#,
                call("get_weather", r#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}},"required":["city"]}"#),
                call("get_time", r#"{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}"#)),
        format!(r#"{{"seq": [{{"until": "</think>", "end_token": {}}}, {{"lit": "\n\n"}}, {{"schema": {{"type": "object", "properties": {{"a": {{"type": "integer"}}}}, "required": ["a"]}}}}]}}"#, open + 2),
        r#"{"seq": [{"lit": "<p>"}, {"until": "</p>"}, {"lit": "!"}]}"#.to_string(),
        r#"{"seq": [{"lit": "x\n\n"}, {"schema": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}]}"#.to_string(),
    ]
}

/// Recorded on 0.1.0 (520d8c1), `cargo test --test golden_masks -- --nocapture` printing them: SCHEMAS in order,
/// then `descriptions` in order.
const RECORDED: &[u64] = &[
    0xc66a3278283112d9, 0xcc2c8f66302b965a, 0x1091f8c2904115f9, 0xc40d77d8ee0c5baa, 0xf7a5ca5d00d7e279,
    0xacc0f85a822d9e69, 0x4b394f1524c0381e, 0x55aca81ceee9c65e, 0x5a9aaaa0593e13a1, 0x0334e4b61afce9c9,
];

#[test]
fn the_masks_of_fixed_walks_are_the_recorded_ones() {
    let mut toks = tokens();
    let open = toks.len() as u32;
    toks.extend([None, None, None]);
    let eos = toks.len() as u32;
    toks.push(None);
    let n = toks.len() as u32;
    let v = Vocab::new(toks, vec![eos]);
    let mut got = Vec::new();
    for (i, src) in SCHEMAS.iter().enumerate() {
        let g = compile(&parse(src).unwrap()).unwrap();
        got.push(walk_hash(&g, &v, n, eos, 0x9E3779B97F4A7C15 ^ (i as u64 + 1), 30, 80));
    }
    for (i, d) in descriptions(open).iter().enumerate() {
        let g = compile_description(&parse(d).unwrap()).unwrap();
        got.push(walk_hash(&g, &v, n, eos, 0xD1B54A32D192ED03 ^ (i as u64 + 1), 30, 150));
    }
    println!("RECORDED: {}", got.iter().map(|h| format!("0x{h:016x}")).collect::<Vec<_>>().join(", "));
    assert_eq!(got, RECORDED, "a mask, a chosen token or an acceptance changed on a fixed walk");
}
