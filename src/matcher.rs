//! The byte-level matcher: a stack of small Copy frames fed one byte (or one token step) at a time - cheap to clone for
//! the trie walk. JSON values (schema nodes) take bounded whitespace before them; description parts never do.
use crate::schema::{Node, Schema, MAX_NUMBER};

/// Whitespace inside a JSON value: at most ONE space at each place JSON allows whitespace, never a newline or a tab
/// (Outlines' rule: small models loop on wider whitespace - TinyLlama wrote 32 newlines before its object in our tests,
/// 2026-11-10; vLLM issue #38696); none before the root value and none after it (llguidance's default).
pub const WS_MAX: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Frame {
    Done,
    Value { node: u32, ws: u8 },
    Lit { node: u32, pos: u16, cand: u64 },
    Text { node: u32, pos: u16 },
    Str { esc: u8, need: u8, lo: u8, hi: u8 },
    Num { int: bool, phase: u8, len: u8 },
    Arr { node: u32, count: u32, phase: u8, ws: u8 },
    Obj { node: u32, seen: u64, phase: u8, ws: u8, pos: u16, cand: u64, cur: u8 },
    Seq { node: u32, idx: u16 },
    Choice { node: u32, pos: u16, cand: u64 },
    /// phase 0 before the first item, 1 after an item, 2 inside the separator at `sp`, 3 after the separator
    Rep { node: u32, count: u32, phase: u8, sp: u16 },
    Until { node: u32, k: u16, need: u8, lo: u8, hi: u8 },
}

/// The automaton's state for one output against one compiled grammar. Its frames are its own: other modules ask it
/// questions (`accepting`, `in_plain_string`, `until_clean`, `token_steps`) and step it (`feed`, `feed_token`,
/// `finish`), never read them.
#[derive(Clone)]
pub struct Matcher<'s> {
    schema: &'s Schema,
    stack: Vec<Frame>,
}

/// A matcher's state set aside (the C ABI's handle keeps one between calls): opaque, and bound to the grammar it was
/// made against — `Matcher::resume` refuses any other by name.
pub struct State {
    schema: *const Schema,
    stack: Vec<Frame>,
}

/// The candidates (one bit each, at most 64) among `texts` whose first byte is `b`.
fn first_byte_candidates(texts: &[Vec<u8>], b: u8) -> u64 {
    let mut cand = 0u64;
    for (i, x) in texts.iter().enumerate() { if x[0] == b { cand |= 1 << i; } }
    cand
}

fn is_ws(b: u8) -> bool {
    b == b' '
}

fn num_end(phase: u8) -> bool {
    matches!(phase, 1 | 2 | 4 | 7)
}

fn num_step(int: bool, phase: u8, b: u8) -> Option<u8> {
    let d = b.is_ascii_digit();
    match (phase, b) {
        (0, b'0') => Some(1),
        (0, _) if d => Some(2),
        (1, b'.') | (2, b'.') if !int => Some(3),
        (1, b'e') | (1, b'E') | (2, b'e') | (2, b'E') | (4, b'e') | (4, b'E') if !int => Some(5),
        (2, _) if d => Some(2),
        (3, _) | (4, _) if d => Some(4),
        (5, b'+') | (5, b'-') => Some(6),
        (5, _) | (6, _) | (7, _) if d => Some(7),
        _ => None,
    }
}

/// UTF-8 inside free text: the next state after `b`, or None. (need, lo, hi) as in the Str and Until frames.
fn utf8_step(need: u8, lo: u8, hi: u8, b: u8, text: bool) -> Option<(u8, u8, u8)> {
    if need > 0 {
        return if b >= lo && b <= hi { Some((need - 1, 0x80, 0xBF)) } else { None };
    }
    match b {
        b'\n' | b'\t' if text => Some((0, 0, 0)),
        0x00..=0x1F => None,
        0x20..=0x7F => Some((0, 0, 0)),
        0xC2..=0xDF => Some((1, 0x80, 0xBF)),
        0xE0 => Some((2, 0xA0, 0xBF)),
        0xED => Some((2, 0x80, 0x9F)),
        0xE1..=0xEF => Some((2, 0x80, 0xBF)),
        0xF0 => Some((3, 0x90, 0xBF)),
        0xF4 => Some((3, 0x80, 0x8F)),
        0xF1..=0xF3 => Some((3, 0x80, 0xBF)),
        _ => None,
    }
}

impl<'s> Matcher<'s> {
    pub fn new(schema: &'s Schema) -> Self {
        Matcher { schema, stack: vec![Frame::Done, Frame::Value { node: schema.root as u32, ws: 0 }] }
    }

    fn node(&self, n: u32) -> &'s Node {
        &self.schema.nodes[n as usize]
    }

    fn lits(&self, node: u32) -> &'s [Vec<u8>] {
        match self.node(node) { Node::Literals(l) => l, _ => unreachable!() }
    }

    fn lit_complete(&self, node: u32, pos: u16, cand: u64) -> bool {
        let l = self.lits(node);
        (0..l.len()).any(|i| cand >> i & 1 == 1 && l[i].len() == pos as usize)
    }

    fn json_value(&self, n: u32) -> bool {
        matches!(self.node(n), Node::Literals(_) | Node::Integer | Node::Number | Node::Str | Node::Array { .. }
                 | Node::Object { .. } | Node::AnyOf(_))
    }

    fn starts(s: &Schema, n: usize, b: u8) -> bool {
        match &s.nodes[n] {
            Node::Literals(l) => l.iter().any(|x| x[0] == b),
            Node::Text(t) => t[0] == b,
            Node::Str => b == b'"',
            Node::Integer | Node::Number => b == b'-' || b.is_ascii_digit(),
            Node::Array { .. } => b == b'[',
            Node::Object { .. } => b == b'{',
            Node::AnyOf(a) => a.iter().any(|&j| Self::starts(s, j, b)),
            Node::Seq(p) => Self::starts(s, p[0], b),
            Node::Choice { prefixes, .. } => prefixes.iter().any(|x| x[0] == b),
            Node::Repeat { node, min, .. } => *min > 0 && Self::starts(s, *node, b),
            Node::Until { .. } => true,
            Node::Token(_) => false,
        }
    }

    fn first_token(s: &Schema, n: usize) -> Option<u32> {
        match &s.nodes[n] {
            Node::Token(id) => Some(*id),
            Node::Until { token, .. } => *token,
            Node::Seq(p) => Self::first_token(s, p[0]),
            Node::Repeat { node, min, .. } if *min > 0 => Self::first_token(s, *node),
            _ => None,
        }
    }

    /// The frames on top that may end without another byte, popped on a copy: what follows them is the parent's.
    fn settle(&self) -> Vec<Frame> {
        let mut st = self.stack.clone();
        loop {
            let pop = match st.last() {
                Some(Frame::Num { phase, .. }) => num_end(*phase),
                Some(Frame::Lit { node, pos, cand }) => self.lit_complete(*node, *pos, *cand),
                Some(Frame::Seq { node, idx }) => match self.node(*node) { Node::Seq(p) => *idx as usize == p.len(), _ => false },
                Some(Frame::Rep { node, count, phase, .. }) => match self.node(*node) {
                    Node::Repeat { min, sep, .. } => *count >= *min && (*phase == 0 || *phase == 1) && (sep.is_empty() || *phase != 2),
                    _ => false,
                },
                _ => false,
            };
            if !pop { return st; }
            st.pop();
        }
    }

    /// The output may end here.
    /// The state set aside; `resume` gives the matcher back.
    pub fn suspend(self) -> State {
        State { schema: self.schema as *const Schema, stack: self.stack }
    }

    /// The matcher of a state set aside, against the grammar it was made with; any other grammar is refused (the
    /// frames index that grammar's nodes: resumed elsewhere they would be a wrong automaton, silently).
    pub fn resume(schema: &'s Schema, state: State) -> Result<Self, String> {
        if !std::ptr::eq(schema, state.schema) {
            return Err("a matcher state resumed against another grammar than the one it was made with".to_string());
        }
        Ok(Matcher { schema, stack: state.stack })
    }

    /// True inside a JSON string at a plain position (no escape in flight, no partial UTF-8 character): the
    /// vocabulary's cached string-safe mask applies as it is.
    pub fn in_plain_string(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Str { esc: 0, need: 0, .. }))
    }

    /// The output ends here (the end of sequence was chosen at an accepting point): nothing more is taken.
    pub fn finish(&mut self) {
        self.stack.clear();
        self.stack.push(Frame::Done);
    }

    pub fn accepting(&self) -> bool {
        self.settle() == vec![Frame::Done]
    }

    /// The token steps the grammar allows next (a dialect marker given as its id).
    pub fn token_steps(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut st = self.stack.clone();
        loop {
            match st.last().copied() {
                Some(Frame::Value { node, .. }) => {
                    if let Some(t) = Self::first_token(self.schema, node as usize) { out.push(t); }
                    return out;
                }
                Some(Frame::Seq { node, idx }) => {
                    let p = match self.node(node) { Node::Seq(p) => p, _ => unreachable!() };
                    if (idx as usize) < p.len() {
                        if let Some(t) = Self::first_token(self.schema, p[idx as usize]) { out.push(t); }
                        return out;
                    }
                    st.pop();
                }
                Some(Frame::Rep { node, count, phase, .. }) => {
                    let (item, min, max, sep) = match self.node(node) {
                        Node::Repeat { node, min, max, sep } => (*node, *min, *max, sep),
                        _ => unreachable!(),
                    };
                    let item_next = phase == 0 || phase == 3 || (phase == 1 && sep.is_empty());
                    if item_next && count < max {
                        if let Some(t) = Self::first_token(self.schema, item) { out.push(t); }
                    }
                    if (phase == 0 || phase == 1) && count >= min { st.pop(); } else { return out; }
                }
                Some(Frame::Num { phase, .. }) if num_end(phase) => { st.pop(); }
                Some(Frame::Lit { node, pos, cand }) if self.lit_complete(node, pos, cand) => { st.pop(); }
                Some(Frame::Until { node, need: 0, .. }) => {
                    if let Node::Until { token: Some(t), .. } = self.node(node) { out.push(*t); }
                    return out;
                }
                _ => return out,
            }
        }
    }

    /// The first byte of the terminator when the top frame is free text at a clean point (no partial match, no partial
    /// UTF-8 character): the vocabulary's cached until-safe tokens for that byte apply as they are.
    pub fn until_clean(&self) -> Option<u8> {
        match self.stack.last() {
            Some(Frame::Until { node, k: 0, need: 0, .. }) => match self.node(*node) { Node::Until { term, .. } => Some(term[0]), _ => None },
            _ => None,
        }
    }

    /// The token step `id`; false when the grammar does not take it here.
    pub fn feed_token(&mut self, id: u32) -> bool {
        loop {
            let top = self.stack.len() - 1;
            match self.stack[top] {
                Frame::Until { node, need: 0, .. } => return match self.node(node) {
                    Node::Until { token: Some(t), .. } if *t == id => { self.stack.pop(); true }
                    _ => false,
                },
                Frame::Value { node, .. } => match self.node(node) {
                    Node::Token(t) if *t == id => { self.stack.pop(); return true; }
                    Node::Until { token: Some(t), .. } if *t == id => { self.stack.pop(); return true; }
                    Node::Seq(_) => self.stack[top] = Frame::Seq { node, idx: 0 },
                    Node::Repeat { .. } => self.stack[top] = Frame::Rep { node, count: 0, phase: 0, sp: 0 },
                    _ => return false,
                },
                Frame::Seq { node, idx } => {
                    let p = match self.node(node) { Node::Seq(p) => p, _ => unreachable!() };
                    if (idx as usize) < p.len() {
                        self.stack[top] = Frame::Seq { node, idx: idx + 1 };
                        self.stack.push(Frame::Value { node: p[idx as usize] as u32, ws: 0 });
                    } else { self.stack.pop(); }
                }
                Frame::Rep { node, count, phase, .. } => {
                    let (item, min, max, sep) = match self.node(node) {
                        Node::Repeat { node, min, max, sep } => (*node, *min, *max, sep.clone()),
                        _ => unreachable!(),
                    };
                    let item_next = phase == 0 || phase == 3 || (phase == 1 && sep.is_empty());
                    if item_next && count < max && Self::first_token(self.schema, item) == Some(id) {
                        self.stack[top] = Frame::Rep { node, count: count + 1, phase: 1, sp: 0 };
                        self.stack.push(Frame::Value { node: item as u32, ws: 0 });
                    } else if (phase == 0 || phase == 1) && count >= min { self.stack.pop(); } else { return false; }
                }
                Frame::Num { phase, .. } if num_end(phase) => { self.stack.pop(); }
                Frame::Lit { node, pos, cand } if self.lit_complete(node, pos, cand) => { self.stack.pop(); }
                _ => return false,
            }
        }
    }

    /// A number opens with a minus sign or its first digit.
    fn start_number(&mut self, top: usize, int: bool, b: u8) -> bool {
        let phase = if b == b'-' { 0 } else { match num_step(int, 0, b) { Some(p) => p, None => return false } };
        self.stack[top] = Frame::Num { int, phase, len: 1 };
        true
    }

    fn start_value(&mut self, node: u32, ws: u8, b: u8) -> bool {
        let top = self.stack.len() - 1;
        // a space only before a value INSIDE a JSON array or object: never before a JSON root, whether it is the grammar's
        // root or a part of a description (after a literal, in a sequence: a model's '\n\n {' after its reasoning was measured)
        let inside = self.stack.len() >= 2
            && matches!(self.stack[self.stack.len() - 2], Frame::Arr { .. } | Frame::Obj { .. });
        if is_ws(b) && self.json_value(node) && inside {
            if ws >= WS_MAX { return false; }
            self.stack[top] = Frame::Value { node, ws: ws + 1 };
            return true;
        }
        match self.node(node) {
            Node::Literals(l) => {
                let cand = first_byte_candidates(l, b);
                if cand == 0 { return false; }
                self.stack[top] = Frame::Lit { node, pos: 0, cand: u64::MAX };
                self.lit_byte(top, node, 0, cand, b)
            }
            Node::Text(_) => { self.stack[top] = Frame::Text { node, pos: 0 }; self.feed(b) }
            Node::Str => {
                if b != b'"' { return false; }
                self.stack[top] = Frame::Str { esc: 0, need: 0, lo: 0, hi: 0 };
                true
            }
            Node::Integer | Node::Number => self.start_number(top, matches!(self.node(node), Node::Integer), b),
            Node::Array { .. } => {
                if b != b'[' { return false; }
                self.stack[top] = Frame::Arr { node, count: 0, phase: 0, ws: 0 };
                true
            }
            Node::Object { .. } => {
                if b != b'{' { return false; }
                self.stack[top] = Frame::Obj { node, seen: 0, phase: 0, ws: 0, pos: 0, cand: 0, cur: 0 };
                true
            }
            Node::AnyOf(alts) => match alts.iter().find(|&&a| Self::starts(self.schema, a, b)) {
                Some(&a) => { self.stack[top] = Frame::Value { node: a as u32, ws }; self.feed(b) }
                None => false,
            },
            Node::Seq(_) => { self.stack[top] = Frame::Seq { node, idx: 0 }; self.feed(b) }
            Node::Choice { prefixes, then } => {
                let cand = first_byte_candidates(prefixes, b);
                if cand == 0 { return false; }
                self.stack[top] = match (0..prefixes.len()).find(|&i| cand >> i & 1 == 1 && prefixes[i].len() == 1) {
                    Some(i) => Frame::Value { node: then[i] as u32, ws: 0 },
                    None => Frame::Choice { node, pos: 1, cand },
                };
                true
            }
            Node::Repeat { .. } => { self.stack[top] = Frame::Rep { node, count: 0, phase: 0, sp: 0 }; self.feed(b) }
            Node::Until { .. } => { self.stack[top] = Frame::Until { node, k: 0, need: 0, lo: 0, hi: 0 }; self.feed(b) }
            Node::Token(_) => false,
        }
    }

    /// A literal frame at `top` reading `b` with `cand` still matching after `pos` bytes; completes eagerly when no
    /// candidate goes on, lazily (at the next byte) when one may.
    fn lit_byte(&mut self, top: usize, node: u32, pos: u16, cand: u64, b: u8) -> bool {
        let l = self.lits(node);
        let mut next = 0u64;
        for i in 0..l.len() {
            if cand >> i & 1 == 1 && l[i].len() > pos as usize && l[i][pos as usize] == b { next |= 1 << i; }
        }
        if next == 0 { return false; }
        let p = pos + 1;
        let longer = (0..l.len()).any(|i| next >> i & 1 == 1 && l[i].len() > p as usize);
        if longer { self.stack[top] = Frame::Lit { node, pos: p, cand: next }; } else { self.stack.pop(); }
        true
    }

    /// One byte; false (the matcher unusable after) when the grammar refuses it. The top frame steps (one method per
    /// frame kind below); a frame that ends without taking the byte hands it to the frame under it.
    pub fn feed(&mut self, b: u8) -> bool {
        loop {
            let top = self.stack.len() - 1;
            let took = match self.stack[top] {
                Frame::Done => Some(false),
                Frame::Value { node, ws } => Some(self.start_value(node, ws, b)),
                Frame::Lit { node, pos, cand } => self.step_lit(top, node, pos, cand, b),
                Frame::Text { node, pos } => self.step_text(top, node, pos, b),
                Frame::Str { esc, need, lo, hi } => self.step_str(top, esc, need, lo, hi, b),
                Frame::Num { int, phase, len } => self.step_num(top, int, phase, len, b),
                Frame::Arr { node, count, phase, ws } => self.step_arr(top, node, count, phase, ws, b),
                Frame::Obj { node, seen, phase, ws, pos, cand, cur } => self.step_obj(top, node, seen, phase, ws, pos, cand, cur, b),
                Frame::Seq { node, idx } => self.step_seq(top, node, idx),
                Frame::Choice { node, pos, cand } => self.step_choice(top, node, pos, cand, b),
                Frame::Rep { node, count, phase, sp } => self.step_rep(top, node, count, phase, sp, b),
                Frame::Until { node, k, need, lo, hi } => self.step_until(top, node, k, need, lo, hi, b),
            };
            if let Some(r) = took { return r; }
        }
    }

    /// A literal among candidates: the byte goes on with one of them, or the literal is complete and the byte is the
    /// parent's. `None`: the frame ended without taking the byte.
    fn step_lit(&mut self, top: usize, node: u32, pos: u16, cand: u64, b: u8) -> Option<bool> {
        let l = self.lits(node);
        let goes_on = (0..l.len()).any(|i| cand >> i & 1 == 1 && l[i].len() > pos as usize && l[i][pos as usize] == b);
        if goes_on { return Some(self.lit_byte(top, node, pos, cand, b)); }
        if self.lit_complete(node, pos, cand) { self.stack.pop(); return None; }
        return Some(false);
    }

    /// A fixed text: its next byte or nothing. `None`: the frame ended without taking the byte.
    fn step_text(&mut self, top: usize, node: u32, pos: u16, b: u8) -> Option<bool> {
        let t = match self.node(node) { Node::Text(t) => t, _ => unreachable!() };
        if t[pos as usize] != b { return Some(false); }
        if pos as usize + 1 == t.len() { self.stack.pop(); } else { self.stack[top] = Frame::Text { node, pos: pos + 1 }; }
        return Some(true);
    }

    /// Inside a JSON string: an escape, a \\u sequence, a UTF-8 continuation, the closing quote or a character. `None`: the frame ended without taking the byte.
    fn step_str(&mut self, top: usize, esc: u8, need: u8, lo: u8, hi: u8, b: u8) -> Option<bool> {
        let f = if need > 0 {
            match utf8_step(need, lo, hi, b, false) { Some((n, l, h)) => Frame::Str { esc: 0, need: n, lo: l, hi: h }, None => return Some(false) }
        } else if esc == 1 {
            match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Frame::Str { esc: 0, need: 0, lo: 0, hi: 0 },
                b'u' => Frame::Str { esc: 2, need: 0, lo: 0, hi: 0 },
                _ => return Some(false),
            }
        } else if esc >= 2 {
            if !b.is_ascii_hexdigit() { return Some(false); }
            Frame::Str { esc: if esc == 5 { 0 } else { esc + 1 }, need: 0, lo: 0, hi: 0 }
        } else {
            match b {
                b'"' => { self.stack.pop(); return Some(true); }
                b'\\' => Frame::Str { esc: 1, need: 0, lo: 0, hi: 0 },
                _ => match utf8_step(0, 0, 0, b, false) { Some((n, l, h)) => Frame::Str { esc: 0, need: n, lo: l, hi: h }, None => return Some(false) },
            }
        };
        self.stack[top] = f;
        return Some(true);
    }

    /// Inside a number: one more byte of it, or the number is complete and the byte is the parent's. `None`: the frame ended without taking the byte.
    fn step_num(&mut self, top: usize, int: bool, phase: u8, len: u8, b: u8) -> Option<bool> {
        if let Some(p) = num_step(int, phase, b) {
            if (len as usize) < MAX_NUMBER {
                self.stack[top] = Frame::Num { int, phase: p, len: len + 1 };
                return Some(true);
            }
        }
        if num_end(phase) { self.stack.pop(); return None; }
        return Some(false);
    }

    /// Inside an array: bounded whitespace, the closing bracket, a comma, or the first byte of an item. `None`: the frame ended without taking the byte.
    fn step_arr(&mut self, top: usize, node: u32, count: u32, phase: u8, ws: u8, b: u8) -> Option<bool> {
        let (items, min, max) = match self.node(node) { Node::Array { items, min, max } => (*items as u32, *min, *max), _ => unreachable!() };
        if is_ws(b) {
            if ws >= WS_MAX { return Some(false); }
            self.stack[top] = Frame::Arr { node, count, phase, ws: ws + 1 };
            return Some(true);
        }
        match (phase, b) {
            (0, b']') | (1, b']') => { if count < min { return Some(false); } self.stack.pop(); return Some(true); }
            (1, b',') => { if count >= max { return Some(false); } self.stack[top] = Frame::Arr { node, count, phase: 2, ws: 0 }; return Some(true); }
            (0, _) | (2, _) => {
                if count >= max { return Some(false); }
                self.stack[top] = Frame::Arr { node, count: count + 1, phase: 1, ws: 0 };
                self.stack.push(Frame::Value { node: items, ws: 0 });
                return Some(self.feed(b));
            }
            _ => return Some(false),
        }
    }

    /// Inside an object: bounded whitespace, the closing brace, a key among the unseen ones byte by byte, the colon,
    /// a comma. `None`: the frame ended without taking the byte.
    fn step_obj(&mut self, top: usize, node: u32, seen: u64, phase: u8, ws: u8, pos: u16, cand: u64, cur: u8, b: u8) -> Option<bool> {
        let (keys, values, required) = match self.node(node) { Node::Object { keys, values, required } => (keys, values, *required), _ => unreachable!() };
        let all = if keys.len() == 64 { u64::MAX } else { (1u64 << keys.len()) - 1 };
        if phase != 1 && is_ws(b) {
            if ws >= WS_MAX { return Some(false); }
            self.stack[top] = Frame::Obj { node, seen, phase, ws: ws + 1, pos, cand, cur };
            return Some(true);
        }
        match (phase, b) {
            (0, b'}') | (4, b'}') => { if required & !seen != 0 { return Some(false); } self.stack.pop(); return Some(true); }
            (0, b'"') | (5, b'"') => {
                let unseen = all & !seen;
                if unseen == 0 { return Some(false); }
                self.stack[top] = Frame::Obj { node, seen, phase: 1, ws: 0, pos: 1, cand: unseen, cur };
                return Some(true);
            }
            (1, _) => {
                let mut next = 0u64;
                let mut done: Option<usize> = None;
                for i in 0..keys.len() {
                    if cand >> i & 1 == 1 && keys[i].len() > pos as usize && keys[i][pos as usize] == b {
                        next |= 1 << i;
                        if keys[i].len() == pos as usize + 1 { done = Some(i); }
                    }
                }
                if next == 0 { return Some(false); }
                self.stack[top] = match done {
                    Some(i) => Frame::Obj { node, seen: seen | 1 << i, phase: 2, ws: 0, pos: 0, cand: 0, cur: i as u8 },
                    None => Frame::Obj { node, seen, phase: 1, ws: 0, pos: pos + 1, cand: next, cur },
                };
                return Some(true);
            }
            (2, b':') => {
                self.stack[top] = Frame::Obj { node, seen, phase: 4, ws: 0, pos: 0, cand: 0, cur };
                self.stack.push(Frame::Value { node: values[cur as usize] as u32, ws: 0 });
                return Some(true);
            }
            (4, b',') => {
                if all & !seen == 0 { return Some(false); }
                self.stack[top] = Frame::Obj { node, seen, phase: 5, ws: 0, pos: 0, cand: 0, cur };
                return Some(true);
            }
            _ => return Some(false),
        }
    }

    /// A sequence: its next part is opened for the byte, or it is complete and the byte is the parent's. `None`: the frame ended without taking the byte.
    fn step_seq(&mut self, top: usize, node: u32, idx: u16) -> Option<bool> {
        let p = match self.node(node) { Node::Seq(p) => p, _ => unreachable!() };
        if (idx as usize) < p.len() {
            self.stack[top] = Frame::Seq { node, idx: idx + 1 };
            self.stack.push(Frame::Value { node: p[idx as usize] as u32, ws: 0 });
            return None;
        }
        self.stack.pop();
        return None;
    }

    /// A choice by prefix: the byte narrows the candidates; a complete prefix hands over to its continuation. `None`: the frame ended without taking the byte.
    fn step_choice(&mut self, top: usize, node: u32, pos: u16, cand: u64, b: u8) -> Option<bool> {
        let (prefixes, then) = match self.node(node) { Node::Choice { prefixes, then } => (prefixes, then), _ => unreachable!() };
        let mut next = 0u64;
        let mut done = None;
        for i in 0..prefixes.len() {
            if cand >> i & 1 == 1 && prefixes[i].len() > pos as usize && prefixes[i][pos as usize] == b {
                next |= 1 << i;
                if prefixes[i].len() == pos as usize + 1 { done = Some(i); }
            }
        }
        if next == 0 { return Some(false); }
        self.stack[top] = match done {
            Some(i) => Frame::Value { node: then[i] as u32, ws: 0 },
            None => Frame::Choice { node, pos: pos + 1, cand: next },
        };
        return Some(true);
    }

    /// A repetition: the byte opens an item, walks the separator, or the repetition is complete and the byte is the
    /// parent's. `None`: the frame ended without taking the byte.
    fn step_rep(&mut self, top: usize, node: u32, count: u32, phase: u8, sp: u16, b: u8) -> Option<bool> {
        let (item, min, max, sep) = match self.node(node) { Node::Repeat { node, min, max, sep } => (*node, *min, *max, sep), _ => unreachable!() };
        let item_next = phase == 0 || phase == 3 || (phase == 1 && sep.is_empty());
        if item_next && count < max && Self::starts(self.schema, item, b) {
            self.stack[top] = Frame::Rep { node, count: count + 1, phase: 1, sp: 0 };
            self.stack.push(Frame::Value { node: item as u32, ws: 0 });
            return None;
        }
        if phase == 1 && !sep.is_empty() && count < max && sep[0] == b {
            self.stack[top] = if sep.len() == 1 { Frame::Rep { node, count, phase: 3, sp: 0 } } else { Frame::Rep { node, count, phase: 2, sp: 1 } };
            return Some(true);
        }
        if phase == 2 {
            if sep[sp as usize] != b { return Some(false); }
            self.stack[top] = if sp as usize + 1 == sep.len() { Frame::Rep { node, count, phase: 3, sp: 0 } } else { Frame::Rep { node, count, phase: 2, sp: sp + 1 } };
            return Some(true);
        }
        if (phase == 0 || phase == 1) && count >= min { self.stack.pop(); return None; }
        return Some(false);
    }

    /// Free text up to a terminator (KMP over the terminator, whole UTF-8 characters). `None`: the frame ended without taking the byte.
    fn step_until(&mut self, top: usize, node: u32, k: u16, need: u8, lo: u8, hi: u8, b: u8) -> Option<bool> {
        let (term, fail) = match self.node(node) { Node::Until { term, fail, .. } => (term, fail), _ => unreachable!() };
        let (n, l, h) = match utf8_step(need, lo, hi, b, true) { Some(x) => x, None => return Some(false) };
        let mut k = k as usize;
        while k > 0 && term[k] != b { k = fail[k - 1] as usize; }
        if term[k] == b { k += 1; }
        if k == term.len() { self.stack.pop(); return Some(true); }
        self.stack[top] = Frame::Until { node, k: k as u16, need: n, lo: l, hi: h };
        return Some(true);
    }

    pub fn feed_all(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.feed(b))
    }
}
