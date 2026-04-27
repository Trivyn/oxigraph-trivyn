//! Trivia-preserving concrete syntax tree (CST) for Turtle.
//!
//! Use [`TurtleCstParser`] when you need to parse a Turtle document, mutate it
//! (rename a class, swap a parent, add a label, etc.), and re-serialize the
//! result with comments and whitespace intact. The semantic-only
//! [`crate::TurtleParser`] path is unaffected by this module.
//!
//! Round-trip invariant: for any well-formed Turtle input that has not been
//! mutated, parsing and re-serializing produces byte-exact output.

// The CST is a self-contained, complex module. We split its inherent impls
// (display/serialize vs. mutation) into multiple blocks for clarity, and use
// some builder-style methods that take ownership of small values for ergonomic
// reasons. The lints suppressed below reflect those intentional choices.
#![allow(
    clippy::allow_attributes,
    clippy::multiple_inherent_impl,
    clippy::needless_pass_by_value,
    clippy::manual_let_else,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

use crate::lexer::{N3Lexer, N3LexerMode, N3LexerOptions, N3Token};
use crate::toolkit::{Lexer, TextPosition, TokenOrLineJump, TurtleSyntaxError};
use crate::{MAX_BUFFER_SIZE, MIN_BUFFER_SIZE};
use oxiri::{Iri, IriParseError};
use oxrdf::vocab::{rdf, xsd};
use oxrdf::{BlankNode, Literal, NamedNode, NamedOrBlankNode, Term};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::io::{self, Write};

// =====================================================================
// Public entry point
// =====================================================================

/// Builder for parsing a Turtle document into a trivia-preserving [`TurtleCst`].
///
/// ```
/// use oxttl::TurtleCstParser;
/// let input = b"@prefix ex: <http://example.com/> .\n# hello\nex:Foo a ex:Bar .\n";
/// let cst = TurtleCstParser::new().parse_slice(input).unwrap();
/// assert_eq!(cst.to_string().as_bytes(), input);
/// ```
#[derive(Default)]
pub struct TurtleCstParser {
    base_iri: Option<Iri<String>>,
    lenient: bool,
}

impl TurtleCstParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base_iri = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Permit IRIs and language tags that fail strict validation.
    #[must_use]
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Parse a slice of bytes into a [`TurtleCst`].
    pub fn parse_slice(self, input: &[u8]) -> Result<TurtleCst, TurtleSyntaxError> {
        let events = lex_to_events(input, self.lenient)?;
        let mut cursor = EventCursor::new(events);
        let mut prefixes = HashMap::new();
        let mut base = self.base_iri.clone();
        let items = parse_document(&mut cursor, &mut prefixes, &mut base, self.lenient)?;
        Ok(TurtleCst {
            items,
            prefixes,
            base,
        })
    }
}

// =====================================================================
// CST node types
// =====================================================================

/// Top-level Turtle document.
#[derive(Debug, Clone)]
pub struct TurtleCst {
    items: Vec<DocItem>,
    /// Prefixes declared in the document, by prefix name → resolved IRI.
    prefixes: HashMap<String, Iri<String>>,
    /// Active base IRI as last set by a `@base` / `BASE` directive (or the
    /// builder's initial base). Retained so future versions can synthesize
    /// relative IRIs.
    #[allow(dead_code)]
    base: Option<Iri<String>>,
}

/// One top-level item in the document.
#[derive(Debug, Clone)]
pub enum DocItem {
    /// A run of comments and whitespace not attached to any directive or
    /// statement. Used for paragraph-style separators between top-level items.
    FreeTrivia(Vec<Trivia>),
    Directive(Directive),
    Statement(Statement),
}

/// A piece of trivia.
#[derive(Debug, Clone)]
pub enum Trivia {
    /// Whitespace (any combination of spaces, tabs, and newlines). Stored
    /// verbatim so that round-tripping is byte-exact.
    Whitespace(String),
    /// A line comment, including the leading `#` and the trailing newline (if
    /// present in the source).
    Comment(String),
}

#[derive(Debug, Clone)]
pub struct Directive {
    pub leading_trivia: Vec<Trivia>,
    pub kind: DirectiveKind,
    pub trailing_trivia: Vec<Trivia>,
}

#[derive(Debug, Clone)]
pub enum DirectiveKind {
    /// `@prefix pre: <iri> .` or `PREFIX pre: <iri>`.
    Prefix {
        keyword: SourceText, // `@prefix` or `PREFIX`
        leading_name_trivia: Vec<Trivia>,
        prefix: SourceText,  // `pre:`
        prefix_name: String, // empty string for default prefix
        leading_iri_trivia: Vec<Trivia>,
        iri: IriNode,
        leading_terminator_trivia: Vec<Trivia>,
        terminator: Option<SourceText>, // `.` for `@prefix`, None for sparql `PREFIX`
    },
    /// `@base <iri> .` or `BASE <iri>`.
    Base {
        keyword: SourceText,
        leading_iri_trivia: Vec<Trivia>,
        iri: IriNode,
        leading_terminator_trivia: Vec<Trivia>,
        terminator: Option<SourceText>,
    },
    /// `@version "..." .` (RDF 1.2). Kept as opaque source for v1.
    Version { source: SourceText },
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub leading_trivia: Vec<Trivia>,
    pub subject: SubjectNode,
    /// One [`PredicateObjectGroup`] per `;`-separated section. Always at least one.
    /// The first element has no leading separator; later elements have separator `;`.
    pub pog: Vec<PredicateObjectGroup>,
    pub leading_terminator_trivia: Vec<Trivia>,
    pub terminator: SourceText, // `.`
    pub trailing_trivia: Vec<Trivia>,
}

#[derive(Debug, Clone)]
pub struct PredicateObjectGroup {
    /// trivia after the subject (or after a previous `;`) and before the predicate.
    pub leading_trivia: Vec<Trivia>,
    /// `;` separator from the previous group, or `None` for the first.
    pub separator: Option<SourceText>,
    pub leading_predicate_trivia: Vec<Trivia>,
    pub predicate: PredicateNode,
    /// One element per `,`-separated object. Always at least one.
    /// The first element has no leading separator.
    pub objects: Vec<ObjectEntry>,
    /// Trivia after the last object, before the next `;` or the terminator.
    pub trailing_trivia: Vec<Trivia>,
}

#[derive(Debug, Clone)]
pub struct ObjectEntry {
    /// `,` for non-first entries; `None` for the first.
    pub separator: Option<SourceText>,
    /// Trivia between the separator (or end of predicate, for the first object)
    /// and the object itself.
    pub leading_object_trivia: Vec<Trivia>,
    pub object: ObjectNode,
    /// Trivia between the object and the next `,` (or `;` / `.` if this is the
    /// last entry in the group).
    pub trailing_trivia: Vec<Trivia>,
}

#[derive(Debug, Clone)]
pub struct ObjectNode {
    pub term: TermNode,
    /// RDF-1.2 reifier `~` clause (kept verbatim).
    pub reifier: Option<Reifier>,
    /// RDF-1.2 annotation block `{| ... |}` (kept as a CST node, not desugared).
    pub annotation: Option<AnnotationBlock>,
}

#[derive(Debug, Clone)]
pub enum SubjectNode {
    Iri(IriNode),
    BlankNodeLabel(BlankNodeLabelNode),
    AnonBlankNode(AnonBlankNode),
    BlankNodePropertyList(BlankNodePropertyList),
    Collection(Collection),
    ReifiedTriple(Box<ReifiedTriple>),
}

#[derive(Debug, Clone)]
pub enum PredicateNode {
    Iri(IriNode),
    /// The keyword `a` (sugar for `rdf:type`).
    A(SourceText),
}

#[derive(Debug, Clone)]
pub enum TermNode {
    Iri(IriNode),
    BlankNodeLabel(BlankNodeLabelNode),
    AnonBlankNode(AnonBlankNode),
    Literal(LiteralNode),
    BlankNodePropertyList(BlankNodePropertyList),
    Collection(Collection),
    ReifiedTriple(Box<ReifiedTriple>),
}

/// IRI leaf, either `<...>` or `prefix:local`.
#[derive(Debug, Clone)]
pub struct IriNode {
    /// Original source bytes (e.g. `<http://x>` or `ex:Foo`). Empty if synthesized.
    pub source: String,
    /// Resolved value.
    pub value: NamedNode,
    /// `true` if this token was rewritten by a mutation. The serializer will
    /// regenerate `source` from `value` using the document's prefix table.
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub struct BlankNodeLabelNode {
    pub source: String, // `_:foo`
    pub value: BlankNode,
}

#[derive(Debug, Clone)]
pub struct AnonBlankNode {
    pub open: SourceText, // `[`
    pub inner_trivia: Vec<Trivia>,
    pub close: SourceText, // `]`
    pub value: BlankNode,
}

#[derive(Debug, Clone)]
pub struct BlankNodePropertyList {
    pub open: SourceText, // `[`
    pub leading_pog_trivia: Vec<Trivia>,
    pub pog: Vec<PredicateObjectGroup>,
    pub leading_close_trivia: Vec<Trivia>,
    pub close: SourceText, // `]`
    pub value: BlankNode,
}

#[derive(Debug, Clone)]
pub struct Collection {
    pub open: SourceText, // `(`
    pub items: Vec<CollectionItem>,
    pub leading_close_trivia: Vec<Trivia>,
    pub close: SourceText, // `)`
}

#[derive(Debug, Clone)]
pub struct CollectionItem {
    pub leading_trivia: Vec<Trivia>,
    pub object: ObjectNode,
}

#[derive(Debug, Clone)]
pub struct ReifiedTriple {
    pub open: SourceText, // `<<`
    pub leading_subject_trivia: Vec<Trivia>,
    pub subject: SubjectNode,
    pub leading_predicate_trivia: Vec<Trivia>,
    pub predicate: PredicateNode,
    pub leading_object_trivia: Vec<Trivia>,
    pub object: ObjectNode,
    pub reifier: Option<Reifier>,
    pub leading_close_trivia: Vec<Trivia>,
    pub close: SourceText, // `>>`
}

#[derive(Debug, Clone)]
pub struct Reifier {
    pub leading_trivia: Vec<Trivia>,
    pub tilde: SourceText, // `~`
    pub identifier: Option<ReifierIdentifier>,
}

#[derive(Debug, Clone)]
pub enum ReifierIdentifier {
    Iri {
        leading_trivia: Vec<Trivia>,
        iri: IriNode,
    },
    BlankNode {
        leading_trivia: Vec<Trivia>,
        node: BlankNodeLabelNode,
    },
}

#[derive(Debug, Clone)]
pub struct AnnotationBlock {
    pub leading_trivia: Vec<Trivia>,
    pub open: SourceText, // `{|`
    pub leading_pog_trivia: Vec<Trivia>,
    pub pog: Vec<PredicateObjectGroup>,
    pub leading_close_trivia: Vec<Trivia>,
    pub close: SourceText, // `|}`
}

#[derive(Debug, Clone)]
pub struct LiteralNode {
    pub quoted: SourceText, // `"..."` or `"""..."""`
    pub kind: LiteralKind,
    pub suffix: Option<LiteralSuffix>,
    pub value: Literal,
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub enum LiteralKind {
    /// `"..."` short string literal.
    String,
    /// `"""..."""` long string literal.
    LongString,
    /// Numeric or boolean literal — `quoted` carries the entire source (e.g. `42`,
    /// `3.14`, `1e10`, `true`). `suffix` is always `None` for these.
    Plain,
}

#[derive(Debug, Clone)]
pub enum LiteralSuffix {
    Lang {
        leading_trivia: Vec<Trivia>,
        tag: SourceText, // e.g. `@en` or `@en--LTR`
    },
    Datatype {
        leading_trivia: Vec<Trivia>,
        caret: SourceText, // `^^`
        leading_iri_trivia: Vec<Trivia>,
        iri: IriNode,
    },
}

/// A small wrapper over a verbatim string of source bytes for a single token.
#[derive(Debug, Clone)]
pub struct SourceText {
    pub source: String,
}

impl SourceText {
    fn new(s: impl Into<String>) -> Self {
        Self { source: s.into() }
    }
}

// =====================================================================
// Internal: lexer-event buffer
// =====================================================================

#[derive(Debug)]
enum RawEvent {
    Token(OwnedToken),
    Whitespace(String),
    LineJump(String),
    Comment(String),
}

#[derive(Debug)]
struct OwnedToken {
    /// Original source bytes for this token.
    source: String,
    kind: OwnedTokenKind,
    /// Position of the token in the input. Used for error reporting.
    location: std::ops::Range<TextPosition>,
}

#[derive(Debug, Clone)]
enum OwnedTokenKind {
    /// `<...>`. Holds the unescaped IRI string from the lexer.
    IriRef(String),
    PrefixedName {
        prefix: String,
        local: String,
    },
    BlankNodeLabel(String),
    /// `"..."` short string. Holds the unescaped value.
    String(String),
    /// `"""..."""` long string. Holds the unescaped value.
    LongString(String),
    Integer,
    Decimal,
    Double,
    /// e.g. `@en`, with optional RDF-1.2 base direction.
    LangTag(String),
    /// One of `.`, `;`, `,`, `[`, `]`, `(`, `)`, `<<`, `>>`, `{|`, `|}`, `^^`, etc.
    Punctuation(String),
    /// Bare keyword like `a`, `BASE`, `PREFIX`, `VERSION`, `true`, `false`.
    Keyword(String),
}

fn lex_to_events(input: &[u8], lenient: bool) -> Result<Vec<RawEvent>, TurtleSyntaxError> {
    let mut lexer = Lexer::new_with_trivia(
        N3Lexer::new(N3LexerMode::Turtle, lenient),
        input,
        true, // is_ending — we have the entire slice
        MIN_BUFFER_SIZE,
        MAX_BUFFER_SIZE,
        Some(b"#"),
        true, // emit_trivia
    );
    let options = N3LexerOptions::default();
    let mut events = Vec::new();
    enum Tag {
        Token(OwnedTokenKind),
        Whitespace,
        LineJump,
        Comment,
    }
    while let Some(result) = lexer.parse_next(&options) {
        let event = result?;
        // Capture the owned token kind first so we can drop the borrow before
        // calling other methods on the lexer.
        let tag = match event {
            TokenOrLineJump::Token(t) => Tag::Token(token_kind_from(&t)),
            TokenOrLineJump::Whitespace => Tag::Whitespace,
            TokenOrLineJump::LineJump => Tag::LineJump,
            TokenOrLineJump::Comment => Tag::Comment,
        };
        let source = lexer.last_token_source().to_string();
        let location = lexer.last_token_location();
        events.push(match tag {
            Tag::Token(kind) => RawEvent::Token(OwnedToken {
                source,
                kind,
                location,
            }),
            Tag::Whitespace => RawEvent::Whitespace(source),
            Tag::LineJump => RawEvent::LineJump(source),
            Tag::Comment => RawEvent::Comment(source),
        });
    }
    Ok(events)
}

fn token_kind_from(token: &N3Token<'_>) -> OwnedTokenKind {
    match token {
        N3Token::IriRef(value) => OwnedTokenKind::IriRef(value.clone()),
        N3Token::PrefixedName { prefix, local, .. } => OwnedTokenKind::PrefixedName {
            prefix: (*prefix).to_owned(),
            local: local.to_string(),
        },
        N3Token::Variable(_) => {
            // Variables are an N3 extension, not valid in Turtle. Treat as
            // punctuation to surface a clear error from the parser.
            OwnedTokenKind::Punctuation(String::new())
        }
        N3Token::BlankNodeLabel(label) => OwnedTokenKind::BlankNodeLabel((*label).to_owned()),
        N3Token::String(s) => OwnedTokenKind::String(s.clone()),
        N3Token::LongString(s) => OwnedTokenKind::LongString(s.clone()),
        N3Token::Integer(_) => OwnedTokenKind::Integer,
        N3Token::Decimal(_) => OwnedTokenKind::Decimal,
        N3Token::Double(_) => OwnedTokenKind::Double,
        N3Token::LangTag { language, .. } => OwnedTokenKind::LangTag((*language).to_owned()),
        N3Token::Punctuation(p) => OwnedTokenKind::Punctuation((*p).to_owned()),
        N3Token::PlainKeyword(k) => OwnedTokenKind::Keyword((*k).to_owned()),
    }
}

// =====================================================================
// Internal: cursor over events
// =====================================================================

struct EventCursor {
    events: Vec<RawEvent>,
    idx: usize,
}

impl EventCursor {
    fn new(events: Vec<RawEvent>) -> Self {
        Self { events, idx: 0 }
    }

    /// Drain all leading trivia events at the cursor into a `Vec<Trivia>`.
    fn drain_trivia(&mut self) -> Vec<Trivia> {
        let mut out = Vec::new();
        while self.idx < self.events.len() {
            match &self.events[self.idx] {
                RawEvent::Whitespace(s) | RawEvent::LineJump(s) => {
                    extend_whitespace(&mut out, s);
                    self.idx += 1;
                }
                RawEvent::Comment(s) => {
                    out.push(Trivia::Comment(s.clone()));
                    self.idx += 1;
                }
                RawEvent::Token(_) => break,
            }
        }
        out
    }

    /// Peek the next non-trivia token without consuming trivia.
    fn peek_token(&self) -> Option<&OwnedToken> {
        let mut i = self.idx;
        while let Some(ev) = self.events.get(i) {
            match ev {
                RawEvent::Token(t) => return Some(t),
                _ => i += 1,
            }
        }
        None
    }

    /// Take the next token (must be present), returning the trivia that
    /// preceded it. Errors if EOF.
    fn take_token(&mut self) -> Result<(Vec<Trivia>, OwnedToken), TurtleSyntaxError> {
        let trivia = self.drain_trivia();
        if self.idx >= self.events.len() {
            return Err(syntax_error_eof());
        }
        // Replace the slot with a placeholder so we can move out.
        let placeholder = RawEvent::Whitespace(String::new());
        let event = std::mem::replace(&mut self.events[self.idx], placeholder);
        self.idx += 1;
        match event {
            RawEvent::Token(t) => Ok((trivia, t)),
            _ => unreachable!("drain_trivia should leave a Token at idx"),
        }
    }

    /// Drain "trailing trivia" events (whitespace, comments) up to but not
    /// including the first line ending. Used after consuming a closer like `.`
    /// or `;` to attach the same-line trailing comment to that closer.
    fn drain_same_line_trailing(&mut self) -> Vec<Trivia> {
        let mut out = Vec::new();
        while self.idx < self.events.len() {
            match &self.events[self.idx] {
                RawEvent::Whitespace(s) => {
                    out.push(Trivia::Whitespace(s.clone()));
                    self.idx += 1;
                }
                RawEvent::Comment(s) => {
                    // A comment terminates the line, so it belongs in this run
                    // along with the (already-included) trailing newline that
                    // the lexer baked into the comment source.
                    out.push(Trivia::Comment(s.clone()));
                    self.idx += 1;
                    break;
                }
                RawEvent::LineJump(_) | RawEvent::Token(_) => break,
            }
        }
        out
    }
}

/// Append whitespace bytes to a trivia vector, coalescing with a previous
/// `Trivia::Whitespace` if one is at the tail.
fn extend_whitespace(out: &mut Vec<Trivia>, src: &str) {
    if let Some(Trivia::Whitespace(last)) = out.last_mut() {
        last.push_str(src);
    } else {
        out.push(Trivia::Whitespace(src.to_owned()));
    }
}

fn syntax_error_eof() -> TurtleSyntaxError {
    let pos = TextPosition {
        line: 0,
        column: 0,
        offset: 0,
    };
    TurtleSyntaxError::new(pos..pos, "Unexpected end of file")
}

fn syntax_error_at(token: &OwnedToken, msg: impl Into<String>) -> TurtleSyntaxError {
    TurtleSyntaxError::new(token.location.clone(), msg.into())
}

// =====================================================================
// Internal: recursive-descent parser
// =====================================================================

fn parse_document(
    cursor: &mut EventCursor,
    prefixes: &mut HashMap<String, Iri<String>>,
    base: &mut Option<Iri<String>>,
    lenient: bool,
) -> Result<Vec<DocItem>, TurtleSyntaxError> {
    let mut items: Vec<DocItem> = Vec::new();
    loop {
        let trivia = cursor.drain_trivia();
        if cursor.peek_token().is_none() {
            // Trailing trivia at EOF: emit as a final FreeTrivia item.
            if !trivia.is_empty() {
                items.push(DocItem::FreeTrivia(trivia));
            }
            break;
        }
        // Decide whether to attach the trivia we just drained to:
        //   (a) the trailing_trivia of the previous DocItem (if any)
        //   (b) FreeTrivia between the previous and the next DocItem
        //   (c) leading_trivia of the next DocItem
        let (trailing_for_prev, between, leading_for_next) = split_inter_doc_trivia(trivia);
        if let Some(last) = items.last_mut() {
            attach_trailing(last, trailing_for_prev);
        } else {
            // No previous item — fold the "trailing" portion into FreeTrivia at the start.
            // (There's no statement to attach to.)
            if !trailing_for_prev.is_empty() {
                items.push(DocItem::FreeTrivia(trailing_for_prev));
            }
        }
        if !between.is_empty() {
            items.push(DocItem::FreeTrivia(between));
        }
        // Now parse the next directive or statement, with `leading_for_next` as its leading trivia.
        let item = parse_doc_item(cursor, leading_for_next, prefixes, base, lenient)?;
        items.push(item);
    }
    Ok(items)
}

/// Split an inter-doc-item trivia run into `(trailing, free, leading)`:
/// - `trailing` = trivia up to and including the first line ending (the closer's same line).
/// - `free` = the middle FreeTrivia chunk (zero or more comments separated from the next item by a blank line).
/// - `leading` = trivia tightly attached to the next item (a comment immediately above with no blank line below it, plus any whitespace).
fn split_inter_doc_trivia(items: Vec<Trivia>) -> (Vec<Trivia>, Vec<Trivia>, Vec<Trivia>) {
    if items.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    // First, split off the "same-line trailing" portion: everything up to and
    // including the first newline.
    let (trailing, mut after_first_nl) = split_after_first_newline(items);
    if after_first_nl.is_empty() {
        return (trailing, Vec::new(), Vec::new());
    }
    // Now scan from the END of `after_first_nl` backward to find the largest
    // suffix that has no "blank line" inside it (≥2 consecutive newlines). That
    // suffix becomes `leading`; the prefix becomes `free`.
    let split_idx = find_blank_line_split(&after_first_nl);
    let leading = after_first_nl.split_off(split_idx);
    (trailing, after_first_nl, leading)
}

/// Split `items` at (and including) the first whitespace entry containing a
/// newline. Returns `(prefix_with_newline, remainder)`.
fn split_after_first_newline(items: Vec<Trivia>) -> (Vec<Trivia>, Vec<Trivia>) {
    let mut prefix = Vec::new();
    let mut iter = items.into_iter();
    let mut found = false;
    for tr in iter.by_ref() {
        match &tr {
            Trivia::Whitespace(s) => {
                if let Some(nl_idx) = first_newline_end(s) {
                    // Split this whitespace at the newline boundary.
                    let (head, tail) = s.split_at(nl_idx);
                    if !head.is_empty() {
                        prefix.push(Trivia::Whitespace(head.to_owned()));
                    }
                    prefix.push(Trivia::Whitespace(s[head.len()..nl_idx].to_owned()));
                    let tail_owned = tail.to_owned();
                    let mut remainder: Vec<Trivia> = iter.collect();
                    if !tail_owned.is_empty() {
                        remainder.insert(0, Trivia::Whitespace(tail_owned));
                    }
                    return (prefix, remainder);
                }
                prefix.push(tr);
            }
            Trivia::Comment(_) => {
                // A comment includes its trailing newline (if any). Push it,
                // then everything after is the remainder.
                prefix.push(tr);
                found = true;
                break;
            }
        }
    }
    if found {
        let remainder = iter.collect();
        (prefix, remainder)
    } else {
        (prefix, Vec::new())
    }
}

/// Index (in bytes, but converted to char boundary) immediately AFTER the first
/// newline character (`\n` or the `\n` of `\r\n`) in `s`. Returns `None` if
/// there is no newline.
fn first_newline_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            return Some(i + 1);
        }
        if *b == b'\r' {
            // Skip past \r and any following \n.
            if bytes.get(i + 1).copied() == Some(b'\n') {
                return Some(i + 2);
            }
            return Some(i + 1);
        }
    }
    None
}

/// Find the index in `items` such that `items[..idx]` contains the last "blank
/// line" boundary and `items[idx..]` is the leading-tight-trivia of the next
/// item. The split point is "right after the last whitespace entry containing a
/// blank line (≥2 newlines)". If no blank line exists, returns 0 (everything is
/// leading).
fn find_blank_line_split(items: &[Trivia]) -> usize {
    let mut last_blank_after = 0_usize;
    for (i, tr) in items.iter().enumerate() {
        if let Trivia::Whitespace(s) = tr {
            if count_newlines(s) >= 2 {
                last_blank_after = i + 1;
            }
        }
    }
    last_blank_after
}

fn count_newlines(s: &str) -> usize {
    s.bytes().filter(|b| *b == b'\n').count()
}

fn attach_trailing(item: &mut DocItem, trailing: Vec<Trivia>) {
    if trailing.is_empty() {
        return;
    }
    match item {
        DocItem::Statement(s) => s.trailing_trivia.extend(trailing),
        DocItem::Directive(d) => d.trailing_trivia.extend(trailing),
        DocItem::FreeTrivia(items) => items.extend(trailing),
    }
}

fn parse_doc_item(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    prefixes: &mut HashMap<String, Iri<String>>,
    base: &mut Option<Iri<String>>,
    lenient: bool,
) -> Result<DocItem, TurtleSyntaxError> {
    let token = cursor.peek_token().ok_or_else(syntax_error_eof)?;
    let is_directive = match &token.kind {
        OwnedTokenKind::Keyword(k) => {
            matches!(
                k.to_ascii_uppercase().as_str(),
                "BASE" | "PREFIX" | "VERSION"
            )
        }
        OwnedTokenKind::LangTag(name) => {
            // `@prefix`/`@base`/`@version` come through as LangTag tokens whose
            // `language` field is the directive name.
            matches!(name.as_str(), "prefix" | "base" | "version")
        }
        _ => false,
    };
    if is_directive {
        Ok(DocItem::Directive(parse_directive(
            cursor,
            leading_trivia,
            prefixes,
            base,
            lenient,
        )?))
    } else {
        Ok(DocItem::Statement(parse_statement(
            cursor,
            leading_trivia,
            prefixes,
            base.as_ref(),
            lenient,
        )?))
    }
}

fn parse_directive(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    prefixes: &mut HashMap<String, Iri<String>>,
    base: &mut Option<Iri<String>>,
    lenient: bool,
) -> Result<Directive, TurtleSyntaxError> {
    let (_, head) = cursor.take_token()?;
    match &head.kind {
        OwnedTokenKind::LangTag(name) => match name.as_str() {
            "prefix" => {
                parse_prefix_directive(cursor, leading_trivia, head, true, prefixes, lenient)
            }
            "base" => parse_base_directive(cursor, leading_trivia, head, true, base, lenient),
            "version" => parse_version_directive(cursor, leading_trivia, head),
            _ => Err(syntax_error_at(&head, format!("Unknown directive @{name}"))),
        },
        OwnedTokenKind::Keyword(k) => match k.to_ascii_uppercase().as_str() {
            "PREFIX" => {
                parse_prefix_directive(cursor, leading_trivia, head, false, prefixes, lenient)
            }
            "BASE" => parse_base_directive(cursor, leading_trivia, head, false, base, lenient),
            "VERSION" => parse_version_directive(cursor, leading_trivia, head),
            _ => Err(syntax_error_at(
                &head,
                format!("Expected a directive keyword, got {k}"),
            )),
        },
        _ => Err(syntax_error_at(&head, "Expected a directive")),
    }
}

fn parse_prefix_directive(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    keyword_token: OwnedToken,
    needs_terminator: bool,
    prefixes: &mut HashMap<String, Iri<String>>,
    lenient: bool,
) -> Result<Directive, TurtleSyntaxError> {
    let keyword = SourceText::new(keyword_token.source);
    let (leading_name_trivia, name_token) = cursor.take_token()?;
    let (prefix_name, prefix_source) = match name_token.kind {
        OwnedTokenKind::PrefixedName { prefix, local } => {
            if !local.is_empty() {
                return Err(syntax_error_at(
                    &OwnedToken {
                        source: name_token.source.clone(),
                        kind: OwnedTokenKind::PrefixedName {
                            prefix: prefix.clone(),
                            local: local.clone(),
                        },
                        location: name_token.location.clone(),
                    },
                    "Prefix declaration must use a name with no local part",
                ));
            }
            (prefix, name_token.source)
        }
        _ => {
            return Err(syntax_error_at(&name_token, "Expected a prefix name"));
        }
    };
    let (leading_iri_trivia, iri_token) = cursor.take_token()?;
    let iri_node = iri_node_from_iriref(&iri_token, None, lenient)?;
    if let Ok(iri) = Iri::parse(iri_node.value.as_str().to_owned()) {
        prefixes.insert(prefix_name.clone(), iri);
    }
    let (leading_terminator_trivia, terminator) = if needs_terminator {
        let (trivia, t) = cursor.take_token()?;
        match &t.kind {
            OwnedTokenKind::Punctuation(p) if p == "." => (trivia, Some(SourceText::new(t.source))),
            _ => return Err(syntax_error_at(&t, "Expected `.` to terminate @prefix")),
        }
    } else {
        (Vec::new(), None)
    };
    Ok(Directive {
        leading_trivia,
        kind: DirectiveKind::Prefix {
            keyword,
            leading_name_trivia,
            prefix: SourceText::new(prefix_source),
            prefix_name,
            leading_iri_trivia,
            iri: iri_node,
            leading_terminator_trivia,
            terminator,
        },
        trailing_trivia: cursor.drain_same_line_trailing(),
    })
}

fn parse_base_directive(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    keyword_token: OwnedToken,
    needs_terminator: bool,
    base: &mut Option<Iri<String>>,
    lenient: bool,
) -> Result<Directive, TurtleSyntaxError> {
    let keyword = SourceText::new(keyword_token.source);
    let (leading_iri_trivia, iri_token) = cursor.take_token()?;
    let iri_node = iri_node_from_iriref(&iri_token, base.as_ref(), lenient)?;
    if let Ok(iri) = Iri::parse(iri_node.value.as_str().to_owned()) {
        *base = Some(iri);
    }
    let (leading_terminator_trivia, terminator) = if needs_terminator {
        let (trivia, t) = cursor.take_token()?;
        match &t.kind {
            OwnedTokenKind::Punctuation(p) if p == "." => (trivia, Some(SourceText::new(t.source))),
            _ => return Err(syntax_error_at(&t, "Expected `.` to terminate @base")),
        }
    } else {
        (Vec::new(), None)
    };
    Ok(Directive {
        leading_trivia,
        kind: DirectiveKind::Base {
            keyword,
            leading_iri_trivia,
            iri: iri_node,
            leading_terminator_trivia,
            terminator,
        },
        trailing_trivia: cursor.drain_same_line_trailing(),
    })
}

fn parse_version_directive(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    keyword_token: OwnedToken,
) -> Result<Directive, TurtleSyntaxError> {
    // For v1, store as opaque source: keyword + version string + optional `.`
    let mut source = String::from(&keyword_token.source);
    let (between_trivia, version_token) = cursor.take_token()?;
    write_trivia_to_string(&mut source, &between_trivia);
    source.push_str(&version_token.source);
    // Optional terminator for `@version`; sparql `VERSION` has none.
    let was_at_directive = matches!(keyword_token.kind, OwnedTokenKind::LangTag(_));
    if was_at_directive {
        let (term_trivia, term_token) = cursor.take_token()?;
        write_trivia_to_string(&mut source, &term_trivia);
        match &term_token.kind {
            OwnedTokenKind::Punctuation(p) if p == "." => source.push_str(&term_token.source),
            _ => {
                return Err(syntax_error_at(
                    &term_token,
                    "Expected `.` to terminate @version",
                ));
            }
        }
    }
    Ok(Directive {
        leading_trivia,
        kind: DirectiveKind::Version {
            source: SourceText::new(source),
        },
        trailing_trivia: cursor.drain_same_line_trailing(),
    })
}

fn write_trivia_to_string(out: &mut String, trivia: &[Trivia]) {
    for t in trivia {
        match t {
            Trivia::Whitespace(s) | Trivia::Comment(s) => out.push_str(s),
        }
    }
}

fn parse_statement(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<Statement, TurtleSyntaxError> {
    let subject = parse_subject(cursor, prefixes, base, lenient)?;
    let mut pog: Vec<PredicateObjectGroup> = Vec::new();
    let subject_starts_pog = !matches!(
        subject,
        SubjectNode::BlankNodePropertyList(_) | SubjectNode::Collection(_)
    );
    let mut pending_separator: Option<SourceText> = None;
    loop {
        let trivia_before = cursor.drain_trivia();
        let peek = cursor.peek_token().ok_or_else(syntax_error_eof)?;
        match &peek.kind {
            OwnedTokenKind::Punctuation(p) if p == "." => {
                if pog.is_empty() && subject_starts_pog {
                    let location = peek.location.clone();
                    return Err(TurtleSyntaxError::new(
                        location,
                        "Expected a predicate-object list".to_owned(),
                    ));
                }
                if pending_separator.is_some() {
                    // A `;` was followed by `.` with no intervening POG. Reject for v1.
                    let location = peek.location.clone();
                    return Err(TurtleSyntaxError::new(
                        location,
                        "Trailing `;` before `.` is not yet supported".to_owned(),
                    ));
                }
                let leading_terminator_trivia = trivia_before;
                let (_, dot) = cursor.take_token()?;
                let trailing_trivia = cursor.drain_same_line_trailing();
                return Ok(Statement {
                    leading_trivia,
                    subject,
                    pog,
                    leading_terminator_trivia,
                    terminator: SourceText::new(dot.source),
                    trailing_trivia,
                });
            }
            OwnedTokenKind::Punctuation(p) if p == ";" => {
                if pog.is_empty() {
                    let location = peek.location.clone();
                    return Err(TurtleSyntaxError::new(
                        location,
                        "Unexpected `;`".to_owned(),
                    ));
                }
                if pending_separator.is_some() {
                    // Repeated `;`. Reject for v1 (can be modelled later if needed).
                    let location = peek.location.clone();
                    return Err(TurtleSyntaxError::new(
                        location,
                        "Repeated `;` is not yet supported by the CST parser".to_owned(),
                    ));
                }
                // Consume `;` and stash it as the separator for the *next* POG.
                let (_, sep_tok) = cursor.take_token()?;
                pog.last_mut()
                    .unwrap()
                    .trailing_trivia
                    .extend(trivia_before);
                pending_separator = Some(SourceText::new(sep_tok.source));
            }
            _ => {
                let group = parse_predicate_object_group(
                    cursor,
                    trivia_before,
                    pending_separator.take(),
                    prefixes,
                    base,
                    lenient,
                )?;
                pog.push(group);
            }
        }
    }
}

fn parse_subject(
    cursor: &mut EventCursor,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<SubjectNode, TurtleSyntaxError> {
    let (_, token) = cursor.take_token()?;
    parse_subject_from_token(cursor, token, prefixes, base, lenient)
}

fn parse_subject_from_token(
    cursor: &mut EventCursor,
    token: OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<SubjectNode, TurtleSyntaxError> {
    match &token.kind {
        OwnedTokenKind::IriRef(_) | OwnedTokenKind::PrefixedName { .. } => Ok(SubjectNode::Iri(
            iri_node_from_token(&token, prefixes, base, lenient)?,
        )),
        OwnedTokenKind::BlankNodeLabel(label) => {
            Ok(SubjectNode::BlankNodeLabel(BlankNodeLabelNode {
                source: token.source.clone(),
                value: BlankNode::new(label.clone())
                    .map_err(|e| syntax_error_at(&token, e.to_string()))?,
            }))
        }
        OwnedTokenKind::Punctuation(p) if p == "[" => {
            // Either anon `[ ]` or `[ predicateObjectList ]`.
            parse_bracket_subject(cursor, &token, prefixes, base, lenient)
        }
        OwnedTokenKind::Punctuation(p) if p == "(" => Ok(SubjectNode::Collection(
            parse_collection(cursor, &token, prefixes, base, lenient)?,
        )),
        OwnedTokenKind::Punctuation(p) if p == "<<" => Ok(SubjectNode::ReifiedTriple(Box::new(
            parse_reified_triple(cursor, &token, prefixes, base, lenient)?,
        ))),
        _ => Err(syntax_error_at(&token, "Expected a subject term")),
    }
}

fn parse_bracket_subject(
    cursor: &mut EventCursor,
    open: &OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<SubjectNode, TurtleSyntaxError> {
    // Peek: if next is `]`, it's an anon blank node. Otherwise a property list.
    let inner_trivia = cursor.drain_trivia();
    let next = cursor.peek_token().ok_or_else(syntax_error_eof)?;
    match &next.kind {
        OwnedTokenKind::Punctuation(p) if p == "]" => {
            let (_, close) = cursor.take_token()?;
            Ok(SubjectNode::AnonBlankNode(AnonBlankNode {
                open: SourceText::new(open.source.clone()),
                inner_trivia,
                close: SourceText::new(close.source),
                value: BlankNode::default(),
            }))
        }
        _ => {
            let (pog, leading_close_trivia, close) =
                parse_pog_list_until_close(cursor, "]", prefixes, base, lenient)?;
            Ok(SubjectNode::BlankNodePropertyList(BlankNodePropertyList {
                open: SourceText::new(open.source.clone()),
                leading_pog_trivia: inner_trivia,
                pog,
                leading_close_trivia,
                close,
                value: BlankNode::default(),
            }))
        }
    }
}

fn parse_pog_list_until_close(
    cursor: &mut EventCursor,
    closer: &str,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<(Vec<PredicateObjectGroup>, Vec<Trivia>, SourceText), TurtleSyntaxError> {
    let mut pog: Vec<PredicateObjectGroup> = Vec::new();
    loop {
        let leading = cursor.drain_trivia();
        let peek = cursor.peek_token().ok_or_else(syntax_error_eof)?;
        match &peek.kind {
            OwnedTokenKind::Punctuation(p) if p == closer => {
                let (_, close_tok) = cursor.take_token()?;
                return Ok((pog, leading, SourceText::new(close_tok.source)));
            }
            OwnedTokenKind::Punctuation(p) if p == ";" => {
                if pog.is_empty() {
                    return Err(syntax_error_at(peek, "Unexpected `;`"));
                }
                let (lead2, sep_tok) = cursor.take_token()?;
                pog.last_mut().unwrap().trailing_trivia.extend(leading);
                pog.last_mut().unwrap().trailing_trivia.extend(lead2);
                let sep = SourceText::new(sep_tok.source);
                let next_leading = cursor.drain_trivia();
                let peek2 = cursor.peek_token().ok_or_else(syntax_error_eof)?;
                if let OwnedTokenKind::Punctuation(p) = &peek2.kind {
                    if p == closer {
                        // Trailing `;` before closer.
                        return Err(syntax_error_at(
                            peek2,
                            "Trailing `;` before closer not yet supported",
                        ));
                    }
                }
                let group = parse_predicate_object_group(
                    cursor,
                    next_leading,
                    Some(sep),
                    prefixes,
                    base,
                    lenient,
                )?;
                pog.push(group);
            }
            _ => {
                let group =
                    parse_predicate_object_group(cursor, leading, None, prefixes, base, lenient)?;
                pog.push(group);
            }
        }
    }
}

fn parse_predicate_object_group(
    cursor: &mut EventCursor,
    leading_trivia: Vec<Trivia>,
    separator: Option<SourceText>,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<PredicateObjectGroup, TurtleSyntaxError> {
    // After (optional) `;`, parse: predicate (then objectList).
    let leading_predicate_trivia = cursor.drain_trivia();
    let predicate = parse_predicate(cursor, prefixes, base, lenient)?;
    let mut objects: Vec<ObjectEntry> = Vec::new();
    // First object is mandatory.
    let leading_object_trivia0 = cursor.drain_trivia();
    let object0 = parse_object(cursor, prefixes, base, lenient)?;
    objects.push(ObjectEntry {
        separator: None,
        leading_object_trivia: leading_object_trivia0,
        object: object0,
        trailing_trivia: Vec::new(),
    });
    // Subsequent objects: while we see `,`, parse another.
    loop {
        let trivia_before_comma = cursor.drain_trivia();
        let peek = cursor.peek_token().ok_or_else(syntax_error_eof)?;
        match &peek.kind {
            OwnedTokenKind::Punctuation(p) if p == "," => {
                // Trivia before `,` belongs to the PREVIOUS object's trailing.
                objects.last_mut().unwrap().trailing_trivia = trivia_before_comma;
                let (_, comma_tok) = cursor.take_token()?;
                let leading_object_trivia = cursor.drain_trivia();
                let object = parse_object(cursor, prefixes, base, lenient)?;
                objects.push(ObjectEntry {
                    separator: Some(SourceText::new(comma_tok.source)),
                    leading_object_trivia,
                    object,
                    trailing_trivia: Vec::new(),
                });
            }
            _ => {
                // No more objects in this group. Trivia between the last object
                // and the next `;`/`.` goes in the GROUP's trailing_trivia.
                return Ok(PredicateObjectGroup {
                    leading_trivia,
                    separator,
                    leading_predicate_trivia,
                    predicate,
                    objects,
                    trailing_trivia: trivia_before_comma,
                });
            }
        }
    }
}

fn parse_predicate(
    cursor: &mut EventCursor,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<PredicateNode, TurtleSyntaxError> {
    let (_, token) = cursor.take_token()?;
    match &token.kind {
        OwnedTokenKind::Keyword(k) if k == "a" => {
            Ok(PredicateNode::A(SourceText::new(token.source)))
        }
        OwnedTokenKind::IriRef(_) | OwnedTokenKind::PrefixedName { .. } => Ok(PredicateNode::Iri(
            iri_node_from_token(&token, prefixes, base, lenient)?,
        )),
        _ => Err(syntax_error_at(&token, "Expected a predicate")),
    }
}

fn parse_object(
    cursor: &mut EventCursor,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<ObjectNode, TurtleSyntaxError> {
    let (_, token) = cursor.take_token()?;
    let term = parse_term_from_token(cursor, token, prefixes, base, lenient)?;
    // Reifier?
    let reifier = parse_optional_reifier(cursor, prefixes, base, lenient)?;
    // Annotation block?
    let annotation = parse_optional_annotation(cursor, prefixes, base, lenient)?;
    Ok(ObjectNode {
        term,
        reifier,
        annotation,
    })
}

fn parse_term_from_token(
    cursor: &mut EventCursor,
    token: OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<TermNode, TurtleSyntaxError> {
    match &token.kind {
        OwnedTokenKind::IriRef(_) | OwnedTokenKind::PrefixedName { .. } => Ok(TermNode::Iri(
            iri_node_from_token(&token, prefixes, base, lenient)?,
        )),
        OwnedTokenKind::BlankNodeLabel(label) => Ok(TermNode::BlankNodeLabel(BlankNodeLabelNode {
            source: token.source.clone(),
            value: BlankNode::new(label.clone())
                .map_err(|e| syntax_error_at(&token, e.to_string()))?,
        })),
        OwnedTokenKind::Punctuation(p) if p == "[" => {
            let inner_trivia = cursor.drain_trivia();
            let next = cursor.peek_token().ok_or_else(syntax_error_eof)?;
            match &next.kind {
                OwnedTokenKind::Punctuation(c) if c == "]" => {
                    let (_, close) = cursor.take_token()?;
                    Ok(TermNode::AnonBlankNode(AnonBlankNode {
                        open: SourceText::new(token.source.clone()),
                        inner_trivia,
                        close: SourceText::new(close.source),
                        value: BlankNode::default(),
                    }))
                }
                _ => {
                    let (pog, leading_close_trivia, close) =
                        parse_pog_list_until_close(cursor, "]", prefixes, base, lenient)?;
                    Ok(TermNode::BlankNodePropertyList(BlankNodePropertyList {
                        open: SourceText::new(token.source.clone()),
                        leading_pog_trivia: inner_trivia,
                        pog,
                        leading_close_trivia,
                        close,
                        value: BlankNode::default(),
                    }))
                }
            }
        }
        OwnedTokenKind::Punctuation(p) if p == "(" => Ok(TermNode::Collection(parse_collection(
            cursor, &token, prefixes, base, lenient,
        )?)),
        OwnedTokenKind::Punctuation(p) if p == "<<" => Ok(TermNode::ReifiedTriple(Box::new(
            parse_reified_triple(cursor, &token, prefixes, base, lenient)?,
        ))),
        OwnedTokenKind::String(_) | OwnedTokenKind::LongString(_) => Ok(TermNode::Literal(
            parse_string_literal(cursor, token, prefixes, base, lenient)?,
        )),
        OwnedTokenKind::Integer => {
            let value = Literal::new_typed_literal(token.source.as_str(), xsd::INTEGER);
            Ok(TermNode::Literal(LiteralNode {
                quoted: SourceText::new(token.source.clone()),
                kind: LiteralKind::Plain,
                suffix: None,
                value,
                dirty: false,
            }))
        }
        OwnedTokenKind::Decimal => {
            let value = Literal::new_typed_literal(token.source.as_str(), xsd::DECIMAL);
            Ok(TermNode::Literal(LiteralNode {
                quoted: SourceText::new(token.source.clone()),
                kind: LiteralKind::Plain,
                suffix: None,
                value,
                dirty: false,
            }))
        }
        OwnedTokenKind::Double => {
            let value = Literal::new_typed_literal(token.source.as_str(), xsd::DOUBLE);
            Ok(TermNode::Literal(LiteralNode {
                quoted: SourceText::new(token.source.clone()),
                kind: LiteralKind::Plain,
                suffix: None,
                value,
                dirty: false,
            }))
        }
        OwnedTokenKind::Keyword(k) => {
            let lower = k.to_ascii_lowercase();
            if lower == "true" || lower == "false" {
                let value = Literal::new_typed_literal(lower.as_str(), xsd::BOOLEAN);
                Ok(TermNode::Literal(LiteralNode {
                    quoted: SourceText::new(token.source.clone()),
                    kind: LiteralKind::Plain,
                    suffix: None,
                    value,
                    dirty: false,
                }))
            } else {
                Err(syntax_error_at(&token, format!("Unexpected keyword `{k}`")))
            }
        }
        _ => Err(syntax_error_at(&token, "Expected a term")),
    }
}

fn parse_string_literal(
    cursor: &mut EventCursor,
    token: OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<LiteralNode, TurtleSyntaxError> {
    let (kind, raw) = match &token.kind {
        OwnedTokenKind::String(s) => (LiteralKind::String, s.clone()),
        OwnedTokenKind::LongString(s) => (LiteralKind::LongString, s.clone()),
        _ => unreachable!(),
    };
    // Check for suffix: peek next token, but DON'T consume trivia yet — if there's
    // no suffix, the trivia belongs to the surrounding grammar context.
    let mut peek_idx = cursor.idx;
    let mut suffix_leading_trivia = Vec::new();
    while let Some(ev) = cursor.events.get(peek_idx) {
        match ev {
            RawEvent::Whitespace(s) | RawEvent::LineJump(s) => {
                extend_whitespace(&mut suffix_leading_trivia, s);
                peek_idx += 1;
            }
            RawEvent::Comment(s) => {
                suffix_leading_trivia.push(Trivia::Comment(s.clone()));
                peek_idx += 1;
            }
            RawEvent::Token(t) => {
                match &t.kind {
                    OwnedTokenKind::LangTag(_) => {
                        // Consume trivia + tag.
                        cursor.idx = peek_idx + 1;
                        let value = literal_with_lang(&raw, &t.source).map_err(|e| {
                            syntax_error_at(&token, format!("Invalid language tag: {e}"))
                        })?;
                        return Ok(LiteralNode {
                            quoted: SourceText::new(token.source),
                            kind,
                            suffix: Some(LiteralSuffix::Lang {
                                leading_trivia: suffix_leading_trivia,
                                tag: SourceText::new(t.source.clone()),
                            }),
                            value,
                            dirty: false,
                        });
                    }
                    OwnedTokenKind::Punctuation(p) if p == "^^" => {
                        cursor.idx = peek_idx + 1;
                        // Now read the datatype IRI.
                        let caret_source = t.source.clone();
                        let (leading_iri_trivia, iri_token) = cursor.take_token()?;
                        let iri_node = iri_node_from_token(&iri_token, prefixes, base, lenient)?;
                        let value =
                            Literal::new_typed_literal(raw.as_str(), iri_node.value.clone());
                        return Ok(LiteralNode {
                            quoted: SourceText::new(token.source),
                            kind,
                            suffix: Some(LiteralSuffix::Datatype {
                                leading_trivia: suffix_leading_trivia,
                                caret: SourceText::new(caret_source),
                                leading_iri_trivia,
                                iri: iri_node,
                            }),
                            value,
                            dirty: false,
                        });
                    }
                    _ => break,
                }
            }
        }
    }
    // No suffix.
    let value = Literal::new_simple_literal(raw.as_str());
    Ok(LiteralNode {
        quoted: SourceText::new(token.source),
        kind,
        suffix: None,
        value,
        dirty: false,
    })
}

fn literal_with_lang(raw: &str, tag_source: &str) -> Result<Literal, &'static str> {
    // tag_source like "@en" or "@en--LTR". Parse out the language portion.
    let after_at = tag_source.strip_prefix('@').unwrap_or(tag_source);
    // Direction (RDF 1.2) appears as `--LTR`/`--RTL` suffix; for v1 we accept any.
    Literal::new_language_tagged_literal(raw.to_owned(), after_at.to_owned())
        .map_err(|_| "Invalid language tag")
}

fn parse_collection(
    cursor: &mut EventCursor,
    open: &OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<Collection, TurtleSyntaxError> {
    let mut items: Vec<CollectionItem> = Vec::new();
    loop {
        let leading = cursor.drain_trivia();
        let peek = cursor.peek_token().ok_or_else(syntax_error_eof)?;
        match &peek.kind {
            OwnedTokenKind::Punctuation(p) if p == ")" => {
                let (_, close) = cursor.take_token()?;
                return Ok(Collection {
                    open: SourceText::new(open.source.clone()),
                    items,
                    leading_close_trivia: leading,
                    close: SourceText::new(close.source),
                });
            }
            _ => {
                let object = parse_object(cursor, prefixes, base, lenient)?;
                items.push(CollectionItem {
                    leading_trivia: leading,
                    object,
                });
            }
        }
    }
}

fn parse_reified_triple(
    cursor: &mut EventCursor,
    open: &OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<ReifiedTriple, TurtleSyntaxError> {
    let leading_subject_trivia = cursor.drain_trivia();
    let subject = parse_subject(cursor, prefixes, base, lenient)?;
    let leading_predicate_trivia = cursor.drain_trivia();
    let predicate = parse_predicate(cursor, prefixes, base, lenient)?;
    let leading_object_trivia = cursor.drain_trivia();
    let object = parse_object(cursor, prefixes, base, lenient)?;
    let reifier = parse_optional_reifier(cursor, prefixes, base, lenient)?;
    let leading_close_trivia = cursor.drain_trivia();
    let (_, close) = cursor.take_token()?;
    match &close.kind {
        OwnedTokenKind::Punctuation(p) if p == ">>" => Ok(ReifiedTriple {
            open: SourceText::new(open.source.clone()),
            leading_subject_trivia,
            subject,
            leading_predicate_trivia,
            predicate,
            leading_object_trivia,
            object,
            reifier,
            leading_close_trivia,
            close: SourceText::new(close.source),
        }),
        _ => Err(syntax_error_at(&close, "Expected `>>`")),
    }
}

fn parse_optional_reifier(
    cursor: &mut EventCursor,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<Option<Reifier>, TurtleSyntaxError> {
    // Reifier is `~` optionally followed by an iri or blank node label.
    let mut peek_idx = cursor.idx;
    let mut leading = Vec::new();
    while let Some(ev) = cursor.events.get(peek_idx) {
        match ev {
            RawEvent::Whitespace(s) | RawEvent::LineJump(s) => {
                extend_whitespace(&mut leading, s);
                peek_idx += 1;
            }
            RawEvent::Comment(s) => {
                leading.push(Trivia::Comment(s.clone()));
                peek_idx += 1;
            }
            RawEvent::Token(t) => {
                if let OwnedTokenKind::Punctuation(p) = &t.kind {
                    if p == "~" {
                        cursor.idx = peek_idx + 1;
                        let tilde_source = t.source.clone();
                        // Optional identifier
                        let mut id_peek_idx = cursor.idx;
                        let mut id_leading = Vec::new();
                        while let Some(ev2) = cursor.events.get(id_peek_idx) {
                            match ev2 {
                                RawEvent::Whitespace(s) | RawEvent::LineJump(s) => {
                                    extend_whitespace(&mut id_leading, s);
                                    id_peek_idx += 1;
                                }
                                RawEvent::Comment(s) => {
                                    id_leading.push(Trivia::Comment(s.clone()));
                                    id_peek_idx += 1;
                                }
                                RawEvent::Token(t2) => match &t2.kind {
                                    OwnedTokenKind::IriRef(_)
                                    | OwnedTokenKind::PrefixedName { .. } => {
                                        cursor.idx = id_peek_idx + 1;
                                        let iri_node =
                                            iri_node_from_token(t2, prefixes, base, lenient)?;
                                        return Ok(Some(Reifier {
                                            leading_trivia: leading,
                                            tilde: SourceText::new(tilde_source),
                                            identifier: Some(ReifierIdentifier::Iri {
                                                leading_trivia: id_leading,
                                                iri: iri_node,
                                            }),
                                        }));
                                    }
                                    OwnedTokenKind::BlankNodeLabel(label) => {
                                        cursor.idx = id_peek_idx + 1;
                                        let node = BlankNodeLabelNode {
                                            source: t2.source.clone(),
                                            value: BlankNode::new(label.clone())
                                                .map_err(|e| syntax_error_at(t2, e.to_string()))?,
                                        };
                                        return Ok(Some(Reifier {
                                            leading_trivia: leading,
                                            tilde: SourceText::new(tilde_source),
                                            identifier: Some(ReifierIdentifier::BlankNode {
                                                leading_trivia: id_leading,
                                                node,
                                            }),
                                        }));
                                    }
                                    _ => break,
                                },
                            }
                        }
                        return Ok(Some(Reifier {
                            leading_trivia: leading,
                            tilde: SourceText::new(tilde_source),
                            identifier: None,
                        }));
                    }
                }
                return Ok(None);
            }
        }
    }
    Ok(None)
}

fn parse_optional_annotation(
    cursor: &mut EventCursor,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<Option<AnnotationBlock>, TurtleSyntaxError> {
    let mut peek_idx = cursor.idx;
    let mut leading = Vec::new();
    while let Some(ev) = cursor.events.get(peek_idx) {
        match ev {
            RawEvent::Whitespace(s) | RawEvent::LineJump(s) => {
                extend_whitespace(&mut leading, s);
                peek_idx += 1;
            }
            RawEvent::Comment(s) => {
                leading.push(Trivia::Comment(s.clone()));
                peek_idx += 1;
            }
            RawEvent::Token(t) => {
                if let OwnedTokenKind::Punctuation(p) = &t.kind {
                    if p == "{|" {
                        cursor.idx = peek_idx + 1;
                        let open_source = t.source.clone();
                        let (pog, leading_close_trivia, close) =
                            parse_pog_list_until_close(cursor, "|}", prefixes, base, lenient)?;
                        return Ok(Some(AnnotationBlock {
                            leading_trivia: leading,
                            open: SourceText::new(open_source),
                            leading_pog_trivia: Vec::new(),
                            pog,
                            leading_close_trivia,
                            close,
                        }));
                    }
                }
                return Ok(None);
            }
        }
    }
    Ok(None)
}

// =====================================================================
// Internal: IRI resolution
// =====================================================================

fn iri_node_from_token(
    token: &OwnedToken,
    prefixes: &HashMap<String, Iri<String>>,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<IriNode, TurtleSyntaxError> {
    match &token.kind {
        OwnedTokenKind::IriRef(value) => iri_node_from_iriref(token, base, lenient).map(|mut n| {
            n.value = NamedNode::new_unchecked(
                resolve_iri(value.as_str(), base).unwrap_or_else(|| value.clone()),
            );
            n
        }),
        OwnedTokenKind::PrefixedName { prefix, local } => {
            let prefix_iri = prefixes
                .get(prefix.as_str())
                .ok_or_else(|| syntax_error_at(token, format!("Undefined prefix `{prefix}:`")))?;
            let mut full = String::with_capacity(prefix_iri.as_str().len() + local.len());
            full.push_str(prefix_iri.as_str());
            full.push_str(local);
            let value = if lenient {
                NamedNode::new_unchecked(full)
            } else {
                NamedNode::new(full).map_err(|e| syntax_error_at(token, e.to_string()))?
            };
            Ok(IriNode {
                source: token.source.clone(),
                value,
                dirty: false,
            })
        }
        _ => Err(syntax_error_at(token, "Expected an IRI")),
    }
}

fn iri_node_from_iriref(
    token: &OwnedToken,
    base: Option<&Iri<String>>,
    lenient: bool,
) -> Result<IriNode, TurtleSyntaxError> {
    let value = match &token.kind {
        OwnedTokenKind::IriRef(v) => v,
        _ => return Err(syntax_error_at(token, "Expected `<...>`")),
    };
    let resolved = resolve_iri(value.as_str(), base).unwrap_or_else(|| value.clone());
    let nn = if lenient {
        NamedNode::new_unchecked(resolved)
    } else {
        NamedNode::new(resolved).map_err(|e| syntax_error_at(token, e.to_string()))?
    };
    Ok(IriNode {
        source: token.source.clone(),
        value: nn,
        dirty: false,
    })
}

fn resolve_iri(value: &str, base: Option<&Iri<String>>) -> Option<String> {
    Some(base?.resolve(value).ok()?.into_inner())
}

// =====================================================================
// Display / serialization
// =====================================================================

impl fmt::Display for TurtleCst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix_table = self.prefix_table_sorted();
        for item in &self.items {
            write_doc_item(f, item, &prefix_table)?;
        }
        Ok(())
    }
}

impl TurtleCst {
    /// Items in source order.
    pub fn items(&self) -> &[DocItem] {
        &self.items
    }

    /// Mutable access to top-level items.
    pub fn items_mut(&mut self) -> &mut Vec<DocItem> {
        &mut self.items
    }

    pub fn serialize_to_string(&self) -> String {
        self.to_string()
    }

    pub fn serialize_to_writer<W: Write>(&self, mut w: W) -> io::Result<()> {
        let s = self.to_string();
        w.write_all(s.as_bytes())
    }

    fn prefix_table_sorted(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .prefixes
            .iter()
            .map(|(k, iri)| (k.clone(), iri.as_str().to_owned()))
            .collect();
        v.sort_unstable_by_key(|(_, iri)| Reverse(iri.len()));
        v
    }
}

fn write_doc_item(
    f: &mut fmt::Formatter<'_>,
    item: &DocItem,
    prefixes: &[(String, String)],
) -> fmt::Result {
    match item {
        DocItem::FreeTrivia(items) => write_trivia(f, items),
        DocItem::Directive(d) => write_directive(f, d, prefixes),
        DocItem::Statement(s) => write_statement(f, s, prefixes),
    }
}

fn write_trivia(f: &mut fmt::Formatter<'_>, items: &[Trivia]) -> fmt::Result {
    for t in items {
        match t {
            Trivia::Whitespace(s) | Trivia::Comment(s) => f.write_str(s)?,
        }
    }
    Ok(())
}

fn write_directive(
    f: &mut fmt::Formatter<'_>,
    d: &Directive,
    prefixes: &[(String, String)],
) -> fmt::Result {
    write_trivia(f, &d.leading_trivia)?;
    match &d.kind {
        DirectiveKind::Prefix {
            keyword,
            leading_name_trivia,
            prefix,
            leading_iri_trivia,
            iri,
            leading_terminator_trivia,
            terminator,
            ..
        } => {
            f.write_str(&keyword.source)?;
            write_trivia(f, leading_name_trivia)?;
            f.write_str(&prefix.source)?;
            write_trivia(f, leading_iri_trivia)?;
            write_iri_node(f, iri, prefixes)?;
            write_trivia(f, leading_terminator_trivia)?;
            if let Some(t) = terminator {
                f.write_str(&t.source)?;
            }
        }
        DirectiveKind::Base {
            keyword,
            leading_iri_trivia,
            iri,
            leading_terminator_trivia,
            terminator,
        } => {
            f.write_str(&keyword.source)?;
            write_trivia(f, leading_iri_trivia)?;
            write_iri_node(f, iri, prefixes)?;
            write_trivia(f, leading_terminator_trivia)?;
            if let Some(t) = terminator {
                f.write_str(&t.source)?;
            }
        }
        DirectiveKind::Version { source } => f.write_str(&source.source)?,
    }
    write_trivia(f, &d.trailing_trivia)?;
    Ok(())
}

fn write_statement(
    f: &mut fmt::Formatter<'_>,
    s: &Statement,
    prefixes: &[(String, String)],
) -> fmt::Result {
    write_trivia(f, &s.leading_trivia)?;
    write_subject(f, &s.subject, prefixes)?;
    for g in &s.pog {
        write_predicate_object_group(f, g, prefixes)?;
    }
    write_trivia(f, &s.leading_terminator_trivia)?;
    f.write_str(&s.terminator.source)?;
    write_trivia(f, &s.trailing_trivia)?;
    Ok(())
}

fn write_predicate_object_group(
    f: &mut fmt::Formatter<'_>,
    g: &PredicateObjectGroup,
    prefixes: &[(String, String)],
) -> fmt::Result {
    if let Some(sep) = &g.separator {
        f.write_str(&sep.source)?;
    }
    write_trivia(f, &g.leading_trivia)?;
    write_trivia(f, &g.leading_predicate_trivia)?;
    write_predicate(f, &g.predicate, prefixes)?;
    for obj in &g.objects {
        write_object_entry(f, obj, prefixes)?;
    }
    write_trivia(f, &g.trailing_trivia)?;
    Ok(())
}

fn write_object_entry(
    f: &mut fmt::Formatter<'_>,
    o: &ObjectEntry,
    prefixes: &[(String, String)],
) -> fmt::Result {
    if let Some(sep) = &o.separator {
        f.write_str(&sep.source)?;
    }
    write_trivia(f, &o.leading_object_trivia)?;
    write_object(f, &o.object, prefixes)?;
    write_trivia(f, &o.trailing_trivia)?;
    Ok(())
}

fn write_subject(
    f: &mut fmt::Formatter<'_>,
    s: &SubjectNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    match s {
        SubjectNode::Iri(i) => write_iri_node(f, i, prefixes),
        SubjectNode::BlankNodeLabel(b) => f.write_str(&b.source),
        SubjectNode::AnonBlankNode(a) => write_anon_blank_node(f, a),
        SubjectNode::BlankNodePropertyList(b) => write_bnpl(f, b, prefixes),
        SubjectNode::Collection(c) => write_collection(f, c, prefixes),
        SubjectNode::ReifiedTriple(r) => write_reified_triple(f, r, prefixes),
    }
}

fn write_predicate(
    f: &mut fmt::Formatter<'_>,
    p: &PredicateNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    match p {
        PredicateNode::Iri(i) => write_iri_node(f, i, prefixes),
        PredicateNode::A(s) => f.write_str(&s.source),
    }
}

fn write_object(
    f: &mut fmt::Formatter<'_>,
    o: &ObjectNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    write_term(f, &o.term, prefixes)?;
    if let Some(r) = &o.reifier {
        write_reifier(f, r, prefixes)?;
    }
    if let Some(a) = &o.annotation {
        write_annotation_block(f, a, prefixes)?;
    }
    Ok(())
}

fn write_term(
    f: &mut fmt::Formatter<'_>,
    t: &TermNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    match t {
        TermNode::Iri(i) => write_iri_node(f, i, prefixes),
        TermNode::BlankNodeLabel(b) => f.write_str(&b.source),
        TermNode::AnonBlankNode(a) => write_anon_blank_node(f, a),
        TermNode::Literal(l) => write_literal(f, l, prefixes),
        TermNode::BlankNodePropertyList(b) => write_bnpl(f, b, prefixes),
        TermNode::Collection(c) => write_collection(f, c, prefixes),
        TermNode::ReifiedTriple(r) => write_reified_triple(f, r, prefixes),
    }
}

fn write_iri_node(
    f: &mut fmt::Formatter<'_>,
    i: &IriNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    if i.dirty || i.source.is_empty() {
        // Regenerate from value using the prefix table.
        let s = i.value.as_str();
        for (pname, piri) in prefixes {
            if let Some(local) = s.strip_prefix(piri.as_str()) {
                if is_valid_local_name(local) {
                    return write!(f, "{pname}:{local}");
                }
            }
        }
        write!(f, "<{}>", escape_iri(s))
    } else {
        f.write_str(&i.source)
    }
}

fn is_valid_local_name(s: &str) -> bool {
    // Conservative subset: ASCII alphanumerics, underscore, dash, dot — no leading dot.
    if s.is_empty() {
        return true;
    }
    let bytes = s.as_bytes();
    if bytes[0] == b'.' || bytes[bytes.len() - 1] == b'.' {
        return false;
    }
    bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(*b, b'_' | b'-' | b'.'))
}

fn escape_iri(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        // Writes to `String` are infallible, so we ignore the `fmt::Result`.
        if matches!(c, '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\') || (c as u32) < 0x20 {
            write!(out, "\\u{:04X}", c as u32).expect("writing to String never fails");
        } else {
            out.push(c);
        }
    }
    out
}

fn write_anon_blank_node(f: &mut fmt::Formatter<'_>, a: &AnonBlankNode) -> fmt::Result {
    f.write_str(&a.open.source)?;
    write_trivia(f, &a.inner_trivia)?;
    f.write_str(&a.close.source)
}

fn write_bnpl(
    f: &mut fmt::Formatter<'_>,
    b: &BlankNodePropertyList,
    prefixes: &[(String, String)],
) -> fmt::Result {
    f.write_str(&b.open.source)?;
    write_trivia(f, &b.leading_pog_trivia)?;
    for g in &b.pog {
        write_predicate_object_group(f, g, prefixes)?;
    }
    write_trivia(f, &b.leading_close_trivia)?;
    f.write_str(&b.close.source)
}

fn write_collection(
    f: &mut fmt::Formatter<'_>,
    c: &Collection,
    prefixes: &[(String, String)],
) -> fmt::Result {
    f.write_str(&c.open.source)?;
    for it in &c.items {
        write_trivia(f, &it.leading_trivia)?;
        write_object(f, &it.object, prefixes)?;
    }
    write_trivia(f, &c.leading_close_trivia)?;
    f.write_str(&c.close.source)
}

fn write_reified_triple(
    f: &mut fmt::Formatter<'_>,
    r: &ReifiedTriple,
    prefixes: &[(String, String)],
) -> fmt::Result {
    f.write_str(&r.open.source)?;
    write_trivia(f, &r.leading_subject_trivia)?;
    write_subject(f, &r.subject, prefixes)?;
    write_trivia(f, &r.leading_predicate_trivia)?;
    write_predicate(f, &r.predicate, prefixes)?;
    write_trivia(f, &r.leading_object_trivia)?;
    write_object(f, &r.object, prefixes)?;
    if let Some(rf) = &r.reifier {
        write_reifier(f, rf, prefixes)?;
    }
    write_trivia(f, &r.leading_close_trivia)?;
    f.write_str(&r.close.source)
}

fn write_reifier(
    f: &mut fmt::Formatter<'_>,
    r: &Reifier,
    prefixes: &[(String, String)],
) -> fmt::Result {
    write_trivia(f, &r.leading_trivia)?;
    f.write_str(&r.tilde.source)?;
    if let Some(id) = &r.identifier {
        match id {
            ReifierIdentifier::Iri {
                leading_trivia,
                iri,
            } => {
                write_trivia(f, leading_trivia)?;
                write_iri_node(f, iri, prefixes)?;
            }
            ReifierIdentifier::BlankNode {
                leading_trivia,
                node,
            } => {
                write_trivia(f, leading_trivia)?;
                f.write_str(&node.source)?;
            }
        }
    }
    Ok(())
}

fn write_annotation_block(
    f: &mut fmt::Formatter<'_>,
    a: &AnnotationBlock,
    prefixes: &[(String, String)],
) -> fmt::Result {
    write_trivia(f, &a.leading_trivia)?;
    f.write_str(&a.open.source)?;
    write_trivia(f, &a.leading_pog_trivia)?;
    for g in &a.pog {
        write_predicate_object_group(f, g, prefixes)?;
    }
    write_trivia(f, &a.leading_close_trivia)?;
    f.write_str(&a.close.source)
}

fn write_literal(
    f: &mut fmt::Formatter<'_>,
    l: &LiteralNode,
    prefixes: &[(String, String)],
) -> fmt::Result {
    f.write_str(&l.quoted.source)?;
    if let Some(suffix) = &l.suffix {
        match suffix {
            LiteralSuffix::Lang {
                leading_trivia,
                tag,
            } => {
                write_trivia(f, leading_trivia)?;
                f.write_str(&tag.source)?;
            }
            LiteralSuffix::Datatype {
                leading_trivia,
                caret,
                leading_iri_trivia,
                iri,
            } => {
                write_trivia(f, leading_trivia)?;
                f.write_str(&caret.source)?;
                write_trivia(f, leading_iri_trivia)?;
                write_iri_node(f, iri, prefixes)?;
            }
        }
    }
    Ok(())
}

// =====================================================================
// Mutation API
// =====================================================================

impl TurtleCst {
    /// Iterator over mutable references to statements whose subject (after IRI
    /// resolution) equals `iri`.
    pub fn statements_for_subject<'a>(
        &'a mut self,
        iri: &'a NamedNode,
    ) -> impl Iterator<Item = &'a mut Statement> + 'a {
        self.items.iter_mut().filter_map(move |it| match it {
            DocItem::Statement(s) => match &s.subject {
                SubjectNode::Iri(i) if &i.value == iri => Some(s),
                _ => None,
            },
            _ => None,
        })
    }

    /// Replace every IRI matching `old` with `new`. Returns the number of
    /// occurrences replaced.
    pub fn rename_iri(&mut self, old: &NamedNode, new: &NamedNode) -> usize {
        let mut count = 0;
        for it in &mut self.items {
            if let DocItem::Statement(s) = it {
                count += rename_in_statement(s, old, new);
            } else if let DocItem::Directive(_) = it {
                // Renaming inside `@prefix` IRIs is intentionally not handled.
            }
        }
        count
    }

    /// Append a freshly synthesized statement for `subject`. Returns a mutable
    /// reference to the new statement so the caller can attach predicate-object
    /// pairs.
    pub fn add_statement(&mut self, subject: NamedOrBlankNode) -> &mut Statement {
        let separator = if self.has_prior_statement() {
            "\n".to_owned()
        } else {
            String::new()
        };
        if !separator.is_empty() {
            self.items
                .push(DocItem::FreeTrivia(vec![Trivia::Whitespace(separator)]));
        }
        let subject_node = subject_node_from_value(&subject);
        self.items.push(DocItem::Statement(Statement {
            leading_trivia: Vec::new(),
            subject: subject_node,
            pog: Vec::new(),
            leading_terminator_trivia: Vec::new(),
            terminator: SourceText::new(" .\n"),
            trailing_trivia: Vec::new(),
        }));
        match self.items.last_mut().unwrap() {
            DocItem::Statement(s) => s,
            _ => unreachable!(),
        }
    }

    fn has_prior_statement(&self) -> bool {
        self.items
            .iter()
            .any(|i| matches!(i, DocItem::Statement(_)))
    }

    /// Remove every top-level statement whose subject equals `iri`. Returns the
    /// number of statements removed.
    pub fn remove_statements_for_subject(&mut self, iri: &NamedNode) -> usize {
        let mut count = 0;
        let mut kept: Vec<DocItem> = Vec::with_capacity(self.items.len());
        for item in self.items.drain(..) {
            match item {
                DocItem::Statement(s) if matches!(&s.subject, SubjectNode::Iri(i) if &i.value == iri) =>
                {
                    count += 1;
                }
                other => kept.push(other),
            }
        }
        self.items = kept;
        count
    }
}

fn subject_node_from_value(v: &NamedOrBlankNode) -> SubjectNode {
    match v {
        NamedOrBlankNode::NamedNode(n) => SubjectNode::Iri(IriNode {
            source: String::new(),
            value: n.clone(),
            dirty: true,
        }),
        NamedOrBlankNode::BlankNode(b) => SubjectNode::BlankNodeLabel(BlankNodeLabelNode {
            source: format!("_:{}", b.as_str()),
            value: b.clone(),
        }),
    }
}

impl Statement {
    /// Replace the object of any `(predicate, old_object)` pair with `new_object`.
    /// Returns true if any object was replaced.
    pub fn replace_object(
        &mut self,
        predicate: &NamedNode,
        old_object: &Term,
        new_object: Term,
    ) -> bool {
        let mut found = false;
        for g in &mut self.pog {
            if !predicate_matches(&g.predicate, predicate) {
                continue;
            }
            for entry in &mut g.objects {
                if term_matches(&entry.object, old_object) {
                    entry.object = object_node_from_term(&new_object);
                    found = true;
                }
            }
        }
        found
    }

    /// Append a new predicate-object pair as a new POG, preserving the
    /// statement's existing layout (indent, newline-style) by inheriting the
    /// previous POG's `leading_trivia`.
    ///
    /// **Layout note.** Inter-POG whitespace (the `\n      ` between a `;`
    /// separator and the next predicate) is parsed into the *following*
    /// POG's `leading_trivia` field, not `leading_predicate_trivia` — the
    /// latter is drained empty by the time `parse_predicate_object_group`
    /// runs. Cloning the wrong field would produce output like
    /// `... ;next_pred` with no indent. This implementation clones
    /// `leading_trivia` so a multi-line statement stays multi-line and a
    /// single-line statement stays single-line.
    ///
    /// **Separator invariant.** First POG of a statement has
    /// `separator: None`; subsequent POGs have `Some(";")`. If the statement
    /// currently has no POGs, the new one becomes the first and is emitted
    /// without a leading `;`; otherwise it gets a `;` separator.
    pub fn add_predicate_object(&mut self, predicate: NamedNode, object: &Term) {
        let separator = if self.pog.is_empty() {
            None
        } else {
            Some(SourceText::new(";"))
        };
        let leading_trivia = self.pog.last().map_or_else(
            || vec![Trivia::Whitespace(" ".to_owned())],
            |prev| prev.leading_trivia.clone(),
        );
        let new_group = PredicateObjectGroup {
            leading_trivia,
            separator,
            leading_predicate_trivia: Vec::new(),
            predicate: PredicateNode::Iri(IriNode {
                source: String::new(),
                value: predicate,
                dirty: true,
            }),
            objects: vec![ObjectEntry {
                separator: None,
                leading_object_trivia: vec![Trivia::Whitespace(" ".to_owned())],
                object: object_node_from_term(object),
                trailing_trivia: Vec::new(),
            }],
            trailing_trivia: Vec::new(),
        };
        // Move any leading_terminator_trivia from the statement into the new group's
        // trailing_trivia so the terminator stays on its own line if it was before.
        let saved = std::mem::take(&mut self.leading_terminator_trivia);
        let mut group_with_trailing = new_group;
        group_with_trailing.trailing_trivia.extend(saved);
        self.pog.push(group_with_trailing);
    }

    /// Remove the matched `(predicate, object)` pair. Returns true if any pair
    /// was removed.
    ///
    /// If removal demotes a previously non-first POG into the first position,
    /// its leading `;` separator is cleared so the serialized output remains
    /// valid Turtle (`subject pred obj .`, not `subject ; pred obj .`).
    pub fn remove_predicate_object(&mut self, predicate: &NamedNode, object: &Term) -> bool {
        let mut removed = false;
        let mut new_pog: Vec<PredicateObjectGroup> = Vec::new();
        for mut g in self.pog.drain(..) {
            if !predicate_matches(&g.predicate, predicate) {
                new_pog.push(g);
                continue;
            }
            g.objects.retain(|e| {
                if term_matches(&e.object, object) {
                    removed = true;
                    false
                } else {
                    true
                }
            });
            if !g.objects.is_empty() {
                new_pog.push(g);
            }
        }
        self.pog = new_pog;
        if let Some(first) = self.pog.first_mut() {
            first.separator = None;
        }
        removed
    }
}

fn rename_in_statement(s: &mut Statement, old: &NamedNode, new: &NamedNode) -> usize {
    let mut count = 0;
    rename_in_subject(&mut s.subject, old, new, &mut count);
    for g in &mut s.pog {
        rename_in_predicate(&mut g.predicate, old, new, &mut count);
        for entry in &mut g.objects {
            rename_in_object(&mut entry.object, old, new, &mut count);
        }
    }
    count
}

fn rename_in_subject(s: &mut SubjectNode, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    match s {
        SubjectNode::Iri(i) => rename_iri_node(i, old, new, count),
        SubjectNode::BlankNodeLabel(_) | SubjectNode::AnonBlankNode(_) => {}
        SubjectNode::BlankNodePropertyList(b) => {
            for g in &mut b.pog {
                rename_in_predicate(&mut g.predicate, old, new, count);
                for entry in &mut g.objects {
                    rename_in_object(&mut entry.object, old, new, count);
                }
            }
        }
        SubjectNode::Collection(c) => {
            for it in &mut c.items {
                rename_in_object(&mut it.object, old, new, count);
            }
        }
        SubjectNode::ReifiedTriple(r) => rename_in_reified(r, old, new, count),
    }
}

fn rename_in_predicate(p: &mut PredicateNode, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    if let PredicateNode::Iri(i) = p {
        rename_iri_node(i, old, new, count);
    }
}

fn rename_in_object(o: &mut ObjectNode, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    rename_in_term(&mut o.term, old, new, count);
    if let Some(a) = &mut o.annotation {
        for g in &mut a.pog {
            rename_in_predicate(&mut g.predicate, old, new, count);
            for entry in &mut g.objects {
                rename_in_object(&mut entry.object, old, new, count);
            }
        }
    }
    if let Some(r) = &mut o.reifier {
        if let Some(ReifierIdentifier::Iri { iri, .. }) = &mut r.identifier {
            rename_iri_node(iri, old, new, count);
        }
    }
}

fn rename_in_term(t: &mut TermNode, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    match t {
        TermNode::Iri(i) => rename_iri_node(i, old, new, count),
        TermNode::BlankNodeLabel(_) | TermNode::AnonBlankNode(_) => {}
        TermNode::Literal(l) => {
            if let Some(LiteralSuffix::Datatype { iri, .. }) = &mut l.suffix {
                rename_iri_node(iri, old, new, count);
            }
        }
        TermNode::BlankNodePropertyList(b) => {
            for g in &mut b.pog {
                rename_in_predicate(&mut g.predicate, old, new, count);
                for entry in &mut g.objects {
                    rename_in_object(&mut entry.object, old, new, count);
                }
            }
        }
        TermNode::Collection(c) => {
            for it in &mut c.items {
                rename_in_object(&mut it.object, old, new, count);
            }
        }
        TermNode::ReifiedTriple(r) => rename_in_reified(r, old, new, count),
    }
}

fn rename_in_reified(r: &mut ReifiedTriple, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    rename_in_subject(&mut r.subject, old, new, count);
    rename_in_predicate(&mut r.predicate, old, new, count);
    rename_in_object(&mut r.object, old, new, count);
    if let Some(rf) = &mut r.reifier {
        if let Some(ReifierIdentifier::Iri { iri, .. }) = &mut rf.identifier {
            rename_iri_node(iri, old, new, count);
        }
    }
}

fn rename_iri_node(i: &mut IriNode, old: &NamedNode, new: &NamedNode, count: &mut usize) {
    if &i.value == old {
        i.value = new.clone();
        i.dirty = true;
        *count += 1;
    }
}

fn predicate_matches(p: &PredicateNode, target: &NamedNode) -> bool {
    match p {
        PredicateNode::Iri(i) => &i.value == target,
        PredicateNode::A(_) => target.as_ref() == rdf::TYPE,
    }
}

fn term_matches(o: &ObjectNode, target: &Term) -> bool {
    match (&o.term, target) {
        (TermNode::Iri(i), Term::NamedNode(n)) => &i.value == n,
        (TermNode::BlankNodeLabel(b), Term::BlankNode(bn)) => &b.value == bn,
        (TermNode::Literal(l), Term::Literal(lt)) => &l.value == lt,
        _ => false,
    }
}

fn object_node_from_term(t: &Term) -> ObjectNode {
    let term = match t {
        Term::NamedNode(n) => TermNode::Iri(IriNode {
            source: String::new(),
            value: n.clone(),
            dirty: true,
        }),
        Term::BlankNode(b) => TermNode::BlankNodeLabel(BlankNodeLabelNode {
            source: format!("_:{}", b.as_str()),
            value: b.clone(),
        }),
        Term::Literal(l) => {
            let value = l.clone();
            let quoted = format!("\"{}\"", escape_string_literal(value.value()));
            let kind = LiteralKind::String;
            let suffix = if let Some(lang) = value.language() {
                Some(LiteralSuffix::Lang {
                    leading_trivia: Vec::new(),
                    tag: SourceText::new(format!("@{lang}")),
                })
            } else if value.datatype() != xsd::STRING {
                Some(LiteralSuffix::Datatype {
                    leading_trivia: Vec::new(),
                    caret: SourceText::new("^^"),
                    leading_iri_trivia: Vec::new(),
                    iri: IriNode {
                        source: String::new(),
                        value: value.datatype().into_owned(),
                        dirty: true,
                    },
                })
            } else {
                None
            };
            TermNode::Literal(LiteralNode {
                quoted: SourceText::new(quoted),
                kind,
                suffix,
                value,
                dirty: true,
            })
        }
        #[cfg(feature = "rdf-12")]
        Term::Triple(_) => panic!("Triple terms are not supported in synthesized ObjectNodes yet"),
    };
    ObjectNode {
        term,
        reifier: None,
        annotation: None,
    }
}

fn escape_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
#[allow(clippy::panic, clippy::tests_outside_test_module)]
mod tests {
    use super::*;

    fn roundtrip(input: &str) {
        let cst = TurtleCstParser::new()
            .parse_slice(input.as_bytes())
            .unwrap_or_else(|e| panic!("parse failed: {e}\ninput:\n{input}"));
        let out = cst.to_string();
        assert_eq!(out, input, "round-trip mismatch");
    }

    #[test]
    fn rt_empty() {
        roundtrip("");
    }

    #[test]
    fn rt_single_statement() {
        roundtrip("<http://a/x> <http://a/p> <http://a/y> .\n");
    }

    #[test]
    fn rt_prefix_then_statement() {
        roundtrip("@prefix ex: <http://example.com/> .\nex:Foo a ex:Class .\n");
    }

    #[test]
    fn rt_comment_after_dot() {
        roundtrip("@prefix ex: <http://x/> .\nex:a a ex:b . # trailing\n");
    }

    #[test]
    fn rt_freestanding_comment() {
        roundtrip("@prefix ex: <http://x/> .\n\n# header\n\nex:a a ex:b .\n");
    }

    #[test]
    fn rt_predicate_object_list() {
        roundtrip("@prefix ex: <http://x/> .\nex:a a ex:b ;\n    ex:p \"v\" ;\n    ex:q 42 .\n");
    }

    #[test]
    fn rt_object_list_with_comma() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p ex:b , ex:c , ex:d .\n");
    }

    #[test]
    fn rt_long_string() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p \"\"\"line1\nline2\"\"\" .\n");
    }

    #[test]
    fn rt_lang_literal() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p \"hi\"@en .\n");
    }

    #[test]
    fn rt_typed_literal() {
        roundtrip(
            "@prefix ex: <http://x/> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\nex:a ex:p \"42\"^^xsd:integer .\n",
        );
    }

    #[test]
    fn rt_bnpl() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p [ ex:q ex:r ] .\n");
    }

    #[test]
    fn rt_collection() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p ( ex:b ex:c ex:d ) .\n");
    }

    #[test]
    fn rt_anon_blank_node() {
        roundtrip("@prefix ex: <http://x/> .\nex:a ex:p [] .\n");
    }

    #[test]
    fn rt_full_iri_kept_full() {
        // Even though `ex:` would match, the original full IRI should be preserved.
        roundtrip("@prefix ex: <http://x/> .\n<http://x/Foo> a <http://x/Class> .\n");
    }

    #[test]
    fn rt_crlf_line_endings() {
        roundtrip("@prefix ex: <http://x/> .\r\nex:a a ex:b .\r\n");
    }

    #[test]
    fn rt_only_comments() {
        roundtrip("# just a comment\n# another\n");
    }

    #[test]
    fn rename_iri_basic() {
        let input = "@prefix ex: <http://x/> .\nex:Foo a ex:Class ;\n    ex:p ex:Foo .\nex:Bar ex:q ex:Foo .\n";
        let mut cst = TurtleCstParser::new()
            .parse_slice(input.as_bytes())
            .unwrap();
        let old = NamedNode::new_unchecked("http://x/Foo");
        let new = NamedNode::new_unchecked("http://x/Renamed");
        let n = cst.rename_iri(&old, &new);
        assert_eq!(n, 3);
        let out = cst.to_string();
        assert!(!out.contains("ex:Foo"));
        assert!(out.contains("ex:Renamed"));
    }

    #[test]
    fn replace_object_basic() {
        let input = "@prefix ex: <http://x/> .\nex:a ex:p ex:b ;\n    ex:p ex:c .\n";
        let mut cst = TurtleCstParser::new()
            .parse_slice(input.as_bytes())
            .unwrap();
        let pred = NamedNode::new_unchecked("http://x/p");
        let old: Term = NamedNode::new_unchecked("http://x/b").into();
        let new: Term = NamedNode::new_unchecked("http://x/B2").into();
        let subject = NamedNode::new_unchecked("http://x/a");
        let stmts: Vec<&mut Statement> = cst.statements_for_subject(&subject).collect();
        for s in stmts {
            assert!(s.replace_object(&pred, &old, new.clone()));
        }
        let out = cst.to_string();
        assert!(out.contains("ex:B2"));
    }

    #[test]
    fn remove_statement_basic() {
        let input = "@prefix ex: <http://x/> .\nex:a a ex:Foo .\nex:b a ex:Bar .\n";
        let mut cst = TurtleCstParser::new()
            .parse_slice(input.as_bytes())
            .unwrap();
        let n = cst.remove_statements_for_subject(&NamedNode::new_unchecked("http://x/a"));
        assert_eq!(n, 1);
        let out = cst.to_string();
        assert!(!out.contains("ex:a "));
        assert!(out.contains("ex:b a ex:Bar"));
    }

    #[test]
    fn add_predicate_object_basic() {
        let input = "@prefix ex: <http://x/> .\nex:a a ex:Foo .\n";
        let mut cst = TurtleCstParser::new()
            .parse_slice(input.as_bytes())
            .unwrap();
        let pred = NamedNode::new_unchecked("http://x/label");
        let obj: Term = Literal::new_simple_literal("hi").into();
        let subject = NamedNode::new_unchecked("http://x/a");
        let stmts: Vec<&mut Statement> = cst.statements_for_subject(&subject).collect();
        for s in stmts {
            s.add_predicate_object(pred.clone(), &obj);
        }
        let out = cst.to_string();
        assert!(out.contains("ex:label"));
        assert!(out.contains("\"hi\""));
    }
}
