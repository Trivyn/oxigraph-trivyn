//! Integration tests for [`oxttl::TurtleCstParser`].
//!
//! These tests exercise the trivia-preserving parse/mutate/serialize loop on a
//! realistic, comment-heavy ontology and check that:
//!   1. Round-trip is byte-exact for unmodified parses.
//!   2. Each editor-style mutation primitive produces the expected output.
//!   3. Mutations preserve trivia in the regions of the document they don't touch.

#![allow(
    clippy::allow_attributes,
    clippy::tests_outside_test_module,
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used
)]

use oxrdf::{Literal, NamedNode, Term};
use oxttl::turtle_cst::Statement;
use oxttl::{TurtleCstParser, TurtleParser};

const ONT: &str = include_str!("turtle_cst_fixtures/commented_ontology.ttl");

fn parse(input: &str) -> oxttl::TurtleCst {
    TurtleCstParser::new()
        .parse_slice(input.as_bytes())
        .unwrap_or_else(|e| panic!("parse failed: {e}"))
}

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n")
}

#[test]
fn round_trip_byte_exact() {
    let cst = parse(ONT);
    let out = cst.to_string();
    assert_eq!(out, ONT, "round-trip mismatch on commented_ontology.ttl");
}

#[test]
fn rename_class_preserves_comments_and_layout() {
    let mut cst = parse(ONT);
    let old = NamedNode::new_unchecked("http://example.com/ont#Mammal");
    let new = NamedNode::new_unchecked("http://example.com/ont#Tetrapod");
    let n = cst.rename_iri(&old, &new);
    // Mammal appears: as subject (1), as object of rdfs:subClassOf in Dog (1) — total 2.
    assert_eq!(n, 2, "expected to rename 2 occurrences of ex:Mammal");
    let out = normalize_line_endings(&cst.to_string());
    assert!(!out.contains("ex:Mammal"));
    assert!(out.contains("ex:Tetrapod"));
    // Surrounding comments are intact.
    assert!(out.contains("# Top-level classes in the example ontology."));
    assert!(out.contains("# Mammals are a subclass of Animal."));
    assert!(out.contains("# Dogs are mammals."));
    // Other class definitions are untouched.
    assert!(out.contains("ex:Animal a owl:Class ;\n    rdfs:label \"Animal\""));
}

#[test]
fn swap_parent_class() {
    // Replace the parent class of Dog: Mammal -> Animal.
    let mut cst = parse(ONT);
    let dog = NamedNode::new_unchecked("http://example.com/ont#Dog");
    let sub_class_of = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    let mammal: Term = NamedNode::new_unchecked("http://example.com/ont#Mammal").into();
    let animal: Term = NamedNode::new_unchecked("http://example.com/ont#Animal").into();
    let stmts: Vec<&mut Statement> = cst.statements_for_subject(&dog).collect();
    assert_eq!(stmts.len(), 1);
    for s in stmts {
        assert!(s.replace_object(&sub_class_of, &mammal, animal.clone()));
    }
    let out = normalize_line_endings(&cst.to_string());
    assert!(out.contains("ex:Dog a owl:Class ;\n    rdfs:subClassOf ex:Animal ;"));
    // Sibling class definitions and comments are unchanged.
    assert!(out.contains("# Mammals are a subclass of Animal."));
    assert!(out.contains("ex:Mammal a owl:Class ;\n    rdfs:subClassOf ex:Animal ;"));
}

#[test]
fn add_label_preserves_layout() {
    let mut cst = parse(ONT);
    let dog = NamedNode::new_unchecked("http://example.com/ont#Dog");
    let alt_label = NamedNode::new_unchecked("http://example.com/ont#altLabel");
    let lit: Term = Literal::new_simple_literal("Canis familiaris").into();
    let stmts: Vec<&mut Statement> = cst.statements_for_subject(&dog).collect();
    assert_eq!(stmts.len(), 1);
    for s in stmts {
        s.add_predicate_object(alt_label.clone(), &lit);
    }
    let out = cst.to_string();
    assert!(out.contains("ex:altLabel"));
    assert!(out.contains("\"Canis familiaris\""));
    // Existing predicates on Dog still present.
    assert!(out.contains("rdfs:subClassOf ex:Mammal"));
    assert!(out.contains("rdfs:label \"Dog\""));
}

#[test]
fn add_parent_class() {
    let mut cst = parse(ONT);
    let dog = NamedNode::new_unchecked("http://example.com/ont#Dog");
    let sub_class_of = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    let extra_parent: Term =
        NamedNode::new_unchecked("http://example.com/ont#DomesticAnimal").into();
    let stmts: Vec<&mut Statement> = cst.statements_for_subject(&dog).collect();
    for s in stmts {
        s.add_predicate_object(sub_class_of.clone(), &extra_parent);
    }
    let out = cst.to_string();
    assert!(out.contains("ex:DomesticAnimal"));
    // Existing subClassOf line is intact.
    assert!(out.contains("rdfs:subClassOf ex:Mammal"));
}

#[test]
fn remove_parent_class() {
    let mut cst = parse(ONT);
    let dog = NamedNode::new_unchecked("http://example.com/ont#Dog");
    let sub_class_of = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    let mammal: Term = NamedNode::new_unchecked("http://example.com/ont#Mammal").into();
    let stmts: Vec<&mut Statement> = cst.statements_for_subject(&dog).collect();
    for s in stmts {
        assert!(s.remove_predicate_object(&sub_class_of, &mammal));
    }
    let out = normalize_line_endings(&cst.to_string());
    // Dog no longer subclasses Mammal.
    let dog_section = out
        .split("ex:Dog a owl:Class")
        .nth(1)
        .expect("Dog section present");
    assert!(!dog_section.contains("rdfs:subClassOf"));
    // Other classes still subclass Mammal/Animal.
    assert!(out.contains("ex:Mammal a owl:Class ;\n    rdfs:subClassOf ex:Animal"));
}

#[test]
fn add_class_appends_statement() {
    let mut cst = parse(ONT);
    let cat_iri = NamedNode::new_unchecked("http://example.com/ont#Cat");
    let rdf_type = NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let owl_class: Term = NamedNode::new_unchecked("http://www.w3.org/2002/07/owl#Class").into();
    let sub_class_of = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    let mammal: Term = NamedNode::new_unchecked("http://example.com/ont#Mammal").into();
    let label_iri = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#label");
    let label_obj: Term = Literal::new_simple_literal("Cat").into();

    let s = cst.add_statement(cat_iri.clone().into());
    s.add_predicate_object(rdf_type, &owl_class);
    s.add_predicate_object(sub_class_of, &mammal);
    s.add_predicate_object(label_iri, &label_obj);

    let out = cst.to_string();
    // The pre-existing classes and comments are untouched.
    assert!(out.contains("ex:Animal a owl:Class"));
    assert!(out.contains("ex:Mammal a owl:Class"));
    assert!(out.contains("ex:Dog a owl:Class"));
    assert!(out.contains("# Dogs are mammals."));
    // The new class appears at the end.
    assert!(out.contains("ex:Cat"));
    assert!(out.contains("\"Cat\""));
}

#[test]
fn remove_class_removes_subject_section_only() {
    let mut cst = parse(ONT);
    let dog = NamedNode::new_unchecked("http://example.com/ont#Dog");
    let n = cst.remove_statements_for_subject(&dog);
    assert_eq!(n, 1);
    let out = normalize_line_endings(&cst.to_string());
    // Dog's own section is gone.
    assert!(!out.contains("ex:Dog a owl:Class"));
    assert!(!out.contains("rdfs:label \"Dog\""));
    // Other classes are untouched.
    assert!(out.contains("ex:Animal a owl:Class"));
    assert!(out.contains("ex:Mammal a owl:Class"));
    // Object-position references to Dog (none here, but check Mammal still subclasses Animal).
    assert!(out.contains("ex:Mammal a owl:Class ;\n    rdfs:subClassOf ex:Animal"));
}

/// Regression: `add_predicate_object` on a statement whose `pog` is empty
/// must produce a leading POG with no `;` separator. Previously emitted
/// `subject ; predicate object .` (invalid Turtle) because the separator
/// was unconditionally `Some(";")`.
#[test]
fn add_predicate_object_to_empty_pog_omits_leading_separator() {
    let input = "@prefix ex: <http://example.com/> .\n\n\
                 ex:A a ex:Class .\n";
    let mut cst = TurtleCstParser::new()
        .parse_slice(input.as_bytes())
        .expect("parse");
    let subj = NamedNode::new_unchecked("http://example.com/A");
    let rdf_type = NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let class = Term::from(NamedNode::new_unchecked("http://example.com/Class"));
    let pred = NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    let parent = Term::from(NamedNode::new_unchecked("http://example.com/B"));

    for s in cst.statements_for_subject(&subj) {
        // Drain to empty, then add. New POG must become first with no `;`.
        s.remove_predicate_object(&rdf_type, &class);
        s.add_predicate_object(pred.clone(), &parent);
    }
    let out = cst.to_string();
    assert!(
        !out.contains("ex:A ;") && !out.contains("ex:A;"),
        "no stray ';' immediately after subject:\n{out}"
    );
    // And the result must re-parse cleanly via the regular Turtle parser.
    for r in TurtleParser::new().for_slice(out.as_bytes()) {
        r.unwrap_or_else(|e| panic!("output must re-parse: {e}\n--- out ---\n{out}"));
    }
}

/// Regression: `add_predicate_object` on a multi-line statement must inherit
/// the statement's existing indent (newline + leading spaces) so the new POG
/// lands on its own line. Previously the implementation cloned
/// `leading_predicate_trivia` from the previous POG, but the parser stashes
/// inter-POG whitespace in `leading_trivia` (between `;` and the next
/// predicate); `leading_predicate_trivia` is drained empty. The bug surfaced
/// as `vs:term_status "x" ;rdfs:comment "y"` with no whitespace after `;`.
#[test]
fn add_predicate_object_preserves_multi_line_indent() {
    let input = "@prefix ex: <http://example.com/> .\n\n\
                 ex:Foo\n      a ex:Class ;\n      ex:p1 \"x\" .\n";
    let mut cst = TurtleCstParser::new()
        .parse_slice(input.as_bytes())
        .expect("parse");
    let subj = NamedNode::new_unchecked("http://example.com/Foo");
    let new_pred = NamedNode::new_unchecked("http://example.com/p2");
    let new_obj = Term::from(Literal::new_simple_literal("y"));

    for s in cst.statements_for_subject(&subj) {
        s.add_predicate_object(new_pred.clone(), &new_obj);
    }
    let out = cst.to_string();
    // The new POG must land on its own line with the same 6-space indent.
    assert!(
        out.contains(";\n      ex:p2"),
        "added POG must inherit the statement's indent (`;\\n      ex:p2`):\n{out}"
    );
    // Previous POGs must still parse cleanly via the regular parser.
    for r in TurtleParser::new().for_slice(out.as_bytes()) {
        r.unwrap_or_else(|e| panic!("output must re-parse: {e}\n--- out ---\n{out}"));
    }
}

/// Regression: `remove_predicate_object` that drops the original first POG
/// must clear the leading `;` of the new first POG (which was previously a
/// non-first POG and therefore had `Some(";")`).
#[test]
fn remove_first_pog_clears_separator_on_promoted_pog() {
    let input = "@prefix ex: <http://example.com/> .\n\n\
                 ex:A a ex:Class ;\n    ex:p ex:o .\n";
    let mut cst = TurtleCstParser::new()
        .parse_slice(input.as_bytes())
        .expect("parse");
    let subj = NamedNode::new_unchecked("http://example.com/A");
    let rdf_type = NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let class = Term::from(NamedNode::new_unchecked("http://example.com/Class"));

    for s in cst.statements_for_subject(&subj) {
        // Drop `a ex:Class` (the first POG); `ex:p ex:o` now becomes first
        // and must lose its leading `;`.
        s.remove_predicate_object(&rdf_type, &class);
    }
    let out = cst.to_string();
    assert!(
        !out.contains("ex:A ;") && !out.contains("ex:A;"),
        "promoted POG must not carry leading ';':\n{out}"
    );
    for r in TurtleParser::new().for_slice(out.as_bytes()) {
        r.unwrap_or_else(|e| panic!("output must re-parse: {e}\n--- out ---\n{out}"));
    }
}
