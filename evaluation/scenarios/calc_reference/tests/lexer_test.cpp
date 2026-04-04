#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "lexer.h"

using lexer::TokenKind;

TEST_CASE("tokenize arithmetic expression") {
    lexer::Lexer lex("2 + 3");
    auto t1 = lex.next();
    CHECK(t1.kind == TokenKind::Number);
    CHECK(t1.number_value == 2.0);

    CHECK(lex.next().kind == TokenKind::Plus);

    auto t3 = lex.next();
    CHECK(t3.kind == TokenKind::Number);
    CHECK(t3.number_value == 3.0);

    CHECK(lex.next().kind == TokenKind::End);
}

TEST_CASE("tokenize all operator types") {
    lexer::Lexer lex("+-*/()=,");
    CHECK(lex.next().kind == TokenKind::Plus);
    CHECK(lex.next().kind == TokenKind::Minus);
    CHECK(lex.next().kind == TokenKind::Star);
    CHECK(lex.next().kind == TokenKind::Slash);
    CHECK(lex.next().kind == TokenKind::LParen);
    CHECK(lex.next().kind == TokenKind::RParen);
    CHECK(lex.next().kind == TokenKind::Equals);
    CHECK(lex.next().kind == TokenKind::Comma);
    CHECK(lex.next().kind == TokenKind::End);
}

TEST_CASE("tokenize identifier and function call") {
    lexer::Lexer lex("sin(x)");
    auto t1 = lex.next();
    CHECK(t1.kind == TokenKind::Identifier);
    CHECK(t1.text == "sin");

    CHECK(lex.next().kind == TokenKind::LParen);

    auto t3 = lex.next();
    CHECK(t3.kind == TokenKind::Identifier);
    CHECK(t3.text == "x");

    CHECK(lex.next().kind == TokenKind::RParen);
    CHECK(lex.next().kind == TokenKind::End);
}

TEST_CASE("tokenize assignment") {
    lexer::Lexer lex("x = 7");
    auto t1 = lex.next();
    CHECK(t1.kind == TokenKind::Identifier);
    CHECK(t1.text == "x");
    CHECK(lex.next().kind == TokenKind::Equals);
    auto t3 = lex.next();
    CHECK(t3.kind == TokenKind::Number);
    CHECK(t3.number_value == 7.0);
}

TEST_CASE("tokenize decimal number") {
    lexer::Lexer lex("3.14");
    auto t = lex.next();
    CHECK(t.kind == TokenKind::Number);
    CHECK(t.number_value == doctest::Approx(3.14));
}

TEST_CASE("peek does not consume token") {
    lexer::Lexer lex("42");
    auto peeked = lex.peek();
    CHECK(peeked.kind == TokenKind::Number);
    auto consumed = lex.next();
    CHECK(consumed.kind == TokenKind::Number);
    CHECK(consumed.number_value == 42.0);
}

TEST_CASE("identifiers can contain underscores and digits") {
    lexer::Lexer lex("foo_bar2");
    auto t = lex.next();
    CHECK(t.kind == TokenKind::Identifier);
    CHECK(t.text == "foo_bar2");
}
