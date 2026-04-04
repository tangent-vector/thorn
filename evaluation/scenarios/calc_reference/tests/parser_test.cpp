#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ast.h"
#include "parser.h"

TEST_CASE("parse number") {
    auto result = parser::parse_line("42");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "42");
}

TEST_CASE("parse addition") {
    auto result = parser::parse_line("2 + 3");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "2 + 3");
}

TEST_CASE("operator precedence preserved in AST") {
    auto result = parser::parse_line("2 + 3 * 4");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "2 + 3 * 4");
}

TEST_CASE("parenthesised subexpression") {
    auto result = parser::parse_line("(2 + 3) * 4");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "(2 + 3) * 4");
}

TEST_CASE("parse assignment") {
    auto result = parser::parse_line("x = 7");
    REQUIRE(std::holds_alternative<parser::Assignment>(result));
    auto& assign = std::get<parser::Assignment>(result);
    CHECK(assign.name == "x");
    CHECK(ast::format(assign.value) == "7");
}

TEST_CASE("parse function call") {
    auto result = parser::parse_line("sin(0)");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "sin(0)");
}

TEST_CASE("parse multi-arg function call") {
    auto result = parser::parse_line("f(1, 2, 3)");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
    CHECK(ast::format(std::get<ast::Expr>(result)) == "f(1, 2, 3)");
}

TEST_CASE("parse unary negation") {
    auto result = parser::parse_line("-5");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
}

TEST_CASE("parse nested unary and binary") {
    auto result = parser::parse_line("-2 + 3");
    REQUIRE(std::holds_alternative<ast::Expr>(result));
}

TEST_CASE("parse error on empty input") {
    auto result = parser::parse_line("");
    CHECK(std::holds_alternative<parser::ParseError>(result));
}

TEST_CASE("parse error on unmatched paren") {
    auto result = parser::parse_line("(2 + 3");
    CHECK(std::holds_alternative<parser::ParseError>(result));
}
