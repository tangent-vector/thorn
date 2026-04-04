#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ast.h"

TEST_CASE("format integer numbers without decimal point") {
    CHECK(ast::format(ast::Number{5.0}) == "5");
    CHECK(ast::format(ast::Number{0.0}) == "0");
    CHECK(ast::format(ast::Number{-3.0}) == "-3");
    CHECK(ast::format(ast::Number{100.0}) == "100");
}

TEST_CASE("format fractional numbers") {
    CHECK(ast::format(ast::Number{2.5}) == "2.5");
    CHECK(ast::format(ast::Number{0.1}) == "0.1");
}

TEST_CASE("format variables") {
    CHECK(ast::format(ast::Variable{"x"}) == "x");
    CHECK(ast::format(ast::Variable{"foo"}) == "foo");
}

TEST_CASE("format binary expressions with correct precedence") {
    // 2 + 3  (no parens needed at top level)
    ast::Expr add = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Add, ast::Number{2}, ast::Number{3}});
    CHECK(ast::format(add) == "2 + 3");

    // 2 + 3 * 4  (mul binds tighter, no parens needed)
    ast::Expr mul = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Mul, ast::Number{3}, ast::Number{4}});
    ast::Expr expr = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Add, ast::Number{2}, std::move(mul)});
    CHECK(ast::format(expr) == "2 + 3 * 4");

    // (2 + 3) * 4  (add has lower precedence, needs parens)
    ast::Expr add2 = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Add, ast::Number{2}, ast::Number{3}});
    ast::Expr expr2 = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Mul, std::move(add2), ast::Number{4}});
    CHECK(ast::format(expr2) == "(2 + 3) * 4");
}

TEST_CASE("format right-associative subtraction") {
    // a - (b - c) should parenthesise the right child
    ast::Expr inner = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Sub, ast::Variable{"b"}, ast::Variable{"c"}});
    ast::Expr outer = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Sub, ast::Variable{"a"}, std::move(inner)});
    CHECK(ast::format(outer) == "a - (b - c)");
}

TEST_CASE("format function calls") {
    std::vector<ast::Expr> args;
    args.push_back(ast::Variable{"x"});
    ast::Expr call =
        std::make_unique<ast::CallExpr>(ast::CallExpr{"sin", std::move(args)});
    CHECK(ast::format(call) == "sin(x)");
}

TEST_CASE("format unary negation") {
    ast::Expr neg = std::make_unique<ast::UnaryExpr>(
        ast::UnaryExpr{ast::Number{5}});
    CHECK(ast::format(neg) == "(-5)");
}

TEST_CASE("clone produces structurally identical tree") {
    ast::Expr original = std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{ast::BinOp::Add, ast::Number{1}, ast::Variable{"x"}});
    ast::Expr copy = ast::clone(original);
    CHECK(ast::format(copy) == ast::format(original));
}
