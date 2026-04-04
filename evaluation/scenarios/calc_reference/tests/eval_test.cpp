#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ast.h"
#include "eval.h"
#include "parser.h"

#include <cmath>
#include <string>

namespace {

std::string eval_str(std::string_view input, eval::Environment& env) {
    auto parsed = parser::parse_line(input);
    auto& expr = std::get<ast::Expr>(parsed);
    auto value = eval::evaluate(expr, env);
    return ast::format(value);
}

std::string eval_str(std::string_view input) {
    eval::Environment env;
    return eval_str(input, env);
}

} // namespace

TEST_CASE("basic arithmetic") {
    CHECK(eval_str("2 + 3") == "5");
    CHECK(eval_str("10 - 4") == "6");
    CHECK(eval_str("3 * 7") == "21");
    CHECK(eval_str("10 / 4") == "2.5");
}

TEST_CASE("operator precedence") {
    CHECK(eval_str("2 + 3 * 4") == "14");
    CHECK(eval_str("(2 + 3) * 4") == "20");
}

TEST_CASE("variable substitution") {
    eval::Environment env;
    auto parsed = parser::parse_line("x = 7");
    auto& assign = std::get<parser::Assignment>(parsed);
    env[assign.name] = eval::evaluate(assign.value, env);

    CHECK(eval_str("x * 3", env) == "21");
    CHECK(eval_str("x + x", env) == "14");
}

TEST_CASE("built-in functions") {
    CHECK(eval_str("sin(0)") == "0");
    CHECK(eval_str("cos(0)") == "1");
    CHECK(eval_str("sqrt(16)") == "4");
    double sqrt2 = std::stod(eval_str("sqrt(2)"));
    CHECK(sqrt2 == doctest::Approx(std::sqrt(2.0)).epsilon(1e-10));
}

TEST_CASE("unary negation") {
    CHECK(eval_str("-5") == "-5");
    CHECK(eval_str("-2 + 3") == "1");
    CHECK(eval_str("-(2 + 3)") == "-5");
}

TEST_CASE("algebraic simplifications") {
    eval::Environment env;
    CHECK(eval_str("x + 0", env) == "x");
    CHECK(eval_str("0 + x", env) == "x");
    CHECK(eval_str("x - 0", env) == "x");
    CHECK(eval_str("x * 1", env) == "x");
    CHECK(eval_str("1 * x", env) == "x");
    CHECK(eval_str("x * 0", env) == "0");
    CHECK(eval_str("x / 1", env) == "x");
}

TEST_CASE("nested function calls") {
    CHECK(eval_str("sqrt(sqrt(16))") == "2");
}
