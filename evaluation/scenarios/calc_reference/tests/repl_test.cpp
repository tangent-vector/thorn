#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "repl.h"

#include <sstream>

TEST_CASE("basic expression") {
    std::istringstream in("2 + 3\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "5\n");
}

TEST_CASE("division produces decimal") {
    std::istringstream in("10 / 4\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "2.5\n");
}

TEST_CASE("variable assignment and subsequent use") {
    std::istringstream in("x = 7\nx * 3\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "x = 7\n21\n");
}

TEST_CASE("built-in functions") {
    std::istringstream in("sin(0)\nsqrt(16)\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "0\n4\n");
}

TEST_CASE("comments are ignored") {
    std::istringstream in("# this is a comment\n2 + 2\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "4\n");
}

TEST_CASE("quit terminates the session") {
    std::istringstream in("1 + 1\nquit\n999\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "2\n");
}

TEST_CASE("exit terminates the session") {
    std::istringstream in("1 + 1\nexit\n999\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "2\n");
}

TEST_CASE("empty and whitespace-only lines are skipped") {
    std::istringstream in("\n   \n3 + 4\n");
    std::ostringstream out;
    repl::run(in, out);
    CHECK(out.str() == "7\n");
}

TEST_CASE("validation cases from evaluate_scenario.py") {
    SUBCASE("addition") {
        std::istringstream in("2 + 3\n");
        std::ostringstream out;
        repl::run(in, out);
        CHECK(out.str() == "5\n");
    }
    SUBCASE("division") {
        std::istringstream in("10 / 4\n");
        std::ostringstream out;
        repl::run(in, out);
        CHECK(out.str() == "2.5\n");
    }
    SUBCASE("variables") {
        std::istringstream in("x = 7\nx * 3\n");
        std::ostringstream out;
        repl::run(in, out);
        CHECK(out.str() == "x = 7\n21\n");
    }
    SUBCASE("sin") {
        std::istringstream in("sin(0)\n");
        std::ostringstream out;
        repl::run(in, out);
        CHECK(out.str() == "0\n");
    }
    SUBCASE("sqrt") {
        std::istringstream in("sqrt(16)\n");
        std::ostringstream out;
        repl::run(in, out);
        CHECK(out.str() == "4\n");
    }
}
