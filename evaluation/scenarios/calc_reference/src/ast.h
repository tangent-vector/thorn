#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace ast {

struct BinaryExpr;
struct UnaryExpr;
struct CallExpr;

struct Number {
    double value;
};

struct Variable {
    std::string name;
};

using Expr = std::variant<
    Number,
    Variable,
    std::unique_ptr<BinaryExpr>,
    std::unique_ptr<UnaryExpr>,
    std::unique_ptr<CallExpr>>;

enum class BinOp { Add, Sub, Mul, Div };

struct BinaryExpr {
    BinOp op;
    Expr left;
    Expr right;
};

struct UnaryExpr {
    Expr operand;
};

struct CallExpr {
    std::string name;
    std::vector<Expr> args;
};

Expr clone(const Expr& e);
std::string format(const Expr& e);

} // namespace ast
