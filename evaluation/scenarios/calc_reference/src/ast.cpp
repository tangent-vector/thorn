#include "ast.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace ast {

Expr clone(const Expr& e) {
    return std::visit(
        [](const auto& v) -> Expr {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Number>) {
                return v;
            } else if constexpr (std::is_same_v<T, Variable>) {
                return v;
            } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryExpr>>) {
                return std::make_unique<BinaryExpr>(
                    BinaryExpr{v->op, clone(v->left), clone(v->right)});
            } else if constexpr (std::is_same_v<T, std::unique_ptr<UnaryExpr>>) {
                return std::make_unique<UnaryExpr>(
                    UnaryExpr{clone(v->operand)});
            } else if constexpr (std::is_same_v<T, std::unique_ptr<CallExpr>>) {
                std::vector<Expr> args;
                args.reserve(v->args.size());
                for (const auto& a : v->args)
                    args.push_back(clone(a));
                return std::make_unique<CallExpr>(
                    CallExpr{v->name, std::move(args)});
            }
        },
        e);
}

namespace {

std::string format_number(double value) {
    if (std::floor(value) == value && std::abs(value) < 1e15) {
        return std::to_string(static_cast<long long>(value));
    }
    std::ostringstream oss;
    oss << std::setprecision(15) << value;
    return oss.str();
}

int precedence(BinOp op) {
    switch (op) {
    case BinOp::Add:
    case BinOp::Sub: return 1;
    case BinOp::Mul:
    case BinOp::Div: return 2;
    }
    return 0;
}

char op_char(BinOp op) {
    switch (op) {
    case BinOp::Add: return '+';
    case BinOp::Sub: return '-';
    case BinOp::Mul: return '*';
    case BinOp::Div: return '/';
    }
    return '?';
}

void format_impl(std::ostringstream& out, const Expr& e,
                 int parent_prec, bool is_rhs) {
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Number>) {
                out << format_number(v.value);
            } else if constexpr (std::is_same_v<T, Variable>) {
                out << v.name;
            } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryExpr>>) {
                int prec = precedence(v->op);
                // Parenthesise when our precedence is lower, or when we are
                // the right-hand child of an equally-precedent non-commutative
                // operator (subtraction, division).
                bool parens =
                    prec < parent_prec ||
                    (prec == parent_prec && is_rhs &&
                     (v->op == BinOp::Sub || v->op == BinOp::Div));
                if (parens)
                    out << '(';
                format_impl(out, v->left, prec, false);
                out << ' ' << op_char(v->op) << ' ';
                format_impl(out, v->right, prec, true);
                if (parens)
                    out << ')';
            } else if constexpr (std::is_same_v<T, std::unique_ptr<UnaryExpr>>) {
                out << "(-";
                format_impl(out, v->operand, 100, false);
                out << ')';
            } else if constexpr (std::is_same_v<T, std::unique_ptr<CallExpr>>) {
                out << v->name << '(';
                for (size_t i = 0; i < v->args.size(); ++i) {
                    if (i > 0)
                        out << ", ";
                    format_impl(out, v->args[i], 0, false);
                }
                out << ')';
            }
        },
        e);
}

} // namespace

std::string format(const Expr& e) {
    std::ostringstream out;
    format_impl(out, e, 0, false);
    return out.str();
}

} // namespace ast
