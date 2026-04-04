#include "eval.h"

#include <cmath>

namespace eval {

namespace {

const double* as_number(const ast::Expr& e) {
    if (auto* n = std::get_if<ast::Number>(&e))
        return &n->value;
    return nullptr;
}

ast::Expr eval_binary(ast::BinOp op, ast::Expr left, ast::Expr right) {
    const double* lv = as_number(left);
    const double* rv = as_number(right);

    if (lv && rv) {
        switch (op) {
        case ast::BinOp::Add: return ast::Number{*lv + *rv};
        case ast::BinOp::Sub: return ast::Number{*lv - *rv};
        case ast::BinOp::Mul: return ast::Number{*lv * *rv};
        case ast::BinOp::Div: return ast::Number{*lv / *rv};
        }
    }

    switch (op) {
    case ast::BinOp::Add:
        if (rv && *rv == 0) return left;
        if (lv && *lv == 0) return right;
        break;
    case ast::BinOp::Sub:
        if (rv && *rv == 0) return left;
        break;
    case ast::BinOp::Mul:
        if (rv && *rv == 1) return left;
        if (lv && *lv == 1) return right;
        if ((rv && *rv == 0) || (lv && *lv == 0))
            return ast::Number{0};
        break;
    case ast::BinOp::Div:
        if (rv && *rv == 1) return left;
        break;
    }

    return std::make_unique<ast::BinaryExpr>(
        ast::BinaryExpr{op, std::move(left), std::move(right)});
}

ast::Expr eval_call(const std::string& name,
                    std::vector<ast::Expr> args) {
    if (args.size() == 1) {
        if (const double* v = as_number(args[0])) {
            if (name == "sin")  return ast::Number{std::sin(*v)};
            if (name == "cos")  return ast::Number{std::cos(*v)};
            if (name == "tan")  return ast::Number{std::tan(*v)};
            if (name == "sqrt") return ast::Number{std::sqrt(*v)};
            if (name == "abs")  return ast::Number{std::abs(*v)};
            if (name == "exp")  return ast::Number{std::exp(*v)};
            if (name == "log")  return ast::Number{std::log(*v)};
        }
    }
    return std::make_unique<ast::CallExpr>(
        ast::CallExpr{name, std::move(args)});
}

} // namespace

ast::Expr evaluate(const ast::Expr& e, const Environment& env) {
    return std::visit(
        [&](const auto& v) -> ast::Expr {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ast::Number>) {
                return v;
            } else if constexpr (std::is_same_v<T, ast::Variable>) {
                auto it = env.find(v.name);
                if (it != env.end())
                    return evaluate(it->second, env);
                return v;
            } else if constexpr (std::is_same_v<T,
                                     std::unique_ptr<ast::BinaryExpr>>) {
                auto left = evaluate(v->left, env);
                auto right = evaluate(v->right, env);
                return eval_binary(v->op, std::move(left),
                                   std::move(right));
            } else if constexpr (std::is_same_v<T,
                                     std::unique_ptr<ast::UnaryExpr>>) {
                auto operand = evaluate(v->operand, env);
                if (const double* val = as_number(operand))
                    return ast::Number{-*val};
                return std::make_unique<ast::UnaryExpr>(
                    ast::UnaryExpr{std::move(operand)});
            } else if constexpr (std::is_same_v<T,
                                     std::unique_ptr<ast::CallExpr>>) {
                std::vector<ast::Expr> args;
                args.reserve(v->args.size());
                for (const auto& a : v->args)
                    args.push_back(evaluate(a, env));
                return eval_call(v->name, std::move(args));
            }
        },
        e);
}

} // namespace eval
