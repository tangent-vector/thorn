#include "parser.h"
#include "lexer.h"

namespace parser {

namespace {

using ExprOrError = std::variant<ast::Expr, ParseError>;

class Parser {
public:
    explicit Parser(std::string_view src) : lex_(src) { advance(); }

    ParseResult parse_line() {
        if (cur_.kind == lexer::TokenKind::Identifier) {
            auto name_tok = cur_;
            auto after = lex_.peek();
            if (after.kind == lexer::TokenKind::Equals) {
                advance();
                advance();
                auto expr = parse_expr();
                if (auto* err = std::get_if<ParseError>(&expr))
                    return *err;
                if (cur_.kind != lexer::TokenKind::End)
                    return ParseError{"unexpected token after expression"};
                return Assignment{
                    std::string(name_tok.text),
                    std::move(std::get<ast::Expr>(expr))};
            }
        }

        auto expr = parse_expr();
        if (auto* err = std::get_if<ParseError>(&expr))
            return *err;
        if (cur_.kind != lexer::TokenKind::End)
            return ParseError{"unexpected token after expression"};
        return std::move(std::get<ast::Expr>(expr));
    }

private:
    lexer::Lexer lex_;
    lexer::Token cur_;

    void advance() { cur_ = lex_.next(); }

    ExprOrError parse_expr() { return parse_additive(); }

    ExprOrError parse_additive() {
        auto left = parse_multiplicative();
        if (std::holds_alternative<ParseError>(left))
            return left;

        while (cur_.kind == lexer::TokenKind::Plus ||
               cur_.kind == lexer::TokenKind::Minus) {
            auto op = (cur_.kind == lexer::TokenKind::Plus)
                          ? ast::BinOp::Add
                          : ast::BinOp::Sub;
            advance();
            auto right = parse_multiplicative();
            if (std::holds_alternative<ParseError>(right))
                return right;
            ast::Expr node = std::make_unique<ast::BinaryExpr>(
                ast::BinaryExpr{op,
                                std::move(std::get<ast::Expr>(left)),
                                std::move(std::get<ast::Expr>(right))});
            left = ExprOrError{std::move(node)};
        }
        return left;
    }

    ExprOrError parse_multiplicative() {
        auto left = parse_unary();
        if (std::holds_alternative<ParseError>(left))
            return left;

        while (cur_.kind == lexer::TokenKind::Star ||
               cur_.kind == lexer::TokenKind::Slash) {
            auto op = (cur_.kind == lexer::TokenKind::Star)
                          ? ast::BinOp::Mul
                          : ast::BinOp::Div;
            advance();
            auto right = parse_unary();
            if (std::holds_alternative<ParseError>(right))
                return right;
            ast::Expr node = std::make_unique<ast::BinaryExpr>(
                ast::BinaryExpr{op,
                                std::move(std::get<ast::Expr>(left)),
                                std::move(std::get<ast::Expr>(right))});
            left = ExprOrError{std::move(node)};
        }
        return left;
    }

    ExprOrError parse_unary() {
        if (cur_.kind == lexer::TokenKind::Minus) {
            advance();
            auto operand = parse_unary();
            if (std::holds_alternative<ParseError>(operand))
                return operand;
            ast::Expr node = std::make_unique<ast::UnaryExpr>(
                ast::UnaryExpr{std::move(std::get<ast::Expr>(operand))});
            return ExprOrError{std::move(node)};
        }
        if (cur_.kind == lexer::TokenKind::Plus) {
            advance();
            return parse_unary();
        }
        return parse_primary();
    }

    ExprOrError parse_primary() {
        if (cur_.kind == lexer::TokenKind::Number) {
            ast::Expr e = ast::Number{cur_.number_value};
            advance();
            return ExprOrError{std::move(e)};
        }

        if (cur_.kind == lexer::TokenKind::Identifier) {
            std::string name(cur_.text);
            advance();
            if (cur_.kind == lexer::TokenKind::LParen) {
                advance();
                std::vector<ast::Expr> args;
                if (cur_.kind != lexer::TokenKind::RParen) {
                    auto arg = parse_expr();
                    if (std::holds_alternative<ParseError>(arg))
                        return arg;
                    args.push_back(std::move(std::get<ast::Expr>(arg)));
                    while (cur_.kind == lexer::TokenKind::Comma) {
                        advance();
                        arg = parse_expr();
                        if (std::holds_alternative<ParseError>(arg))
                            return arg;
                        args.push_back(
                            std::move(std::get<ast::Expr>(arg)));
                    }
                }
                if (cur_.kind != lexer::TokenKind::RParen)
                    return ParseError{"expected ')'"};
                advance();
                ast::Expr call = std::make_unique<ast::CallExpr>(
                    ast::CallExpr{name, std::move(args)});
                return ExprOrError{std::move(call)};
            }
            ast::Expr var = ast::Variable{name};
            return ExprOrError{std::move(var)};
        }

        if (cur_.kind == lexer::TokenKind::LParen) {
            advance();
            auto inner = parse_expr();
            if (std::holds_alternative<ParseError>(inner))
                return inner;
            if (cur_.kind != lexer::TokenKind::RParen)
                return ParseError{"expected ')'"};
            advance();
            return inner;
        }

        if (cur_.kind == lexer::TokenKind::End)
            return ParseError{"unexpected end of input"};
        return ParseError{
            "unexpected token: '" + std::string(cur_.text) + "'"};
    }
};

} // namespace

ParseResult parse_line(std::string_view input) {
    Parser p(input);
    return p.parse_line();
}

} // namespace parser
