#pragma once

#include <string_view>

namespace lexer {

enum class TokenKind {
    Number,
    Identifier,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Equals,
    Comma,
    End,
    Error,
};

struct Token {
    TokenKind kind;
    std::string_view text;
    double number_value = 0;
};

class Lexer {
public:
    explicit Lexer(std::string_view src);

    Token next();
    Token peek();

private:
    std::string_view src_;
    size_t pos_;

    void skip_whitespace();
    Token single(TokenKind k);
    Token lex_number();
    Token lex_identifier();
};

} // namespace lexer
