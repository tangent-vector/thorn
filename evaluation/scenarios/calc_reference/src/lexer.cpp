#include "lexer.h"

#include <cctype>
#include <charconv>

namespace lexer {

Lexer::Lexer(std::string_view src) : src_(src), pos_(0) {}

void Lexer::skip_whitespace() {
    while (pos_ < src_.size() &&
           (src_[pos_] == ' ' || src_[pos_] == '\t'))
        ++pos_;
}

Token Lexer::single(TokenKind k) {
    return {k, src_.substr(pos_++, 1)};
}

Token Lexer::lex_number() {
    size_t start = pos_;
    while (pos_ < src_.size() &&
           (std::isdigit(static_cast<unsigned char>(src_[pos_])) ||
            src_[pos_] == '.'))
        ++pos_;
    if (pos_ < src_.size() &&
        (src_[pos_] == 'e' || src_[pos_] == 'E')) {
        ++pos_;
        if (pos_ < src_.size() &&
            (src_[pos_] == '+' || src_[pos_] == '-'))
            ++pos_;
        while (pos_ < src_.size() &&
               std::isdigit(static_cast<unsigned char>(src_[pos_])))
            ++pos_;
    }
    auto text = src_.substr(start, pos_ - start);
    double val = 0;
    auto [ptr, ec] =
        std::from_chars(text.data(), text.data() + text.size(), val);
    if (ec != std::errc{})
        return {TokenKind::Error, text};
    return {TokenKind::Number, text, val};
}

Token Lexer::lex_identifier() {
    size_t start = pos_;
    while (pos_ < src_.size() &&
           (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
            src_[pos_] == '_'))
        ++pos_;
    return {TokenKind::Identifier, src_.substr(start, pos_ - start)};
}

Token Lexer::next() {
    skip_whitespace();
    if (pos_ >= src_.size())
        return {TokenKind::End, {}};

    char c = src_[pos_];
    switch (c) {
    case '+': return single(TokenKind::Plus);
    case '-': return single(TokenKind::Minus);
    case '*': return single(TokenKind::Star);
    case '/': return single(TokenKind::Slash);
    case '(': return single(TokenKind::LParen);
    case ')': return single(TokenKind::RParen);
    case '=': return single(TokenKind::Equals);
    case ',': return single(TokenKind::Comma);
    default: break;
    }

    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
        return lex_number();
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
        return lex_identifier();

    return {TokenKind::Error, src_.substr(pos_++, 1)};
}

Token Lexer::peek() {
    size_t saved = pos_;
    Token t = next();
    pos_ = saved;
    return t;
}

} // namespace lexer
