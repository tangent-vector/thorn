#pragma once

#include "ast.h"

#include <string>
#include <string_view>
#include <variant>

namespace parser {

struct ParseError {
    std::string message;
};

struct Assignment {
    std::string name;
    ast::Expr value;
};

using ParseResult = std::variant<ast::Expr, Assignment, ParseError>;

ParseResult parse_line(std::string_view input);

} // namespace parser
