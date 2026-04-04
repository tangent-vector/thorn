#pragma once

#include "ast.h"

#include <map>
#include <string>

namespace eval {

using Environment = std::map<std::string, ast::Expr>;

// Substitute variables from env into e, then constant-fold arithmetic
// and apply built-in functions (sin, cos, sqrt, …).
ast::Expr evaluate(const ast::Expr& e, const Environment& env);

} // namespace eval
