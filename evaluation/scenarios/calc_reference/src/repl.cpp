#include "repl.h"
#include "ast.h"
#include "eval.h"
#include "parser.h"

#include <iostream>
#include <string>

namespace repl {

void run(std::istream& in, std::ostream& out) {
    eval::Environment env;
    std::string line;

    while (std::getline(in, line)) {
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        std::string_view trimmed(line.data() + start, line.size() - start);

        if (trimmed.starts_with("#"))
            continue;
        if (trimmed.starts_with("quit") || trimmed.starts_with("exit"))
            break;

        auto result = parser::parse_line(trimmed);

        if (auto* err = std::get_if<parser::ParseError>(&result)) {
            std::cerr << "error: " << err->message << "\n";
            continue;
        }

        if (auto* assign = std::get_if<parser::Assignment>(&result)) {
            auto value = eval::evaluate(assign->value, env);
            out << assign->name << " = " << ast::format(value) << "\n";
            env[assign->name] = std::move(value);
            continue;
        }

        auto value = eval::evaluate(std::get<ast::Expr>(result), env);
        out << ast::format(value) << "\n";
    }
}

} // namespace repl
