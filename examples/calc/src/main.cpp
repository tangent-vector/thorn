// ============================================================================
// calc — a simple command-line calculator
// ============================================================================
//
// Purpose:
//   A command-line calculator that supports symbolic evaluation and
//   simplification of arithmetic expressions involving constants,
//   variables, and standard mathematical functions.
//
// Responsibilities:
//   - Read lines from stdin and evaluate each as an expression
//   - Expressions are simplified/evaluated and the result is printed
//   - Lines of the form `<variable> = <expression>` define variables;
//     subsequent expressions referring to defined variables substitute
//     the stored definition during simplification/evaluation
//   - Lines beginning with `#` are comments and are ignored
//   - Lines beginning with `quit` or `exit` terminate the program
//   - Support standard arithmetic operators: +, -, *, /
//   - Support standard functions: sin, cos, sqrt (at minimum)
//
// Requirements/Constraints:
//   - C++20 standard
//   - No third-party dependencies (standard library only)
//   - Clean separation of concerns across modules
//

#include <iostream>
#include <string>

#include "parser.h"
#include "evaluator.h"
#include "environment.h"

int main()
{
    environment::Environment env;
    std::string line;

    while (std::getline(std::cin, line)) {
        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Skip comments (lines starting with '#')
        if (line[0] == '#') {
            continue;
        }

        // Check for quit/exit commands
        if (line == "quit" || line == "exit") {
            break;
        }

        try {
            // Parse the line (could be assignment or expression)
            parser::Parser p(line);
            auto [var_name, expr] = p.parse_assignment();

            // Evaluate the expression
            double result = evaluator::evaluate(expr, env);

            // If it's an assignment, store the result in the environment
            if (var_name.has_value()) {
                env.set(var_name.value(), result);
            }

            // Print the result
            std::cout << result << std::endl;

        } catch (const std::runtime_error& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
