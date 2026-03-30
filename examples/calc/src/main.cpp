// ============================================================================
// calc — a simple command-line calculator
// ============================================================================
//
// Purpose:
//   A command-line calculator that reads arithmetic expressions from stdin,
//   parses them into expression trees, evaluates them, and prints results.
//
// Responsibilities:
//   - Read expression lines from stdin
//   - Parse each line into an expression tree using the parser module
//   - Evaluate the expression tree and print the result
//   - Handle basic error reporting for invalid expressions
//   - Support basic arithmetic operators: +, -, *, /
//   - Support parentheses for grouping
//   - Support integer and decimal numbers
//
// Dependencies:
//   - parser: tokenizing and parsing input into expression trees
//   - expression: expression tree data structure and evaluation
//
// Requirements/Constraints:
//   - C++20 standard
//   - No third-party dependencies (standard library only)
//

#include "parser.h"
#include "expression.h"
#include <iostream>
#include <string>

int main()
{
    std::string line;

    while (true) {
        std::cout << "> " << std::flush;

        if (!std::getline(std::cin, line)) {
            // EOF reached
            break;
        }

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Skip comment lines (starting with #)
        if (line[0] == '#') {
            continue;
        }

        // Exit commands
        if (line == "quit" || line == "exit") {
            break;
        }

        try {
            auto expr = parser::parse(line);
            double result = expr->evaluate();
            std::cout << "= " << result << std::endl;
        } catch (const parser::ParseError& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
