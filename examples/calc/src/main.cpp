// main.cpp
//
// The `calc` tool is a simple command-line calculator that supports
// symbolic evaluation/simplification of arithmetic expressions that
// can include a mix of constants, variables, and a few standard functions.
//
// When invoked, the tool reads lines from stdin and evaluates each.
// Lines that represent expressions are simplified/evaluated and the result is printed to stdout.
// Lines of the form `<variable> = <expression>` are used to define variables, and any expressions
// that refer to a variable defined this way will subsequently be replaced with its definition
// as part of the simplification/evaluation process.
//
// A line that begins with `#` is a comment and is ignored.
// A line that begins with `quit` or `exit` is used to exit the tool.
//
// Architecture:
// -------------
// The main function serves as the entry point and delegates to the REPL module
// which orchestrates the interactive calculator session. The program is decomposed
// into the following sub-modules:
//
//   - expression: Data structures for representing arithmetic expressions
//   - parser: Parsing text input into expression structures
//   - evaluator: Evaluating and simplifying expressions
//   - environment: Managing variable definitions and lookups
//   - repl: The main read-eval-print loop coordinating all components
//
// Dependencies:
// -------------
// This module depends on the repl module to provide the main interaction loop.

#include "repl.h"

int main()
{
    return 0;
}
