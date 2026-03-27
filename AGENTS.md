- Make sure any code you add builds, has no linter issues, and passes all tests.

- If you see build/lint/test failures that don't seem related to your own work, you are still responsible for addressing them or (if you cannot fix them) bringing them to the attention of the user.
  Do NOT shrug off any kind of issues as not your problem.

- Make sure to add tests for code you introduce, whenever possible.
  Ensure that you are only adding tests for the functionality a user of your code/API would actually care about; we don't need to pad out test counts with fluff.

- Flat code is better than deeply nested code.
  Handle-early-out cases in functions and loops first, so that the main or most complicated path can remain less nested.

- Comments should be used to explain *why* you are doing something in a particular way, or using a particular design/architecture approach. They should discuss alternatives considered, where appropriate.

  Comments that just state *what* the code is doing are a code smell.
  If code is complicated enough that you need a comment to explain what it's doing, then you should be defining cleanly named helper routines, temporaries, or whatever it takes to make the code more obvious to somebody reading the code itself.
  