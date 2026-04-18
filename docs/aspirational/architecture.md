Thorn System Vision
===================

This document describes the architecture and design of the Thorn system as we imagine it should look.
The descriptions here may not match what is actually implemented at present, both in details and in the broad strokes.
The goal of this document is to help provide a "point on the horizon" that can guide incremental implementation choices toward the eventual goal.

Concepts
--------

### Sessions

A **session** is, first and foremost, a sequence of messages in a conversation between a user and AI, which can be used to query for completions from an LLM, thus driving tool calls, etc.

### Agencies

An **agency** represents