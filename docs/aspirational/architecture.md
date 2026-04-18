Thorn System Vision
===================

This document describes the architecture and design of the Thorn system as we imagine it should look.
The descriptions here may not match what is actually implemented at present, both in details and in the broad strokes.
The goal of this document is to help provide a "point on the horizon" that can guide incremental implementation choices toward the eventual goal.

Concepts
--------

### Agencies

An **agency** is a collection of agents, their sessions, and supporting services.
An agency can be persisted to disk and version-controlled as a single repository, and it can be loaded into memory by a runtime gateway/daemon process so that its agents can take action, services can send/receive events, etc.

An agency's durable state is peristed as a directory containing files and subdirectories for the state of its agents, services, etc.
We often refer to that directory as the **agency root** directory.

We can broadly distinguish two main cases of agencies that get created:

- **Local** agencies are typically created for use by a single user. By convention, these will often be persisted as `~username/.thorn/`

- **Server**-based agencies are typically hosted on their own machine (or inside a container/VM, etc.), in which case they might have any directory name/path (TODO: define a conventional path to have be the default).

Throughout this document we will make note of important differences in how local vs. server agencies are typically configured or run.

### Agents

An **agent** is an AI individual, having its own memory, accounts on various services, permissions, etc.
Distinct agents should typically use distinct accounts on things like messaging services, just as distinct humans typically should.

A single agent can sustain multiple conversational sessions at once, and can be performing multiple tasks, of different kinds, at once.
Thus it is not necessary to define new agents to enable concurrency, or to perform different kinds of tasks.
The main reason to define a new agent rather than use an existing one should be because of differences in the permissions that are appropriate to each agent: permissions to perform actions and access services/information.

The state of an agent can be divided into a few key pieces:

- The static *configuration* of the agent: e.g., what tools it has access to, what user accounts it has on what services
- The persistent/durable state of the agent's *memory*.
- The various *sessions* of conversation/activity that the agent is (or has been) engaged in.
- The contents of a filesystem *workspace* where the agent authors and manipulates data.

The memory and sessions of an agent together define its durable state, and these are persisted as a subdirectory under a given agency.
If `helper` is an agent and the agency root is `.../.thorn`, then the state of the agent would be persisted as the directory `.../.thorn/agents/helper/`.
Under the agent's directory there are its *home* directory (`.../.thorn/agents/helper/home/`) and a directory for its sessions (`.../.thorn/agents/helper/sessions/`).

The workspace of an agent is more like the desk or computer of a human: a place for in-progress work that, while important, is not as vital to save as the agent's own memory.
The runtime system for an agency should strive not to throw away an agent's workspace, but at the same time agents are expected to be able to recover from loss of workspace data in much the same way a human would: if something you're working on is truly important, it should be backed up, in source control, etc.

### Sessions

A **session** is single logical "thread" of conversation and/or action that an agent is engaged in.
All of the sessions for a given agent share the same memory, but each session has a distinct sequence of user/assistant/system messages that are used when requesting completions from an LLM.

Each session also has a logical **inbox** of notifications

### Agencies

An **agency** represents