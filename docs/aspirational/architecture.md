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

The static state of an agent (the ports of itself it shouldn't be allowed to modify freely) are stored under the agent directory as a simple configuration file (`.../.thorn/agents/helper/agent.json`).

The workspace of an agent is more like the desk or computer of a human: a place for in-progress work that, while important, is not as vital to save as the agent's own memory.
The runtime system for an agency should strive not to throw away an agent's workspace, but at the same time agents are expected to be able to recover from loss of workspace data in much the same way a human would: if something you're working on is truly important, it should be backed up, in source control, etc.

### Sessions

A **session** is single logical "thread" of conversation and/or action that an agent is engaged in.
All of the sessions for a given agent share the same memory, but each session has a distinct sequence of messages that are used when requesting completions from an LLM.

Each session also has a logical **inbox** of **notifications**.
Notifications can represent incoming events from chat services, other agents, etc.
A session is active/runnable when it has unhandled notifications in its inbox; the system will schedule runnable sessions, prompting an LLM to read and handle the remaining notifications.

#### Session Keys

Every session of an agent has a unique **session key**.

When a notification is to be sent to an agent, an appropriate session key is determined and then the notification is placed in the session of that agent corresponding to the key (potentially creating such a session if it did not already exist).

At the simplest, session keys are just strings, but in practice they use a path-like notation that ends up being reflected in how sessiosn are actually stored.
For example for a session with a key like `a/b/c` on an agent `helper`, the state of the session would be stored at `.thorn/agents/helper/sessions/a/b/c/`.

#### Session Workspaces

Each session has a workspace within the overall workspace of the containing agent, with a relative path based on the session key.
For example, if the workspace of agent `helper` is at `/workspace/helper/` then the workspace for a session with key `a/b/c` would be `/workspace/helper/a/b/c/`.

Note that all sessions of an agent share the same home directory (that of the agent), but distinct sessions always have distinct session workspaces.

The workspace of a session is effectively just the "current working directory" for the purpose of paths used in tool calls made for that session.
Tool calls made by the session may still access/modify files outside of its specific workspace, unless other measures are taken to prevent such operations.

### Services

A **service** represents an messaging platform, server, etc. that is external to the agency.
An example of a service would be Telegram or a GitLab server.
An agent can have an **account** (including the relevant **credentials**) on one or more services.

An agency is explicitly configured with a list of services it should be aware of, and the static configuration of an agent includes its accounts/credentials on the various services it can receive events from, send messages/requests to, etc.

### Peers

A **peer** is a logical individual (typically a person, but not always) that agents in a given agency might have reason to communicate or interact with.

The configuration of an agency includes a list of known peers.
For each peer, the agency includes information on their account names/numbers on various services, allowing the agency and the agents in it to identify when two accounts (potentially on different services) represent the same individual.

### Routing of Events

When a running agency detects a potentially-interesting **event** coming from a service (e.g., by polling for notifications on GitHub or GitLab), it will route that event to the correct session inbox for handling.

#### Routing to Agents

Most events will clearly indicate the right agent to handle them.
For example, if an event shows a direct message was received for a Discord account, then the correct agent to handle it is the one associated with that account (if any).
When events aren't clearly directed to a single agent, the given service will have to use other criteria to determin which agent(s) should be notified.

#### Routing to Sessions

Once the runtime has determined that a given agent should potentially be notified of an event, it must decide which session on the agent should receive the notificatin in its inbox.

In order to guide routing, we think of each event as producing a combination of:

- A set of string tags defining simple attributes of the event's context (e.g., `direct_message` for events that relate to a direct-messaging channel/conversation).

- A set of key-value pairs (string to string) defining properties of the events context (e.g., mapping the key `peer` to the name of a known peer, or from the key `service` to the name of a service)

As part of the static configuration of an agent, there are **routing rules** that determine where an event should go.
Each routing rule defines:

- A set of tags to look for. The rule matches events that have all the tags, and otherwise doesn't match.

- A set of keys to look for, and either a corresponding value to look for or a wildcard `*`.
  The rule matches on events that have the key, if it has an identical value, or if the rule had a wildcard (`*`) value.

- A template for a session key to use, if this rule matches and is chosen.
  For example, a template might take the form `peers/{peer}/dms/{service}` for a rule that required the `direct_message` tag, and matched both the keys `peer` and `service` with wildcards.

  The `{}`-enclosed names in the session-key template must be keys that the rule matched with a wildcard.
  Not every key that was matched must be present in the session key template.

The runtime matches an event against the declared rules and then picks the *most specific* rule that matched.
A rule that matches on more tags is more specific than one that matches on a subset of those tags.
A rule that matches on more keys is more specific than one that matches on fewer keys.
A rule that matches on a specific value for a key is more specific than one that does a wildcard match.
If no single most-specific rule matches, then an error is logged (the situation represents a configuration problem).

### Journaling

Every agent has access to temporal memory in the form of a **journal**.
The journal for an agent is stored in its home directory as files for the form `journal/YYY/MM/DD.md`.

While an agent's journal is stored as ordinary files, and can be manipulated as such, agents will typically write journal entries with a dedicated tool, which automatically appends a timestamped entry to the current day's journal.
Each journal entry is tagged with both a timestamp, and also the session key of the session that posted the entry.

When prompting an LLM, the system prompt generated for a session will include recent journal entries for the agent, prioritizing entries that are from the same or related sessions.
(TODO: the exact definition of what "related" means here)