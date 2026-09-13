# Plan Format

Write your plan using this structure:

```markdown
# [Plan Title]

## Overview

[Problem statement, chosen approach, and key decisions in 1-2 paragraphs]

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer,
Quality Reviewer). Quality matters: vague entries here produce poor annotations
and missed risks.

### Decision Log

| Decision           | Reasoning Chain                                              |
| ------------------ | ------------------------------------------------------------ |
| [What you decided] | [Multi-step reasoning: premise -> implication -> conclusion] |

Each rationale must contain at least 2 reasoning steps. Single-step rationales
are insufficient.

INSUFFICIENT: "Polling over webhooks | Webhooks are unreliable" SUFFICIENT:
"Polling over webhooks | Third-party API has 30% webhook delivery failure in
testing -> unreliable delivery would require fallback polling anyway -> simpler
to use polling as primary mechanism"

INSUFFICIENT: "500ms timeout | Matches upstream latency" SUFFICIENT: "500ms
timeout | Upstream 95th percentile is 450ms -> 500ms covers 95% of requests
without timeout -> remaining 5% should fail fast rather than queue"

Include BOTH architectural decisions AND implementation-level micro-decisions:

- Architectural: "Event sourcing over CRUD | Need audit trail + replay
  capability -> CRUD would require separate audit log -> event sourcing provides
  both natively"
- Implementation: "Mutex over channel | Single-writer case -> channel
  coordination adds complexity without benefit -> mutex is simpler with
  equivalent safety"

Technical Writer sources ALL code comments from this table. If a micro-decision
isn't here, TW cannot document it.

### Rejected Alternatives

| Alternative          | Why Rejected                                                        |
| -------------------- | ------------------------------------------------------------------- |
| [Approach not taken] | [Concrete reason: performance, complexity, doesn't fit constraints] |

Technical Writer uses this to add "why not X" context to code comments.

### Constraints & Assumptions

- [Technical: API limits, language version, existing patterns to follow]
- [Organizational: timeline, team expertise, approval requirements]
- [Dependencies: external services, libraries, data formats]
- [Default conventions applied: cite any `<default-conventions domain="...">`
  used]

### Known Risks

| Risk            | Mitigation                                    | Anchor                                     |
| --------------- | --------------------------------------------- | ------------------------------------------ |
| [Specific risk] | [Concrete mitigation or "Accepted: [reason]"] | [file:L###-L### if claiming code behavior] |

**Anchor requirement**: If mitigation claims existing code behavior ("no change
needed", "already handles X"), cite the file:line + brief excerpt that proves
the claim. Skip anchors for hypothetical risks or external unknowns.

Quality Reviewer excludes these from findings but will challenge unverified
behavioral claims.

## Invisible Knowledge

This section captures knowledge NOT deducible from reading the code alone.
Technical Writer uses this to create README.md files **in the same directory as
the affected code** during post-implementation.

**Placement principle**: Invisible knowledge must be captured CLOSE to
implementation. README.md files go in the package/directory containing the
relevant code, not in a separate documentation directory.

**Self-contained principle**: Code-adjacent documentation must be
self-contained. Do NOT reference external authoritative sources (doc/
directories, wikis, external documentation). If knowledge exists in an
authoritative source, it must be summarized in the code-adjacent README.md.
Duplication is acceptable; maintenance burden is the cost of locality.

**The test**: Would a new team member understand this from reading the source
files? If no, it belongs here.

**Categories** (non-exhaustive examples -- apply the principle):

1. **Architectural decisions**: Component relationships, data flow, module
   boundaries
2. **Business rules**: Domain constraints that shape implementation choices
3. **System invariants**: Properties that must hold but are not enforced by
   types/compiler
4. **Historical context**: Why alternatives were rejected (links to Decision
   Log)
5. **Performance characteristics**: Non-obvious efficiency properties or
   requirements
6. **Tradeoffs**: Costs and benefits of chosen approaches

## Milestones

Milestone numbering starts at 1 within each plan. Use sequential integers (1, 2, 3),
not phase-prefixed numbers (2.1, 3.1) unless explicitly managing multi-phase plans.

### Milestone 1: [Name]

**Files**: [exact paths - e.g., src/auth/handler.py, not "auth files"]

**Code Intent** (you write this):

Describe WHAT changes are needed. Do NOT include exact code or diffs.
Do NOT read source files -- Developer handles that.

Include:
- Functions/structs to add or modify (names, purposes)
- Behavior to implement
- Key decisions (reference Decision Log entries by name)
```
