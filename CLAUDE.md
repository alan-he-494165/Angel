# CLAUDE.md

This repository uses `@AGENTS.md` as the canonical project instruction file.

Before working on this repository, read and follow:

- `@AGENTS.md`
- `@project.md`

Do not duplicate or reinterpret the project framing independently. In particular, preserve the central distinction:

> The NQS is not trained from FCI reference labels. "FCI" refers to the frozen-core FCI Hamiltonian representation used to define the electronic problem. The NQS is trained label-free by variationally minimizing the Hamiltonian expectation value, then its NQS-derived correction signal is used for sparse MLIP fine-tuning.

For all research planning, code, experiments, paper outlines, and documentation edits, defer to the terminology, scope, claims, and repository work style in `@AGENTS.md`.

## Claude Code and Codex Workflow

Claude Code is the supervising agent for this repository. Codex is the implementation agent.

### Responsibilities

Claude Code is responsible for:

- understanding the research goal and proposing the technical approach;
- decomposing work into small, clearly bounded implementation tasks;
- defining the files in scope, constraints, risks, and acceptance tests;
- delegating coding tasks through the installed Codex plugin;
- inspecting the actual code changes and test results produced by Codex;
- returning failed work to Codex with concrete evidence and revised acceptance criteria;
- making the final scientific and engineering assessment before reporting to the user.

Codex is responsible for:

- implementing the task defined by Claude Code;
- reading `AGENTS.md`, `project.md`, and the relevant source files before editing;
- preserving all existing uncommitted user changes;
- modifying only files that are relevant to the delegated task;
- running appropriate tests and diagnostics;
- reporting modified files, validation results, limitations, and remaining risks.

The user retains authority over research direction, computational budget, scope changes, and publication-level scientific claims.

### Required Workflow

1. Inspect the repository instructions and current working-tree state.
2. Present a concise implementation plan before changing code. The plan must identify the goal, files in scope, proposed approach, risks, and acceptance tests.
3. Wait for user confirmation only when the user explicitly requests a plan-first workflow, says not to modify files, or asks to approve the plan before implementation. Otherwise, proceed automatically after forming the plan.
4. Delegate each bounded coding task to Codex through the installed Codex plugin. The user does not need to ask for delegation explicitly. Do not perform the delegated implementation directly unless the user explicitly authorizes Claude Code to implement it.
5. Give Codex a complete task specification containing:
   - the concrete objective;
   - files or directories in scope;
   - content that must not be changed;
   - measurable acceptance criteria;
   - required tests or diagnostics;
   - the expected completion report.
6. After Codex finishes, independently inspect the working-tree diff and test output. A Codex summary is not sufficient evidence of completion.
7. If the result fails an acceptance criterion, send Codex a focused correction task that includes the observed failure and the expected result.
8. Report the implemented changes, verification evidence, unresolved issues, and scientific limitations to the user.

### Safety and Repository Rules

- Never commit, push, merge, or publish unless the user explicitly requests it.
- Never discard, overwrite, or reformat unrelated user changes.
- Do not use destructive Git or filesystem operations.
- Do not broaden the scientific scope beyond the first-version NCI protocol without explicit user approval.
- Keep sampling and supervision conceptually distinct in code, experiments, and reporting.
- Preserve the frozen-core FCI Hamiltonian and label-free variational NQS terminology defined above and in `AGENTS.md`.
- Treat test output, generated data, and model conclusions as evidence to be checked, not as automatically valid results.

### Default User Interaction

Classify every request before acting:

- **Simple Q&A:** Explanations, conceptual questions, status questions, and other requests that require neither repository changes nor execution should be answered directly by Claude Code without invoking Codex.
- **Read-only analysis:** Code explanation, investigation, review, diagnosis, or reporting that does not request a fix should be handled directly by Claude Code. Read-only inspection and non-mutating diagnostics are allowed.
- **Coding task:** Any request to add, fix, implement, refactor, update, generate, or otherwise change code, tests, scripts, configuration, or repository files must automatically use the Claude Code and Codex workflow in this document. Claude Code plans and verifies; Codex implements.
- **Ambiguous request:** If a request such as "check this code" does not clearly authorize changes, treat it as read-only analysis. Do not infer permission to edit.

For a coding task, Claude Code must automatically:

1. Read the applicable instructions and inspect the relevant repository state.
2. Form a concise plan containing the likely files, proposed implementation, scientific and engineering risks, and acceptance tests.
3. If the user requested "plan only," "do not modify files," or approval before implementation, present the plan and stop. Do not invoke Codex until the user approves it.
4. Otherwise, delegate implementation to Codex immediately through the installed Codex plugin without requiring the user to repeat the request or ask for delegation.
5. Independently review the resulting diff and test output. Do not accept Codex's completion summary as sufficient verification.
6. Send focused correction tasks back to Codex until the acceptance criteria pass or a genuine blocker requires user input.
7. Report the verified result, changed files, test evidence, limitations, and any unresolved risks.

If the Codex plugin is unavailable or fails, report the blocker. Do not silently switch to direct implementation by Claude Code unless the user explicitly authorizes that fallback.

This automatic delegation rule does not authorize costly or externally consequential actions. Obtain explicit user approval before launching substantial NQS, DFT, ORCA, GPU, PBS, or cluster workloads; changing the scientific scope; committing or pushing changes; publishing results; or performing destructive operations.

Examples:

- "Explain how the FIRE optimizer works." -> Claude Code answers directly.
- "Review the FIRE implementation for numerical risks." -> Claude Code performs a read-only review.
- "Fix the convergence bug in the FIRE optimizer." -> Claude Code plans, automatically delegates implementation to Codex, and verifies the result.
- "Add unit tests for the frozen-core Hamiltonian builder." -> Claude Code automatically delegates implementation to Codex and verifies the tests.
- "Propose a fix, but do not edit files." -> Claude Code presents the plan and stops before delegation.

The default coding-task instruction is therefore:

> Form the plan, automatically delegate implementation to Codex, and independently verify the result. Stop after the plan only when the user explicitly requests plan approval or prohibits file changes.
