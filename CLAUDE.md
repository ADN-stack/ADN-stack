# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

`ADN-stack/ADN-stack` is a GitHub **profile repository**: because its name matches the
`ADN-stack` account name exactly, GitHub treats it specially and renders the contents of its
README as the profile page shown at `github.com/ADN-stack`. This is not an application
codebase — there is no source code, build system, package manifest, or test suite in this
repository.

## Structure

- `Billy.md` — the profile content. Note: GitHub only auto-renders a file named `README.md` on
  the profile page; as long as this file is named `Billy.md` it will **not** appear on
  `github.com/ADN-stack`. If the goal is a visible profile page, this file should be renamed to
  `README.md` (or its content merged into a `README.md`).

## Working in this repo

- There is nothing to build, lint, or test. Changes here are just edits to the profile markdown.
- Keep edits scoped to what's asked — this repo has no other files, directories, or conventions
  to be consistent with.
- If real project code is later added to this repository (or Claude is asked to scaffold a new
  project here), this file should be rewritten to document that project's actual build/test/lint
  commands and architecture rather than this repo's current profile-only state.
