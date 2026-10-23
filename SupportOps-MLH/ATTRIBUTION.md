# Attribution and contribution boundaries

Prepared for Abeeb Adesina Adeola's MLH application on 23 October 2026. The purpose is to make the engineering inspectable while stating how it was produced.

## Supplied foundation

The programme supplied `supportops-agent-starter.zip`. On 23 October 2026, Abeeb confirmed that this starter was generated for him in ChatGPT. Its original M0 foundation includes FastAPI endpoints, public demo authentication, strict contracts, read-only synthetic order/policy tools, a rule-based responder, a browser page and 37 baseline test cases. These are credited as a ChatGPT-generated foundation, not independently authored work by the applicant.

The preserved archive's hash and per-file comparisons are in [PROVENANCE.json](docs/PROVENANCE.json). The archive itself is not redistributed in this candidate. No author identity or license file was found in its file inventory; the origin attribution above is based on the owner's direct confirmation, not an embedded archive notice. No third-party repository origin is asserted.

## Codex-assisted extensions

The programme's S01-S03 work added typed LangGraph orchestration, owner-guarded SQLite conversations and checkpoints, atomic failure rollback, server-owned work budgets, failure tests and a repeatable synthetic demonstration. Codex assisted with implementation, tests, debugging, verification and documentation. These are assisted engineering extensions, not evidence that Abeeb personally typed every function or passed an independent technical assessment.

S00 also adapted the test event-loop fixture for Windows and added two checks that application networking remains forbidden. The starter test assertions were retained. S01-S03 test modules establish routing, persistence and execution-boundary behaviours.

## This MLH candidate

This is a separately versioned mock-only derivative of the current working source, not a restored historical Git commit. Current source was copied into a new directory. Configuration and responses now allow only mock mode. Live adapter modules, credentials/authorization machinery, the live extra and the unused provider-call method were excluded. The original working tree, all its tests and its blocked S04 status remain unchanged.

The generic injected route-planner seam remains for local deterministic testing, but no live adapter is included. Five new sample tests check the mock-only distribution boundary. The existing 134-case mock regression files are unchanged from the working source. Packaging, the scope checks, this attribution and the walkthrough are also Codex-assisted. All exclusions and file hashes are reviewable; excluding an unaccepted feature is not a claim that it was repaired.

## Applicant's role and interview wording

Abeeb supplied the portfolio/application goals, directed the assisted work and confirmed his professional history. Personal authorship percentages and independent mastery have not been assessed. Do not replace this record with an unsupported claim of independent implementation.

A suitable description is:

> SupportOps is my portfolio engineering project, developed from a ChatGPT-generated starter with substantial Codex assistance. The extensions cover typed routing, customer-scoped persistence and bounded failure handling. I distinguish the generated foundation and assistance from the code I can personally explain and changes I make myself.

Use the last sentence only after actually completing the walkthrough. Do not claim that this preparation is already a completed technical interview, accepted upstream contribution, deployed product or MLH-approved sample.

## Rights and release

The starter archive has no LICENSE/COPYING file. Its ChatGPT origin is now owner-confirmed; the earlier request for origin clarification is resolved. No blanket MIT/Apache license has been attached or third-party copyright grant invented. The package is prepared locally for owner review; public release still needs an explicitly scoped publishing approval. If an open-source license is desired, the owner should select it as part of that release. Dependency licenses remain their respective authors' licenses and are not replaced by this notice. This provenance record is not a legal determination about generated code.

Third-party libraries include FastAPI, Pydantic, LangGraph, LangGraph SQLite checkpointing, LangSmith (used to disable tracing), Uvicorn, pytest and HTTPX. Their behaviour is not claimed as newly invented project work. Consult installed package metadata and their upstream documentation for their license terms.
