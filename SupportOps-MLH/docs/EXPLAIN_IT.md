# Practice explaining the sample

These tasks are for the applicant to do and explain. None is recorded as completed personal work. Work in a disposable copy if changing behaviour; preserve the verified candidate and do not invent commits or backdate work.

## A five-minute explanation

1. **One minute:** describe the support problem, deterministic scope and starter/Codex credits.
2. **One minute:** trace message validation, server-established principal, graph routing and owner-scoped order lookup.
3. **One minute:** explain conversation UUID versus permission, saved order references and the ownership recheck.
4. **One minute:** demonstrate a failed turn, unchanged count and recovery; explain the transaction.
5. **One minute:** explain bounded waiting versus thread cancellation, then name two production gaps.

## Questions and answer checks

| Question | What a correct explanation should cover |
|---|---|
| Why not put customer ID in the request body? | A client can forge it; identity comes from authentication and is checked inside tools. |
| Why does a frozen principal matter? | It prevents accidental mutation after validation; it does not make a public demo token secure. |
| What does a route plan authorize? | Nothing; it selects a fixed operation whose tool still checks ownership. |
| Why validate the plan again? | Internal/injected callers and constructed models can bypass earlier assumptions. |
| Why is a UUID not an access token? | It identifies a record; `require_owner` must also match the authenticated owner. |
| What is stored for follow-up? | A previously authorized order reference, reauthorized on each use; not trusted cached permission. |
| What happens if the graph fails after writing a checkpoint? | The turn transaction rolls back checkpoints and count together. |
| Why parameterized SQL? | Values stay data instead of becoming SQL syntax. |
| Why not use pickle for memory? | This boundary uses data-only JSON and rejects unsupported object encodings. |
| Is this an LLM agent? | It is a deterministic graph demonstrating application boundaries; no live model is included. |
| Does timeout stop the worker? | No. The caller stops waiting; the slot remains occupied until the read exits. |
| What do the tests not prove? | Production security, every environment/timing case, live-model quality or personal authorship. |
| What did you personally write? | State only your actual work. Attribute the supplied M0 foundation and Codex-assisted extensions. |

## Small exercises you can own

1. Explain `lookup_order` line by line, including its return value when B asks for A's order. Read the corresponding test before changing anything.
2. In a disposable copy, add a test that a blank message is rejected, then compare with the existing coverage. If it is redundant, explain why it adds no new protection.
3. Design a meaningful test for a behaviour you think is missing, explain its expected failure, and obtain a review before altering the verified sample. Do not weaken the network guard or existing assertions.
4. Run the six choices manually. Predict each status/turn count before pressing Enter. Explain why choice 5's policy is a trusted harness setting rather than a client-controlled escape hatch.

Record future personal work honestly: date, file, behaviour, why it was necessary, exact test command/result and the assistance actually used. Understanding assisted code is valuable, but it does not retroactively turn generated code into independently authored code.

## Application project description

> SupportOps is a local Python customer-support workflow built from a ChatGPT-generated starter with substantial Codex assistance. It demonstrates typed routing, customer-scoped tools, SQLite-backed conversations and bounded failure handling. I am using the code walkthrough and regression tests to understand the design and distinguish the generated foundation, assisted extensions and any changes I personally make.

Revise the learning sentence only as your actual progress warrants. If MLH asks specifically for independently authored work, this attribution must remain visible; preparation does not guarantee acceptance of the sample.
