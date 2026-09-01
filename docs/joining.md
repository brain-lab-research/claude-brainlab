# Joining the lab's knowledge base

The base is a shared, typed record of what the lab claims, ran, measured and decided, plus a
library of read papers. Your agents read it and write to it over MCP; you set it up once.

## What a lead does, once per person

    python3 invite.py --email name@university.edu

That adds the person to the base, mails them a one-time link and prints it as a fallback. Any real
address works; nothing requires a lab domain. A second link for the same person replaces their key,
which is also how a lost key is handled.

## What you do, once

Open the link and press the button. That shows your key once, with the block to paste into your MCP
client. There is no password and no account to remember: the key is the credential, one key covers
every tool you run, and if you lose it you ask for another link.

Which of your agents wrote a record is worked out from the connection itself — a client names
itself in the protocol handshake — so nothing is typed by hand and no model is ever asked to sign
its own name.

## What you can do from the first minute

- **Read everything.** Search covers the lab's own records and the library at once. Ask
  `search_lab` a question in plain words; it walks the graph, so a claim arrives with the run that
  tested it and the number that came out.
- **Write to your own project.** Membership is per project: your project is yours, everything else
  is read-only until someone adds you.

## What the base will refuse, and why

- A hypothesis without a falsifier that can actually happen. "Worse in every setting" never
  arrives, so it closes nothing.
- Evidence without a number in `metrics`. A measurement nobody can compare is not a measurement.
- A claim about a paper without a verbatim quote from that paper's stored text.
- A fragment instead of a statement: a table row, half a formula, an item from a numbered list.

These are not style rules. Every one of them exists because a record that breaks it cannot be
cited by anyone but its author.

## When something goes wrong

The refusal names the defect and what to do about it — read it rather than retrying. If the
service itself does not answer, tell the lead: it runs as a single process, and a watchdog
restarts it, but a wedge is worth knowing about.
