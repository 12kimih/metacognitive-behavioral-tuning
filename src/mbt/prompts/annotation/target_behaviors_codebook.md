# Target-behavior codebook v1 (frozen before any trace was read with it)

Source: seven behaviour definitions (`where`, `insert`, `never`, `marker`) of
the MBT editing instructions (`mbt_a_brief.md`). This codebook describes what
each behaviour LOOKS LIKE in a trace.

Unit of analysis: one full trace (the reasoning plus the final answer). An OCCURRENCE is one
passage that meets the definition; two passages at different places are two occurrences even
when they are about the same result. COUNT is the number of occurrences. POSITION: the third
of the trace in which each occurrence starts; the line ranges of the three thirds are given with
the trace. Judge by the definitions only; whether the final answer is right is irrelevant.

A passage that only ANNOUNCES the behaviour without doing it ("let me verify", "double-check:
yes", "I should plan", "checked, it is correct") is NOT an occurrence of any behaviour below.

## B1 subgoal-when-absent (a plan: goal, sub-goals, order)

PRESENT when the trace states, as a plan before doing the work, all three of:
- the goal: what has to be produced;
- sub-goals: what has to be established or be TRUE first for the goal to be reachable;
- the order: which sub-goal comes before which (and ideally why).
Usually near the start, but count it wherever it appears.
NOT an occurrence: restating the problem; naming a method only ("use generating functions");
a to-do list with no dependency between items; a summary of a solution already worked out.

## B2 logical-check-in-deduction (a named premise or missing case, tested)

PRESENT when, right after a step that established a new fact, the trace names a premise that
step rests on (a condition that could have failed: a divisor non-zero, an index in range, a
case assumption, a requirement of a theorem) or a case that the trace's own case split does not
cover, and tests it against the line that established it, a condition the problem states, or a
settling case, saying what the test showed.
Example shape: "This divides by (x - 2), which needs x != 2; the line above fixed x = 5, so it
holds." / "The split covers even n >= 2 and odd n >= 3; n = 1 falls outside both branches."
NOT an occurrence: redoing the step's own working to see if it comes out the same; a premise
named in a planning passage; "which holds here" with nothing after it; naming a premise that
cannot fail.

## B3 checkable-verification (a check of a derived result by a stated route, with its outcome)

PRESENT when the trace checks an already-derived result (a candidate answer or an intermediate
result) by a route a reader could carry out, and states the outcome. Routes:
- substitution: put the result back into the condition it must satisfy;
- constraint: check it against a stated requirement (range, type, count, format, allowed set);
- second route: reach the same result by an independent method; a small or degenerate case
  whose answer is known without the derivation also counts.
NOT an occurrence: "verified", "Yes.", "matches.", "OK" with nothing that could have failed;
redoing the same working (the derivation agreeing with itself); a check with no stated
outcome. Checks that are part of the final closing passage (B7) are coded under B7, not here.

## B4 deep-named-return (a return to a named earlier point, with the assumption given up)

PRESENT when, where the trace abandons a path and turns to another, it says which EARLIER
point it is going back to (identifiable in the trace's own words), which assumption made at
that point is being given up, and (usually) what is tried instead.
Example shape: "Going back to where I assumed the three roots were distinct: that is what made
the factorisation fail, so drop it and allow a repeated root."
NOT an occurrence: a bare switch ("let's try another approach", "alternatively"); naming only
the obstacle; rewinding only to the previous line.

## B5 continue-the-dropped-line (a line the trace was about to leave, carried two or more steps)

PRESENT when the trace takes up an approach or a possibility it had raised and was leaving (or
had left lying: a question asked and not answered, a candidate named and not tested, a formula
written and not used) and carries it forward by TWO OR MORE deductive steps, each a
consequence of the one before ("thus", "hence", "so"), to a stated result or a stated obstacle,
and the trace then goes on.
Example shape: "Before leaving the generating-function route: the coefficient of x^n is the
convolution sum, which for n = 4 gives 8, so the route does reach a number, just not in closed
form."
NOT an occurrence: the trace's main line of work; one step; restating the idea; proposing yet
another approach.

## B6 catch-and-correct (an error caught and repaired)

PRESENT when the trace redoes an earlier step of its own, states that the result differs from
what was written above, and gives the corrected result and carries it forward. Saying which
operation produced the difference is usual but not required.
Example shape: "Recomputing: 17 * 23 = 391, not 371 as written above; the carry was dropped.
Taking 391 forward."
NOT an occurrence: a recheck that agrees with what was written (that is B3 or nothing); "wait,
that's wrong" with no redo and no corrected value; a correction of an interpretation of the
problem rather than of a step's result.

## B7 closing-decision (a decision that the work is finished, with checks of the candidate)

PRESENT when, at the end of the reasoning, after a candidate answer is reached, the trace both
(i) says the work is finished and why it is finished rather than merely stopping, and
(ii) checks the candidate by at least one of the four named checks, naming what it examined
and its outcome:
- conditions: the candidate put back into each stated condition;
- second route: the same result by a different route;
- consistency: a condition that could conflict with another, and whether it does;
- form: what the problem asked for (unit, exactness, one value or a set) and that the
  candidate is in that form.
At most one occurrence per trace. Report which of the four checks the closing passage contains
in `closing_checks`.
NOT an occurrence: "Checked, it is correct" or "Final answer: X" with no named check; a trace
that runs out without reaching a candidate.
