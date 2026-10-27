DELETING THE METACOGNITIVE PASSAGES OF A REASONING TRACE
--------------------------------------------------------
Below is `trajectory.md`: a problem (PROBLEM), the reasoning a model wrote while solving it
(REASONING), and the answer it then gave (RESPONSE). Your work is to find every passage of the
REASONING whose job is one of the eight metacognitive behaviours defined below, and to list each
one for deletion.

What should remain is the same solution with the thinking-about-the-thinking taken out. The
remaining REASONING does the same working, in the same order, and reaches the same result, but it
no longer restates the problem to itself, plans, weighs which method to use, comments on how the
work is going, catches and fixes its own mistakes, checks its results, decides whether it has done
enough, or looks back over what it did.

YOU DO NOT WRITE ANY OF THE TRACE. You only choose what to cut. A program takes your list and
cuts each passage out of the REASONING exactly where you quoted it. Then it closes each gap by
removing the spaces, the line break or the blank line the cut left behind, and it does nothing
else: it adds no character, changes no word and re-cases no letter. So choose passages whose
removal leaves text that still reads. Cut whole sentences, whole clauses together with their own
punctuation, whole lines or whole paragraphs. Never cut part of a word, part of a formula or part
of a display-math, code or table block. Where a passage cannot be removed without leaving broken
text, leave it.

THE EIGHT BEHAVIOURS
--------------------
Each is a kind of passage to delete. A passage is deleted for what it does in the trace, not for
the words it uses: "check" in "check whether n is even" that then decides a case is working, and
"Yes." closing a recomputation of a result is a verification.

  B1  UNDERSTANDING THE PROBLEM
      Restating to itself what the problem asks or gives, listing the problem's conditions,
      settling what the question means, noting which information matters and which does not,
      and remarks about the request itself (its format, what the user wants, how long the answer
      should be).
      "We need to find the number of ordered pairs (a, b) with ..."
      "So the question is asking for the smallest such n."
      "The condition about c is irrelevant here."

  B2  PLANNING AND DECOMPOSITION
      Saying what will be done before doing it: a goal broken into sub-goals, the order steps
      will be taken in, an announcement of the next step, and plans for writing the final answer
      ("Need to present the answer boxed", "Let's craft the final response").
      "First find the roots, then count which lie in the interval."
      "Plan: 1) compute f(0); 2) use symmetry; 3) sum."
      "Now let's compute the area."

  B3  CHOOSING A STRATEGY FOR A REASON
      Naming possible methods, weighing them, choosing one and saying why. The method's own
      working, once it starts, is not B3 and stays.
      "We could use complementary counting or direct casework; complementary is simpler since
      there are fewer bad cases."

  B4  MONITORING PROGRESS
      Judging how the work is going: a summary of where the work stands, what has been found so
      far and what remains, and remarks on difficulty, on being stuck or on whether a line is
      leading anywhere.
      "So far we have x + y = 7 and xy = 10; it remains to find x^2 + y^2."
      "Hmm, this is getting messy."
      "Not obviously helpful."

  B5  DETECTING AND CORRECTING ERRORS
      (a) A mistake caught and fixed: the step that was wrong, the catch ("Wait, that's wrong",
          "Actually ..."), and any remark on it. Delete the wrong step together with the catch,
          so that what remains is the corrected working only, as if it had been right the first
          time. The corrected working stays. Where the corrected value is written only inside
          the sentence that announces the catch, so that cutting the catch would cut the
          correction too, leave that episode.
      (b) A path given up: the sentence that says an approach fails, is abandoned, or that the
          trace goes back to an earlier point or tries something else ("That doesn't work.",
          "Let's go back to the equation for k.", "Let's try another approach."). Delete that
          sentence. The working of the approach that was tried and left is the trace's working,
          and it stays.

  B6  VERIFICATION
      Checking a result, a candidate answer or a step that was already done: substituting back,
      recomputing it, deriving it another way, testing a small or special case to confirm it,
      testing that a premise holds, and the verdict that closes a check ("Yes.", "OK.", "Good.",
      "matches.", "Correct."). Delete the whole check, the announcement, the working done only
      for the check, and its verdict. The result that was checked stays.
      "Check: 2*3^2 - 3 = 15. Yes."
      "Let me double-check with n = 2: the formula gives 6, and listing gives 6. Good."

  B7  JUDGING SUFFICIENCY
      Deciding that the work is or is not enough to answer: that the result is certain enough,
      that nothing else needs checking, that the work can stop, or that more must be done before
      answering.
      "That should be it."
      "I'm fairly confident in this."
      "But we should make sure there are no other solutions before answering."

  B8  REFLECTION AND EVALUATION
      Looking back over the solution as a whole and judging it: what was key, what went well or
      badly, what could have been done differently.
      "Looking back, the trick was to factor early."
      "This approach was more roundabout than needed."

WHAT STAYS
----------
  - Every step of the working that the remaining text goes on to use: equations, values,
    derivations, the cases of a case split, definitions, and the working of approaches that
    were tried. After your deletions, the REASONING still derives the result it derived before,
    by the same working.
  - The statement of the result the working reaches ("So x = 5.", "Thus the answer is 12.").
    That is the end of the working, not a judgement about it.
  - The working inside a passage that is partly a behaviour and partly working the rest relies
    on. Cut the behaviour's sentences or clauses if they come out cleanly, and keep the rest.
  - A short word that only opens a passage ("Wait,", "Hmm,", "OK,", "So,") goes with the passage
    it opens when that passage is deleted, and stays when it is not.

EVERY PASSAGE THAT IS ONE OF THESE IS LISTED, AND NOTHING ELSE IS. A trace that checks its
results forty times has forty B6 deletions, and a trace with no such passage has none. Do not
cut anything to make the trace shorter, and do not leave a passage because many have already
been cut. Read the whole REASONING, from the first paragraph to the last, before you finish.

WHERE EACH PASSAGE IS
---------------------
In the REASONING printed below, the paragraphs are separated by blank lines as they always were,
and before each paragraph there is now a numbered label on a line of its own: `[[#1]]` before
the first paragraph, `[[#2]]` before the second, and so on to the last. The labels are not part
of the trace. They name the paragraphs: paragraph k is the one that follows `[[#k]]`.

THE SHAPE OF YOUR REPLY
-----------------------
Your reply is a list of deletions and then the NOTES, and nothing before the first. Each
deletion is three things, opened by its own line:

    ===== DELETION =====
    PARAGRAPHS: [[#<k>]]                       or, for a passage over several paragraphs,
                                               PARAGRAPHS: [[#<first>]] to [[#<last>]]
    BEHAVIOUR: <B1 to B8>
    <the words to delete, copied exactly as they stand in those paragraphs, one or more lines>

To delete a whole paragraph, name that one paragraph and write instead one line: WHOLE:
followed by its first words, copied exactly, at least six words or the whole paragraph if it is
shorter. A WHOLE deletion is always one paragraph: for several whole paragraphs in a row, list
each as its own deletion. Otherwise copy the words character for character, including their mathematics and
punctuation, and never a label. The words must occur in the paragraphs you name. If the same
words occur more than once there, copy enough of the passage to make it occur once, or list each
occurrence as its own deletion, in order.

After the last deletion:

    ===== NOTES =====
    <one line per behaviour B1 to B8: how many passages you deleted, or why none>

List the deletions in the order of their paragraphs, first to last. A label the REASONING does
not carry or a behaviour other than B1 to B8 makes the whole reply unusable. A deletion whose
words are not in the paragraphs it names, or whose cut goes through the middle of a word, is not
made.
