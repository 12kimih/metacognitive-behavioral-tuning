THIS WORK IS DONE IN ONE REPLY
------------------------------
The brief below was written for an editor who works on `trajectory.md` inside a git repository,
with an editing tool, one insertion and one commit at a time. You have none of that. There is no
repository, no tool and no file to change: `trajectory.md` is printed in full below, and the whole of your work is the one reply you write. The parts of the brief that only
describe the repository, the editing tool and the commits have been taken out. Everything else
is the brief exactly as that editor received it, and every rule in it holds for you.

Some of what remains still speaks of the repository. Read it this way.

  `trajectory.md`              the file printed below
  an insertion, or a commit    one passage you add at one point
    that records one
  a `<name>: none -- <why>`    a line of exactly that form in the NOTES of your reply, one for
    empty commit               each behaviour that has no place in this trace
  a `fix:` commit              does not exist here. Write each passage as you mean it
  the message you finish       the NOTES section of your reply
    with
  `git show`, `git log`,       not available and not needed. The file as it arrived is the one
    `git diff`                 printed below, and nothing else about it exists

WHAT YOU RETURN IS THE WHOLE FILE'S TWO EDITABLE SECTIONS, AS YOU LEAVE THEM. Your reply
carries the REASONING section and the RESPONSE section written out in full: every line of
the original that stays is written out where it stands, and each passage you add is written
at its point, after its host line and a blank line, as its own paragraph. Where the host
line sits inside a block that has to stay whole, a display-math block, a code block or a
table, the passage goes after the line that closes the block. What you leave out of your
reply is gone from the file, so a line you do not write out is a line you removed.

WHAT YOU ARE LOOKING AT
-----------------------
A language model was given one problem and nothing else. It wrote out its thinking on the
way to an answer, and then it wrote an answer. `trajectory.md` holds that problem, that
thinking and that answer, and it is the whole of what you have.

Nobody here knows whether the answer is right. There is no reference answer in this file and
you are not being given one, so you cannot grade this attempt and you are not being asked to.
What you are given is one attempt: somebody worked the problem from the problem alone, and
whether they got it right is not known to you or to anyone else working on this file.

AND YOU MAY NOT GO AND FIND OUT. Do not search for this problem. Do not search for its
source, its contest, its year, its number or its answer. Do not fetch a page, a paper, a
forum thread or a solutions file about it. Do not read any file outside `trajectory.md`
looking for it.

If you notice yourself wanting to check what the answer really is, that is the moment this
rule is for: the answer is not part of this job, and an edit made because you learned it is
an edit nobody can use.

THIS MUST NOT HAPPEN, and it is stated here rather than only in the rules below because it
is a decision you would make in your first minute. The whole of what this work produces is a
comparison between two training sets that differ in your edits and in nothing else. An editor who knows the answer edits towards it -- adds a correction where the
trace disagrees with what was looked up, withholds one where it agrees -- and every one of
those edits is a fact about the reference answer wearing the clothes of a fact about the
trace. Nothing downstream can separate the two afterwards.

IT HAS HAPPENED AND THAT IS WHY THIS PARAGRAPH EXISTS. Measured 2026-09-15 over three rounds
of twenty cells: one cell in each went looking, always the same trace, the one whose own
route to the answer is a memory of the answer. In the third round it put the problem
statement in quotes into a search, found the contest and year, read the answer choices off a
forum page, tried the official solutions file, and then declined a correction it had made to
the same trace the round before. It said so in its closing message; nothing else showed it.

WHAT TO DO INSTEAD. Say in the message you finish with that the question could not be settled
from the file, and leave the trace as it is. A question recorded unanswered is worth more here
than an answer fetched from outside, and the last numbered step says where in that message it
goes.

WHAT YOUR EDITS ARE FOR
-----------------------
The file you leave behind becomes one example in a training set. A smaller model, which
this brief calls THE STUDENT, is trained on the thinking as you leave it, and is then tested
on problems it has never seen. So what is being taught is not this answer. It is the habits
of thinking that the reasoning displays: checking a step before relying on it, noticing that
a result came out wrong, going back far enough to reach the assumption that failed, deciding
that the work is done. Those habits are what the named behaviours below are, and the purpose
of the work is to make this trace show them where it does not.

The student learns from what it reads, including the parts nobody meant to teach it. That is
the single fact behind most of the rules below. A passage that is correct but written in a
different voice teaches the student that thinking changes voice halfway through. A check
that examines nothing teaches it to write checks that examine nothing. A passage that uses
a result the trace has not reached yet teaches it to reason from what it does not have.

HOW YOUR WORK WILL BE MEASURED, which is the reason the rules about what may change are as
strict as they are. Two training sets are built from the same traces: one from the files as
they arrived and one from the files as you leave them. Two copies of the same student are
trained, one on each, and the difference between them is what these edits are worth. That
subtraction only means something if the two sets differ in the edits and in nothing else.
Every rule below about preserving existing text is protecting that subtraction.

WHY YOU ARE ADDING RATHER THAN REWRITING. Asking a model to rewrite a trace was tried and
measured, and it failed twice over: a 10,000-character trace came back at about 2,000
characters and so did a 20,000-character one, and when the paragraphs were numbered and a
model was asked where to insert, it chose the early numbers almost every time. Editing a
real file in place is what escapes both. Two of the numbered steps below ask you to say
WHERE your insertions fell and WHAT you read, not to hit a figure, so that a file whose far
end was skimmed is visible afterwards and a file whose weak steps really were early is not
mistaken for one. The last step is where you report both.

WHY MISTAKES STAY. A step that came out wrong is not damage to be cleaned up. It is the one
place in the trace where catching and repairing an error can be shown, and a student that
never sees an error being caught does not learn to catch one. Where a behaviour below covers
exactly those places, use it; where none does, leave the wrong step alone.

WORDS THIS BRIEF USES
---------------------
  trace       the model's thinking, the middle section of the file. Also called the
              reasoning
  the student the smaller model that will be trained on the file you leave behind
  behaviour   one of the named habits of thinking listed below. Each has one NAME, a
              hyphenated phrase saying what the behaviour is, and that name is what a
              commit is recorded under. There is no code beside the name
  passage     the text you add at one point. Most are a paragraph and some are a
              sentence, and a piece of work that needs more takes more. There is no
              length this brief asks for
  host line   the existing line you attach a passage after
  insertion   one passage, added at one point, recorded by one commit
  planning    a stretch of the trace that says what it intends to do rather than
              establishing anything. Restating the problem, naming a method, listing what
              to try. A behaviour that is refused inside planning says so in its entry
  marker      a word or phrase that signals doubt or a check without carrying one out.
              "Wait", "Hmm", "Let me double-check", "Is that right?". The rule named
              `no-marker-without-a-check` is the one about these
  case split  a point where the reasoning divides the problem into cases and handles them
              separately. It is COVERED when every case the problem allows falls into one
              of the branches
  candidate   a result the trace puts forward as the answer, before anything has
              confirmed it. Also written "candidate answer"

Each behaviour is labelled with two classifications, printed in brackets after its name.
This is what they mean:

  SIDE   what kind of work the entry adds
    exploit     Goes deeper into a line of reasoning the trace already
                has, or checks one. It works with what is in front of it
                rather than opening anything new.
    structure   Names what the reasoning is about to do, before it does
                it, so that the steps which follow have something to
                attach to.

  SCALE  the size of error the addition can catch, and for an
         entry that adds nothing, the reach of the text it acts on
    global      The addition can catch an error in something the whole
                derivation rests on: a premise, a case split, a strategy,
                an assumption carried forward.
    local       The addition can catch an error in one result or one
                requirement: a slip in the working, one value used in
                place of another, an answer given in the wrong form.
                For an entry that adds nothing, the text it acts on is
                a sentence or a clause, and sometimes a short run of them
                where the test in that entry picks out the run. What makes
                it local is that it does not reach the derivation.

Three more words carry the behaviour entries below, and this brief is for mathematics, so this
is what they mean here.

  A STEP is one move in the derivation. It is NOT a line of the file, and the
  difference matters, because a trace that writes its whole reasoning as one
  paragraph still holds as many steps as it makes moves.
  The FACT a step establishes is an equation, a value, a bound, a relation or a
  settled case.
  A PREMISE is a condition the step needed in order to be valid, such as a
  divisor being non-zero or an assumed ordering holding.

Each entry also carries one or more examples. An example shows the shape of the addition
and not a form of words to copy. What to write is what the entry's own INSERT describes, in
the voice the trace already uses.

THE FILE AND THE REPOSITORY
---------------------------
`trajectory.md` has three sections, each opened by a line of the form `===== NAME =====`:

    PROBLEM     The problem exactly as it was put to the model, unchanged. It is the only
                thing the model was given. Do not change this section for any reason.

    REASONING   The model's working: everything it thought on the way to an answer, in the
                order it thought it. This is the section you edit.

    RESPONSE    What the model produced as its reply after it stopped reasoning. It is one
                attempt at a solution and nothing here establishes whether it is right. It
                is not a reference answer and there is no reference answer in this file.
                Edit it only in the one case the rules below describe.

Never add, remove or reword a `===== ... =====` line. Everything downstream reads the file by
those three lines and nothing else.

WHAT THE WORK IS
----------------
The trace already contains two kinds of work. WIDENING the search, where it considers what
might be true, and DEEPENING it, where it takes one possibility and derives from it until
something comes out. Both are already there and both stay. You are adding to whichever is
thin at a given point.

IN THESE TRACES THE THIN ONE IS ALMOST ALWAYS THE DEEPENING, and that is worth knowing
before you start reading. A trace of this kind will name three approaches, ask twice whether
a condition might hold, offer two readings of the wording, and then widen again -- and the
line that takes one of those and carries it two steps forward is the one that is not there.
So most of what you add is a consequence of something already written rather than a new
idea, and the rule `do-not-widen-a-wide-trace` refuses the new idea outright. Where a
behaviour below names a possibility nothing was deduced from, that entry is where the
deduction goes, and its INSERT says how far to carry it.

Most of the work is adding: you find a point where a named behaviour is missing and you put
a passage there. A behaviour that may also CHANGE or REMOVE text states so in its own entry
and names exactly what it may touch, and that named text is the only text you may change or
remove. If no behaviour below states it, nothing in the trace may be removed or reworded.

    DO     Insert new passages at points where a named behaviour is missing.

    NEVER  Change a value, an expression or a conclusion the trace has already
           written, unless a behaviour below states that it may and names that text.
           A WRONG STEP IS NEVER AMONG THE TEXT AN ENTRY NAMES, so a wrong value is
           never repaired in place under any behaviour or any policy. Where one is
           wrong, leave it standing. A behaviour below may put a catch and a
           correction after it, and where none does, the wrong value stays as it is
           and nothing is added.

THE READER YOU ARE WRITING FOR
------------------------------
Everything you leave in this file will be read as one continuous piece of thinking by someone
learning how to reason. So the finished REASONING has to read as one person working, from the
first line to the last: one voice, one tense, one level of formality, one notation. An
insertion that is correct but reads as a different writer is a worse insertion than none.

WHAT NO EDIT MAY DO
-------------------
These hold for every behaviour below, whichever one you are working under. Each is a failure
that was measured on real rounds rather than a matter of taste. Each has a name, so that a
behaviour entry can point at one and you can tell which is meant.

  no-marker-without-a-check
      Never add a marker without a check. "Wait", "Hmm", "Let me double-check", "Is that
      right?" with no named object and no stated finding is not an insertion, it is
      noise. Every addition must say WHAT it examined and WHAT it found.

  a-check-goes-beside-its-object
      Never put a re-check of a result, or a check of the form the answer has to
      take, inside a stretch of planning. Those belong immediately after the step
      that produced the thing they check.

  do-not-widen-a-wide-trace
      Never add a new parallel approach to a trace that already proposes many. Widening
      a trace that is already wide is the structure that transfers to the student and
      costs it accuracy.

  nothing-is-removed-or-reordered
      Never summarise, never truncate, never reorder. THE ORDER IS ABSOLUTE: no behaviour
      and no policy moves a line of this trace, ever. The LENGTH is absolute too except
      in one place, and this rule says where: a behaviour whose own entry names text it
      may take out, under a policy that offers that behaviour. That is a named exception and
      the last paragraph of this rule states it. It is not one of the three refused above --
      it is the removal of text the entry names, in its own commit, with what it was
      recorded in the subject. Everything outside it is add-only.

      Never delete or rewrite REASONING. Everything the model reasoned stays exactly as
      it is, character for character: a dead end that led nowhere, a hypothesis that was
      wrong, a step that came out wrong, a check that found nothing. Those are what the
      trace is, and each is a place where catching and repairing an error can be shown.
      Where a step is wrong, leave it standing. Only add.

      Breaking one line into two at a sentence boundary, to make room for a paragraph in
      the middle of a section that is a single line, is not deleting or rewriting. Every
      word is still there and in the same order. The rule is about the words, not about
      where the line endings fall, and the single space that used to separate the two
      sentences is the one character it lets go, since a paragraph break now stands in
      that space's place.

      If a behaviour below states that it may remove or rewrite text, that entry names
      exactly what it may touch, and that named text is the only exception to this rule.
      Nothing else may be removed or reworded, and if no behaviour you have been given
      states it, nothing may be removed at all.

  density-is-not-the-signal
      Never repeat a behaviour in order to make the trace denser. Density does not
      separate right traces from wrong ones, placement and consistency do, so an
      insertion added to raise a count is added for nothing.

      THIS IS ABOUT THE REASON AND NOT THE NUMBER. A trace with thirty steps that each
      rest on something fragile gets thirty checks and that is correct. What this
      refuses is the thirty-first, put there because thirty looked thin.

  write-from-what-is-established
      Never write a passage that reaches back for something the trace has not got to
      yet. At any point, write from what the trace has established BY THAT POINT.

  a-disqualifier-is-spent-one-place-at-a-time
      Never disqualify a group of places with one sentence. Several entries below close with a
      short list of the things that let you pass a place over, and each of those is a fact
      about ONE place. "Every candidate is checked later by the trace" and "every switch has
      its reason in the line above" are not disqualifications, they are summaries, and a
      summary cannot be checked by anyone.

      SO EACH PLACE IS TAKEN SEPARATELY, and for each one you name the place, name which
      disqualifier, and QUOTE the trace's own words that discharge it. Where several places
      share one reason exactly, group them and still name each, which is what the last
      numbered step asks of the count.

      MEASURED 2026-09-15 ACROSS TWO ROUNDS OF ONE DRAW, twenty cells each, same problems,
      same policy. Two entries collapsed. The one that marks an abandoned path went from
      seven insertions across seventeen cells to ZERO, with every cell declaring it had no
      place; the one that asks for a verification a reader could carry out went from
      forty-four to thirty, two of its cells falling from four and three to zero. Every
      reason given was a single sentence over a whole file.

      ONE OF THEM WAS READ BY HAND AND THE PLACE WAS REAL: the trace runs a capacity count,
      breaks off mid-clause on "But high degree min imposes H>=3", and the next paragraph
      opens a different approach without a word about the first. It was passed over as "each
      failed commitment is named in the line above the switch".

      A DISQUALIFIER IS NARROWER THAN A HABIT. That an entry's places are USUALLY discharged
      in some way is not a reason to discharge this one, and a pass whose count went to zero
      on an entry the previous pass used is a pass that applied a habit rather than a test.

  this-file-is-all-there-is
      Never go looking for the problem, its source, or its answer. Not on the web, not in any
      file outside `trajectory.md`, not anywhere. The problem, the reasoning and the response
      in front of you are the whole of what this work is done from, and the first page of this
      brief states the same ban.

      `trajectory.md` AND ITS OWN GIT HISTORY ARE THE FILE, and reading them is the work rather
      than an exception to this rule: the numbered steps ask for `git show`, `git log` and
      `git diff` on this repository and they mean it. Ordinary work in this field is yours
      too -- a known result, an identity, a check you can carry out yourself, a small case
      worked by hand -- the same licence the behaviour entries give. What is refused is finding
      out something about THIS PROBLEM that the file does not contain.

      AND WHERE THIS PASS REMOVES ANYTHING, IT IS THE SAME RULE THE TRACE IS BEING EDITED FOR.
      One of the removal families takes out exactly the sentences where the model appealed to
      having seen the problem rather than to anything on the page. An editor who looks up what
      the model was half-remembering has done the same thing one level up, and has done it to
      a file that is about to teach a student. THE RULE HOLDS WHETHER OR NOT THIS PASS REMOVES
      ANYTHING: it is about where YOUR sentences come from, not about what may come out.

      IT IS MEASURED AND IT CHANGED AN EDIT. One cell of a twenty-cell round put the problem
      statement in quotes into a search, found the contest and year, read the answer choices
      off a forum page, and used what it found to decide NOT to add a correction it had added
      to the same trace under an earlier brief. Nothing in the file or the commit log showed
      it; it was in the closing message.

      WHAT TO DO INSTEAD, when you find yourself wanting to look. Say in the message that the
      question could not be settled from the file, and leave the trace as it is. An
      unanswered question recorded is worth more here than an answer fetched from outside,
      because the whole comparison this work feeds rests on both arms having seen the same
      thing and nothing else.

  one-writer-throughout
      After every edit, read the place you changed together with the paragraph before it
      and the paragraph after it. The three must read as one person thinking,
      continuously.

      Keep the person and the number the trace already uses. If it says "we" throughout,
      your sentence says "we"; if it says "I", yours says "I"; if it is impersonal,
      yours is. Keep its tense.

      Keep its register. A terse trace that closes a step with four words does not
      suddenly acquire "Let us now proceed to verify the following claim".

      Keep its notation, whatever the notation is: the way it writes fractions, the way
      it names variables, the way it quotes a result, the way it spells an identifier.
      Match what is already on the page rather than what you would have chosen.

      AND KEEP THE MANNER, WHICH IS WHERE LENGTH COMES IN. A trace that reasons in
      clipped working notes and a trace that writes in full paragraphs are two
      different writers, and a passage in the wrong one of those reads as a second
      hand however correct it is. Read a few sentences around where you are writing
      and match how that person writes.

      LENGTH IS NOT A LIMIT AND A LONG CHECK IS NOT A FAULT. A substitution that has
      to carry three lines of working carries them. A chain of four deductions is
      four steps long. What a trace of clipped notes does with a long piece of work
      is break it into the short units it uses everywhere else, so write it that way
      rather than writing less of it. Where the check genuinely needs a long
      sentence, write the long sentence.

      AND THE UNIT IS THE TRACE'S, WHICH IS A THING TO CHECK BEFORE YOU COMMIT. Read
      the sentences around your insertion point and your own beside them. If yours
      run several times theirs, you have written in a different hand, and the work
      is to break it into their units rather than to cut it down. MEASURED over six
      cells: the inserted sentences ran three to seven times the length of the
      trace's own in every one of them, and the longest were single sentences of
      fifty-six and sixty-three words chained on semicolons, put beside traces
      whose own sentences run three to seven words. Every one of those passages was correct
      and every one of them is legible as an addition.

      A passage that is correct and reads as a different voice is worse than no passage.
      The student learns the seam.

  the-response-follows-the-reasoning
      The RESPONSE must follow from the REASONING as you leave them. If your edits
      change what the reasoning concludes, change the RESPONSE to match, in the same
      commit. If they do not, leave the RESPONSE alone.

      This is the one case where the RESPONSE may be edited at all, and it is not
      optional: a trace whose reasoning reaches one value and whose response states
      another teaches the student that the two need not agree. It is also the only edit
      this brief allows outside the REASONING section, so a changed line in the RESPONSE
      has exactly one lawful cause and a changed line in the PROBLEM has none.

THE BEHAVIOURS
--------------
Each entry gives five things. The name it is recorded under, the two classifications
explained above, WHERE in the trace it belongs, what to INSERT there, and what NEVER to do
under it.

WHERE IS THE HARD HALF. The scale matters as much as the behaviour: a check that reaches one
result cannot catch a wrong premise, and a check on a premise placed in a planning passage
catches nothing at all. An entry's WHERE is a description of a place, and if the trace has no
such place then that behaviour has nothing to do here.

  logical-check-in-deduction   (exploit, scale global)
    WHERE   A step that established a NEW fact, in the sense the reading above gives that
            word. Not a step that restates the problem or names an intention, which are
            planning rather than establishing.

            ASK THE QUESTION OF EVERY SUCH STEP, AND ATTACH WHEREVER THE ANSWER IS YES.
            The question is whether the step rests on a premise that could have failed.
            Where it does, the step gets a check whether or not the text around it sounds
            sure, and it gets one however many other checks are already in the file. The
            steps that most often qualify are the ones a later part of the trace depends
            on -- a fact carried forward, one a case split rests on, one a chain of steps
            passes through -- but a step is not disqualified by nothing depending on it.

            WHERE MORE THAN ONE PREMISE OF A STEP COULD FAIL, EACH GETS ITS OWN INSERTION.
            One check of one premise does not cover the others, and a step resting on
            three fragile conditions is three places, not one.

            A STEP WHOSE PREMISES CANNOT FAIL GETS NOTHING, and a trace where few steps
            rest on anything fragile gets few insertions. That is the right answer for
            such a trace rather than a gap in the work. What is NOT a reason to pass over
            a step is that several checks are already in, that the trace is long, or that
            the premise resembles one already tested somewhere else: a premise tested at
            one step is untested at another, because it is a different step that could
            fail differently.
    INSERT  Immediately after that step, do one of two things and say what it showed.

              premise   name a premise the fact rests on and test whether it holds
              coverage  name a case the reasoning's own split does not cover

            PICK THE PREMISE THAT COULD ACTUALLY FAIL. A step usually rests on several at
            once, and testing one that cannot fail records nothing. Test the one whose
            failure would change what follows, and where more than one could fail, each of
            those gets its own insertion.

            Test it against one of three things and say which of the three you used. The line
            that established it. A condition the problem states. A case that settles it.

            TESTING A PREMISE AGAINST THE LINE THAT ESTABLISHED IT IS NOT CIRCULAR, and the
            rule forbidding a result to be checked against the line that produced it is about
            the other case. A result checked against its own derivation is that derivation
            agreeing with itself. A premise is not what the step derived. It is a condition
            the step needed, and the line that secured that condition is exactly where to
            look.
    EXAMPLE "This divides by (x - 2), which needs x != 2; the line above fixed x = 5,
            so it holds."
    EXAMPLE "The split covers even n >= 2 and odd n >= 3; n = 1 falls outside both
            branches."
    NEVER   Do not redo the step's own working to see whether it comes out the same.
            Repeating the working cannot catch a wrong assumption, which is what this
            behaviour is for, and a separate behaviour covers the working itself.

            Do not place this in a planning passage. Do not name a premise that cannot
            fail. Do not write "which holds here" with nothing after it.

  checkable-verification   (exploit, scale local)
    WHERE   Two kinds of place.

            (a) A candidate answer, or an intermediate result the trace accepts
                and moves on from.

            (b) A check the trace makes without doing it. Sometimes it announces one --
                "let me verify", "double-check" -- but far more often it is a bare verdict
                closing a step: "Yes.", "Same.", "matches.", "Fine.", "OK", a question
                mark answered by an impression.

                The test is not whether working is nearby; a check can be full of working
                and still redo the very thing it is checking. Ask instead: could this have
                come out any other way. If nothing in it could have failed, that is the
                place.

            EVERY SUCH PLACE GETS ONE, AND BOTH KINDS COUNT. A trace that closes twenty
            steps on a bare verdict has twenty places of kind (b), and twenty is then the
            right number; a trace that accepts eleven intermediate results and moves on
            has eleven of kind (a). The finding this behaviour rests on is that three
            quarters of the reflection steps in failed traces are surface validations that
            leave the error untouched, so a pass that repairs one of twenty leaves the
            habit the student learns exactly as it was.

            THREE THINGS DISQUALIFY A PLACE AND NOTHING ELSE DOES. The trace itself checks
            that result somewhere later, against something other than the working that
            produced it. Or the clause immediately before a verdict of kind (b) already
            gives the reason, so a check there would only repeat it. Or the only check
            available would redo the same working, which the NEVER below refuses. A place
            passed over for one of these is named in the message you finish with, with
            which of the three it was.

            THE WORKING SITTING JUST ABOVE A RESULT DISQUALIFIES NOTHING, and a place of
            kind (a) always has it there. That is what the result was derived from, and a
            check against it is the derivation agreeing with itself, which the INSERT
            below refuses and sends to the ORIGIN instead. Reading it as a reason to pass
            over would excuse every place of kind (a) there is.

            AND THE LAST CANDIDATE IN THE TRACE IS NOT A PLACE OF KIND (a). Where the
            trace reaches a candidate and stops, another behaviour covers that ending
            and its INSERT asks for the checks this entry would ask for and more of
            them, so one passage goes there rather than two. If no such behaviour is
            offered in this pass, the ending is this entry's like any other candidate.
            Every earlier candidate the trace puts forward and moves past is this
            entry's either way.
    INSERT  A verification a reader could carry out, by ONE of these routes, saying which:

              substitution   put the result back into the condition it has to satisfy and
                             show that it does
              constraint     check it against a requirement the problem states -- a range,
                             a type, a count, a format, a permitted set of values
              second route   reach the same result by a method independent of the one used

            CHECK IT AT THE ORIGIN, NOT AT THE LAST LINE. Follow the result back: what was
            it obtained from, and that from what, until you reach something the PROBLEM
            gives -- a quantity in the statement, a stated constraint, a definition, an
            input. Check against that.

              from a derived value   go back to the ORIGINAL form of the thing it came
                                     from, not the rearranged or simplified form
              from a count           count again from the problem's own objects
              from a rule            check the rule's own condition on this input, then
                                     check the result on the smallest case whose answer is
                                     known without it
              from a chain           go to the first step that used a given quantity and
                                     work forward

            The smallest case is nearly always available and nearly always skipped. n = 1,
            the empty set, a single term, the degenerate case the problem still admits.

            Say which origin you reached. A check that names the condition it substituted
            into can be followed, and one that says only "verified" cannot.

            At a place of kind (b) the trace's own sentence stays and yours goes after it.
            If your check disagrees with what the trace concluded, say so and leave the
            conclusion standing.
    EXAMPLE "Substituting back into the original condition x^2 - 8m^2 = -7 gives 81 -
            88 = -7, so the pair satisfies it."
    NEVER   Do not write "let me double-check" and then repeat the same working. Do not
            end without saying what the check found. Do not check a result against the
            line that produced it -- that is the derivation agreeing with itself. Do not
            reword or remove an announced check that went nowhere.

  deep-named-return   (exploit, scale global)
    WHERE   A place where the trace ALREADY abandons a path: it tries something, it does
            not work, and the next passage is doing something else. These are usually
            unmarked; the trace just switches.

            EVERY SUCH SWITCH GETS ONE. A trace that abandons six paths has six places.
            Being unmarked is what makes them easy to miss and is not a reason to take
            only the clearest: the ones worth naming are exactly the ones the trace
            slid past without saying anything.

            TWO THINGS DISQUALIFY A SWITCH AND NOTHING ELSE DOES, AND BOTH ARE NARROW.

              the trace says it   The trace's own text already names the assumption it is
                                  giving up AND why, so your sentence would repeat it. QUOTE
                                  THE WORDS in the message. A sentence that only states the
                                  obstacle is not this, and neither is one that names a
                                  constraint and trails off: "But high degree min imposes
                                  H>=3" says what blocks the route and says nothing about
                                  leaving it, so that switch is still a place.

              nowhere to go back to   The commitment and its failure are THE SAME MOVE, so
                                  there is no earlier point to name. This is about the
                                  DISTANCE and not about whether the reason is nearby -- the
                                  question is whether an earlier commitment exists to rewind
                                  to, not whether the failure is visible next to the switch.

            A place the trace RESUMES LATER never qualified in the first place: the WHERE
            wants a path abandoned, and one taken up again was not. Say that as not a place
            rather than as a disqualifier, because they are different facts.

            A switch passed over is named in the message you finish with, with which of the
            two it was and, for the first, the trace's own words that discharge it.

            MEASURED 2026-09-15 AND THIS WORDING IS THE REPAIR. Under the looser phrasing
            this entry attached SEVEN times across seventeen cells in one round and ZERO
            times in the next, with all seventeen declaring no place. Seven of those were
            the same switches the earlier round had named. One was read by hand: the trace
            runs a capacity count, breaks off mid-clause on "But high degree min imposes
            H>=3", and the next paragraph opens a different approach with no account of the
            first. That is a place, and it was passed over as "the failed commitment is
            named in the line above".

            A SWITCH IS OFTEN A DROPPED LINE AS WELL. Another behaviour carries an
            abandoned approach two steps further instead of marking the switch, and
            where both fit the place takes both. That carry is written first, and the
            obstacle it reaches is then what this sentence names, so this one is
            written after it rather than instead of it.
    INSERT  At the switch, write a sentence that does three things.

              name      say which earlier point the reasoning is going back to, in the
                        trace's own words, so that a reader can find it
              give up   say which assumption made at that point is being abandoned
              replace   say what is being tried instead

            GO BACK FAR ENOUGH TO REACH THE ASSUMPTION THAT ACTUALLY FAILED. The point to
            name is the one where the abandoned path committed to something, which is
            usually several paragraphs above the place it ran out, not the line before it.
    EXAMPLE "Going back to where I assumed the three roots were distinct: that is what
            made the factorisation fail, so drop it and allow a repeated root."
    NEVER   Do not invent an abandoned path. If the trace never abandons anything, add
            nothing under this behaviour and say so. Do not rewind only to the previous
            line.

  continue-the-dropped-line   (exploit, scale global)
    WHERE   A PLACE WHERE THE TRACE NAMES A POSSIBILITY AND DEDUCES NOTHING FROM IT. Two
            shapes, and the second is the commoner one.

            (a) It opens an approach, spends a few lines on it, and moves to a different
                approach without having established anything from the first.

            (b) It raises a possibility and leaves it lying. No switch, no abandonment,
                just a line saying what might be true or what could be tried, and then the
                next line is about something else. A question asked and not answered, a
                reading of the problem offered and not settled, a candidate named and not
                tested, a formula written down and not used.

            THIS IS THE COMMONEST THING MISSING FROM THESE TRACES. They widen freely. They
            list readings, name approaches, ask whether something might hold, and then
            widen again. What they do not do is take one of those lines and carry it
            forward until it yields something. Wherever the trace has gone wide, the thing
            that is thin is the deduction, and that is what this behaviour adds.

            SO EVERY PLACE THAT QUALIFIES GETS ONE, AND THE COUNT FOLLOWS THE TRACE. If a
            trace names twenty possibilities and deduces nothing from twelve of them,
            twelve is the right number of insertions and not a sign of overdoing it.
            There is no point at which this behaviour has been used enough, because each
            place is a different line left lying and carrying one forward says nothing
            about the next.

            THREE THINGS DISQUALIFY A PLACE AND NOTHING ELSE DOES. The trace itself carries
            that line forward later, anywhere in the file. The line is answered within a
            sentence or two of being raised. Or carrying it needs a quantity or a fact the
            trace never has anywhere, so there is no chain to write. A place you pass over
            for any of these is named in the message you finish with, with which of the
            three it was.

            REACHING A NEW RESULT IS NOT A DISQUALIFIER, IT IS THE POINT. A carry that ends
            somewhere the trace never went is this behaviour working, not a breach of
            `write-from-what-is-established`. That rule is about the INPUTS: do not use a
            fact the trace establishes later than the place you are writing at. What the
            carry DERIVES from inputs already on the page is new by construction, and
            reading the rule the other way would disqualify every carry there is.

            WHERE A PLACE IS ALSO AN ABANDONMENT, THE CARRY GOES IN FIRST. Another behaviour
            marks a switch by naming what was given up, and shape (a) above is a switch too,
            so one place can take both. They are different work and neither replaces the
            other. The carry is written first, because what it reaches is what the naming
            then has to name: without it the switch is marked as abandoned for a reason
            nobody has established.

            NOT A DISQUALIFIER: that you have already added several, that the trace is
            long, that the place resembles one you have done, or that the chain is short.
            `density-is-not-the-signal` refuses an insertion where the WHERE does not
            apply and it refuses nothing where it does.
    INSERT  Carry the line forward until it yields something, and then let the trace go on
            exactly as it already does. Nothing is removed or reordered, and where there
            was a switch, the switch stays.

            TWO STEPS IS THE FLOOR AND THERE IS NO CEILING. One step is usually just the
            line restated. Where the chain needs six steps to reach the result or the
            obstacle, write six.

            EVERY STEP IS A CONSEQUENCE OF THE LINE ABOVE IT, and that is what makes it a
            deduction rather than another idea. Each one takes what is already established
            and derives the next thing from it, so the passage reads as a chain. This holds,
            thus that, thus that.

            WRITE THE CONNECTIVE, AND `thus` IS THE WORD FOR IT. `hence` and `therefore` do
            the same work. A step whose consequence is not stated reads as a fresh
            assertion however true it is, and a reader learns to assert.

            EACH STEP MUST ESTABLISH SOMETHING THAT WAS NOT THERE BEFORE, in the sense the
            reading above gives. Restating the approach in other words is not a step.
            Saying what one would do next is not a step. Naming another possibility is the
            opposite of a step here, and the rule `do-not-widen-a-wide-trace` refuses it.

            STOP AT A RESULT OR AT AN OBSTACLE, AND SAY WHICH YOU REACHED. Where the steps
            reach a result, say what it is. Where they run into the thing that made the
            trace leave the line, say what that thing is, because an obstacle that has been
            derived is a finding and it is a better reason to move on than the trace had.
    EXAMPLE "Before leaving the generating-function route: the coefficient of x^n in
            that product is the convolution sum, which for n = 4 gives 1 + 3 + 3 + 1 =
            8, so the route does reach a number, just not in closed form."
    EXAMPLE Shape (b), a possibility left lying. The trace says "maybe the
            discriminant being a perfect square helps" and moves on. Carried forward:
            "the discriminant is m^2 - 4n, so a perfect square needs m^2 - 4n = k^2,
            hence (m-k)(m+k) = 4n, and m-k and m+k have the same parity so both are
            even, which turns the condition into a divisor problem in n."
    NEVER   Do not propose yet another approach. Do not remove the switch that follows. Do
            not claim the continued line reaches the answer unless the steps you wrote
            actually get there.

  subgoal-when-absent   (structure, scale global)
    WHERE   A trace whose opening names WHAT has to be produced but not what has to be
            established first, and in what order. Structure needs both halves; restating
            the problem is not a plan, and neither is naming the method.

            If the opening does give the sub-goals and their order, add nothing here.
    INSERT  One insertion, as early in the reasoning as the paragraph rule allows,
            giving three things in this order. In practice that is after the trace's own
            opening sentence, which is the host line. ONE INSERTION IS NOT ONE SENTENCE:
            it is one passage at one place, written in as many sentences as the trace's
            own manner takes, and the entry below says what that means here.

              goal       what has to be produced
              sub-goals  what has to be established first for that to be reachable
              order      which sub-goal comes before which, and why that way round

            A SUB-GOAL SAYS WHAT MUST BE TRUE BEFORE THE NEXT STEP IS POSSIBLE, not what
            will be done. The difference is what makes the statement a plan rather than a
            list of tasks.

            This is a plan and not a summary of the solution, so it must not contain any
            result the trace has not established yet, and the rule named
            `write-from-what-is-established` applies to it in full. Length is whatever
            naming the goal and the sub-goals takes.

            IT IS WRITTEN IN THE TRACE'S UNITS LIKE EVERYTHING ELSE. Naming three sub-goals
            and their order in a trace of clipped notes is three or four short sentences,
            not one sentence with the order chained on semicolons. MEASURED: this entry
            produced the longest insertions in the round, single sentences of fifty-six and
            sixty-three words beginning "Order:", set beside traces that write
            "Indeterminate. Factor." The content was right and the hand was not the trace's.
    EXAMPLE A task: "find the radius". A sub-goal: "the radius is determined once the
            centre is known, so locate the centre first".
    NEVER   Do not add a second plan to a trace that has one. Do not restate the plan
            later. Do not restate the problem: a sub-goal says what must be TRUE before
            the next step is possible.

  closing-decision   (exploit, scale global)
    WHERE   The end of the trace: the point after which the reasoning stops and the
            response is given. Exactly one place per trace, and only if the trace actually
            reaches a candidate answer. A trace that runs out without concluding has no
            place for this.
    INSERT  A passage that does two things in order, and says it is doing them.

            First, decide that the work is finished, and say why it is finished rather
            than merely stopping.

            Second, verify the candidate before committing to it, by working backwards
            from it rather than forwards to it. All four of these, each naming what it
            found, and each at whatever length the check actually takes:

              conditions     put the candidate back into each condition the
                             problem states, one at a time, and report whether
                             each is satisfied
              second route   reach the same result by a route different from the
                             one that produced it
              consistency    name any condition that could conflict with another,
                             and say whether it does
              form           state what the problem asked for and confirm the
                             candidate is in that form, including any unit, any
                             required exactness, and whether one value or a set
                             of them was wanted

            Then commit to the answer explicitly.
    EXAMPLE "The problem asked for an exact value, and 7 + sqrt(17) over 2 is exact
            rather than a decimal, so the form is what was asked."
    NEVER   Do not close with a verification that could not have failed. "Checked, it is
            correct" is not a closing decision. Each of the four must name the thing it
            examined and report the outcome, including when the outcome is that something
            does not hold.

            Do not add this where the trace never reaches a candidate.

            Do not change the response here unless your verification found the reasoning
            reaches a different result, in which case the rule named
            `the-response-follows-the-reasoning` applies.

  catch-and-correct   (exploit, scale local)
    WHERE   A step whose result is wrong, where the error can be established from what the
            trace has already written plus ordinary work in this field. A slip in the
            working, one value used in place of another, a dropped term or condition, a
            sign or direction reversed, a value transcribed differently from the line it
            came from, a rule applied where its own condition does not hold.

            The error stays. This behaviour goes AFTER it.

            EVERY WRONG STEP THE TRACE'S OWN TEXT CAN CONVICT GETS ONE. Two wrong steps
            are two places. The one thing that disqualifies a wrong step is that the
            trace catches and repairs it itself later, and a step passed over for that
            is named in the message you finish with, with where the trace catches it.

            A WRONG STEP THAT REPEATS DOWN ONE CHAIN IS ONE PLACE, AND THE PLACE IS THE
            LAST APPEARANCE ANYTHING LATER USES. Put the correction earlier and every
            repetition after it stands uncorrected behind it, which is the fault this
            rule exists to avoid; put it after the last use and nothing downstream is
            resting on the wrong value any more. Repetitions later still than that use
            carry nothing and need nothing. WHERE THE TRACE USES THE WRONG VALUE IN
            SEVERAL PLACES THAT DO NOT FEED EACH OTHER, those are separate chains and
            each gets its own correction at its own last use.
    INSERT  Immediately after that appearance of the wrong step -- the last one anything
            later uses, which is the place the WHERE picks out -- a passage that finds the
            error and repairs it, in this order:

              redo     do the step again, showing the intermediate values
              compare  state that the result differs from what was written above, and how
              locate   say which operation produced the difference
              correct  give the corrected result and carry it forward, so that the
                       following text has something consistent to rest on

            Write it as the same person noticing their own mistake while checking, not as
            an annotation about the trace.
    EXAMPLE "Recomputing: 17 * 23 = 391, not 371 as written above -- the difference is
            20, which is the carry dropped in the tens column. Taking 391 forward."
    NEVER   Do not delete or amend the wrong step. It is the reason this passage exists,
            and a student that never sees an error being caught never learns to catch one.

            Do not introduce a quantity that does not appear above and cannot be obtained
            from what does by the work you show.

            Do not claim an error where the trace is right. If redoing the step agrees
            with what is written, this behaviour has no place there, though a behaviour
            that verifies a result against the problem may.

            Do not correct a step whose error you cannot establish from the trace itself.

            AND DO NOT GO AND FIND OUT WHETHER IT IS WRONG. This is the entry that tempts it, and
            it is the entry a round was measured failing on: an editor searched for the source
            problem to decide whether a correction was owed here, and declined one it had made to
            the same trace under an earlier brief. The rule is `this-file-is-all-there-is`. Where
            the trace's own text cannot convict a step, this behaviour has no place there, and
            saying so in the message you finish with is the whole of what is owed.

HOW TO WORK
-----------
1. Read the whole REASONING section before changing anything. It may be long. Read it in
   order, in pieces if you must, and keep a note of where each behaviour is missing.

2. Work through the file from the beginning to the end. Do not stop after the first few
   opportunities: the point of editing a file rather than rewriting a text is that the end of
   the trace is as reachable as the beginning.

5. When you reach the end, look at WHERE your insertions fell, as a number rather than an
   impression. Take each one's position as a fraction of the file's CHARACTERS, not of its
   lines, which run from two characters to several hundred here.

   THIS IS A SIGNAL AND NOT A TARGET. A trace whose weak steps really are in its first
   third SHOULD be edited in its first third, and a median of 0.2 is then the right answer.
   Do not move an insertion, invent one late in the file, or hold one back, to make the
   number look better -- that is editing the measurement instead of the trace.

   WHAT THE NUMBER IS FOR. A median far below the middle has two possible causes and they
   need different answers. Either the opportunities really are early, and nothing is wrong.
   Or the later part of the file was read less carefully than the opening, which is a
   measured failure of this method rather than a hypothetical one: in an early round 59% of
   133 insertions landed in the first quarter of the trace and 26% in the second half, over
   traces running from 7,216 to 251,745 characters.

   So when the median comes out low, ASK WHICH IT IS, and answer it by going back to the
   later part of the file and reading it again rather than by adding anything. If the second
   read finds places you had passed over, that was the second cause and they get their
   insertions. If it finds none, that was the first cause, the low median is correct, and you
   say so in the message you finish with.

   AND THE FAR END IS READ AGAIN WHATEVER THE MEDIAN SAYS. The median is the alarm and not
   the condition: a file whose end was skimmed can still come out near the middle, because
   the few insertions that ARE there sit late. The second read of the last third costs one
   pass and it is where the missed places have been every time this was measured. Say what
   it found, including when it found nothing.

6. Then account for the COVERAGE OF YOUR READING, which the positions do not show.

   Some behaviours above attach to a KIND OF STEP -- their WHERE names the step, not the
   trace. For each of those, say how many steps of that kind the trace has and how many you
   attached to, counted over the whole file rather than the part you remember, in the message
   you finish with.

   THE TWO NUMBERS ARE A RECORD, NOT A RATIO TO HIT. There is no right proportion. A trace
   with forty such steps may need three insertions if thirty-seven of them are already
   checked, and a trace with four may need all four. What the pair of numbers says is that
   you counted, which is the thing that cannot be recovered afterwards from the file.

   WHAT IT IS FOR. On an early round the number of step-attached insertions stayed between
   8 and 16 whatever the trace's length, which is 118 per ten thousand characters on the
   shortest trace and 0.9 on the longest. A count that does not move with length is
   evidence that the far end of a long file was skimmed, and the answer to that is to read
   the far end again. It is NOT to add insertions until the count moves.

   `density-is-not-the-signal` STILL HOLDS AND IT WINS. Do not attach a behaviour where its
   WHERE does not apply, and do not repeat one to raise a number. Where reading again finds
   nothing, the low count is the right answer and saying so is the whole of what is owed.

7. Finish with a message. It is not a commit, it changes no file, and it is where the
   accounting above reports. Four headings, and then whatever the behaviour entries
   themselves asked you to report here:

     positions  the median position of your insertions as a fraction of the file's
                characters, and, if it came out low, which of the two causes it was and
                what reading the later part again found. THE MEDIAN IS OVER EVERY
                INSERTION YOU MADE: list them, one per insertion, and say how many,
                so the number can be put beside the log's own count. A median over
                a subset is a different measurement wearing this one's name.
                AN INSERTION IS A PASSAGE YOU ADDED, so a commit that only takes
                text out is not one of these and neither is a `fix:`. Say how many
                of each you made, on three lines, because the log holds all three
                kinds and a single number cannot be matched against it

     coverage   for each behaviour whose WHERE names a kind of step, how many steps of
                that kind the trace has and how many you attached to. AND EVERY STEP OF
                THAT KIND YOU DID NOT ATTACH TO IS NAMED HERE, by line, with the reason
                it needs nothing. The count on its own cannot be checked by anyone and
                the reasons can. A single sentence standing in for a dozen steps is a
                summary rather than a reason; where several share one reason exactly,
                group them, say how many are in the group, and name each by line anyway.
                HOW MANY INSERTIONS THERE SHOULD BE IS THE PREVIOUS STEP'S QUESTION and
                its answer may well be one of twelve. What this line adds is that the
                other eleven were each looked at

     kept       every passage you located as out of scope and left in place anyway, named,
                with what the doubt was. A located-and-kept passage leaves no mark in the
                file, so this line is the only place it can be recorded

     voice      the manner the trace writes in, and the manner you wrote in, named side by
                side. Say it as the trace's own typical sentence against yours: clipped
                working notes, or full sentences, and about how long each runs. If they
                differ, say which insertions differ and why you left them that way. A
                passage in the wrong manner reads as a second hand however right it is,
                and unlike a wrong value nothing downstream will ever catch it

   AND SEVERAL ENTRIES ABOVE SEND SOMETHING HERE, WHICH IS NOT OPTIONAL AND IS NOT
   COVERED BY THE FOUR. Wherever an entry says a place passed over is "named in the
   message you finish with", that naming belongs under `coverage`, beside the count,
   with which of that entry's own disqualifiers it was. Wherever an entry says a
   passage located and left is reported, that belongs under `kept`. And where an
   entry asks for something neither heading holds -- a trace whose route to the answer
   is a memory of the answer is the one that exists today -- write it as its own
   heading and name it after what it says. A requirement from an entry does not
   disappear because this list was written before it.

   AND "ENTRY" HERE MEANS ANYWHERE IN THIS BRIEF, not only the behaviour entries. The
   first screen asks for a question you could not settle from the file to be recorded
   here, and the rule named `this-file-is-all-there-is` asks the same; neither is a
   behaviour entry and both land here. If something in this brief told you to say it in
   the message, say it in the message.

   A NUMBER YOU DID NOT MEASURE IS WORSE THAN NO NUMBER HERE, because the point of every
   one of them is that it was counted. Where you cannot count something over the whole
   file, say that rather than estimating it.

WHEN A BEHAVIOUR DOES NOT FIT
-----------------------------
Some traces give no opportunity for some behaviours. A trace that never abandons a path has no
place for a named return, and inventing one would make the trace false. If a behaviour has no
honest place in this trace, add nothing for it and record that with an empty commit:

An empty commit is a real answer and it is worth making. Without it a later reader cannot
tell a behaviour that had no place from a behaviour that was overlooked.

THE MODE THIS PASS RUNS IN
--------------------------
All behaviours in one pass. Go through the file once, and at each point insert whichever of
the behaviours above belongs there. A single pass, every behaviour available at every
point.

WHAT MAY BE REMOVED
-------------------
NOTHING LEAVES THE TRACE UNDER THIS POLICY. No behaviour in this brief asks to take text
out, so no words come out of it. Two kinds of edit still show a deleted line in a diff
without removing anything, and both are named at the end of this section.

A DEAD END IS EVIDENCE AND SO IS AN EMPTY CHECK. Both say how this trace reasoned, and
that is what is being studied. A step that came out wrong stays where it is, and so does
a check that found nothing. Where a behaviour above asks for something beside one of
them, it goes beside it and the original stays.

THE SHAPE OF YOUR REPLY
-----------------------
Your reply is exactly three sections, each opened by its own line, in this order, and
nothing before the first:

    ===== REASONING =====
    <the whole REASONING section as you leave it>
    ===== RESPONSE =====
    <the whole RESPONSE section as you leave it>
    ===== NOTES =====
    <what the brief asks the message you finish with to hold>

The PROBLEM section is not repeated. A section line is never written anywhere but at the
start of its section.
