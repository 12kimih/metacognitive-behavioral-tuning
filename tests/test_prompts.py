import hashlib

from mbt import prompts


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_prompts_are_the_ones_the_paper_sent():
    assert sha(prompts.mbt_a_brief()) == prompts.RECORDED_SHA256["mbt_a_brief"]
    assert sha(prompts.one_call_identity_text("rewrite")) == prompts.RECORDED_SHA256["mbt_r"]
    assert sha(prompts.one_call_identity_text("insert")) == prompts.RECORDED_SHA256["mbt_i"]


def test_every_strategy_names_the_same_seven_behaviors():
    for text in (prompts.mbt_a_brief(), prompts.one_call_instructions("rewrite"),
                 prompts.one_call_instructions("insert")):
        for name in prompts.BEHAVIORS:
            assert name in text


def test_the_one_call_prompts_say_nothing_about_git_commits():
    for method in ("rewrite", "insert"):
        assert "git commit -m" not in prompts.one_call_instructions(method)


def test_the_generation_prompt_is_figure_d1():
    assert prompts.generation("What is 1+1?") == (
        "What is 1+1?\n\nPlease reason step by step, and put your final answer within "
        "\\boxed{}.\n")


def test_the_prompt_files_are_ascii():
    for name in ("mbt_a_brief.md", "mbt_r_instructions.md", "mbt_i_instructions.md",
                 "deletion_instructions.md", "generation.txt",
                 "annotation/undesirable_behaviors_codebook.md",
                 "annotation/target_behaviors_codebook.md",
                 "annotation/undesirable_instructions.txt",
                 "annotation/target_instructions.txt"):
        prompts.text(name).encode("ascii")
