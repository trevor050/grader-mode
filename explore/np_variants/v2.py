exec(open("np_variants/v1.py").read())
VARIANTS = {k: VARIANTS[k] for k in ["rl_full", "no_human_bullet", "not_a_test_note", "real_classroom_sys", "teacher_autograder", "support_bot_rl"]}
