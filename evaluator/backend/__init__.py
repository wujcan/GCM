# import eval_score_matrix_loo
try:
    from evaluator.backend.cpp.evaluate_loo import eval_score_matrix_loo
    print("eval_score_matrix_loo with cpp")
except:
    from evaluator.backend.python.evaluate_loo import eval_score_matrix_loo
    print("eval_score_matrix_loo with python")
