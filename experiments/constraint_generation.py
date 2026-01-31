"""
We basically have two approaches to test the constraint generation
easily for various configurations.

1. First one uses the CLI, by just calling the train command for all
the paramn configurations ...
"""


# Frage von Oli: Wie kriege ich das Programm dazu dort anzukommen wo mein
# Wissen bezüglich cli anfängt?

import subprocess

from pathlib import Path

"""

EXPERIMENTS_DIR = Path(__path__).parent.resolve()

PARAMETERS = ParameterGrid({'a': [1, 2]})

for params in PARAMETERS:
    sorted_params = sorted(params.keys())
    params_str = '-'.join(f'{k}_{params[k]}' for k in sorted_params)

    model_file = EXPERIMENTS_DIR / f'{params_str}.pbl_model'

    train_images = ...

    command = f'pbl train ... {train_images} {model_file}'

    subprocess.run(command)


"""
# 2. The alternative is to build the graph manually just using the classes
# etc. and than optimizing everything
"""

graph =
"""

# if test:
#     opt = ConstraintGeneration(criterion=criterion)
#     graph.weights = np.ones(graph.n_potentials)
#
#     # the graph optimization, that might take some time
#     result = opt.optimize(graph, train_images,
#                           val_images=validation_images,
#                           batch_size=8,
#                           max_time=learning_max_time,
#                           working_dir=working_dir)
#
#     path = Path("C:/Users/ofritsche/Dropbox/Arbeit")
#     f = open("C:/Users/ofritsche/Dropbox/Arbeit/debug.txt", "w")
#     f.write("nothing")
#     f.close()
#
#     copy2(working_dir / 'metrics.pdf',
#           path / f'test.pdf')
# else:
#     count = 0
#     val_rel_error_list = []
#     corresponding_parameterlist = []
#
#     filepath = "C:/Users/ofritsche/Dropbox/Arbeit/Test 7/"
#
#     for params in ParameterGrid({'slack_factor': [1000, 10000],
#                                  'loss_function': [1],
#                                  'margin_defining': [1, True],
#                                  'optimization_accuracy': [1e-9],
#                                  'infinity_replacement': [1e6],
#                                  'constraint_appending': [True],
#                                  'allow_duplicates': [True],
#                                  'solver': [1],
#                                  'remote': [False],
#                                  'starting_value': [0.5],
#                                  'dynamic_starting_value': [False],
#                                  'loss_multiplier': [1 / 100],
#                                  'error_red_func': [1],
#                                  'deletion': [False, 0, 1]}):
#         cg_slack_factor = params['slack_factor']
#         cg_loss_function = params['loss_function']
#         cg_optimization_accuracy = params['optimization_accuracy']
#         cg_infinity_replacement = params['infinity_replacement']
#         cg_allow_dup = params['allow_duplicates']
#         cg_solver = params['solver']
#         cg_remote = params['remote']
#         cg_starting_value = params['starting_value']
#         cg_d_starting_v = params['dynamic_starting_value']
#         cg_loss_multiplier = params['loss_multiplier']
#         cg_error_red_func = params['error_red_func']
#
#         # complicated assignment. I define new help-variables to assign
#         # [True, |R]
#         cg_h_constraint_appending = params['constraint_appending']
#         # [|R, True]
#         cg_h_margin_defining = params['margin_defining']
#         #
#         cg_h_deletion = params['deletion']
#
#         if isinstance(cg_h_constraint_appending, bool):
#             cg_append_all_constraints = True
#             cg_cnab = -1
#         else:
#             cg_append_all_constraints = False
#             cg_cnab = cg_h_constraint_appending
#
#         if isinstance(cg_h_margin_defining, bool):
#             cg_dynamic_margin = True
#             cg_margin = -1
#         else:
#             cg_dynamic_margin = False
#             cg_margin = cg_h_margin_defining
#
#         if isinstance(cg_h_deletion, bool):
#             cg_delete_constraints = cg_h_deletion
#             cg_cdb = -1
#         else:
#             cg_delete_constraints = True
#             cg_cdb = cg_h_deletion
#
#         if learning == Learning.MAXMARGIN_SGD:
#             opt = SgdMaxMarginLearning(criterion=criterion)
#         elif learning == Learning.FULL_MAXMARGIN_SGD:
#             opt = FullSgdMaxMarginLearning(criterion=criterion,
#                                            session=session)
#         elif learning == Learning.MAXMARGIN_CG:
#             opt = ConstraintGeneration(criterion=criterion,
#                                        slack_factor=cg_slack_factor,
#                                        loss_func=cg_loss_function,
#                                        margin=cg_margin,
#                                        opt_acc=cg_optimization_accuracy,
#                                        inf_rep=cg_infinity_replacement,
#                                        aac=cg_append_all_constraints,
#                                        cnab=cg_cnab,
#                                        allow_duplicates=cg_allow_dup,
#                                        solver=cg_solver,
#                                        remote=cg_remote,
#                                        d_margin=cg_dynamic_margin,
#                                        starting_value=cg_starting_value,
#                                        d_starting_value=cg_d_starting_v,
#                                        loss_multiplier=cg_loss_multiplier,
#                                        error_red_func=cg_error_red_func,
#                                        del_con=cg_delete_constraints,
#                                        con_del_barrier=cg_cdb)
#         else:
#             raise RuntimeError('will never happen')
#
#         graph.weights = np.ones(graph.n_potentials)
#
#         # the graph optimization, that might take some time
#         result = opt.optimize(graph, train_images,
#                               val_images=validation_images,
#                               batch_size=8,
#                               max_time=learning_max_time,
#                               working_dir=working_dir)
#
#         path = Path(filepath)
#
#         copy2(working_dir / 'metrics.pdf', path / f'{count}.pdf')
#
#         opt.parameter_string = "val-rel error: " + \
#                                str(result.metrics[REL_ERROR.val]) + \
#                                " \tin iteration" + result.iteration + \
#                                " \t" + opt.parameter_string
#
#         val_rel_error_list.append(result.metrics[REL_ERROR.val])
#         corresponding_parameterlist.append(opt.parameter_string)
#
#         f = open(filepath + "intermediate_result_.txt", "a")
#
#         if count == 0:
#             f.write("Optimization time: " +
#                     "{0:.2f}".format(learning_max_time / 3600) + " h \n")
#
#
#         def output(number, string):
#             return "Run " + number + "\t mit " + string
#
#
#         f.write(output(str(count),
#                        str(corresponding_parameterlist[count])))
#         f.close()
#
#         count += 1
#
#     sorted_indexlist = np.argsort(val_rel_error_list)
#     f = open(filepath + "end_result.txt",
#              "w")
#     f.write("Optimization time: "
#             + "{0:.2f}".format(learning_max_time / 3600) +
#             " h. Runs are sorted by val-rel error:\n")
#     print("Optimization time: " + str(learning_max_time / 3600) +
#           " Runs are sorted by val-rel error:\n")
#     for k in sorted_indexlist:
#         print(output(str(k), str(corresponding_parameterlist[k])))
#         f.write(output(str(k), str(corresponding_parameterlist[k])))
#
#     f.close()
#
# # graph.dump(model_file)
