import numpy as np
import pandas as pd
from dowhy import CausalModel
import dowhy.datasets
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from dowhy import gcm
from networkx.drawing.nx_pydot import write_dot
import dowhy.causal_estimators.linear_regression_estimator

import os, json



def mediation(original_graph, data, all_mediators, treatment, outcome):
    mediation_dict = {}
    for each_mediator in all_mediators:
        mediation_dict[each_mediator] = indv_mediation_analysis(each_mediator, original_graph, data, treatment, outcome)
    
    return mediation_dict

def indv_mediation_analysis(mediator, true_G, data, treatment, outcome):
    found_mediator = False
    # abc = indv_mediation_analysis('either', original_graph, data, args.treatment, args.outcome)
    i = 0 
    while (not found_mediator):
        i+=1
        write_dot(true_G, 'graph.dot')
        model = CausalModel(data, treatment, outcome, graph = "graph.dot")
        identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde", 
                                                    proceed_when_unidentifiable=True)
        identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie",
                                            proceed_when_unidentifiable=True)

        identified_mediators_nde = identified_estimand_nde.get_mediator_variables()
        identified_mediators_nie = identified_estimand_nie.get_mediator_variables()
        if mediator in identified_mediators_nde and mediator in identified_mediators_nie:
            found_mediator = True
            causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
                                                    method_name="mediation.two_stage_regression",
                                                confidence_intervals=False,
                                                test_significance=False,
                                                    method_params = {
                                                        'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                                                        'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                                                    }
                                                )

            causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
                                                    method_name="mediation.two_stage_regression",
                                                confidence_intervals=False,
                                                test_significance=False,
                                                    method_params = {
                                                        'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                                                        'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                                                    }
                                                )
            print(causal_estimate_nde)
            print(causal_estimate_nie)

            return causal_estimate_nde.value, causal_estimate_nie.value, causal_estimate_nie.value/causal_estimate_nde.value
                                                

def ranked_mediation_nodes(args, save_f):
    filen = f'{args.treatment}_{args.outcome}.json'
    save_f = os.path.join(save_f, "mediation_analysis")
    out_file = f'{save_f}/{filen}'

    with open(out_file, 'r') as file:
        data = json.load(file)

    sorted_keys = sorted(data.keys(), key=lambda k: data[k][-1])

    return sorted_keys

