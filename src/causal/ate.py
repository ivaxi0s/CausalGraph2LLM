import dowhy
from src.data.data_generation import generate_dataset
# from dowhy import CausalModel
from networkx.drawing.nx_pydot import to_pydot
from causalinference import CausalModel



def calc_ate(args, codebook, save_f):
    
    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif', n=args.n)
    P = to_pydot(true_G)

    # Save as .dot file
    P.write_raw('graph.dot')

    cm = CausalModel(
    Y=data['asia'].values, 
    D=data['dysp'].values, 
    X=data.drop(['asia', 'dysp'], axis=1).values
    )

    # Estimate the causal effect
    cm.est_via_ols()

    # Print the estimated causal effect
    print(cm.estimates)
    data = data.drop(columns=['tub'])

    cm = CausalModel(
    Y=data['asia'].values, 
    D=data['dysp'].values, 
    X=data.drop(['asia', 'dysp'], axis=1).values
    )

    # Estimate the causal effect
    cm.est_via_ols()

    # Print the estimated causal effect
    print(cm.estimates)

    model= CausalModel(
            data=data,
            treatment='asia',
            outcome='dysp',
            graph="graph.dot" # replace with your actual graph
            )
    
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print("ATE with all variables: ", estimate.value)
    breakpoint()
    true_G.remove_node('bronc')
    data = data.drop(columns=['bronc'])
    P = to_pydot(true_G)

    # Save as .dot file
    P.write_raw('graph.dot')

    model= CausalModel(
            data=data,
            treatment='asia',
            outcome='dysp',
            graph="graph.dot" # replace with your actual graph
            )
    
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print("ATE with all variables: ", estimate.value)

    breakpoint()
