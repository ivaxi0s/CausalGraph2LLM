import bnlearn as bn
from src.data.data_generation import generate_dataset
from src.utils import calculate_shd 


def pc_on_obs(args, codebook, save_f):
    true_G, data = generate_dataset('_raw_bayesian_nets/' + args.dataset + '.bif', n=args.n)  
    model_learned = bn.structure_learning.fit(data, methodtype=args.causal)
    shd = calculate_shd(true_G, model_learned)
    print(model_learned)
    print(shd)

