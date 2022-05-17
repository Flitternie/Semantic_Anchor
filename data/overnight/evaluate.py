import numpy as np
from data.overnight.evaluator.domain_base import Domain

overnight_domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

def evaluate(args, outputs, targets, all_domains, *xargs):
    assert len(outputs) == len(targets)
    data = [[[],[]] for _ in range(len(all_domains))]
    evaluators = [Domain.from_dataset(domain) for domain in overnight_domains]
    for p, g, d in zip(outputs, targets, all_domains):
        data[d][0].append(p)
        data[d][1].append(g)
    scores = []
    for i, evaluator in enumerate(evaluators):
        domain_score = evaluator.compare_logical_form(data[i][0], data[i][1])
        scores += domain_score
        print("{}-domain accuracy: {}".format(overnight_domains[i], np.mean(domain_score)))
    return np.mean(scores)