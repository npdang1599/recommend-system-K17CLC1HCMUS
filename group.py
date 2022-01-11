import numpy as np


class Group:
    def __init__(self, members, candidate_items, ratings):
        # member ids
        self.members = sorted(members)
        
        # List of items that can be recommended.
        # These should not have been watched by any member of group.
        self.candidate_items = candidate_items
        
        self.ratings_per_member = [np.size(ratings[member].nonzero()) for member in self.members]

        self.grp_factors_bf = []
        self.bias_bf = 0
        self.reco_list_bf = []
        

@staticmethod
def find_candidate_items(ratings, members):
    if len(members) == 0: return []

    unwatched_items = np.argwhere(ratings[members[0]] == 0)
    for member in members:
        cur_unwatched = np.argwhere(ratings[member] == 0)
        unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)

    return unwatched_items
Group.find_candidate_items = find_candidate_items
