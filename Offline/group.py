import numpy as np
import configparser

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

#Configuration reader.
class Config:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

        configParser = configparser.RawConfigParser()
        configParser.read(config_file_path)


        self.max_iterations_mf = int(configParser.get('Config', 'max_iterations_mf'))
        self.lambda_mf = float(configParser.get('Config', 'lambda_mf'))
        self.learning_rate_mf = float(configParser.get('Config', 'learning_rate_mf'))
        
        self.num_factors = int(configParser.get('Config', 'num_factors'))
        
        #BF (before factorization)
        self.rating_threshold_bf = float(configParser.get('Config', 'rating_threshold_bf'))
        self.num_recos_bf = int(configParser.get('Config', 'num_recos_bf'))
        
        self.is_debug = configParser.getboolean('Config', 'is_debug')

@staticmethod
def find_candidate_items(ratings, members):
    if len(members) == 0: return []

    unwatched_items = np.argwhere(ratings[members[0]] == 0)
    for member in members:
        cur_unwatched = np.argwhere(ratings[member] == 0)
        unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)

    return unwatched_items
Group.find_candidate_items = find_candidate_items

