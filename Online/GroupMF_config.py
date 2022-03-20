import configparser
#Configuration reader.

class Config:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

        configParser = configparser.RawConfigParser()
        configParser.read(config_file_path)

        self.movie_type = configParser.get('Config', 'type_filter')

        self.max_iterations_mf = int(configParser.get('Config', 'max_iterations_mf'))
        self.lambda_mf = float(configParser.get('Config', 'lambda_mf'))
        self.learning_rate_mf = float(configParser.get('Config', 'learning_rate_mf'))
        
        self.num_factors = int(configParser.get('Config', 'num_factors'))
        
        #BF (before factorization)
        self.rating_threshold_bf = float(configParser.get('Config', 'rating_threshold_bf'))
        self.num_recos_bf = int(configParser.get('Config', 'num_recos_bf'))
        
        self.is_debug = configParser.getboolean('Config', 'is_debug')