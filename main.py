"""
__author__ = "Francesco Cannarile"
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""


import argparse
from utils.config import *
from agents import *

def main():
    arg_parser = argparse.ArgumentParser(description = 'Configuration path')
    arg_parser.add_argument('config', help = 'The Configuration file in json format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    print(config.agent)
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()
    
if __name__ == '__main__':
    main()

# python RNN_Autoencoder/main.py RNN_Autoencoder/configs/config_rnn_ae.json