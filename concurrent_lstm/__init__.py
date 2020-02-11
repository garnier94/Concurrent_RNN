from path import Path
import configparser

DATA_CONCURRENCE =  Path(__file__).parent /  '../data'
RESULTS =  Path(__file__).parent /  '../results'
CONFIG_CONTAINER = configparser.RawConfigParser()
CONFIG_CONTAINER.read(Path(__file__).parent / '../parameters.cfg')