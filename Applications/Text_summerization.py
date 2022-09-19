from datasets import list_datasets, load_dataset
from pprint import pprint

datasets_list = list_datasets() 
pprint(datasets_list,compact=True) 
dataset = load_dataset('cnn_dailymail', '3.0.0')

print("dataset", dataset)


