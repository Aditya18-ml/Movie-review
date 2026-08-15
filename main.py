from train_test import trainmodel
from eda import runeda
if __name__ == "__main__":
    print('---eda---')
    runeda()
    print("Starting pipeline execution...")
    trainmodel()