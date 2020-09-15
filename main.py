from caseData import caseData
from model import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = caseData()
    datasetNew = dataset.newCaseArray()
    dataModel = caseModel(datasetNew)
    dataModel.train()
    x = dataModel.predict(30)
    for entry in x:
        datasetNew.append(entry)
    plt.plot(datasetNew)

    plt.ylabel('vic cases')
    plt.show()

if __name__ == "__main__":
    main()

