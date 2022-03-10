import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas

class CellTypeClassifier(nn.Module):
    
    def __init__(self, h1=(2000,12), h2=(12,8), h3=(8,1)):
        super().__init__()
        self.hidden1 = nn.Linear(*h1)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(*h2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(*h3)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        return self.act_output(
            self.output(
                self.act2(
                    self.hidden2(
                        self.act1(self.hidden1(x))))))


class CellTypePrediction:

    def __init__(self, raw_dataset, ycol, layer_nodes, 
                 n_epochs=100, batch_size=1000,
                 multiclass=False, lossplot=True):
        
        index_of_ycol = raw_dataset.columns.tolist().index(ycol)
        self.raw_dataset = raw_dataset
        self.model = CellTypeClassifier(*layer_nodes)

        if multiclass:
            ## Use a softmax as output layer?
            dataset = self.__makeTensor_OneHotEncoding(ycol)
        else:
            ## Use a sigmoid as output layer?
            dataset = self.__makeTensor_LabelEncoding(ycol)

        dataset = dataset.to_numpy()
        self.X = torch.tensor(np.delete(dataset, index_of_ycol, axis=1), 
                              dtype=torch.float32)
        
        self.Y = torch.tensor(dataset[:,index_of_ycol], 
                              dtype=torch.float32).reshape(-1,1)

        self.doTraining(n_epochs, batch_size)
        if lossplot:
            self.plotLoss()

    ## Public
    def determineAccuracy(self, message=True):
        """Determine how well the training data is predicted"""
        mean_accuracy = (self.model(self.X).round() == self.Y).float().mean()
        if message:
            print(f"Mean Accuracy: {mean_accuracy}")
        else:
            return(float(mean_accuracy))

    def doTraining(self, n_epochs=100, batch_size=1000):
        loss_fn   = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        Loss = [] 

        for epoch in range(n_epochs):
            for i in range(0, len(self.X), batch_size):
                Xbatch = self.X[i:i+batch_size]
                y_pred = self.model(Xbatch)
                ybatch = self.Y[i:i+batch_size]
                loss = loss_fn(y_pred, ybatch) ## diff: O - E
                Loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Finished epoch {epoch}, latest loss {loss}', end="\r")
        self.loss = Loss
        
    def plotLoss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss)
        plt.xlabel("no. of iterations")
        plt.ylabel("total loss")
        plt.show()    

   
    def doLabelPrediction(self, test_dataframe):
        """Grab a test dataframe of Pseudocells and perform prediction"""
        self.test_X = torch.tensor(
            test_dataframe.to_numpy(), 
            dtype=torch.float32)
        
        predictions = [self.ctypes_int[int(x[0])]
                       for x in (self.model(self.test_X) * len(self.ctypes_int))
                       .floor().tolist()]
        
        df = pandas.DataFrame()
        got_right = 0
        for i in range(len(test_dataframe)):
            psd = test_dataframe.iloc[i].name
            prd = predictions[i]
            df = pandas.concat([
                df, 
                pandas.DataFrame({
                    "Pseudo": psd, "Predicted": prd
                }, index=[0])], axis=0)
            got_right += 1 if (psd == prd) else 0
        score = int(100 * got_right / len(test_dataframe))
        return({"score_perc" : score, "result": df})
    
    ## Private
    def __labelEncodingMaps(self, ycol="Celltype"):
        """Defines mappings between cell type labels and integers."""
        ctypes = self.raw_dataset[ycol].unique().tolist()
        ## "0" → "Epiblast"
        self.ctypes_int = {x : ctypes[x] for x in range(len(ctypes))}
        ## "Epiblast" → "0"
        self.ctypes_nam = {y : x for x,y in self.ctypes_int.items()}

    def __makeTensor_LabelEncoding(self, ycol):
        """Converts the dataframe with headers into a raw tensor, with celltype
        annotations switched out to floats [0,1]
        
        Torch does not support this natively, though sklearn does..."""
        self.__labelEncodingMaps(ycol)
        dataset = self.raw_dataset
        ## "Mesoderm"  → 4
        dataset[ycol] = dataset[ycol].map(lambda x: self.ctypes_nam[x])
        dataset = dataset.astype("float32")
        ## 4 → 4/total_celltypes
        dataset[ycol] = dataset[ycol] / len(self.ctypes_nam)
        return(dataset)

    def __makeTensor_OneHotEncoding(self, ycol):
        print("TODO!")
        return(None)
        
    def doOneHotPrediction(self, test_dataframe):
        print("TODO")
        return(None)

    def makePseudoCells(self, message=True):
        df = pandas.DataFrame()
        for ctype in self.ctypes_nam.keys():
            df = pandas.concat(
                [df, self.__makePseudoCell(ctype)], 
                axis=1)
        if message:
            print("Created", len(self.ctypes_nam), " pseudo cells from the averages of cells in the same output group")
        else:
            return(df.T)
        
    def __makePseudoCell(self, name):
        ## Filter by a type and apply the mean, then remove the celltype
        ps = self.raw_dataset[
            self.raw_dataset["Celltype"] == self.ctypes_nam[name]
        ].apply(np.mean).round().drop("Celltype")
        ps.name = name
        return(ps)



## torch.nn.Softmax for probability on output
## We can do one_hot encoding on the model in this case