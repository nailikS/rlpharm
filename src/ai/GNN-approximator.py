import pandas as pd
import numpy as np
import torch
import dgl
import wandb

from sklearn.model_selection import KFold

from dgl.nn.pytorch import GraphConv
from torch.nn import Linear, MSELoss
from torch.optim import Adam


# hyperparameters
learning_rate = 0.01
hidden_feats = 16
num_epochs = 100
batch_size = 32
data_path = r"C:\Users\kilia\MASTER\rlpharm\data\approxFinal.csv"

wandb.init(
    # set the wandb project where this run will be logged
    project="approximator",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "GNN",
    "dataset": "approxFinal",
    "epochs": num_epochs,
    'hidden_feats': hidden_feats,
    }
)
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.graphs = []
        self.labels = []
        for i in range(X.shape[0]):
            # Create a DGL graph from the ith row of the input
            g = dgl.graph((torch.arange(X_train[i].shape[0]), torch.arange(X_train[i].shape[0])))
            g.ndata['feat'] = torch.FloatTensor(X[i])
            self.graphs.append(g)
            self.labels.append(torch.FloatTensor([y[i]]))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
# GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.linear = Linear(hidden_feats, out_feats)

    def forward(self, g, h):
        h = torch.relu(self.conv1(g, h))
        h = torch.relu(self.conv2(g, h))
        h = self.linear(h)
        return h.squeeze(1)

# Load your data
df = pd.read_csv(data_path)

# Define the number of input features and output features
in_feats = 6
out_feats = 1

# Split your data into input features (X) and target output (y)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values.reshape(-1, 1)

# Define your cross-validation split
kf = KFold(n_splits=5, shuffle=True)

# Initialize lists to store the training and validation losses for each fold
train_losses = []
val_losses = []
best_val_loss = float('inf')

# Iterate over each fold
for train_index, val_index in kf.split(X):
    # Split your data into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert your data to DGL graphs
    #train_graph = dgl.graph((np.arange(X_train.shape[0]), np.arange(X_train.shape[0])))
    #val_graph = dgl.graph((np.arange(X_val.shape[0]), np.arange(X_val.shape[0])))
    #train_graph.ndata['feat'] = torch.FloatTensor(X_train)
    #val_graph.ndata['feat'] = torch.FloatTensor(X_val)

    # Define your model, optimizer, and loss function
    model = GNNModel(in_feats, hidden_feats, out_feats)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MSELoss()
    # Initialize dataset and dataloader
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.stack(labels)
        return batched_graph, batched_labels
    
    train_dataset = GraphDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    val_dataset = GraphDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    # train_loader = dgl.dataloading.GraphDataLoader(train_graph, batch_size=X_train.shape[0], shuffle=False, collate_fn=dgl.batch)
    # val_loader = dgl.dataloading.GraphDataLoader(val_graph, batch_size=X_val.shape[0], shuffle=False, collate_fn=dgl.batch)

    # Train your model
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0

        # Train the model on the training set
        for batch, d in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = d.ndata['feat'], torch.FloatTensor(y_train)
            outputs = model(d, inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate the model on the validation set
        with torch.no_grad():
            for batch, d in enumerate(val_loader):
                inputs, labels = d.ndata['feat'], torch.FloatTensor(y_val)
                outputs = model(d, inputs)
                val_loss = loss_fn(outputs, labels)
                val_loss += val_loss.item()

            # Update best_val_loss and save the model if the current validation loss is lower than the previous lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), '/models/approximator/best_model.pth')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
