import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dvclive import Live

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, 20)
        self.output_layer = nn.Linear(20, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        out =  self.output_layer(out)
        return out


df = pd.read_csv('/Users/skyrex/Desktop/my/учеба/прога/neural networks course spbu/iris classification/Iris.csv')
df['Species'] = df['Species'].map({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})
df.drop(['Id'], axis=1, inplace=True)
X = df.drop(["Species"], axis=1).values
y = df["Species"].values

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


def train_network(model, optimizer, criterion, X_train, y_train, 
                  X_test, y_test, num_epochs, train_losses, test_losses):
    
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        
        #forward feed
        output_train = model(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)
        
        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")
        
def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr) != len(original_arr):
        return False
        
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred = []
    
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)

input_dim = 4 
output_dim = 3

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()

with Live() as live:
    live.log_param("epochs", 1)
    
    for hidden_dim in [10, 20, 30]:
        num_epochs = 10
        train_losses = np.zeros(num_epochs)
        test_losses  = np.zeros(num_epochs)
        
        model = NeuralNetworkClassificationModel(input_dim, output_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses)

        predictions_train = []
        predictions_test =  []
        with torch.no_grad():
            predictions_train = model(X_train)
            predictions_test = model(X_test)

        train_acc = get_accuracy_multiclass(predictions_train, y_train)
        test_acc  = get_accuracy_multiclass(predictions_test, y_test)
        live.log_metric('Accuracy train', train_acc)
        live.log_metric('Accuracy test', test_acc)
        
        plt.figure(figsize=(10,10))
        plt.plot(train_losses, label='train loss')
        plt.plot(test_losses, label='test loss')
        plt.legend()
        
        plt.savefig('loss.png')
        live.log_image(f"NN with {hidden_dim} hidden neurons loss.png", 'loss.png')
        live.next_step()
        
                
    
