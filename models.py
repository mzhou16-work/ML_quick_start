import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout, LeakyReLU
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError
from torchsummary import summary
import numpy as np

class NN(LightningModule):
    
    # initialize NN class
    def __init__(self, in_channels: int, out_channels: int, normalization: object, labels: list, label_dict: dict, predict_params: dict={'mlp_layers': 3, 'mlp_dim': 10, 'dropout': 0.2}, learning_rate=2e-4):
        
        # use 'self' to represent an instance of the class 'NN'
        super().__init__()
        self.save_hyperparameters()
        #initialization of the class; putting together the parts around the engine to create a functioning vehicle - essentially have to do a lot of this since our class is from the LightningModule class but we still need to customize it a little bit
        self.predict = MLP(in_channels=in_channels, out_channels=out_channels, **predict_params)
        self.normalization = normalization
        
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        
        self.labels = labels
        self.label_dict = label_dict
        
        self.monitor = 'val_mse'
        self.optmode = 'min'
        
    
    def forward(self, x):
        x = self.predict(x)
        return x
    
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self(x.float())
        
        # use Z score as penalty to force the model to pay more attention to rare cases...
        weights = torch.ones_like(y)
        #calculate the z-score of y
        z_scores = (y - y.mean())/ y.std()
        # for data that is beyond 1 std, triple its possibility
        #weights[torch.abs(z_scores) > 1] = 2
        #weights[torch.abs(z_scores) > 2] = 3
        temperature = 2.7 # hyperparameter to adjust weight; 'after how many std, the lost should be amplified'
        weights = torch.exp(torch.abs(z_scores) / temperature) # want the system to amplify the loss function for less standard values in order to learn better
        
        #weights = torch.pow(z_scores, 2) + 1
        #weights /= weights.sum() #normalize sum to 1
        
        #weighted loss function
        train_loss = F.mse_loss(y_hat.float(), y.float(), reduction = 'none')
        
        # Calculate the weighted mean squared error
        train_loss = (train_loss * weights).mean()
        
        if batch_idx % 100 == 0: #print every 100 steps
            for device in range(torch.cuda.device_count()):
                print_memory_usage(device)
        
        return train_loss # return the amount of loss for the training step
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        y_hat = self(x.float())
        y, _ = self.normalization(y.cpu().numpy(), self.labels[0], self.label_dict[self.labels[0]], 'inverse')
        y_hat, _ = self.normalization(y_hat.cpu().numpy(), self.labels[0], self.label_dict[self.labels[0]], 'inverse')
        
        idx = np.where(y_hat == y_hat)
        #y = torch.from_numpy(y[idx]).to('cuda:0')
        #y_hat = torch.from_numpy(y_hat[idx]).to('cuda:0')
        #y = torch.from_numpy(y).to('cuda:0')
        #y_hat = torch.from_numpy(y_hat).to('cuda:0')
        
        y = torch.from_numpy(y[idx]).to(self.device)
        y_hat = torch.from_numpy(y_hat[idx]).to(self.device)
        
        #evaluation of the model's performance - recall that 'y_hat' is the value predicted by the model, 'y' is the actual value
        self.val_r2(y_hat, y)
        self.val_rmse(y_hat, y)
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        
        self.log('val_r2', self.val_r2, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_rmse', self.val_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_mse', self.val_mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_mae', self.val_mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
    
    
    def test_step(self, batch, batch_idx):#basically same thing as before
        
        x, y = batch
        y_hat = self(x.float())
        y, _ = self.normalization(y.cpu().numpy(), self.labels[0], self.label_dict[self.labels[0]], 'inverse')
        y_hat, _ = self.normalization(y_hat.cpu().numpy(), self.labels[0], self.label_dict[self.labels[0]], 'inverse')
        
        idx = np.where(y_hat == y_hat)
        y = torch.from_numpy(y[idx]).to(self.device)
        y_hat = torch.from_numpy(y_hat[idx]).to(self.device)
        
        self.test_r2(y_hat, y)
        self.test_rmse(y_hat, y)
        self.test_mse(y_hat, y)
        self.test_mae(y_hat, y)
        self.log('test_r2', self.test_r2, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_rmse', self.test_rmse, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_mse', self.test_mse, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_mae', self.test_mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_hat.size(0))
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

#this is where we create our multilayer perceptron model
class MLP(LightningModule):
    
    def __init__(self, in_channels: int, out_channels: int, mlp_layers: int = 6, mlp_dim: int = 10, dropout: float = 0.2):
        
        super().__init__()
        self.save_hyperparameters()
        
        mlp = []
        
        mlp_dim = 2**mlp_dim # number of neurons to be held within largest layer; on the order of 2 (256, 512, 1024, 2048, etc.)
        
        layer_dims = np.zeros(mlp_layers-1).astype(int) # array containing the number of neurons within each layer
        max_layer_id = int((mlp_layers-2)/2) # index of the largest layer
        layer_dims[max_layer_id] = mlp_dim # set the dimension of the largest layer with the previously defined value
        
        # create a structure such that the number of neurons in each layer increases until the halfway point, then decreases; does not have to be the case but this is how Meng prefers to do it
        # for layers prior to the largest layer, the dimension of each consecutive layer increases by a factor of two
        for idx in range(0, max_layer_id):
            layer_dims[idx] = mlp_dim/2**(max_layer_id-idx)
        
        # for layers following the largest layer, the dimension of each consecutive layer decreases by a factor of two
        for idx in range(max_layer_id, (mlp_layers-1)):
            layer_dims[idx] = mlp_dim/2**(idx-max_layer_id)
        
        print(f' - Model Middle Layer Structure {layer_dims}')
        
        
        for i in range(mlp_layers-1):
            
            current_dim = layer_dims[i]
            
            mlp += [Sequential(Linear(in_channels, current_dim), BatchNorm1d(current_dim), LeakyReLU(), Dropout(p=dropout),)] # mlp is a 'unit' - essentially one unit per layer, with each unit processing data first through a linear layer, then a batch normalization layer, then a LeakyReLU (activation function) layer, then a dropout (see training script) layer. This same structure found in each layer in the model
            in_channels = current_dim
        
        mlp += [Linear(in_channels, out_channels)] # at the end, create a linear layer mapping to the number of labels, in our case just one
        
        #print('mlp')
        #print(mlp)
        
        self.mlp = ModuleList(mlp)
        
        
        for i in range(mlp_layers-1):
            torch.nn.init.xavier_normal_(self.mlp[i][0].weight.data, gain=1.0) #xavier normalization - fill each neuron with some value initially to reduce bias or something - search on google
            torch.nn.init.zeros_(self.mlp[i][0].bias.data)
        
    def forward(self, x):
        for nn in self.mlp:
            x = nn(x) # forward process basically subjects data to the same nn structure (linear, batch norm, LeakyReLU, dropout) for each layer in the multilayer perceptron
        return x


def print_memory_usage(device):
    print(f'GPU memory allocated on device {device}: {torch.cuda.memory_allocated(device) / 1024**3} / {torch.cuda.memory_reserved(device) / 1024**3} GB')

#'''