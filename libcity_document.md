# Add a new model to LibCity

Here, we present how to develop a new model, and apply it to the `LibCity`.

## Create a New Model Class

To begin with, we should create a new model implementing from `AbstractModel` or `AbstractTrafficStateModel` . Note that for traffic state prediction tasks, please inherit Class `AbstractTrafficStateModel` and for trajectory location prediction tasks, please inherit Class `AbstractModel`.

Here we take the traffic state prediction task as an example. We would like to develop a model for traffic speed prediction task named as `NewModel`. 

First please create a new file `NewModel.py` in the directory `libcity/model/traffic_speed_prediction/` and write the following code into the file.

```python
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

class NewModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        pass
    
    def predict(self, batch):
        pass
    
    def calculate_loss(self, batch):
        pass
```

## Implement \_\_init\_\_()

Then we implement `__init__()` method, `__init__()` is used to define the model structure according to the data feature and the configuration information.

The input parameters of  `__init__()` are `config` and `data_feature`, where `config` contains various configuration information, including model parameters and so on,  and `data_feature` contains features of the dataset to build our model.

Here we take a simple LSTM model as an example for traffic prediction. You can define `__init__()` like this:

```python
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class NewModel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()
        # section 2: model config 
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0)
        # section 3: model structure
        self.rnn = nn.LSTM(input_size=self.num_nodes * self.feature_dim, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_nodes * self.output_dim)
```

You can see that in the section 1 of the code, we take the necessary parameters from `data_feature`, including the number of nodes(`num_nodes`), data input dimensions(`feature_dim`), output dimensions(`output_dim`), and data normalized objects(`scaler`), and initialize the logger. 

In the section 2 of the code, we take the necessary parameters from `config`, including hidden layer dimensions(`hidden_size`), network layers(`num_layers`), input time length(`input_window`), output time length(`output_window`), etc.

In the section 3 of the code, we define model structures, including a LSTM layer and a full connectivity layer.

## Implement predict()

Then we define the `predict()` method, which is used to compute the prediction results of the model. The input parameters of  `predict()` is `batch`, which is an object of class [Batch](../user_guide/data/batch.md). 

Both the  `AbstractModel` and `AbstractTrafficStateModel`  inherit from class `torch.nn.Module`. If you are familiar with the Pytorch framework, you will find that this function is similar to the `forward()` function in `torch.nn.Module`.  In most cases, you can call the `forward()` function directly inside this function to get the output of the model. 

The purpose of this function is that you can do some extra processing on the basis of the model output calculated by the `forward()` function. For example, when the `forward()`  function calculates the result of single-step prediction of the model and you need the result of multi-step prediction, you can use the `predict()` function to implement it.

For example, you can define `predict()` like this:

```python
class NewModel(AbstractTrafficStateModel):
    def forward(self, batch):
        src = batch['X'].clone()
        src = src.permute(1, 0, 2, 3)
        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)
        outputs = []
        for i in range(self.output_window):
            out, _ = self.rnn(src)
            out = self.fc(out[-1])
            out = out.reshape(batch_size, self.num_nodes, self.output_dim)
            outputs.append(out.clone())
            src = torch.cat((src[1:, :, :], out.reshape(
                batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)
        outputs = torch.stack(outputs)
        return outputs.permute(1, 0, 2, 3)

    def predict(self, batch):
        return self.forward(batch)
```

It can be seen that the `predict` function calls the `forward` function directly, and in the `forward` function, we define the calculation process of the model. This model uses LSTM to make multiple predictions, and takes the output of each predicted as the next input.  Here we assume that the input data dimension and output data dimension are equal, that is `feature_dim=output_dim`.

## Implement calcualte_loss()

Finally we define the `calculate_loss()` method, `calculate_loss()` is used to compute the loss between the prediction results and the ground-truth value. 

The input parameters of  `calculate_loss()` is `batch`, which is an object of class [Batch](../user_guide/data/batch.md). And the method return a `torch.Tensor` for back propagation.

You can customize the loss function or call the loss function we defined in the `libcity/model/loss.py` file.

For example, you can define `calcualte_loss()` like this:

```python
from libcity.model import loss

class NewModel(AbstractTrafficStateModel):
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)
```

You can see that here we directly call the `loss.masked_mae_torch` function that has been defined in `loss.py`, its function is to calculate the MAE loss.

Now we have completed the definition of the model structure, and there are some simple configurations left.

## Import The Model

After adding the model, you need to modify the `__init__.py` file in the task folder where your model belongs.  In the example above, the file you need to modify is `libcity/model/traffic_speed_prediction/__init__.py`. 

Please add code like this:

```python
from libcity.model.traffic_speed_prediction.NewModel import NewModel

__all__ = [
    "NewModel",
]
```

## Add Model Config

Finally, you need to modify some relevant `config` files.

- First, you need to modify the `libcity/config/task_config.json`, which is used to set the models and datasets supported by each task, and specify the basic parameters (data module, execution module, evaluation module) used by the model. 

  For example, you can add codes like the follows, which means the data module class used by `NewModel` is `TrafficStatePointDataset`, the execution module class is `TrafficStateExecutor`, and the evaluation module class is `TrafficStateEvaluator`. 

```json
{
    "traffic_state_pred": {
        "allowed_model": ["NewModel"],
        "allowed_dataset": [""],
        "NewModel": {
            "dataset_class": "TrafficStatePointDataset",
            "executor": "TrafficStateExecutor",
            "evaluator": "TrafficStateEvaluator"
        }
    }
}
```

- Second, you need to add a file in the `libcity/config/model/` directory to set the default parameters of your model. You can also set parameters of other modules which you want to cover as the parameters of the model module have the highest priority than other modules. 

  For example, you can add this file `libcity/config/model/traffic_state_pred/NewModel.json` and add codes like the follows. You can see that in addition to the three parameters related to the model structure, we also define the number of training rounds(`max_epoch`), optimizer(`learner`), and learning rate(`learning_rate`) to cover the default execution configuration.

```json
{
  "hidden_size": 64,
  "num_layers": 1,
  "dropout": 0,

  "max_epoch": 100,
  "learner": "adam",
  "learning_rate": 0.001
}
```

**Note: The filename of the config and the value in `allowed_model` list must be the same as the class name of the model you added.**  Just like the `NewModel` above.

Now that you have learned how to add a new model, try the following commands to run this model!

```shell
python run_model.py --task traffic_state_pred --model NewModel --dataset METR_LA
```

# Customize Dataset


If we have a new model, and if there is no suitable dataset class, then we need to design a new dataset. Here, we present how to develop a new dataset, and apply it to the `LibCity`.

## Create a New Dataset Class

To begin with, we should create a new dataset implementing from `AbstractDataset` or one of the subclass of `AbstractDataset`.

For example, we would like to develop a dataset for traffic state prediction task named as `NewDataset` and write the code to `newdataset.py` in the directory `libcity/data/dataset/`. 

Here we inherit subclass `TrafficStatePointDataset` of class `AbstractDataset`.

```python
from libcity.data.dataset import TrafficStatePointDataset

class NewDatasets(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        pass
```

Or you can inherit class `AbstractDataset` directly.

```python
from libcity.data.dataset import AbstractDataset

class NewDatasets(AbstractDataset):
    def __init__(self, config):
        pass
```

## Rewrite Corresponding Methods

The function `get_data()` in `AbstractDataset` is used to get the divided `train_dataloader`, `eval_dataloader` and `test_ dataloader`. You need to call the function `libcity.data.utils.generate_dataloader` to get data-loader from list of input data, where the generated data-loader contains [Batch](../user_guide/data/batch.md) object.

The function `get_data_feature() ` in `AbstractDataset` is used to return the features of some datasets for use by the model and executor.

Other interfaces defined in subclasses of `AbstractDataset` will not be described here.

If there is no suitable dataset class, then you can rewrite the corresponding interface mentioned above.


### Example 1

Here we explain how to inherit `AbstractDataset` directly and rewrite the function `get_data_feature()` to return some values we want.

```python
from libcity.data.dataset import AbstractDataset

class NewDatasets(AbstractDataset):
    def __init__(self, config):
        pass
    
    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim}
```

### Example 2

Here we explain how to inherit a subclass of `AbstractDataset` and rewrite one of its methods.

```python
from libcity.data.dataset import TrafficStatePointDataset

class NewDatasets(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        pass
    
    # We will rewrite this method which is used to calculate `self.adj_mx` based on the atmoic file `rel_file.rel`.
    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)))
        self.adj_mx[:] = 1  # set all one
```
### Example 3

Here we explain how to inherit a subclass of `AbstractDataset` and return different keys in batch from the origin. Specifically, we intend to return three key values: X, Y and Z. This is just an example, for more details, you can refer to `TrafficStateCPTDataset` whose has four keys in batch.

```python
from libcity.data.dataset import TrafficStateDataset

class NewDatasets(TrafficStateDataset):
    def __init__(self, config):
        super().__init__(config)
        # the origin code
        # self.feature_name = {'X': 'float', 'y': 'float'}
        # the modified code
        self.feature_name = {'X': 'float', 'Y': 'float', 'Z': 'int'}
        pass
    
    def get_data(self):
        # Load datset for the keys x,y,z, generate [x|y|z]_[train|val|test].
        # ... (implement it yourself)
        # Data normalization using self.scaler.
        # ... (implement it yourself)
        # Aggregate X, Y, Z into a list.
        # The i-th element in train_data(a list) is a tuple, consists of x_train[i], y_train[i] and z_train[i].
        train_data = list(zip(x_train, y_train, z_train))
        eval_data = list(zip(x_val, y_val, z_val))
        test_data = list(zip(x_test, y_test, z_test))
        # Get dataloader by libcity.data.utils.generate_dataloader.
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers)
        # Return the dataloader
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
```