"""
DeepMapMatchingExecutor: Executor for neural network-based map matching models

This executor is designed for deep learning map matching models that inherit from
AbstractModel and implement forward(), predict(), and calculate_loss() methods.

Models supported:
- TRMMA (Trajectory Recovery with Multi-Modal Alignment)
- DeepMM (Deep Learning-based Map Matching)
- DiffMM (Diffusion-based Map Matching)

The executor is adapted from TrajLocPredExecutor but customized for map matching tasks.
"""

from ray import tune
import torch
import torch.optim as optim
import numpy as np
import os
from logging import getLogger

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class DeepMapMatchingExecutor(AbstractExecutor):
    """Executor for deep learning-based map matching models.

    This executor handles training, validation, and evaluation of neural network
    map matching models that use MapMatchingDataset and MapMatchingEvaluator.
    """

    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.model = model.to(self.config['device'])
        self.tmp_path = './libcity/tmp/checkpoint/'
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.loss_func = None  # Use model's calculate_loss
        self._logger = getLogger()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Metrics for map matching (based on config or defaults)
        # Get metrics configuration - could be string or list
        metrics_config = config.get('metrics', 'accuracy')
        if isinstance(metrics_config, list):
            self.primary_metric = metrics_config[0] if metrics_config else 'accuracy'
        else:
            self.primary_metric = metrics_config
        # Keep metrics config for potential other uses
        self.metrics = metrics_config

    def _move_batch_to_device(self, batch, device):
        """Move batch dictionary tensors to specified device.

        Args:
            batch: Dictionary containing tensors and other data
            device: Target device

        Returns:
            batch: Dictionary with tensors moved to device
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            else:
                result[key] = value
        return result

    def train(self, train_dataloader, eval_dataloader):
        """Train the neural map matching model.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
        """
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        metrics = {}
        metrics['accuracy'] = []
        metrics['loss'] = []
        lr = self.config['learning_rate']

        for epoch in range(self.config['max_epoch']):
            # Notify model of current epoch if needed
            if hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch)

            self._logger.info('start train')
            self.model, avg_loss = self.run(train_dataloader, self.model,
                                            self.config['learning_rate'], self.config['clip'])
            self._logger.info('==>Train Epoch:{:4d} Loss:{:.5f} learning_rate:{}'.format(
                epoch, avg_loss, lr))

            # Eval stage
            self._logger.info('start evaluate')
            avg_eval_acc, avg_eval_loss = self._valid_epoch(eval_dataloader, self.model)
            self._logger.info('==>Eval Acc:{:.5f} Eval Loss:{:.5f}'.format(avg_eval_acc, avg_eval_loss))
            metrics['accuracy'].append(avg_eval_acc)
            metrics['loss'].append(avg_eval_loss)

            if self.config['hyper_tune']:
                # Use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # Ray tune use loss to determine which params are best
                tune.report(loss=avg_eval_loss, accuracy=avg_eval_acc)
            else:
                save_name_tmp = 'ep_' + str(epoch) + '.m'
                torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)

            self.scheduler.step(avg_eval_acc)
            # Early stop if learning rate too small
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                break

        if not self.config['hyper_tune'] and self.config['load_best_epoch']:
            best = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(best) + '.m'
            self.model.load_state_dict(
                torch.load(self.tmp_path + load_name_tmp))

        # Clean up temporary files
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)

    def load_model(self, cache_name):
        """Load model from checkpoint.

        Args:
            cache_name: Path to checkpoint file
        """
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, cache_name):
        """Save model checkpoint.

        Args:
            cache_name: Path to save checkpoint
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def evaluate(self, test_dataloader):
        """Evaluate the model on test data.

        Calculates accuracy directly by comparing predictions with ground truth,
        bypassing the incompatible MapMatchingEvaluator interface.

        Args:
            test_dataloader: DataLoader for test data
        """
        import json
        import datetime

        self.model.train(False)
        total_correct = 0
        total_valid = 0
        total_loss = []
        loss_func = self.loss_func or self.model.calculate_loss

        for batch in test_dataloader:
            # Move batch to device
            if hasattr(batch, 'to_tensor'):
                batch.to_tensor(device=self.config['device'])
            else:
                batch = self._move_batch_to_device(batch, self.config['device'])

            # Get predictions from model
            result = self.model.predict(batch)

            # Calculate loss
            loss = loss_func(batch)
            total_loss.append(loss.data.cpu().numpy().tolist())

            # Calculate accuracy by comparing predictions with ground truth
            tgt_roads = batch.get('tgt_roads')
            if tgt_roads is None:
                tgt_roads = batch.get('target', batch.get('output_trg'))

            if tgt_roads is not None:
                # Convert result to tensor if needed
                if not isinstance(result, torch.Tensor):
                    result = torch.LongTensor(result).to(self.config['device'])

                # Ensure same shape for comparison
                if result.dim() == 1:
                    result = result.unsqueeze(0)
                if tgt_roads.dim() == 1:
                    tgt_roads = tgt_roads.unsqueeze(0)

                # Handle length mismatch by truncating to minimum
                min_len = min(result.shape[1], tgt_roads.shape[1])
                result_trimmed = result[:, :min_len]
                tgt_trimmed = tgt_roads[:, :min_len]

                # Create mask for valid positions (tgt_roads >= 0, ignoring padding -1)
                valid_mask = tgt_trimmed >= 0

                # Count correct predictions where mask is valid
                correct = ((result_trimmed == tgt_trimmed) & valid_mask).sum().item()
                valid_count = valid_mask.sum().item()

                total_correct += correct
                total_valid += valid_count

        # Compute final metrics
        if total_valid > 0:
            accuracy = total_correct / total_valid
        else:
            accuracy = 0.0

        avg_loss = np.mean(total_loss, dtype=np.float64) if total_loss else 0.0

        # Build evaluation results
        evaluate_result = {
            'accuracy': accuracy,
            'loss': float(avg_loss),
            'total_correct': total_correct,
            'total_valid': total_valid,
            'num_batches': len(test_dataloader)
        }

        # Log results
        self._logger.info('Test Evaluation Results:')
        self._logger.info('  Accuracy: {:.4f}'.format(accuracy))
        self._logger.info('  Loss: {:.4f}'.format(avg_loss))
        self._logger.info('  Correct/Total: {}/{}'.format(total_correct, total_valid))

        # Save results to file
        if not os.path.exists(self.evaluate_res_dir):
            os.makedirs(self.evaluate_res_dir)

        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        model_name = self.config.get('model', 'DeepMM')
        dataset_name = self.config.get('dataset', 'unknown')
        filename = '{}_{}_{}_{}'.format(timestamp, model_name, dataset_name, 'evaluate')

        # Save as JSON
        json_path = os.path.join(self.evaluate_res_dir, '{}.json'.format(filename))
        with open(json_path, 'w') as f:
            json.dump(evaluate_result, f, indent=4)
        self._logger.info('Evaluation results saved to {}'.format(json_path))

        # Save as CSV
        csv_path = os.path.join(self.evaluate_res_dir, '{}.csv'.format(filename))
        with open(csv_path, 'w') as f:
            headers = list(evaluate_result.keys())
            f.write(','.join(headers) + '\n')
            values = [str(evaluate_result[h]) for h in headers]
            f.write(','.join(values) + '\n')
        self._logger.info('Evaluation results saved to {}'.format(csv_path))

    def run(self, data_loader, model, lr, clip):
        """Training loop for one epoch.

        Args:
            data_loader: DataLoader for training data
            model: The neural network model
            lr: Learning rate
            clip: Gradient clipping threshold

        Returns:
            model: Updated model
            avg_loss: Average loss for the epoch
        """
        model.train(True)
        if self.config['debug']:
            torch.autograd.set_detect_anomaly(True)

        total_loss = []
        loss_func = self.loss_func or model.calculate_loss

        self._logger.info("num_batches: {}".format(len(data_loader)))

        for batch in data_loader:
            # One batch, one step
            self.optimizer.zero_grad()
            # Handle both LibCity Batch objects and plain dictionaries
            if hasattr(batch, 'to_tensor'):
                batch.to_tensor(device=self.config['device'])
            else:
                # Dictionary batch from DeepMapMatchingDataset - tensors already in batch
                batch = self._move_batch_to_device(batch, self.config['device'])
            loss = loss_func(batch)

            if self.config['debug']:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            total_loss.append(loss.data.cpu().numpy().tolist())

            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            except:
                pass

            self.optimizer.step()

        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader, model):
        """Validation loop for one epoch.

        Calculates accuracy directly by comparing predictions with ground truth
        (tgt_roads), bypassing the incompatible evaluator interface.

        Args:
            data_loader: DataLoader for validation data
            model: The neural network model

        Returns:
            avg_acc: Average accuracy (ratio of correct predictions)
            avg_loss: Average loss
        """
        model.train(False)
        total_loss = []
        total_correct = 0
        total_valid = 0
        loss_func = self.loss_func or model.calculate_loss

        for batch in data_loader:
            if hasattr(batch, 'to_tensor'):
                batch.to_tensor(device=self.config['device'])
            else:
                batch = self._move_batch_to_device(batch, self.config['device'])

            # Get predictions from model
            # predict() returns road segment IDs [batch_size, seq_len]
            result = model.predict(batch)

            # Calculate loss
            loss = loss_func(batch)
            total_loss.append(loss.data.cpu().numpy().tolist())

            # Calculate accuracy by comparing predictions with ground truth
            # Ground truth is in 'tgt_roads' with -1 as padding mask
            tgt_roads = batch.get('tgt_roads')
            if tgt_roads is None:
                # Fallback to other possible keys
                tgt_roads = batch.get('target', batch.get('output_trg'))

            if tgt_roads is not None:
                # Convert result to tensor if needed
                if not isinstance(result, torch.Tensor):
                    result = torch.LongTensor(result).to(self.config['device'])

                # Ensure same shape for comparison
                if result.dim() == 1:
                    result = result.unsqueeze(0)
                if tgt_roads.dim() == 1:
                    tgt_roads = tgt_roads.unsqueeze(0)

                # Handle length mismatch by truncating to minimum
                min_len = min(result.shape[1], tgt_roads.shape[1])
                result_trimmed = result[:, :min_len]
                tgt_trimmed = tgt_roads[:, :min_len]

                # Create mask for valid positions (tgt_roads >= 0, ignoring padding -1)
                valid_mask = tgt_trimmed >= 0

                # Count correct predictions where mask is valid
                correct = ((result_trimmed == tgt_trimmed) & valid_mask).sum().item()
                valid_count = valid_mask.sum().item()

                total_correct += correct
                total_valid += valid_count

        # Compute average accuracy
        if total_valid > 0:
            avg_acc = total_correct / total_valid
        else:
            avg_acc = 0.0

        avg_loss = np.mean(total_loss, dtype=np.float64)
        return avg_acc, avg_loss

    def _build_optimizer(self):
        """Build optimizer based on config.

        Returns:
            optimizer: PyTorch optimizer
        """
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                        weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.config['learning_rate'])
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler.

        Returns:
            scheduler: PyTorch learning rate scheduler
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                         patience=self.config['lr_step'],
                                                         factor=self.config['lr_decay'],
                                                         threshold=self.config['schedule_threshold'])
        return scheduler
