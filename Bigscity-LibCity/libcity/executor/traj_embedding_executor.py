"""
TrajEmbeddingExecutor for Trajectory Embedding Models

This executor is designed for self-supervised trajectory embedding models
that learn representations through contrastive learning or other unsupervised
objectives. Unlike supervised prediction tasks, these models:
- Don't require ground truth labels
- Use model.calculate_loss(batch) for self-supervised objectives
- Return embeddings via model.predict(batch)

Suitable for models like:
- JCLRNT (Joint Contrastive Learning for Road Network and Trajectory)
- BERT-based trajectory models
- Other self-supervised trajectory representation learning models

Based on TrafficStateExecutor but simplified for embedding tasks.
"""

import os
import time
import numpy as np
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator, ensure_dir


class TrajEmbeddingExecutor(AbstractExecutor):
    """
    Executor for self-supervised trajectory embedding models.

    This executor handles training and evaluation of models that:
    1. Learn trajectory representations through self-supervised learning
    2. Implement calculate_loss(batch) for training objectives
    3. Return embeddings through predict(batch)
    """

    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        # Setup directories
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}/'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._logger.info(self.model)

        # Log model parameters
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        # Training configuration
        self.epochs = self.config.get('max_epoch', 100)
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)

        # Learning rate scheduling
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)

        # Gradient clipping
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)

        # Early stopping
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)

        # Logging and saving
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 100)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.hyper_tune = self.config.get('hyper_tune', False)

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        # Load from checkpoint if specified
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)

    def _build_optimizer(self):
        """Build optimizer based on configuration."""
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))

        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                betas=self.lr_betas, eps=self.lr_epsilon, weight_decay=self.weight_decay
            )
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate,
                betas=self.lr_betas, eps=self.lr_epsilon, weight_decay=self.weight_decay
            )
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate,
                momentum=self.lr_momentum, weight_decay=self.weight_decay
            )
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=self.learning_rate,
                alpha=self.lr_alpha, eps=self.lr_epsilon,
                momentum=self.lr_momentum, weight_decay=self.weight_decay
            )
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate,
                betas=self.lr_betas, eps=self.lr_epsilon, weight_decay=self.weight_decay
            )
        return optimizer

    def _build_lr_scheduler(self):
        """Build learning rate scheduler based on configuration."""
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))

            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio
                )
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio
                )
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio
                )
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min
                )
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda
                )
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold
                )
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def save_model(self, cache_name):
        """Save current model to file."""
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """Load model from cache."""
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model_with_epoch(self, epoch):
        """Save model with epoch number in filename."""
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """Load model from specific epoch."""
        model_path = self.cache_dir + '/' + self.config['model'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def train(self, train_dataloader, eval_dataloader):
        """
        Train the model using self-supervised learning.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader

        Returns:
            Minimum validation loss achieved
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches: {}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)
            self._writer.add_scalar('training loss', train_loss, epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)

            # Update learning rate
            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            # Logging
            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.format(
                    epoch_idx, self.epochs, train_loss, val_loss, log_lr, (end_time - start_time)
                )
                self._logger.info(message)

            # Early stopping and model saving
            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

        # Training summary
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.format(
                                  len(train_time), sum(train_time) / len(train_time),
                                  sum(eval_time) / len(eval_time)))

        # Load best model
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        """
        Train for one epoch.

        Args:
            train_dataloader: Training data loader
            epoch_idx: Current epoch index

        Returns:
            List of losses for each batch
        """
        self.model.train()
        losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)

            # Model computes its own loss via calculate_loss
            loss = self.model.calculate_loss(batch)

            losses.append(loss.item())
            loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Batch-level logging
            if batch_idx % self.log_batch == 0:
                self._logger.debug('Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}'.format(
                    epoch_idx, self.epochs, batch_idx, len(train_dataloader), loss.item()
                ))

        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx):
        """
        Validate for one epoch.

        Args:
            eval_dataloader: Evaluation data loader
            epoch_idx: Current epoch index

        Returns:
            Average validation loss
        """
        with torch.no_grad():
            self.model.eval()
            losses = []

            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = self.model.calculate_loss(batch)
                losses.append(loss.item())

            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss

    def evaluate(self, test_dataloader):
        """
        Evaluate the model on test data.

        For embedding models, this generates embeddings and can optionally
        evaluate on downstream tasks if the evaluator supports it.

        Args:
            test_dataloader: Test data loader

        Returns:
            Evaluation results
        """
        self._logger.info('Start evaluating ...')

        with torch.no_grad():
            self.model.eval()

            # Collect embeddings
            embeddings = []

            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)

                # Handle different output formats
                if isinstance(output, dict):
                    if 'trajectory_embedding' in output:
                        emb = output['trajectory_embedding']
                    elif 'seq_rep' in output:
                        emb = output['seq_rep']
                    else:
                        # Use first value in dict
                        emb = list(output.values())[0]
                else:
                    emb = output

                embeddings.append(emb.cpu().numpy())

            # Concatenate embeddings
            embeddings = np.concatenate(embeddings, axis=0)

            # Save embeddings
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' + \
                       self.config['model'] + '_' + self.config['dataset'] + '_embeddings.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename),
                                embeddings=embeddings)

            self._logger.info('Saved {} embeddings to {}'.format(
                embeddings.shape[0], filename))

            # If evaluator can handle embeddings, use it
            # Otherwise just return embedding statistics
            try:
                self.evaluator.clear()
                # Some evaluators may support embedding quality metrics
                test_result = self.evaluator.save_result(self.evaluate_res_dir)
                return test_result
            except:
                self._logger.info('Evaluator does not support embedding evaluation')
                return {
                    'num_embeddings': embeddings.shape[0],
                    'embedding_dim': embeddings.shape[1] if embeddings.ndim > 1 else 1,
                    'mean_norm': np.linalg.norm(embeddings, axis=1).mean()
                }
