# -*- coding: utf-8 -*-
"""
KgContextExecutor for CKGGNN Model

This executor extends TrafficStateExecutor to support Knowledge Graph context
during training and evaluation.

Adapted from: /home/wangwenrui/shk/AgentCity/repos/CKGGNN/libcity/executor/kg_context_executor.py

Key features:
- Custom training loop that passes KG embeddings (dict_kge) to the model
- Custom evaluation that handles KG context
- Loss function that works with 'y_goal' instead of 'y'
"""

import copy
import os
import time
import pickle
import numpy as np
import torch
from datetime import datetime
from ray import tune
from libcity.model import loss
from functools import partial
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.pipeline.embedkg_template import (
    obatin_spatial_pickle,
    obatin_temporal_pickle,
    generate_spatial_kg,
    generate_temporal_kg,
    generate_kgsub_spat,
    generate_kgsub_temp_notcover
)


class KgContextExecutor(TrafficStateExecutor):
    """
    Executor for training and evaluating models with Knowledge Graph context.

    Inherits from TrafficStateExecutor and adds support for:
    - Passing dict_kge (KG embeddings dictionary) to model during training/evaluation
    - Custom loss function that handles 'y_goal' key from batch
    """

    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        # Store reference to dataset for later KG embedding loading
        self._dataset = None
        self._dict_kge = None

    def _load_kg_embeddings(self, dataloader):
        """
        Load or generate KG embeddings.

        This method constructs the dict_kge dictionary required by CKGGNN model
        by loading KG data from pickle files and generating embeddings.

        Args:
            dataloader: The dataloader (used to access the dataset)

        Returns:
            dict: Dictionary containing KG embeddings and labels, or None if KG data unavailable
        """
        kg_switch = self.config.get('kg_switch', False)
        if not kg_switch:
            self._logger.warning("KG switch is off, model will run without KG embeddings")
            return None

        # Return cached dict_kge if already loaded
        if self._dict_kge is not None:
            self._logger.info("Using cached KG embeddings")
            return self._dict_kge

        try:
            self._logger.info("Loading KG embeddings...")

            # Try to get dataset from dataloader
            dataset = None
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'get_kge_template'):
                dataset = dataloader.dataset
            elif hasattr(dataloader, 'data_source') and hasattr(dataloader.data_source, 'get_kge_template'):
                dataset = dataloader.data_source

            if dataset is None:
                self._logger.warning("Dataset does not support KG embeddings (missing get_kge_template method)")
                return None

            # Get KG embeddings from dataset
            np_goal, np_auxi, spat_file, temp_file, spat_ent_kge, spat_rel_kge, temp_ent_kge, temp_rel_kge = \
                dataset.get_kge_template()
            self._logger.info("KG template loaded from dataset")

            # Load spatial and temporal dictionaries from pickle files
            dictKG_spatial = obatin_spatial_pickle(self.config, self._logger)
            dictKG_temporal = obatin_temporal_pickle(self.config, self._logger)

            # Generate entity and relation labels from triples factories
            tf_spat = generate_spatial_kg(self.config, self._logger)
            spat_ent_label = tf_spat.entity_labeling.label_to_id
            spat_rel_label = tf_spat.relation_labeling.label_to_id

            tf_temp = generate_temporal_kg(self.config, self._logger)
            temp_ent_label = tf_temp.entity_labeling.label_to_id
            temp_rel_label = tf_temp.relation_labeling.label_to_id

            # Generate spatial sub-KG
            subdict_spat = generate_kgsub_spat(self.config, self._logger, dictKG_spatial)

            # Initialize empty sub-KG embeddings (will be populated during forward pass)
            subdict_spat_kge = {}
            subdict_temp_kge = {}
            subdict_temp = {}

            # Generate temporal datetime list and sub-KG embeddings
            kg_context = self.config.get('kg_context', 'spat-temp')
            kg_weight = self.config.get('kg_weight', 'times')
            kg_weight_temp = 'add' if kg_weight == 'add' else 'times'

            if kg_context != 'spat':
                # Generate temporal datetime list
                temp_datetime_list = []
                for _dim1 in range(np_goal.shape[0]):  # len_time
                    # Convert auxiliary data to datetime
                    x_auxi_0 = np_auxi[_dim1, 0, :]
                    part1 = int(x_auxi_0[0])
                    part2 = int(x_auxi_0[1])
                    str_len = int(x_auxi_0[2])
                    str_num = str(part1) + str(part2).zfill(str_len - len(str(part1)))
                    long_num = int(str_num)
                    temp_datetime = datetime.fromtimestamp(long_num)
                    temp_datetime_list.append(temp_datetime)
                temp_datetime_list = list(dict.fromkeys(temp_datetime_list))
                self._logger.info('temp_datetime from[{}] end[{}]'.format(
                    temp_datetime_list[0], temp_datetime_list[-1]))

                # Try to load from cache or generate temporal embeddings
                temp_kge_emd_file = 'temp_kge_emd_{}_{}'.format(
                    self.config.get('dataset'), kg_weight_temp)
                temp_kge_emd_pickle = os.path.join(
                    './raw_data/{}'.format(self.config.get('dataset')),
                    '{}.pickle'.format(temp_kge_emd_file))

                if os.path.exists(temp_kge_emd_pickle):
                    self._logger.info('[MP]Loading temporal embeddings from pickle')
                    with open(temp_kge_emd_pickle, 'rb') as f_pickle:
                        dict_kge_part = pickle.load(f_pickle)
                        subdict_temp = copy.deepcopy(dict_kge_part.get('sub_temp', {}))
                        subdict_temp_kge = copy.deepcopy(dict_kge_part.get('sub_temp_emd', {}))
                    self._logger.info('[MP]Load successfully from pickle')
                else:
                    self._logger.info('[MP]Generating temporal embeddings (this may take a while)')
                    # Generate temporal sub-KG embeddings for each datetime
                    for temp_datetime in temp_datetime_list:
                        if temp_datetime not in dictKG_temporal:
                            continue
                        subdict_temp_single = generate_kgsub_temp_notcover(
                            self.config, self._logger, dictKG_temporal,
                            temp_datetime, temp_datetime_list, kg_weight_temp)
                        subdict_temp[temp_datetime] = copy.deepcopy(subdict_temp_single)
                        # Note: subdict_temp_kge is populated during model forward pass
                    self._logger.info('[MP]Temporal sub-KG generated')

            # Organize all data into dict_kge
            dict_kge = {
                'dict_spat': dictKG_spatial,
                'dict_temp': dictKG_temporal,
                'spat_ent_kge': spat_ent_kge,
                'spat_rel_kge': spat_rel_kge,
                'temp_ent_kge': temp_ent_kge,
                'temp_rel_kge': temp_rel_kge,
                'spat_ent_label': spat_ent_label,
                'spat_rel_label': spat_rel_label,
                'temp_ent_label': temp_ent_label,
                'temp_rel_label': temp_rel_label,
                'sub_spat': subdict_spat,
                'sub_temp': subdict_temp,
                'sub_spat_emd': subdict_spat_kge,
                'sub_temp_emd': subdict_temp_kge
            }

            self._dict_kge = dict_kge
            self._logger.info("KG embeddings loaded successfully")
            return dict_kge

        except Exception as e:
            self._logger.warning(f"Failed to load KG embeddings: {e}")
            import traceback
            self._logger.debug(traceback.format_exc())
            return None

    def train(self, train_dataloader, eval_dataloader):
        """
        Override train to load KG embeddings and call kg_train.

        This method ensures that KG embeddings are loaded before training
        starts, fixing the issue where dict_kge=None is passed to the model.

        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader

        Returns:
            float: Minimum validation loss achieved during training
        """
        # Load KG embeddings
        dict_kge = self._load_kg_embeddings(train_dataloader)

        if dict_kge is None:
            self._logger.warning("Training without KG embeddings - model may not work correctly")

        # Call kg_train with embeddings
        return self.kg_train(train_dataloader, eval_dataloader, dict_kge)

    def evaluate(self, test_dataloader):
        """
        Override evaluate to load KG embeddings and call kg_evaluate.

        This method ensures that KG embeddings are loaded before evaluation
        starts, fixing the issue where dict_kge=None is passed to the model.

        Args:
            test_dataloader: Test dataloader

        Returns:
            dict: Test evaluation results
        """
        # Load KG embeddings (use cached if available)
        dict_kge = self._load_kg_embeddings(test_dataloader)

        if dict_kge is None:
            self._logger.warning("Evaluating without KG embeddings - model may not work correctly")

        # Call kg_evaluate with embeddings
        return self.kg_evaluate(test_dataloader, dict_kge)

    def _build_train_loss(self):
        """
        Build training loss function based on config['train_loss']
        If 'none', will use model's built-in calculate_loss method.

        Returns:
            callable: Loss function that accepts batch and returns loss tensor
        """
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch):
            y_true = batch['y_goal']  # Use y_goal for CKGGNN
            y_predicted = self.model.predict(batch)
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)
        return func

    def kg_evaluate(self, test_dataloader, dict_kge):
        """
        Evaluate model on test data with KG context.

        Args:
            test_dataloader(torch.Dataloader): Test dataloader
            dict_kge(dict): Dictionary containing KG embeddings and labels

        Returns:
            dict: Test evaluation results
        """
        self._logger.info('Start evaluating ...')

        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch, dict_kge)
                try:
                    y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                except:
                    y_true = self._scaler.inverse_transform(batch['y_goal'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def kg_train(self, train_dataloader, eval_dataloader, dict_kge):
        """
        Train model with KG context.

        Args:
            train_dataloader(torch.Dataloader): Training dataloader
            eval_dataloader(torch.Dataloader): Evaluation dataloader
            dict_kge(dict): Dictionary containing KG embeddings and labels

        Returns:
            float: Minimum validation loss achieved
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func, dict_kge)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func, dict_kge)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

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
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None, dict_kge=None):
        """
        Train model for one epoch.

        Args:
            train_dataloader: Training data
            epoch_idx: Epoch number
            loss_func: Loss function
            dict_kge: KG embeddings dictionary

        Returns:
            list: List of losses for each batch
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch, dict_kge)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None, dict_kge=None):
        """
        Validate model for one epoch.

        Args:
            eval_dataloader: Evaluation data
            epoch_idx: Epoch number
            loss_func: Loss function
            dict_kge: KG embeddings dictionary

        Returns:
            float: Average validation loss
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch, dict_kge)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
