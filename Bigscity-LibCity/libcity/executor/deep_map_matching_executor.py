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
import datetime
import json
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
            if eval_dataloader is not None:
                self._logger.info('start evaluate')
                avg_eval_acc, avg_eval_loss = self._valid_epoch(eval_dataloader, self.model)
                self._logger.info('==>Eval Acc:{:.5f} Eval Loss:{:.5f}'.format(avg_eval_acc, avg_eval_loss))
            else:
                self._logger.warning('No evaluation data available - skipping validation')
                avg_eval_acc, avg_eval_loss = 0.0, float('inf')
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
        """
        Modified evaluate function to support Map Matching metrics (RMF, AN, AL).
        Flow: Init -> Collect(Inference) -> Process(LCS/Merge) -> Calculate Metrics -> Log/Save
        """
        self.model.train(False)
        
        # 1. 初始化评估所需的数据结构
        self._init_eval_resource()
        
        # 2. 收集阶段：遍历 DataLoader，进行推理并将 Tensor 转换为轨迹字典
        # 对应你提供的 collect 逻辑，但适配了 DataLoader 的循环
        self._collect_epoch_results(test_dataloader)

        # 3. 处理阶段：计算 LCS (最长公共子序列) 和 Merge Result
        # 这是计算 RMF 等指标的前置条件
        self.merge_result()
        if self.route is not None:
            self.find_lcs()

        # 4. 评估阶段：执行你提供的 evaluate 核心逻辑计算 RMF, AN, AL
        evaluate_result = self._calculate_metrics()

        # 5. 记录与保存
        self._log_and_save(evaluate_result)

    def _init_eval_resource(self):
        """
        初始化路网权重信息和结果容器
        """
        self.route = {}  # Ground Truth: {usr_id: {traj_id: [edge_list]}}
        self.result = {} # Prediction: {usr_id: {traj_id: [edge_list]}}
        self.rel_info = {} # Road Network Edge Distance info
        self.metrics = self.config.get('metrics', ['RMF', 'AN', 'AL'])
        self.evaluate_result = {}
        
        # 从 dataset 中获取路网信息 (假设 dataset 有 get_data_feature 方法或 adj_mx)
        # 这里假设 self.dataset.adj_mx 存储了路网图结构
        # 如果你的路网信息存储在其他地方，请修改此处
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'adj_mx'):
             rd_nwk_adj = self.dataset.adj_mx
             # 构造 rel_info，结构为 {geo_id: {'distance': dist, ...}}
             # 注意：这里的 geo_id 必须与模型输出的 ID 一致
             for start_node in rd_nwk_adj:
                 for end_node in rd_nwk_adj[start_node]:
                     edge_data = rd_nwk_adj[start_node][end_node]
                     # 假设 edge_data 中包含 'id' (edge_id) 和 'weight' (distance)
                     # 如果模型输出的是 edge_id，这里需要用 edge_id 做 key
                     if isinstance(edge_data, dict):
                         edge_id = edge_data.get('geo_id', f"{start_node}-{end_node}")
                         distance = edge_data.get('distance', 1.0)
                         self.rel_info[edge_id] = {'distance': distance}
        else:
            self._logger.warning("Road network info not found in dataset. Metrics relying on distance (RMF, AL) may fail.")

    def _collect_epoch_results(self, test_dataloader):
        """
        推理循环：将 Batch Tensor 转换为 {usr: {traj: [route]}} 格式
        """
        with torch.no_grad():
            for batch in test_dataloader:
                # 移动数据到设备
                if hasattr(batch, 'to_tensor'):
                    batch.to_tensor(device=self.config['device'])
                else:
                    batch = self._move_batch_to_device(batch, self.config['device'])

                # 模型推理
                output = self.model.predict(batch)
                
                # 获取 Ground Truth
                tgt_roads = batch.get('tgt_roads')
                if tgt_roads is None:
                    tgt_roads = batch.get('target', batch.get('output_trg'))

                # 转换 Tensor 为 Python List (CPU)
                if isinstance(output, dict):
                    output = output['pred_rid']
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy().tolist()
                if isinstance(tgt_roads, torch.Tensor):
                    tgt_roads = tgt_roads.cpu().numpy().tolist()

                # 获取 Batch 中的 ID 信息 (usr_id, traj_id) 用于构建字典 key
                # 如果 dataloader 没有提供这些 ID，我们使用自增索引生成伪 ID
                batch_uids = batch.get('uid') # 假设 batch 中有 uid
                batch_traj_ids = batch.get('trace_id') # 假设 batch 中有 trace_id
                
                batch_size = len(output)
                for i in range(batch_size):
                    # 获取当前样本的 Key
                    u_id = int(batch_uids[i]) if batch_uids is not None else 0
                    t_id = int(batch_traj_ids[i]) if batch_traj_ids is not None else i  # 这是一个临时的处理，实际应确保 batch 包含 ID
                    
                    # 初始化字典结构
                    if u_id not in self.result:
                        self.result[u_id] = {}
                        self.route[u_id] = {}
                    
                    # 存入预测结果 (过滤掉 padding 值，假设 padding 为 -1 或 0，视具体情况定)
                    #pred_seq = [x for x in output[i] if x > 0] 
                    self.result[u_id][t_id] = output[i]
                    
                    # 存入真实路径
                    if tgt_roads is not None and i < len(tgt_roads):
                        # 注意：为了适配 evaluate 逻辑，这里需要存为 numpy array 或 list，
                        # evaluate 代码中用 route[:, 1] 取值，暗示 route 可能是 [[idx, edge_id], ...] 结构
                        # 但为了通用性，这里我假设 evaluate 代码中的 route 就是 edge_id 列表。
                        # 如果你的 target 包含时间戳等额外信息，请在这里保留。
                        true_seq = [x for x in tgt_roads[i] if x > 0]
                        # 为了适配下方 evaluate 代码的 `route = route[:, 1]` 写法：
                        # 这里我们构造一个伪结构，或者你需要修改下方的 evaluate 代码去掉 `[:, 1]`
                        # 假设我们修改下方的 evaluate 代码使其更通用，这里直接存 list
                        self.route[u_id][t_id] = true_seq

    def merge_result(self):
        """
        简单合并逻辑：去除连续重复的边 (Map Matching 常用后处理)
        """
        self.merged_result = {}
        for usr_id, trajectories in self.result.items():
            self.merged_result[usr_id] = {}
            for traj_id, path in trajectories.items():
                merged_path = []
                if len(path) > 0:
                    merged_path.append(path[0])
                    for i in range(1, len(path)):
                        if path[i] != path[i-1]:
                            merged_path.append(path[i])
                self.merged_result[usr_id][traj_id] = merged_path

    def find_lcs(self):
        """
        计算最长公共子序列 (Longest Common Subsequence)
        """
        self.lcs = {}
        for usr_id, trajectories in self.route.items():
            self.lcs[usr_id] = {}
            for traj_id, ground_truth in trajectories.items():
                if traj_id not in self.result[usr_id]:
                    continue
                
                prediction = self.merged_result[usr_id][traj_id] # 通常用 merged 后的结果计算
                
                # 经典的 DP 求解 LCS
                m, n = len(ground_truth), len(prediction)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if ground_truth[i - 1] == prediction[j - 1]:
                            dp[i][j] = dp[i - 1][j - 1] + 1
                        else:
                            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
                # 回溯找序列
                lcs_seq = []
                i, j = m, n
                while i > 0 and j > 0:
                    if ground_truth[i - 1] == prediction[j - 1]:
                        lcs_seq.append(ground_truth[i - 1])
                        i -= 1
                        j -= 1
                    elif dp[i - 1][j] > dp[i][j - 1]:
                        i -= 1
                    else:
                        j -= 1
                self.lcs[usr_id][traj_id] = lcs_seq[::-1]

    def _calculate_metrics(self):
        """
        核心评估逻辑 (基于你提供的代码，稍作适配以防止 KeyErrors)
        """
        final_metrics = {'RMF': [], 'AN': [], 'AL': []}
        
        for usr_id, usr_value in self.route.items():
            if usr_id not in self.evaluate_result:
                self.evaluate_result[usr_id] = {}
                
            for traj_id, route in usr_value.items():
                # 适配：如果 route 是 list，就不需要 [:, 1]。如果是 numpy array 带时间戳，则保留
                # 这里假设它是 list，如果报错请根据实际数据结构调整
                # route = route[:, 1] 
                
                if traj_id not in self.lcs[usr_id] or traj_id not in self.merged_result[usr_id]:
                    continue

                lcs = self.lcs[usr_id][traj_id]
                merged_result = self.merged_result[usr_id][traj_id]
                self.evaluate_result[usr_id][traj_id] = {}

                # --- Metric: RMF ---
                if 'RMF' in self.metrics:
                    d_plus = 0
                    d_sub = 0
                    d_total = 0
                    
                    # 累加 Ground Truth 总距离
                    for rel_id in route:
                        d_total += self.rel_info.get(rel_id, {}).get('distance', 1.0)
                    
                    if d_total == 0: d_total = 1e-5 # 防止除零

                    i = j = k = 0
                    # 计算 d_sub (漏掉的距离)
                    while i < len(lcs) and j < len(route):
                        while j < len(route) and route[j] != lcs[i]:
                            d_sub += self.rel_info.get(route[j], {}).get('distance', 1.0)
                            j += 1
                        i += 1
                        j += 1
                    # 处理尾部剩余
                    while j < len(route):
                        d_sub += self.rel_info.get(route[j], {}).get('distance', 1.0)
                        j += 1
                    
                    # 计算 d_plus (多余的/错误的距离)
                    i = k = 0
                    while i < len(lcs) and k < len(merged_result):
                        while k < len(merged_result) and merged_result[k] != lcs[i]:
                            if isinstance(merged_result[k],list):
                                for m in range(len(merged_result[k])):
                                    if merged_result[k][m] != lcs[i]:
                                        break
                                    d_plus += self.rel_info.get(merged_result[k][m], {}).get('distance', 1.0)
                            else:
                                d_plus += self.rel_info.get(merged_result[k], {}).get('distance', 1.0)
                            k += 1
                        i += 1
                        k += 1
                    # 处理尾部剩余
                    while k < len(merged_result):
                        if isinstance(merged_result[k],list):
                                for m in range(len(merged_result[k])):
                                    d_plus += self.rel_info.get(merged_result[k][m], {}).get('distance', 1.0)
                        else:
                            d_plus += self.rel_info.get(merged_result[k], {}).get('distance', 1.0)
                        k += 1

                    RMF = (d_plus + d_sub) / d_total
                    self.evaluate_result[usr_id][traj_id]['RMF'] = RMF
                    final_metrics['RMF'].append(RMF)

                # --- Metric: AN (Accuracy by Number) ---
                if 'AN' in self.metrics:
                    len_route = len(route)
                    AN = len(lcs) / len_route if len_route > 0 else 0
                    self.evaluate_result[usr_id][traj_id]['AN'] = AN
                    final_metrics['AN'].append(AN)

                # --- Metric: AL (Accuracy by Length) ---
                if 'AL' in self.metrics:
                    d_lcs = 0
                    d_tru = 0
                    for rel_id in lcs:
                        d_lcs += self.rel_info.get(rel_id, {}).get('distance', 1.0)
                    for rel_id in route:
                        d_tru += self.rel_info.get(rel_id, {}).get('distance', 1.0)

                    AL = d_lcs / d_tru if d_tru > 0 else 0
                    self.evaluate_result[usr_id][traj_id]['AL'] = AL
                    final_metrics['AL'].append(AL)

        # 汇总平均值
        avg_results = {}
        for m in self.metrics:
            avg_results[m] = np.mean(final_metrics[m]) if final_metrics[m] else 0.0
            
        return avg_results

    def _log_and_save(self, avg_results):
        """
        日志打印与文件保存
        """
        self._logger.info('Test Evaluation Results (Map Matching):')
        for k, v in avg_results.items():
            self._logger.info(f'  {k}: {v:.4f}')

        # 保存结果
        if not os.path.exists(self.evaluate_res_dir):
            os.makedirs(self.evaluate_res_dir)

        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        filename = '{}_{}_{}_{}'.format(
            timestamp, 
            self.config.get('model', 'DeepMM'), 
            self.config.get('dataset', 'unknown'), 
            'evaluate'
        )

        # 保存详细的 per-trajectory JSON
        json_path = os.path.join(self.evaluate_res_dir, '{}.json'.format(filename))
        output_dict = {
            'summary': {k: float(v) for k, v in avg_results.items()},
            'details': self.evaluate_result
        }
        with open(json_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
        
        self._logger.info(f'Evaluation results saved to {json_path}')

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
