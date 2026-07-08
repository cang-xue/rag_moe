import logging
import os
import time
import numpy as np
import torch


from src.base.trainer import BaseTrainer
from src.utils import graph_algo

from src.utils import metrics as mc
import pandas as pd
import csv
from src.utils.helper import move_batch_meta, split_batch


class IMPEL_Trainer(BaseTrainer):
    def __init__(self,
                 unknown_set,
                 known_set,
                 n_m,
                 llm_encoding,
                 **args):
        self.direct_expert = args.pop("direct_expert", "")
        self.allow_raw_prior_direct = bool(args.pop("allow_raw_prior_direct", False))
        self._retriever_aux_weight = float(args.pop("retriever_aux_weight", 0.0))
        self._random_unknown_nodes_each_batch = bool(args.pop("random_unknown_nodes_each_batch", False))
        self._num_random_unknown_nodes = args.pop("num_random_unknown_nodes", None)
        super(IMPEL_Trainer, self).__init__(**args)
        self._supports = self._calculate_supports(args['adj_mat'], args['filter_type'])
        self._unknown_set = unknown_set
        self._known_set = known_set
        self._all_nodes = set(self._unknown_set) | set(self._known_set)
        if self._num_random_unknown_nodes is None:
            self._num_random_unknown_nodes = len(self._unknown_set)
        self._n_m = n_m
        self._llm_encoding = llm_encoding
        

    def _calculate_supports(self, adj_mat, filter_type):
        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "scalap":
            supports = [graph_algo.calculate_scaled_laplacian(new_adj).todense()]
        elif filter_type == "normlap":
            supports = [graph_algo.calculate_normalized_laplacian(
                new_adj).astype(np.float32).todense()]
        elif filter_type == "symnadj":
            supports = [graph_algo.sym_adj(new_adj)]
        elif filter_type == "transition":
            supports = [graph_algo.asym_adj(new_adj)]
        elif filter_type == "doubletransition":
            supports = [graph_algo.asym_adj(new_adj),
                        graph_algo.asym_adj(np.transpose(new_adj))]
        elif filter_type == "identity":
            supports = [np.diag(np.ones(new_adj.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"
        supports = [torch.tensor(i).cuda() for i in supports]
        return supports

    def _training_known_set(self):
        if not self._random_unknown_nodes_each_batch:
            return set(self._known_set)
        unknown_count = min(int(self._num_random_unknown_nodes), len(self._all_nodes))
        sampled_unknown = set(np.random.choice(sorted(self._all_nodes), unknown_count, replace=False).tolist())
        return set(self._all_nodes) - sampled_unknown

    # Rewrite the training and testing procedure for masked training
    def train_batch(self, X, label, iter, batch_meta=None):
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()

        ##Unknown sensors for testing##
        train_known_set = self._training_known_set()
        train_known_nodes = sorted(train_known_set)
        X = X[:, :, train_known_nodes, :]  # [B,S,N,C]
        label = label[:, :, train_known_nodes, :]
        supports = [support[:, train_known_nodes][train_known_nodes, :] for support in supports]
        llm_encoding = self._llm_encoding
        llm_encoding = llm_encoding[train_known_nodes, :]
        ##Masked sensors for inductive training##
        missing_index = np.ones(X.shape)
        for j in range(X.shape[0]):
            missing_mask = np.random.choice(range(0, len(train_known_set)), self._n_m, replace=False)  # Masked locations
            missing_index[j, :, missing_mask, :] = 0
        missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
        X = X * missing_index
        ###############################

        output = self._model_forward_train(
            X,
            supports,
            llm_encoding,
            label=label,
            batch_meta=batch_meta,
        )
        pred, retriever_aux_loss = self._split_prediction_and_aux(output)
        pred, label = self._inverse_transform([pred, label])

        loss = self.loss_fn(pred, label, self.null_value)
        loss = self._add_retriever_aux_loss(loss, retriever_aux_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, data in enumerate(self.data['train_loader']):
                X, label, batch_meta = split_batch(data, self.data.get('batch_meta_keys'))
                X, label = self._check_device([X, label])
                batch_meta = move_batch_meta(batch_meta, X.device)
                train_losses.append(self.train_batch(X, label, iter, batch_meta=batch_meta))
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                                                                                  self._max_epochs,
                                                                                                  iter,
                                                                                                  np.mean(train_losses),
                                                                                                  val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch

            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                      '{:.1f}s'.format(epoch,
                                       self._max_epochs,
                                       iter,
                                       np.mean(train_losses),
                                       val_loss,
                                       new_lr,
                                       (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for data in self.data['val_loader']:
                X, label, batch_meta = split_batch(data, self.data.get('batch_meta_keys'))
                X, label = self._check_device([X, label])
                batch_meta = move_batch_meta(batch_meta, X.device)

                ##Unknown sensors for testing##
                missing_index = np.ones(X.shape)
                missing_index[:, :, list(self._unknown_set), :] = 0
                missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
                X = X * missing_index
                ###############################

                pred, label = self.test_batch(X, label, batch_meta=batch_meta)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels, self.null_value).item()
        return mae

    def test_batch(self, X, label, batch_meta=None):
        pred = self._model_forward(X, self.supports, self._llm_encoding, batch_meta=batch_meta)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

    def _model_forward(self, X, supports, llm_encoding, batch_meta=None):
        direct_expert = getattr(self, "direct_expert", "")
        if getattr(self.model, "name", "") == "rag_moe_impel" and direct_expert:
            return self.model.forward_direct_expert(
                direct_expert,
                X,
                supports,
                llm_encoding,
                batch_meta=batch_meta or {},
                allow_raw_prior=self.allow_raw_prior_direct,
            )
        if direct_expert:
            raise ValueError("--direct_expert requires model_name='rag_moe_impel'")
        if getattr(self.model, "name", "") == "rag_moe_impel":
            return self.model(X, supports, llm_encoding, batch_meta=batch_meta or {})
        return self.model(X, supports, llm_encoding)

    def _model_forward_train(self, X, supports, llm_encoding, label, batch_meta=None):
        if self._should_request_retriever_aux():
            meta = batch_meta or {}
            return self.model(
                X,
                supports,
                llm_encoding,
                x_hour=meta.get("x_hour"),
                x_minute=meta.get("x_minute"),
                x_weekday=meta.get("x_weekday"),
                sample_idx=meta.get("sample_ids", meta.get("sample_idx", meta.get("rag_index"))),
                query_future=label,
                teacher_forcing=True,
                return_aux=True,
            )
        return self._model_forward(X, supports, llm_encoding, batch_meta=batch_meta)

    def _should_request_retriever_aux(self):
        return (
            self._retriever_aux_weight > 0
            and hasattr(self.model, "rag_memory")
            and getattr(self.model, "name", "") != "rag_moe_impel"
        )

    def _split_prediction_and_aux(self, output):
        if isinstance(output, dict):
            return output["pred"], output.get("retriever_aux_loss")
        return output, None

    def _add_retriever_aux_loss(self, loss, retriever_aux_loss):
        if self._retriever_aux_weight > 0 and retriever_aux_loss is not None:
            return loss + self._retriever_aux_weight * retriever_aux_loss
        return loss

    def test(self, epoch, mode='test'):
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []

        start_time = time.time()

        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data[mode + '_loader']):
                X, label, batch_meta = split_batch(data, self.data.get('batch_meta_keys'))
                X, label = self._check_device([X, label])
                batch_meta = move_batch_meta(batch_meta, X.device)

                ##Unknown sensors for testing##
                missing_index = np.ones(X.shape)
                missing_index[:, :, list(self._unknown_set), :] = 0
                missing_index = torch.from_numpy(missing_index.astype('float32')).to(X.device)
                X = X * missing_index
                ###############################

                pred, label = self.test_batch(X, label, batch_meta=batch_meta)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        end_time = time.time()

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        # ###Save the preds and labels##
        # np.save('results/pred_stid_hz.npy', preds)
        # np.save('results/label_stid_hz.npy', labels)
        # ###Save the preds and labels##


        amae = []
        armse = []

        for i in range(self.model.horizon):
            pred = preds[:, i]
            real = labels[:, i]
            metrics = mc.compute_all_metrics(pred, real, self.null_value)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, teat_time {:.1f}s'
            print(log.format(i + 1, metrics[0], metrics[1], (end_time - start_time)))
            amae.append(metrics[0])
            armse.append(metrics[1])

        log = 'On average over {} horizons, Average Test MAE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(self.model.horizon, np.mean(amae), np.mean(armse)))

        result_name = self.model_name
        columns = ['hp', 'end_time', 'time', 'mae', 'rmse']
        if self.direct_expert:
            result_name = '{}_{}_direct'.format(self.model_name, self.direct_expert)
            columns = ['hp', 'direct_expert', 'end_time', 'time', 'mae', 'rmse']
        csv_path = self.result_path + '/{}.csv'.format(result_name)
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, index=False)

        with open(csv_path, 'a+') as f:
            csv_write = csv.writer(f)
            data_row = [self.hp, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), round(end_time - start_time, 2),
                        np.mean(amae), np.mean(armse)]
            if self.direct_expert:
                data_row = [self.hp, self.direct_expert, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                            round(end_time - start_time, 2), np.mean(amae), np.mean(armse)]
            csv_write.writerow(data_row)

        return np.mean(amae)
