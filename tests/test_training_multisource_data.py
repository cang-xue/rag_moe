import unittest

import numpy as np
import torch


class TrainingMultiSourceDataTest(unittest.TestCase):
    def test_build_city_contexts_excludes_target_and_keeps_per_city_runtime_state(self):
        from experiments.training.multisource_data import build_city_contexts

        def fake_loader(datapath, batch_size, input_dim, output_dim, include_metadata=False):
            city = datapath.split("/")[-1]
            return {
                "train_loader": [city + "_train"],
                "val_loader": [city + "_val"],
                "test_loader": [city + "_test"],
                "scalers": [city + "_scaler"],
                "batch_meta_keys": ["sample_ids"] if include_metadata else [],
            }

        def fake_num_nodes(city):
            return {"Delivery_SH": 3, "Delivery_HZ": 4, "Delivery_CQ": 2}[city]

        def fake_llm_loader(city):
            return np.ones((fake_num_nodes(city), 5), dtype="float32") * len(city)

        def fake_support_loader(city):
            return np.eye(fake_num_nodes(city), dtype="float32")

        protocol, contexts = build_city_contexts(
            ["Delivery_SH", "Delivery_HZ", "Delivery_CQ"],
            "Delivery_HZ",
            batch_size=2,
            input_dim=1,
            output_dim=1,
            num_unknown_nodes=1,
            num_masked_nodes=1,
            dataloader_fn=fake_loader,
            num_nodes_fn=fake_num_nodes,
            null_value_fn=lambda city: -1.0,
            llm_loader_fn=fake_llm_loader,
            support_loader_fn=fake_support_loader,
        )

        self.assertEqual(protocol.source_cities, ["Delivery_SH", "Delivery_CQ"])
        self.assertNotIn("Delivery_HZ", contexts)
        self.assertEqual(contexts["Delivery_SH"].city_id, 0)
        self.assertEqual(contexts["Delivery_CQ"].city_id, 1)
        self.assertEqual(contexts["Delivery_SH"].num_nodes, 3)
        self.assertEqual(contexts["Delivery_CQ"].llm_encoding.shape, torch.Size([2, 5]))
        self.assertEqual(contexts["Delivery_SH"].supports[0].shape, torch.Size([3, 3]))
        self.assertEqual(len(contexts["Delivery_SH"].unknown_set), 1)
        self.assertEqual(contexts["Delivery_SH"].num_masked_nodes, 1)


if __name__ == "__main__":
    unittest.main()
