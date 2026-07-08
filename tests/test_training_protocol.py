import unittest


class TrainingProtocolTest(unittest.TestCase):
    def test_split_source_target_excludes_target_city(self):
        from experiments.training.protocol import DEFAULT_CITIES, split_source_target

        sources, target = split_source_target(DEFAULT_CITIES, "Delivery_HZ")

        self.assertEqual(target, "Delivery_HZ")
        self.assertNotIn("Delivery_HZ", sources)
        self.assertEqual(
            sources,
            ["Delivery_SH", "Delivery_CQ", "Delivery_YT", "Delivery_JL"],
        )

    def test_split_source_target_rejects_unknown_target(self):
        from experiments.training.protocol import DEFAULT_CITIES, split_source_target

        with self.assertRaisesRegex(ValueError, "target_city"):
            split_source_target(DEFAULT_CITIES, "Delivery_UNKNOWN")

    def test_parse_city_list_trims_and_rejects_duplicates(self):
        from experiments.training.protocol import parse_city_list

        self.assertEqual(
            parse_city_list("Delivery_SH, Delivery_HZ"),
            ["Delivery_SH", "Delivery_HZ"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_city_list("Delivery_SH,Delivery_SH")


if __name__ == "__main__":
    unittest.main()
