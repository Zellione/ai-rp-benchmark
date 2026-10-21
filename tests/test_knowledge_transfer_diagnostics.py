import unittest

from rpbench.knowledge_transfer_diagnostics import match_case


def transfer(
    kind="explicit_transfer",
    source="mara-vey",
    recipient="finn-calder",
    prior="unknown",
    fact="The key is hidden.",
):
    return {
        "kind": kind,
        "source_character_id": source,
        "recipient_character_id": recipient,
        "recipient_prior_knowledge": prior,
        "fact": fact,
    }


class TransferDiagnosticsTests(unittest.TestCase):

    def test_exact_match(self):
        result = match_case(
            [transfer()], [transfer()]
        )
        self.assertEqual(len(result["matches"]), 1)
        self.assertTrue(
            result["matches"][0]["fact_exact_match"]
        )

    def test_reported_kind_error_detected(self):
        result = match_case(
            [transfer("reported_transfer")],
            [transfer("explicit_transfer")],
        )
        self.assertFalse(
            result["matches"][0]["kind_match"]
        )

    def test_fact_paraphrase_remains_separate(self):
        result = match_case(
            [transfer(fact="The code is 7319.")],
            [transfer(fact="Vault combination: 7319")],
        )
        self.assertFalse(
            result["matches"][0]["fact_exact_match"]
        )

    def test_wrong_recipient_not_aligned(self):
        result = match_case(
            [transfer()],
            [transfer(recipient="alden-voss")],
        )
        self.assertEqual(result["matches"], [])
        self.assertEqual(
            len(result["unmatched_gold"]), 1
        )
        self.assertEqual(
            len(result["unmatched_predicted"]), 1
        )

    def test_rumor_false_positive(self):
        result = match_case(
            [],
            [transfer(kind="reported_transfer")],
        )
        self.assertEqual(result["matches"], [])
        self.assertEqual(
            len(result["unmatched_predicted"]), 1
        )


if __name__ == "__main__":
    unittest.main()
