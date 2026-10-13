import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service, gtm_agent


class ProspectTechnologyUpdateTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.original_record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.original_profile = data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        data_service.PROSPECTS[self.prospect_id] = self.original_record
        data_service._PROFILES.pop(self.prospect_id, None)
        if self.original_profile is not None:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_technology_and_preserves_result_shape(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertEqual(
            result,
            {
                "updated": True,
                "found": True,
                "tech_stack": [*self.original_record["tech_stack"], "Kafka"],
            },
        )
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))

    def test_update_invalidates_cached_profile(self):
        original = gtm_agent.build_prospect_profile.func(self.prospect_id)
        self.assertNotIn("Kafka", original["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        updated = gtm_agent.build_prospect_profile.func(self.prospect_id)

        self.assertIn("Kafka", updated["prospect_profile"]["tech_stack"])

    def test_adding_existing_technology_does_not_duplicate_it(self):
        existing_technology = self.original_record["tech_stack"][0]

        result = data_service.update_prospect_info(self.prospect_id, existing_technology)

        self.assertEqual(result["updated"], True)
        self.assertEqual(result["found"], True)
        self.assertEqual(result["tech_stack"].count(existing_technology), 1)

    def test_unknown_prospect_does_not_change_records_or_profiles(self):
        other_id = "LEAD-12853"
        original_other_record = copy.deepcopy(data_service.PROSPECTS[other_id])
        original_other_profile = data_service._PROFILES.get(other_id)
        result = data_service.update_prospect_info("UNKNOWN", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertEqual(data_service.PROSPECTS[other_id], original_other_record)
        self.assertEqual(data_service._PROFILES.get(other_id), original_other_profile)
        self.assertNotIn("UNKNOWN", data_service._PROFILES)

    def test_update_then_score_prompt_uses_new_technology(self):
        profile = gtm_agent.build_prospect_profile.func(self.prospect_id)["prospect_profile"]
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 1,
            "description": "Test offering",
        }
        score_result = {
            "score": 90,
            "max_score": 100,
            "justification": "Kafka is present in the prospect's tech stack.",
            "rubric_breakdown": {
                "revenue_fit": 30,
                "tech_stack_match": 30,
                "segment_fit": 30,
                "component_max": 100,
            },
        }

        mock_llm = SimpleNamespace(
            invoke=Mock(return_value=SimpleNamespace(model_dump=lambda: score_result))
        )
        with patch.object(gtm_agent, "_scoring_llm", mock_llm):
            result = gtm_agent.score_prospect.func(profile, offering)

        messages = mock_llm.invoke.call_args.args[0]
        prompt_profile = json.loads(
            messages[1]["content"].split("\n\nProspect profile:\n", 1)[1]
        )
        self.assertIn("Kafka", prompt_profile["tech_stack"])
        self.assertIn("Kafka", offering["required_tech_stack"])
        self.assertIn("Kafka is present", result["justification"])


if __name__ == "__main__":
    unittest.main()
