import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service
from gtm_agent.gtm_agent import SYSTEM_PROMPT, get_prospect, send_prospect_email


def _send(prospect, runtime=None):
    return send_prospect_email.func(
        prospect=prospect,
        subject="Hello",
        body="Checking in.",
        runtime=runtime or SimpleNamespace(config={}),
        from_rep={"name": "Rep", "email": "rep@example.com"},
    )


class SendProspectEmailTests(unittest.TestCase):
    def test_disqualified_prospect_is_blocked_and_rep_can_be_told_why(self):
        prospect = get_prospect.func("LEAD-50001")["prospect"]

        result = _send(prospect)

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })
        self.assertIn("explain to the rep", SYSTEM_PROMPT)
        self.assertIn("the email was not sent", SYSTEM_PROMPT)

    def test_missing_authoritative_record_is_blocked(self):
        with patch.object(data_service, "get_prospect_record", return_value=None):
            result = _send({"prospect_id": "UNKNOWN", "email": "lead@example.com"})

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect record was not found; email not sent.",
        })

    def test_eligible_prospect_with_email_is_sent(self):
        with patch.object(
            data_service,
            "get_prospect_record",
            return_value={"disqualified": False},
        ):
            result = _send({
                "prospect_id": "LEAD-ELIGIBLE",
                "name": "Lead",
                "email": "lead@example.com",
            })

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "lead@example.com")

    def test_eligible_prospect_without_email_keeps_validation(self):
        with patch.object(
            data_service,
            "get_prospect_record",
            return_value={"disqualified": False},
        ):
            result = _send({"prospect_id": "LEAD-ELIGIBLE", "name": "Lead"})

        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })
