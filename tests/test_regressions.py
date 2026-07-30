# tests/test_regressions.py
"""
Regression tests for bugs found and fixed in this project's beta hardening
pass. Each test class is named after the observed symptom (not the fix), so
a future reader can match a bug report back to the test that guards it.

Run with: python -m unittest discover -s tests -v
No network/DB/Ollama required - every external dependency (LLM client, RAG
service, ecommerce client) is mocked; only real PIIMasker/GuardrailService
logic runs unmocked, since that's exactly what these regressions are about.
"""
import unittest
from unittest.mock import MagicMock, patch

from common.models import AgentTask, EscalationAgentInputParams, Message
from services.data_pipeline import DataIngestionPipeline
from services.pii_masker import PIIMasker
from services.guardrails import GuardrailService
from services.agents.escalation_agent import EscalationAgent, _SUPPORT_CONTACT


class JohnDoeNameMaskingRegressionTest(unittest.TestCase):
    """Bug: a customer was addressed as 'John Doe' because seed conversation
    text containing that name was ingested into the RAG store unmasked.
    Two independent causes were fixed - both are covered here.
    """

    def test_name_survives_lowercasing_is_masked_not_leaked(self):
        # Root cause 1: data_pipeline.py used to lowercase text *before*
        # masking, and PIIMasker's name pattern only fires on a capitalized
        # name - so lowercasing first silently defeated masking entirely.
        masker = PIIMasker()
        original = "Hi, my name is John Doe, and I want to know about my order 12345."
        masked = masker.mask_text(original.strip()).masked_text
        self.assertNotIn("john doe", masked.lower())
        self.assertIn("[name]", masked.lower())

    def test_ingest_customer_conversations_never_stores_a_literal_name(self):
        # End-to-end through the actual ingestion path (the real regression
        # surface), not just the masker in isolation.
        pii_masker = PIIMasker()
        llm_client = MagicMock()
        llm_client.call_embeddings.return_value = [0.0] * 8
        rag_service = MagicMock()
        classifier_client = MagicMock()

        pipeline = DataIngestionPipeline(pii_masker, llm_client, rag_service, classifier_client)
        from common.models import RawCustomerConversation

        cleaned = pipeline.ingest_customer_conversations([
            RawCustomerConversation(
                id="conv_regress_1",
                text="Hi, my name is John Doe, and I want to know about my order 12345.",
                metadata={"source": "twitter", "user_id": "jd_123"},
            )
        ])

        self.assertNotIn("john doe", cleaned[0].cleaned_text.lower())
        # The chunk actually handed to the RAG store must be clean too -
        # this is what a real customer query would eventually retrieve.
        ingested_doc = rag_service.ingest_document.call_args[0][0]
        self.assertNotIn("john doe", ingested_doc.content.lower())

    def test_live_api_no_longer_seeds_fabricated_customer_conversations(self):
        # Root cause 2: api/dependencies.py's warm_up_services() used to
        # ingest a hardcoded "John Doe" sample conversation into the *live*
        # API's shared RAG store on every startup. That seeding step (and
        # the fixture list backing it) must be gone.
        import api.dependencies as deps

        self.assertFalse(
            hasattr(deps, "_SAMPLE_CONVERSATIONS"),
            "fabricated sample conversations should no longer be defined in the live API's dependency wiring",
        )
        with patch.object(deps, "get_orchestrator"), \
             patch.object(deps, "get_data_pipeline") as mock_get_pipeline, \
             patch.object(deps, "_build_catalog_products", return_value=[]), \
             patch.object(deps, "get_ecommerce_client"):
            mock_pipeline = MagicMock()
            mock_get_pipeline.return_value = mock_pipeline
            deps.warm_up_services()
            mock_pipeline.ingest_customer_conversations.assert_not_called()


class GarbledOllamaOutputRegressionTest(unittest.TestCase):
    """Bug: the local Ollama fallback model degenerated into token-salad
    output (e.g. a fabricated, garbled 'contact us at <mojibake>' span)
    that reached the customer verbatim.
    """

    def test_garbled_output_is_replaced_with_safe_fallback(self):
        guardrails = GuardrailService()
        garbled = (
            "Additionally, please contact our support team directly at "
            "×°ùÁáţęăòƨíŚluħÅöÜl. "
            "A human agent will follow up with you."
        )
        verdict = guardrails.screen_output(garbled)
        self.assertTrue(verdict.flagged)
        self.assertIn("garbled_output", verdict.categories)
        self.assertNotIn("×", verdict.safe_text)

    def test_normal_reply_with_real_contact_info_is_not_flagged_as_garbled(self):
        # Guards against the detector being too aggressive: real support
        # replies (including a real email/phone and the occasional accented
        # product/city name) must pass through untouched.
        guardrails = GuardrailService()
        clean = (
            "Your order ord-1234 has shipped. For anything else, email "
            "support@shopassist.com or call +91-7777777777. Enjoy your café-quality espresso maker!"
        )
        verdict = guardrails.screen_output(clean)
        self.assertNotIn("garbled_output", verdict.categories)


class EscalationInventedContactRegressionTest(unittest.TestCase):
    """Bug (contributing cause): the generative LLM had no real support
    contact to cite on an escalation, so it invented one - which is what
    then came out garbled from the local model. EscalationAgent must now
    supply a real, grounded value.
    """

    def test_escalation_result_carries_the_real_support_contact(self):
        agent = EscalationAgent(MagicMock(), MagicMock(), MagicMock())
        task = AgentTask(
            session_id="sess_1",
            user_id="user_1",
            original_query="my package never arrived",
            intent="escalation_due_to_sub_task_error",
            params=EscalationAgentInputParams(reason="repeated agent failure"),
            conversation_context=[Message(role="user", content="my package never arrived")],
        )
        result = agent.process_task(task)
        self.assertEqual(result.result_data.support_contact, _SUPPORT_CONTACT)
        self.assertIn("@", result.result_data.support_contact)


class RepeatedGreetingRegressionTest(unittest.TestCase):
    """Bug: every reply opened with 'Hi <Name>!' because the customer's name
    stays in the (untrimmed) conversation history for the whole session, and
    the system prompt had no rule against re-greeting every turn (it only
    had one for the bot's own name).
    """

    def test_system_prompt_forbids_greeting_by_name_every_turn(self):
        from services.llm_inference import LLMInferenceService

        with patch.object(LLMInferenceService, "__init__", return_value=None):
            service = LLMInferenceService()
        service.local_client = MagicMock()
        service.generative_client = MagicMock()
        service.generative_model = "test-model"
        service.generative_mode = "json_object"
        service.generative_provider = "local"
        service.generative_local_model = "test-model"

        captured = {}

        def fake_dispatch(*, client, model, mode, messages, schema, temperature, seed=None):
            captured["messages"] = messages
            return schema(response_text="ok")

        with patch.object(service, "_dispatch_structured", side_effect=fake_dispatch):
            from common.models import NLGRequest

            service.call_generative(NLGRequest(
                session_id="sess_1",
                conversation_history=[
                    Message(role="user", content="Hi, my name is Vikas."),
                    Message(role="assistant", content="Hi Vikas! How can I help?"),
                ],
                agent_results=[],
                final_user_intent="general question",
            ))

        system_prompt = captured["messages"][0]["content"]
        self.assertIn("not open with", system_prompt.lower())


class LangfuseNotFlushedOnDockerShutdownRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Bug: the FastAPI/Docker path initialized Langfuse at startup but never
    explicitly flushed it at shutdown, relying solely on an atexit hook that
    isn't guaranteed to run inside Docker's SIGTERM -> SIGKILL grace window.
    main.py's lifespan shutdown phase must now flush explicitly.
    """

    async def test_lifespan_shutdown_flushes_langfuse_client(self):
        import api.main as main

        mock_langfuse_client = MagicMock()
        with patch.object(main, "initialize_langfuse_client"), \
             patch.object(main, "warm_up_services"), \
             patch.object(main, "get_langfuse_client_instance", return_value=mock_langfuse_client):
            async with main.lifespan(main.app):
                pass

        mock_langfuse_client.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
