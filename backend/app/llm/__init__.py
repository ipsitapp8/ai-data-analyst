"""Hybrid multi-provider LLM routing.

- ``families``: which model family a provider/model belongs to (the Critic's independence is judged on this);
- ``providers``: the hybrid providers' settings and their OpenAI-compatible adapter (Groq, NVIDIA, Mistral,
  OpenRouter). Gemini and the legacy Llama endpoint keep their own adapters in app/agents/;
- ``routing``: each agent's ordered provider chain, and the failover engine that walks it.

All of it is inert unless LLM_HYBRID_ROUTING_ENABLED is on; app/agents/llm_client.py then hands calls here.
Circuit breakers, deadlines and error kinds are app/llm_gateway.py's, shared with the existing path.
"""
