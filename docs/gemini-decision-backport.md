# Gemini Decision 1 backport

This adds text decisions to the existing 1.102.1 Decisions API backport. It
adapts Google's `/v1beta/decisions` endpoint without upgrading dependencies,
changing the database schema, or changing existing provider request formats.
The provider contract follows upstream PR #51220 and Google's model and
Decisions API documentation, checked on November 14, 2026.

Configure a deployment with `model: gemini/gemini-decision-1` and the existing
Gemini named credential. Call `POST /v1/decisions` (or `/decisions`) with the
existing `state` and named `questions` request format. The adapter converts
state to Google text input, maps yes/no criteria to `true_description` and
`false_description`, and supplies required instructions for criteria-only
questions. Choice and score questions retain their existing application shape.

The Google response's `total_input_tokens` and `total_output_tokens` are
normalized for LiteLLM's existing billing path. Missing input usage or nonzero
output tokens fail validation. The price is $0.10 per million input tokens and
zero output cost, with a 262,144-token input limit. Proxy tests cover virtual-key
authentication, the cost header, SpendLogs, and budget enforcement.

This backport accepts text and JSON state only. The existing gateway limit of
128 questions remains. Google's image, audio, video, PDF and 1,000-question
features require a later gateway update and must not be advertised by the
public catalog for this backport. It does not add the newer OpenAI question
list format or a `/v1/systemone` alias. Existing clients keep using `state`.
Custom API base URLs require an explicit key; the shared Google environment
key is never sent to a caller-selected host.

## Production rollout

This branch alone does not publish a model. Before promotion, verify a full
database backup, record the current immutable image and configuration, rehearse
the candidate's migration mechanism on a private clone, and run the canonical
read-only billing audit. There are no migration or dependency changes in this
backport. Preserve existing live prices: the proxy fetches upstream pricing at
startup, which can change unrelated models. Verify the complete deployed model
price delta before restarting or reloading.

After exact-commit image CI succeeds, switch only the shared LiteLLM service to
the verified digest and check migrations, readiness, logs and billing. Stage the
catalog row hidden, register the model through `/model/new` using the existing
secure Gemini credential reference, verify routing and live prices, then
publish it as an experimental decision model supporting only `decisions`.
Run immediate and settled-window billing audits. No paid synthetic inference
is authorized by the model registration workflow.

Rollback hides the exact public catalog entry first, removes only its
deployment through the management API, and restores the previous immutable
image/configuration if needed. No schema rollback is required by this change.
