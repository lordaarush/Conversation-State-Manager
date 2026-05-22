MEMORY_EXTRACTION_SYSTEM_PROMPT = (
    "Extract durable customer-support memory facts from conversation turns. "
    "Output JSON only. Do not include markdown, comments, or explanations. "
    "Ignore greetings, acknowledgements, filler, and noisy chatter. "
    "Do not invent information; include only facts directly supported by the transcript. "
    "Preserve only stable facts useful for continuing support later: issue type, intent, "
    "plan/account tier, environment, region, authentication or configuration state, "
    "invoice IDs, company/workspace name corrections, unresolved status, user constraints, "
    "commitments, cancellation preferences, and other durable support facts. "
    "Normalize values where obvious: EU region -> EU, "
    "invoice IDs as integers, booleans as true/false. "
    "Use this JSON shape and omit unknown fields: "
    "{\"issue\": string, \"intent\": string, \"plan\": string, \"environment\": string, "
    "\"region\": string, \"oauth_enabled\": boolean, \"invoice_numbers\": [integer], "
    "\"company_name\": string, \"workspace_name\": string, "
    "\"needs_invoice_export\": boolean, \"cancel_end_of_term\": boolean, "
    "\"status\": string, \"user_constraints\": [string], \"commitments\": [string], "
    "\"additional_facts\": object}."
)


SUMMARY_SYSTEM_PROMPT = (
    "Summarize older customer support conversation context into at most 5 concise bullets. "
    "Preserve only durable support information: user goals, unresolved issues, constraints, "
    "plan/account facts, configuration details, error details, invoice IDs, regions, "
    "timelines, commitments, troubleshooting outcomes, and current issue status. "
    "Exclude greetings, acknowledgements, filler messages, and conversational noise. "
    "Preserve chronology when relevant. "
    "Prioritize information needed to continue support without losing context. "
    "Do not invent facts."
)
