# Security Policy

## Supported version

Security fixes are applied to the current `0.2.x` line.

Release artifacts must match their published `SHA256SUMS`. Install production
builds through `memagent-release`; do not install an unverified wheel copied
from chat, email, or an unknown shared folder.

## Secrets

- Keep API keys in `.env` or process environment variables. Never commit `.env`.
- Rotate a key immediately if it appears in logs, prompts, exported memories, or source control.
- Treat memory JSON and generated works as private user data.

## Network behavior

- HTTPS certificates are verified with the operating system trust store.
- Model endpoints and web research are optional; the memory engine works offline.
- Production deployments should restrict outbound hosts and set request budgets.

## Reporting

Report vulnerabilities privately to the product owner with a minimal reproduction. Do not attach real memory stores or API keys.
