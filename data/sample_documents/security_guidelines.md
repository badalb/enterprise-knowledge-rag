# Security Guidelines

Synthetic documentation for a fictional enterprise.

## Authentication

Internal APIs validate JWT signatures, issuer, audience and expiry. OAuth2 access tokens are used for authorization. Never log access tokens or credentials.

## Authorization

Apply least privilege and service-specific scopes. Rotate secrets through the approved secret manager. Report suspected credential exposure to security on-call.
