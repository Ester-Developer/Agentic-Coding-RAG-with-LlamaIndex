"""TLS setup for networks with HTTPS inspection (e.g. NetFree).

Python 3.13 verifies certificates in strict X.509 mode and rejects some inspection CAs
("CA cert does not include key usage extension"). With `network.use_system_cert_store: true`,
verification is delegated to the OS certificate store via `truststore` (what pip does):
certificates are still fully verified — by Windows, against the CAs the machine trusts.
Must run before any Cohere/Pinecone client is created.
"""
from __future__ import annotations

_done = False


def configure_tls(config: dict) -> None:
    global _done
    if _done or not config.get("network", {}).get("use_system_cert_store", False):
        return
    import truststore

    truststore.inject_into_ssl()
    _done = True
