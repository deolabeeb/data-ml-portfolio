"""DEMO ONLY: published fixture tokens, not production authentication."""
from secrets import compare_digest
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supportops.schemas import Principal

bearer = HTTPBearer(auto_error=False)
DEMO_CUSTOMERS = {
    "demo-customer-a": "cust_demo_a",
    "demo-customer-b": "cust_demo_b",
}


def require_customer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if credentials is not None and credentials.scheme.lower() == "bearer":
        for token, customer_id in DEMO_CUSTOMERS.items():
            if compare_digest(credentials.credentials.encode(), token.encode()):
                return Principal(customer_id=customer_id)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid local-demo customer token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
