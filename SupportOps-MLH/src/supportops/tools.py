"""Read-only synthetic tools. Check customer ownership inside the tool boundary."""
from importlib.resources import files
import json
from supportops.schemas import Citation, OrderView, Principal


def lookup_order(principal: Principal, order_id: str) -> OrderView | None:
    rows = json.loads(files("supportops").joinpath("data/orders.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["order_id"] == order_id and row["customer_id"] == principal.customer_id:
            return OrderView(
                order_id=row["order_id"],
                status=row["status"],
                estimated_delivery=row["estimated_delivery"],
            )
    # Do not distinguish another customer's order from an absent order.
    return None


def read_policy(source_id: str) -> Citation:
    policies = json.loads(
        files("supportops").joinpath("data/policies.json").read_text(encoding="utf-8")
    )
    return Citation.model_validate(policies[source_id])
