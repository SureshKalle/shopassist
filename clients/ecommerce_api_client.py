# clients/ecommerce_api_client.py
import logging
import os
import uuid
from pathlib import Path
from typing import Any

# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

try:
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover - optional dependency for dev/editor
    # Provide lightweight fallbacks so linters/editors don't flag unresolved
    # import and to give a clear error if runtime use is attempted.
    def create_engine(*args, **kwargs):
        raise ImportError(
            "sqlalchemy is required to use EcommerceClient: install sqlalchemy"
        )

    def text(stmt):
        return stmt

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{(BASE_DIR / 'db' / 'shopassist.db').as_posix()}"

# orders.status CHECK constraint (db/schema_sqlite.sql) also allows
# 'delivered', 'cancelled', 'returned' - those are terminal, not cancellable.
_CANCELLABLE_STATUSES = {"pending", "confirmed", "shipped"}

# delete_order() is a hard DELETE (unlike cancel_order()'s status change), so
# it's stricter: only while there's no fulfillment history worth keeping -
# still 'pending' (never confirmed), or already 'cancelled' (nothing to
# lose). A 'delivered' order (the case that prompted this) can never be
# deleted outright - same for 'confirmed'/'shipped'/'returned'.
_DELETABLE_STATUSES = {"pending", "cancelled"}

# add_item()/remove_item() guard against an item mid-fulfillment - same set
# as _CANCELLABLE_STATUSES, reused here under its own name since the two
# guards are conceptually different (one's about the order, one's about the
# item) even though the status set happens to match today.
_UNFULFILLED_ORDER_STATUSES = {"pending", "confirmed", "shipped"}


class EcommerceClient:
    """Data-access layer for the orders/customers/items DB - the one place
    every other module goes through instead of touching SQL or DATABASE_URL
    directly. Locally that's the SQLite dev DB built by db/init_db.py; point
    DATABASE_URL at a real PostgreSQL instance to switch, once that
    database's schema is aligned with this one (see db/README.md).
    Everything here keys on user_id - shopassist's single identifier, sent
    by shopassist-client at login - end to end.
    """

    def __init__(self, database_url: str = None):
        self.engine = create_engine(database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL))
        logger.info("EcommerceClient initialised (engine=%s)", self.engine.url)

    def is_reachable(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Database reachability check: OK")
            return True
        except Exception:
            logger.warning("Database not reachable", exc_info=True)
            return False

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_get_order_details")
    # --- Langfuse Integration End ---
    def get_order_details(self, user_id: str, order_id: str) -> dict[str, Any]:
        logger.info("get_order_details: user_id=%s order_id=%s", user_id, order_id)
        logger.debug("Using database URL: %s", self.engine.url)

        with self.engine.connect() as conn:
            order_row = conn.execute(
                text("SELECT order_id, user_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                logger.warning("get_order_details: order_id=%s not found in orders table", order_id)
                return {"error": "Order not found", "order_id": order_id}

            logger.debug("get_order_details: row found - %s", dict(order_row))

            # Ownership check: only enforce when user_id actually looks like a
            # seeded business key (alum-1001 style - see db/README.md), so it
            # degrades gracefully instead of rejecting every real lookup.
            #
            # TODO: Temporarily commented out - re-enable once shopassist-client
            # sends the real logged-in user_id on every request instead of this
            # being hardcoded (api/routers/chat.py, main_simulation.py).
            #if user_id.startswith("alum-") and user_id != order_row["user_id"]:
            #    return {"error": "Order not found", "order_id": order_id}

            item_rows = conn.execute(
                text(
                    """
                    SELECT i.name, oi.quantity
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    WHERE oi.order_id = :order_id
                    """
                ),
                {"order_id": order_id},
            ).mappings().all()
            logger.debug("get_order_details: %d line item(s) for order_id=%s", len(item_rows), order_id)

        logger.info("get_order_details: order_id=%s status=%s", order_row["order_id"], order_row["status"])
        return {
            "order_id": order_row["order_id"],
            "user_id": order_row["user_id"],
            "status": order_row["status"].capitalize(),
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            # schema_sqlite.sql has no ETA column yet; NLG/StructuredOrderSummary
            # already treat this as optional, so report unknown rather than
            # fabricate a date.
            "estimated_delivery": None,
        }

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_cancel_order")
    # --- Langfuse Integration End ---
    def cancel_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        """Cancel an order in place. Returns the same shape as
        get_order_details() (order_id/user_id/status/items/estimated_delivery)
        so callers can feed either tool's result through the same
        diagnose/summarize pipeline without branching on which one ran - see
        services/agents/order_tracking_agent.py's tool registry.
        """
        logger.info("cancel_order: user_id=%s order_id=%s", user_id, order_id)

        with self.engine.begin() as conn:
            order_row = conn.execute(
                text("SELECT order_id, user_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                logger.warning("cancel_order: order_id=%s not found in orders table", order_id)
                return {"error": "Order not found", "order_id": order_id}

            current_status = order_row["status"].lower()
            if current_status not in _CANCELLABLE_STATUSES:
                logger.info("cancel_order: order_id=%s status=%s is not cancellable", order_id, current_status)
                return {"error": f"Order is already {current_status} and can no longer be cancelled", "order_id": order_id}

            conn.execute(
                text("UPDATE orders SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE order_id = :order_id"),
                {"order_id": order_id},
            )

            item_rows = conn.execute(
                text(
                    """
                    SELECT i.name, oi.quantity
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    WHERE oi.order_id = :order_id
                    """
                ),
                {"order_id": order_id},
            ).mappings().all()

        logger.info("cancel_order: order_id=%s cancelled (was %s)", order_id, current_status)
        return {
            "order_id": order_row["order_id"],
            "user_id": order_row["user_id"],
            "status": "Cancelled",
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            "estimated_delivery": None,
        }

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_create_order")
    # --- Langfuse Integration End ---
    def create_order(self, user_id: str, line_items: list[dict[str, Any]], shipping_address: str | None = None) -> dict[str, Any]:
        """Place a new order. line_items is a list of {"item_id": ..., "quantity": ...}.

        Guards: user_id must exist, every item must exist/be active with
        enough stock_quantity for the requested quantity - the "can this
        actually happen" checking cancel_order() does for order status,
        applied here to inventory instead. Transactional: either every line
        item is inserted and stock decremented, or nothing is (engine.begin()).
        """
        logger.info("create_order: user_id=%s line_items=%s", user_id, line_items)
        if not line_items:
            return {"error": "Order must contain at least one item"}

        with self.engine.begin() as conn:
            customer_row = conn.execute(
                text("SELECT user_id FROM customers WHERE user_id = :user_id"),
                {"user_id": user_id},
            ).first()
            if not customer_row:
                logger.warning("create_order: user_id=%s not found", user_id)
                return {"error": "Customer not found", "user_id": user_id}

            subtotal = 0.0
            resolved_items = []
            for line in line_items:
                item_id = line["item_id"]
                quantity = line["quantity"]
                if quantity <= 0:
                    return {"error": f"Invalid quantity for {item_id}: {quantity}"}

                item_row = conn.execute(
                    text("SELECT item_id, name, price, stock_quantity, is_active FROM items WHERE item_id = :item_id"),
                    {"item_id": item_id},
                ).mappings().first()
                if not item_row or not item_row["is_active"]:
                    return {"error": f"Item not found or unavailable: {item_id}"}
                if item_row["stock_quantity"] < quantity:
                    return {
                        "error": f"Insufficient stock for {item_row['name']}: "
                                 f"{item_row['stock_quantity']} available, {quantity} requested"
                    }

                unit_price = float(item_row["price"])
                subtotal += unit_price * quantity
                resolved_items.append({"item_id": item_id, "name": item_row["name"], "quantity": quantity, "unit_price": unit_price})

            # Same shipping rule as seed_sqlite.sql's own seed data: free at/above
            # Rs. 999 subtotal, flat Rs. 49 below.
            shipping_fee = 0.0 if subtotal >= 999 else 49.0
            total_amount = subtotal + shipping_fee

            order_id = f"ord-{uuid.uuid4().hex[:10]}"
            conn.execute(
                text(
                    """
                    INSERT INTO orders (order_id, user_id, status, subtotal, discount, shipping_fee, total_amount, shipping_address)
                    VALUES (:order_id, :user_id, 'pending', :subtotal, 0, :shipping_fee, :total_amount, :shipping_address)
                    """
                ),
                {
                    "order_id": order_id, "user_id": user_id, "subtotal": subtotal,
                    "shipping_fee": shipping_fee, "total_amount": total_amount, "shipping_address": shipping_address,
                },
            )

            for line in resolved_items:
                conn.execute(
                    text(
                        """
                        INSERT INTO order_items (order_id, item_id, quantity, unit_price, line_total)
                        VALUES (:order_id, :item_id, :quantity, :unit_price, :line_total)
                        """
                    ),
                    {
                        "order_id": order_id, "item_id": line["item_id"], "quantity": line["quantity"],
                        "unit_price": line["unit_price"], "line_total": line["unit_price"] * line["quantity"],
                    },
                )
                conn.execute(
                    text("UPDATE items SET stock_quantity = stock_quantity - :quantity WHERE item_id = :item_id"),
                    {"quantity": line["quantity"], "item_id": line["item_id"]},
                )

        logger.info("create_order: order_id=%s created for user_id=%s (total_amount=%.2f)", order_id, user_id, total_amount)
        return {
            "order_id": order_id,
            "user_id": user_id,
            "status": "Pending",
            "items": [{"name": line["name"], "qty": line["quantity"]} for line in resolved_items],
            "total_amount": total_amount,
            "estimated_delivery": None,
        }

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_delete_order")
    # --- Langfuse Integration End ---
    def delete_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        """Permanently delete an order and its line items (ON DELETE CASCADE -
        schema_sqlite.sql/schema_postgres.sql). Unlike cancel_order() (a status
        change that keeps the row as a permanent record), this removes it
        outright - only allowed for _DELETABLE_STATUSES (module-level constant
        above): still 'pending', or already 'cancelled'. A 'delivered' order -
        or 'confirmed'/'shipped'/'returned' - can never be deleted this way,
        only cancelled via the normal flow.

        Returns the same shape as get_order_details()/cancel_order()
        (order_id/user_id/status/items/estimated_delivery) - status="Deleted" -
        so callers (services/agents/order_tracking_agent.py's tool registry)
        can feed any of the three through the same diagnose/summarize pipeline.
        """
        logger.info("delete_order: user_id=%s order_id=%s", user_id, order_id)

        with self.engine.begin() as conn:
            order_row = conn.execute(
                text("SELECT order_id, user_id, status FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                logger.warning("delete_order: order_id=%s not found in orders table", order_id)
                return {"error": "Order not found", "order_id": order_id}

            current_status = order_row["status"].lower()
            if current_status not in _DELETABLE_STATUSES:
                logger.info("delete_order: order_id=%s status=%s cannot be deleted", order_id, current_status)
                return {
                    "error": f"Order is {current_status} and can no longer be deleted - it can only be cancelled",
                    "order_id": order_id,
                }

            # Ownership check intentionally left disabled, same as
            # get_order_details()/cancel_order() above - see the TODO on
            # get_order_details() for why.
            #if user_id.startswith("alum-") and user_id != order_row["user_id"]:
            #    return {"error": "Order not found", "order_id": order_id}

            item_rows = conn.execute(
                text(
                    """
                    SELECT i.name, oi.quantity
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    WHERE oi.order_id = :order_id
                    """
                ),
                {"order_id": order_id},
            ).mappings().all()

            conn.execute(text("DELETE FROM orders WHERE order_id = :order_id"), {"order_id": order_id})

        logger.info("delete_order: order_id=%s deleted (was %s)", order_id, current_status)
        return {
            "order_id": order_row["order_id"],
            "user_id": order_row["user_id"],
            "status": "Deleted",
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            "estimated_delivery": None,
        }

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_get_customer_history")
    # --- Langfuse Integration End ---
    def get_customer_history(self, user_id: str) -> dict[str, Any]:
        logger.info("get_customer_history: user_id=%s", user_id)

        with self.engine.connect() as conn:
            last_purchase = conn.execute(
                text(
                    """
                    SELECT i.name
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.user_id = :user_id
                    ORDER BY o.placed_at DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id},
            ).scalar()

            if last_purchase is None:
                logger.warning("get_customer_history: no order history for user_id=%s", user_id)
                return {"error": "Customer history not found", "user_id": user_id}

            favorite_category = conn.execute(
                text(
                    """
                    SELECT i.category
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.user_id = :user_id
                    GROUP BY i.category
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id},
            ).scalar()

        logger.debug(
            "get_customer_history: user_id=%s last_purchase=%s favorite_category=%s",
            user_id, last_purchase, favorite_category,
        )
        return {"last_purchase": last_purchase, "favorite_category": favorite_category}

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_get_customer")
    # --- Langfuse Integration End ---
    def get_customer(self, user_id: str) -> dict[str, Any] | None:
        logger.info("get_customer: user_id=%s", user_id)
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT user_id, first_name, last_name, email, phone, city, state, country
                    FROM customers WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            ).mappings().first()
        if not row:
            logger.warning("get_customer: user_id=%s not found", user_id)
            return None
        return dict(row)

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_get_item")
    # --- Langfuse Integration End ---
    def get_item(self, item_id: str) -> dict[str, Any] | None:
        logger.info("get_item: item_id=%s", item_id)
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT item_id, name, description, category, price, mrp, stock_quantity
                    FROM items WHERE item_id = :item_id
                    """
                ),
                {"item_id": item_id},
            ).mappings().first()
        if not row:
            logger.warning("get_item: item_id=%s not found", item_id)
            return None
        return dict(row)

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_search_items")
    # --- Langfuse Integration End ---
    def search_items(self, category: str | None = None, keyword: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        logger.info("search_items: category=%s keyword=%s limit=%d", category, keyword, limit)
        clauses = ["is_active"]
        params: dict[str, Any] = {"limit": limit}
        if category:
            clauses.append("category = :category")
            params["category"] = category
        if keyword:
            clauses.append("(name LIKE :keyword OR description LIKE :keyword)")
            params["keyword"] = f"%{keyword}%"
        where = " AND ".join(clauses)

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT item_id, name, description, category, price, mrp, stock_quantity
                    FROM items WHERE {where} ORDER BY name LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()
        logger.debug("search_items: %d result(s)", len(rows))
        return [dict(row) for row in rows]

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_add_item")
    # --- Langfuse Integration End ---
    def add_item(
        self,
        item_id: str,
        name: str,
        price: float,
        description: str | None = None,
        category: str | None = None,
        mrp: float | None = None,
        stock_quantity: int = 0,
    ) -> dict[str, Any]:
        """Add a new catalog item. Guard: item_id must not already exist -
        items.item_id is the primary key (same business-key convention as
        orders/customers, db/README.md) - so this returns a clean error
        instead of raising a raw IntegrityError.
        """
        logger.info("add_item: item_id=%s name=%s", item_id, name)
        with self.engine.begin() as conn:
            existing = conn.execute(
                text("SELECT item_id FROM items WHERE item_id = :item_id"), {"item_id": item_id}
            ).first()
            if existing:
                return {"error": f"Item {item_id} already exists", "item_id": item_id}

            conn.execute(
                text(
                    """
                    INSERT INTO items (item_id, name, description, category, price, mrp, stock_quantity)
                    VALUES (:item_id, :name, :description, :category, :price, :mrp, :stock_quantity)
                    """
                ),
                {
                    "item_id": item_id, "name": name, "description": description, "category": category,
                    "price": price, "mrp": mrp, "stock_quantity": stock_quantity,
                },
            )
        logger.info("add_item: item_id=%s added", item_id)
        return {"item_id": item_id, "name": name, "added": True}

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_remove_item")
    # --- Langfuse Integration End ---
    def remove_item(self, item_id: str) -> dict[str, Any]:
        """Deactivate a catalog item (is_active = FALSE) rather than a hard
        DELETE - order_items.item_id has ON DELETE RESTRICT (schema_sqlite.sql),
        so a real delete would fail once anything has ever been ordered anyway.
        Guard: also blocked while the item is part of any order that hasn't
        reached a terminal status yet (_UNFULFILLED_ORDER_STATUSES) - don't
        pull an item out from under a customer who's still waiting on it.
        """
        logger.info("remove_item: item_id=%s", item_id)
        with self.engine.begin() as conn:
            item_row = conn.execute(
                text("SELECT item_id, is_active FROM items WHERE item_id = :item_id"), {"item_id": item_id}
            ).mappings().first()
            if not item_row:
                return {"error": "Item not found", "item_id": item_id}
            if not item_row["is_active"]:
                return {"error": "Item is already inactive", "item_id": item_id}

            unfulfilled = conn.execute(
                text(
                    """
                    SELECT 1 FROM order_items oi
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE oi.item_id = :item_id AND o.status IN ('pending', 'confirmed', 'shipped')
                    LIMIT 1
                    """
                ),
                {"item_id": item_id},
            ).first()
            if unfulfilled:
                return {"error": "Item is part of an unfulfilled order and cannot be removed yet", "item_id": item_id}

            conn.execute(
                text("UPDATE items SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE item_id = :item_id"),
                {"item_id": item_id},
            )

        logger.info("remove_item: item_id=%s deactivated", item_id)
        return {"item_id": item_id, "removed": True}

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_list_orders_for_customer")
    # --- Langfuse Integration End ---
    def list_orders_for_customer(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        logger.info("list_orders_for_customer: user_id=%s limit=%d", user_id, limit)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT order_id, status, total_amount, placed_at
                    FROM orders WHERE user_id = :user_id
                    ORDER BY placed_at DESC LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": limit},
            ).mappings().all()
        logger.debug("list_orders_for_customer: %d order(s) for user_id=%s", len(rows), user_id)
        return [dict(row) for row in rows]

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_get_popular_category")
    # --- Langfuse Integration End ---
    def get_popular_category(self, limit: int = 1) -> list[dict[str, Any]]:
        """Categories ranked by total units sold, most popular first.

        Excludes cancelled/returned orders so a cancellation doesn't still
        count toward a category's popularity.
        """
        logger.info("get_popular_category: limit=%d", limit)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT i.category, SUM(oi.quantity) AS units_sold
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.status NOT IN ('cancelled', 'returned') AND i.category IS NOT NULL
                    GROUP BY i.category
                    ORDER BY units_sold DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        logger.debug("get_popular_category: %d categor(y/ies)", len(rows))
        return [dict(row) for row in rows]

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_add_review")
    # --- Langfuse Integration End ---
    def add_review(self, item_id: str, user_id: str, review_title: str, review_content: str) -> dict[str, Any]:
        """Add a review. Guard: user_id must actually have bought item_id - an
        order_items row for this item on one of their orders that isn't
        'cancelled' (never delivered) - a "verified purchase" check. This is
        why item_reviews.user_id is a real FK to customers rather than a
        free-text name in the first place (schema_sqlite.sql's header note).
        """
        logger.info("add_review: item_id=%s user_id=%s", item_id, user_id)
        with self.engine.begin() as conn:
            purchase_row = conn.execute(
                text(
                    """
                    SELECT 1 FROM order_items oi
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE oi.item_id = :item_id AND o.user_id = :user_id AND o.status != 'cancelled'
                    LIMIT 1
                    """
                ),
                {"item_id": item_id, "user_id": user_id},
            ).first()
            if not purchase_row:
                return {"error": "You can only review items you've purchased", "item_id": item_id, "user_id": user_id}

            review_id = f"rev-{uuid.uuid4().hex[:10]}"
            conn.execute(
                text(
                    """
                    INSERT INTO item_reviews (review_id, item_id, user_id, review_title, review_content)
                    VALUES (:review_id, :item_id, :user_id, :review_title, :review_content)
                    """
                ),
                {
                    "review_id": review_id, "item_id": item_id, "user_id": user_id,
                    "review_title": review_title, "review_content": review_content,
                },
            )
        logger.info("add_review: review_id=%s added for item_id=%s by user_id=%s", review_id, item_id, user_id)
        return {"review_id": review_id, "item_id": item_id, "user_id": user_id, "added": True}

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="ecommerce_client_remove_review")
    # --- Langfuse Integration End ---
    def remove_review(self, review_id: str, user_id: str) -> dict[str, Any]:
        """Delete a review. Guard: only the review's own author can delete it."""
        logger.info("remove_review: review_id=%s user_id=%s", review_id, user_id)
        with self.engine.begin() as conn:
            review_row = conn.execute(
                text("SELECT review_id, user_id FROM item_reviews WHERE review_id = :review_id"),
                {"review_id": review_id},
            ).mappings().first()
            if not review_row:
                return {"error": "Review not found", "review_id": review_id}
            if review_row["user_id"] != user_id:
                return {"error": "You can only delete your own reviews", "review_id": review_id}

            conn.execute(text("DELETE FROM item_reviews WHERE review_id = :review_id"), {"review_id": review_id})

        logger.info("remove_review: review_id=%s deleted", review_id)
        return {"review_id": review_id, "deleted": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    client = EcommerceClient()
    order = client.get_order_details("alum-1001", "ord-1001")
    print(f"Sample Order Details: {order}")
    history = client.get_customer_history("alum-1001")
    print(f"Sample Customer History: {history}")
    print(f"Reachable: {client.is_reachable()}")
