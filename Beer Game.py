import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Styling to avoid "ChatGPT vibes"
# -----------------------------
APP_TITLE = "Food Supply Chain Ops Console"
st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
div[data-testid="stChatMessage"] { border-radius: 14px; }
div[data-testid="stChatMessage"] p { font-size: 0.98rem; }
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Simulation Model
# -----------------------------
ROLES = ["Retailer (Grocery)", "Regional Distributor", "National DC", "Food Producer"]
ROLE_KEYS = ["retailer", "regional", "dc", "producer"]

LEAD_TIME_WEEKS = 2
SPOILAGE_RATE = 0.10  # 10% of on-hand at end of week (food-specific)
HOLDING_COST = 1.0
BACKORDER_COST = 2.0


def default_demand_schedule(n_weeks: int):
    """
    Food distribution-friendly shock:
      Weeks 1-3: 4
      Weeks 4-6: 6 (promo/holiday)
      Weeks 7+: 4
    """
    schedule = []
    for w in range(1, n_weeks + 1):
        schedule.append(6 if 4 <= w <= 6 else 4)
    return schedule


def init_state(n_weeks: int):
    state = {
        "week": 1,
        "n_weeks": n_weeks,
        "demand_schedule": default_demand_schedule(n_weeks),
        "nodes": {},
        "log": [],
        "history": [],
        "mode": "Human",  # Human or Agentic AI
        "human_role": "retailer",
        "finished": False,
    }

    # Start steady-state: on_hand 12, inbound pipeline has 4 arriving each week for lead time
    for k in ROLE_KEYS:
        state["nodes"][k] = {
            "on_hand": 12.0,
            "backorder": 0.0,
            "inbound_pipeline": [4.0] * LEAD_TIME_WEEKS,
            "order_last": 4.0,
            "waste_cum": 0.0,
            "cost_cum": 0.0,
        }

    state["log"].append(
        ("system", f"{APP_TITLE} initialized. Lead time={LEAD_TIME_WEEKS}w, spoilage={int(SPOILAGE_RATE*100)}%.")
    )
    state["log"].append(("system", "Round 1: Human ordering. Limited visibility. Type your order each week."))
    return state


def inventory_position(node):
    # Inventory position = on_hand + on_order - backorder
    return node["on_hand"] + sum(node["inbound_pipeline"]) - node["backorder"]


def downstream_key(role_key):
    mapping = {
        "producer": "dc",
        "dc": "regional",
        "regional": "retailer",
        "retailer": None,
    }
    return mapping[role_key]


def local_heuristic_order(role_key, state, observed_demand):
    """
    Non-coordinated, 'human-like' local policy:
      - order-up-to inventory position based on observed downstream demand proxy
      - mild panic if backorders exist
    """
    node = state["nodes"][role_key]
    avg = float(observed_demand)
    target = avg * (LEAD_TIME_WEEKS + 1)  # ~3 weeks of demand in position
    pos = inventory_position(node)
    gap = target - pos

    panic = 1.0 + (0.25 if node["backorder"] > 0 else 0.0)
    order = max(0.0, gap) * panic

    # mild smoothing
    order = 0.6 * order + 0.4 * node["order_last"]
    return round(order, 2)


def agentic_ai_orders(state, end_customer_demand):
    """
    Coordinated agentic policy:
      - Everyone 'sees' end-customer demand and system state
      - Rolling 3-week average forecast
      - Dampened order changes (reduces bullwhip + waste)
    """
    hist = [h["end_customer_demand"] for h in state["history"]]
    window = (hist + [end_customer_demand])[-3:]
    forecast = sum(window) / len(window)

    targets = {k: forecast * (LEAD_TIME_WEEKS + 1) for k in ROLE_KEYS}

    orders = {}
    for k in ROLE_KEYS:
        node = state["nodes"][k]
        pos = inventory_position(node)
        raw = max(0.0, targets[k] - pos)
        order = 0.5 * raw + 0.5 * node["order_last"]  # dampen swings
        orders[k] = round(order, 2)
    return orders


def fulfill_and_cost(role_key, demand, state):
    node = state["nodes"][role_key]

    # Receive inbound shipment
    incoming = node["inbound_pipeline"].pop(0)
    node["on_hand"] += incoming

    # Fulfill backlog first, then current demand
    total_required = node["backorder"] + float(demand)
    shipped = min(node["on_hand"], total_required)
    node["on_hand"] -= shipped
    node["backorder"] = total_required - shipped

    # Spoilage (end of week)
    spoilage = node["on_hand"] * SPOILAGE_RATE
    node["on_hand"] -= spoilage
    node["waste_cum"] += spoilage

    # Costs
    holding = node["on_hand"] * HOLDING_COST
    backcost = node["backorder"] * BACKORDER_COST
    total_cost = holding + backcost + spoilage  # treat spoilage as waste cost
    node["cost_cum"] += total_cost

    return {
        "incoming": incoming,
        "shipped": shipped,
        "spoilage": spoilage,
        "holding": holding,
        "backcost": backcost,
        "total_cost": total_cost,
    }


def advance_week(state, human_order=None):
    w = state["week"]
    end_customer_demand = state["demand_schedule"][w - 1]

    # Decide orders
    if state["mode"] == "Agentic AI":
        orders = agentic_ai_orders(state, end_customer_demand)
    else:
        orders = {}
        for k in ROLE_KEYS:
            if k == state["human_role"]:
                orders[k] = float(human_order) if human_order is not None else state["nodes"][k]["order_last"]
            else:
                # Observe last week's downstream order as a proxy (limited visibility)
                if len(state["history"]) == 0:
                    observed = 4.0
                else:
                    dk = downstream_key(k)
                    observed = state["history"][-1].get(f"order_{dk}", 4.0) if dk else 4.0
                orders[k] = local_heuristic_order(k, state, observed)

    # Save last order
    for k in ROLE_KEYS:
        state["nodes"][k]["order_last"] = orders[k]

    # Demand into each tier
    tier_demand = {
        "retailer": end_customer_demand,
        "regional": orders["retailer"],
        "dc": orders["regional"],
        "producer": orders["dc"],
    }

    # Fulfillment/cost per tier (downstream -> upstream)
    metrics = {}
    for k in ROLE_KEYS:
        metrics[k] = fulfill_and_cost(k, tier_demand[k], state)

    # Shipments arrive after lead time:
    state["nodes"]["dc"]["inbound_pipeline"].append(orders["producer"])
    state["nodes"]["regional"]["inbound_pipeline"].append(orders["dc"])
    state["nodes"]["retailer"]["inbound_pipeline"].append(orders["regional"])
    state["nodes"]["producer"]["inbound_pipeline"].append(0.0)  # producer has no inbound in this simplified model

    # Record history
    record = {
        "week": w,
        "end_customer_demand": end_customer_demand,
        "order_retailer": orders["retailer"],
        "order_regional": orders["regional"],
        "order_dc": orders["dc"],
        "order_producer": orders["producer"],
    }
    for k in ROLE_KEYS:
        node = state["nodes"][k]
        record[f"{k}_on_hand"] = node["on_hand"]
        record[f"{k}_backorder"] = node["backorder"]
        record[f"{k}_waste_cum"] = node["waste_cum"]
        record[f"{k}_cost_cum"] = node["cost_cum"]
        record[f"{k}_incoming"] = metrics[k]["incoming"]
        record[f"{k}_spoilage"] = metrics[k]["spoilage"]
        record[f"{k}_week_cost"] = metrics[k]["total_cost"]
    state["history"].append(record)

    # Log
    state["log"].append(("system", f"Week {w} complete. End-customer demand={end_customer_demand}."))
    state["log"].append(
        ("system", f"Orders: Retailer={orders['retailer']}, Regional={orders['regional']}, DC={orders['dc']}, Producer={orders['producer']}.")
    )

    # Advance time
    if w >= state["n_weeks"]:
        state["finished"] = True
        state["log"].append(("system", "Run finished. Review analysis below, then run Agentic AI replay."))
    else:
        state["week"] += 1


def render_charts(df: pd.DataFrame, title_suffix="", y_max_orders=None):
    # Bullwhip chart (LOCKED SCALE)
    fig1 = plt.figure()
    plt.plot(df["week"], df["end_customer_demand"], label="End-Customer Demand")
    plt.plot(df["week"], df["order_retailer"], label="Retailer Orders")
    plt.plot(df["week"], df["order_regional"], label="Regional Orders")
    plt.plot(df["week"], df["order_dc"], label="DC Orders")
    plt.plot(df["week"], df["order_producer"], label="Producer Orders")
    plt.xlabel("Week")
    plt.ylabel("Units")
    plt.title(f"Bullwhip – Orders Over Time{title_suffix}")
    if y_max_orders is not None:
        plt.ylim(0, y_max_orders)
    plt.legend()
    st.pyplot(fig1)

    # Waste chart (autoscale)
    fig2 = plt.figure()
    plt.plot(df["week"], df["retailer_waste_cum"], label="Retailer Waste (cum)")
    plt.plot(df["week"], df["regional_waste_cum"], label="Regional Waste (cum)")
    plt.plot(df["week"], df["dc_waste_cum"], label="DC Waste (cum)")
    plt.plot(df["week"], df["producer_waste_cum"], label="Producer Waste (cum)")
    plt.xlabel("Week")
    plt.ylabel("Units wasted")
    plt.title(f"Food Waste (Spoilage) – Cumulative{title_suffix}")
    plt.legend()
    st.pyplot(fig2)


def summarize(df: pd.DataFrame):
    last = df.iloc[-1]
    total_waste = (
        last["retailer_waste_cum"]
        + last["regional_waste_cum"]
        + last["dc_waste_cum"]
        + last["producer_waste_cum"]
    )
    total_cost = (
        last["retailer_cost_cum"]
        + last["regional_cost_cum"]
        + last["dc_cost_cum"]
        + last["producer_cost_cum"]
    )
    return float(total_waste), float(total_cost)


def add_post_run_narrative():
    """Narrative: what went wrong in Human round + recommendations."""
    st.subheader("Post-Run Analysis: What Happened & Why")

    st.markdown("""
**1) Demand was stable, but orders were not**  
The demand change was small and temporary, but orders amplified upstream — classic bullwhip behaviour driven by local signals.

**2) Delayed feedback caused over-correction**  
With a 2-week lead time, decisions were made before prior actions showed up, leading to overshoot (stockouts followed by excess inventory).

**3) Food waste was a delayed consequence, not a mistake**  
Perishable inventory arrived after demand normalized, so spoilage increased later. Waste is a structural outcome of delayed over-ordering.

**4) Local optimisation increased system cost**  
Each role behaved rationally with limited visibility, but the system became unstable because decisions were not coordinated end-to-end.
    """)

    st.markdown("### Recommendations")
    st.markdown("""
- **Coordinate decisions, not just forecasts:** better forecasts don’t prevent volatility if replenishment decisions still overreact under lead times.  
- **Make inventory risk explicit:** decide where buffer should live (retail vs DC), because unmanaged risk turns into waste.  
- **Use AI to stabilize the system:** value comes from dampening overreaction and synchronizing replenishment across tiers.
    """)


# -----------------------------
# UI
# -----------------------------
st.title("📦 " + APP_TITLE)
st.caption(
    "Beer Game-style simulation for food distribution: lead time, perishability, bullwhip, and an Agentic AI replay for waste reduction."
)

if "sim" not in st.session_state:
    st.session_state.sim = init_state(n_weeks=12)

sim = st.session_state.sim

with st.sidebar:
    # --- Logo ---
    st.image("Full Lockup - Colour.png", use_container_width=True)
    st.markdown("---")

    st.header("Run Controls")
    sim["n_weeks"] = st.slider("Weeks", 8, 20, int(sim["n_weeks"]))
    if st.button("Reset Simulation"):
        st.session_state.sim = init_state(n_weeks=int(sim["n_weeks"]))
        st.rerun()

    st.divider()
    sim["human_role"] = st.selectbox(
        "You control (Human Round)",
        options=ROLE_KEYS,
        format_func=lambda k: ROLES[ROLE_KEYS.index(k)],
    )
    sim["mode"] = st.selectbox("Mode", ["Human", "Agentic AI"])
    st.write("Lead time:", LEAD_TIME_WEEKS, "weeks")
    st.write("Spoilage:", f"{int(SPOILAGE_RATE*100)}%/week")
    st.write("Costs: holding=1, backorder=2, waste=1 per unit")

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Operations Feed")

    for who, msg in sim["log"][-40:]:
        with st.chat_message("assistant" if who == "system" else "user"):
            st.markdown(msg)

    st.divider()

    if not sim["finished"]:
        w = sim["week"]
        end_customer_demand = sim["demand_schedule"][w - 1]
        role = sim["human_role"]
        node = sim["nodes"][role]

        with st.chat_message("assistant"):
            st.markdown(
                f"**Week {w}**\n\n"
                f"- End-customer demand (only visible at Retailer): **{end_customer_demand if role=='retailer' else '—'}**\n"
                f"- Your on-hand: **{node['on_hand']:.2f}**\n"
                f"- Your backorders: **{node['backorder']:.2f}**\n"
                f"- Your inbound arriving now: **{node['inbound_pipeline'][0]:.2f}**\n"
                f"- Your last order: **{node['order_last']:.2f}**\n"
            )

        if sim["mode"] == "Human":
            order_text = st.chat_input("Type your order quantity for this week (units)…")
            if order_text is not None and order_text.strip() != "":
                try:
                    order_qty = float(order_text)
                    if order_qty < 0:
                        raise ValueError("Order must be non-negative.")
                    sim["log"].append(("user", f"Order placed: {order_qty:.2f} units"))
                    advance_week(sim, human_order=order_qty)
                    st.rerun()
                except Exception as e:
                    sim["log"].append(("system", f"Input error: {e}"))
                    st.rerun()
        else:
            if st.button("Advance Week (AI)"):
                advance_week(sim)
                st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown("**Simulation complete.** Review analysis on the right, then run the Agentic AI replay.")

with right:
    st.subheader("System View (for professionals)")
    if len(sim["history"]) > 0:
        df = pd.DataFrame(sim["history"])

        total_waste, total_cost = summarize(df)
        st.metric("Total Food Waste (units)", f"{total_waste:.2f}")
        st.metric("Total System Cost", f"{total_cost:.2f}")

        st.caption("Most recent state")
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)

        max_units_human = max(
            df["end_customer_demand"].max(),
            df["order_retailer"].max(),
            df["order_regional"].max(),
            df["order_dc"].max(),
            df["order_producer"].max(),
        )
        max_units_human = float(max_units_human) * 1.1 if max_units_human > 0 else 10.0

        st.divider()
        render_charts(df, y_max_orders=max_units_human)

        # Post-run narrative (only shows after finishing Human round)
        if sim["finished"] and sim["mode"] == "Human":
            st.divider()
            add_post_run_narrative()

        st.divider()
        st.subheader("Agentic AI Replay (same demand)")
        st.caption("Runs a fresh simulation using coordinated agentic ordering policy, then compares waste & cost.")

        if st.button("Run Agentic AI Replay"):
            replay = init_state(n_weeks=int(sim["n_weeks"]))
            replay["mode"] = "Agentic AI"
            replay["demand_schedule"] = sim["demand_schedule"][:]

            while not replay["finished"]:
                advance_week(replay)

            df_ai = pd.DataFrame(replay["history"])

            waste_h, cost_h = summarize(df)
            waste_ai, cost_ai = summarize(df_ai)

            st.success("Replay complete.")
            c1, c2 = st.columns(2)
            c1.metric("Human Waste", f"{waste_h:.2f}")
            c1.metric("AI Waste", f"{waste_ai:.2f}", delta=f"{(waste_h - waste_ai):.2f} less")

            c2.metric("Human Cost", f"{cost_h:.2f}")
            c2.metric("AI Cost", f"{cost_ai:.2f}", delta=f"{(cost_h - cost_ai):.2f} lower")

            st.caption("Agentic AI charts (same units as Human chart)")
            render_charts(df_ai, title_suffix=" (Agentic AI)", y_max_orders=max_units_human)
    else:
        st.info("Start advancing weeks to see KPIs and charts.")



