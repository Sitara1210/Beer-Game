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
/* Make it feel like an ops console */
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { letter-spacing: 0.2px; }
div[data-testid="stChatMessage"] { border-radius: 14px; }
div[data-testid="stChatMessage"] p { font-size: 0.98rem; }
footer {visibility: hidden;}
/* De-emphasize Streamlit menu */
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
        if 4 <= w <= 6:
            schedule.append(6)
        else:
            schedule.append(4)
    return schedule

def init_state(n_weeks: int):
    # Start steady-state: on_hand 12, pipeline has 4 arriving each week (represented by a queue)
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

    for k in ROLE_KEYS:
        state["nodes"][k] = {
            "on_hand": 12.0,
            "backorder": 0.0,
            "inbound_pipeline": [4.0] * LEAD_TIME_WEEKS,   # shipments arriving in future weeks
            "outbound_pipeline": [0.0] * LEAD_TIME_WEEKS,  # not strictly needed, but helps reasoning
            "order_last": 4.0,
            "waste_cum": 0.0,
            "cost_cum": 0.0,
        }

    state["log"].append(("system", f"{APP_TITLE} initialized. Lead time={LEAD_TIME_WEEKS}w, spoilage={int(SPOILAGE_RATE*100)}%."))
    state["log"].append(("system", "Round 1: Human-only ordering. Limited visibility. Type your order each week."))
    return state

def inventory_position(node):
    # Inventory position = on_hand + on_order - backorder
    return node["on_hand"] + sum(node["inbound_pipeline"]) - node["backorder"]

def local_heuristic_order(role_key, state, observed_demand):
    """
    Non-coordinated, 'human-like' local policy:
      - Order-up-to inventory position based on last observed demand
      - Slight overreaction factor when backorders exist
    """
    node = state["nodes"][role_key]
    avg = observed_demand
    target = avg * (LEAD_TIME_WEEKS + 1)  # keep ~3 weeks of demand in position
    pos = inventory_position(node)
    gap = target - pos

    # Overreaction if backorders exist (a realistic bullwhip driver)
    panic = 1.0 + (0.25 if node["backorder"] > 0 else 0.0)

    order = max(0.0, gap) * panic
    # Smooth just a bit
    order = 0.6 * order + 0.4 * node["order_last"]
    return round(order, 2)

def agentic_ai_orders(state, end_customer_demand):
    """
    Coordinated agentic policy (system-level):
      - Everyone sees end-customer demand + all inventory positions
      - Uses same target inventory position across tiers
      - Dampens changes to reduce waste and bullwhip
    """
    # Rolling forecast using last 3 weeks (or available)
    w = state["week"]
    hist = [h["end_customer_demand"] for h in state["history"]]
    window = (hist + [end_customer_demand])[-3:]
    forecast = sum(window) / len(window)

    targets = {k: forecast * (LEAD_TIME_WEEKS + 1) for k in ROLE_KEYS}

    orders = {}
    for k in ROLE_KEYS:
        node = state["nodes"][k]
        pos = inventory_position(node)
        raw = max(0.0, targets[k] - pos)

        # Dampen swings: don't change orders too fast
        order = 0.5 * raw + 0.5 * node["order_last"]
        orders[k] = round(order, 2)
    return orders

def fulfill_and_cost(role_key, demand, state):
    """
    Process a node for one week:
      1) Receive inbound shipment (arrives from pipeline)
      2) Satisfy previous backorders first, then current demand
      3) Compute end inventory & backorders
      4) Spoilage applied to end inventory (food waste)
      5) Costs added
    """
    node = state["nodes"][role_key]

    # Receive inbound shipment
    incoming = node["inbound_pipeline"].pop(0)
    node["on_hand"] += incoming

    # Total demand includes backlog + this week's demand (downstream orders)
    total_required = node["backorder"] + demand
    shipped = min(node["on_hand"], total_required)
    node["on_hand"] -= shipped
    node["backorder"] = total_required - shipped

    # Spoilage on remaining on_hand (end of week)
    spoilage = node["on_hand"] * SPOILAGE_RATE
    node["on_hand"] -= spoilage
    node["waste_cum"] += spoilage

    # Weekly costs
    holding = node["on_hand"] * HOLDING_COST
    backcost = node["backorder"] * BACKORDER_COST
    total_cost = holding + backcost + spoilage  # treat spoilage as a "waste cost"
    node["cost_cum"] += total_cost

    return {
        "incoming": incoming,
        "shipped": shipped,
        "spoilage": spoilage,
        "holding": holding,
        "backcost": backcost,
        "total_cost": total_cost,
        "end_on_hand": node["on_hand"],
        "end_backorder": node["backorder"],
    }

def advance_week(state, human_order=None):
    """
    One complete week advancement. Demand flows downstream->upstream as orders.
    Shipments flow upstream->downstream after lead time.
    """
    w = state["week"]
    end_customer_demand = state["demand_schedule"][w - 1]

    # Decide orders
    if state["mode"] == "Agentic AI":
        orders = agentic_ai_orders(state, end_customer_demand)
    else:
        # Human mode: user controls one role, others use local heuristic (limited visibility)
        orders = {}
        for k in ROLE_KEYS:
            if k == state["human_role"]:
                orders[k] = float(human_order) if human_order is not None else state["nodes"][k]["order_last"]
            else:
                # Each tier only "observes" orders from downstream, not end-customer demand.
                # We'll use last week's downstream order as their observed demand proxy.
                # For week 1, assume steady state 4.
                if len(state["history"]) == 0:
                    observed = 4.0
                else:
                    observed = state["history"][-1][f"order_{downstream_key(k)}"]
                    if pd.isna(observed):
                        observed = 4.0
                orders[k] = local_heuristic_order(k, state, observed)

    # Record last orders
    for k in ROLE_KEYS:
        state["nodes"][k]["order_last"] = orders[k]

    # Demand into each tier is downstream order (except retailer demand is end-customer demand)
    tier_demand = {
        "retailer": end_customer_demand,
        "regional": orders["retailer"],
        "dc": orders["regional"],
        "producer": orders["dc"],
    }

    # Process fulfillment/cost at each tier (downstream to upstream)
    metrics = {}
    for k in ROLE_KEYS:
        metrics[k] = fulfill_and_cost(k, tier_demand[k], state)

    # Shipments: what each tier "orders" becomes inbound shipments for upstream to produce,
    # but in Beer Game, orders travel instantly and shipments arrive after lead time.
    # We'll model shipments arriving after lead time equal to the order placed (idealized capacity).
    # For more realism you can cap production/shipments later.
    # Push orders into the inbound pipeline of downstream nodes:
    # producer produces -> ships to dc, etc.
    state["nodes"]["dc"]["inbound_pipeline"].append(orders["producer"])
    state["nodes"]["regional"]["inbound_pipeline"].append(orders["dc"])
    state["nodes"]["retailer"]["inbound_pipeline"].append(orders["regional"])
    state["nodes"]["producer"]["inbound_pipeline"].append(0.0)  # producer receives no inbound in this simplified model

    # Log + history
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

    # Chat-style system summary
    state["log"].append(("system", f"Week {w} complete. End-customer demand={end_customer_demand}."))
    state["log"].append(("system", f"Orders placed: Retailer={orders['retailer']}, Regional={orders['regional']}, DC={orders['dc']}, Producer={orders['producer']}."))

    # Next week / finish
    if w >= state["n_weeks"]:
        state["finished"] = True
        state["log"].append(("system", "Run finished. You can switch to Agentic AI replay to compare waste & bullwhip."))
    else:
        state["week"] += 1

def downstream_key(role_key):
    # Helper for observation mapping in heuristic mode
    mapping = {
        "producer": "dc",
        "dc": "regional",
        "regional": "retailer",
        "retailer": None,
    }
    return mapping[role_key]

def render_charts(df: pd.DataFrame, title_suffix="", y_max=None):     # Bullwhip chart (locked scale if y_max provided)     fig1 = plt.figure()      plt.plot(df["week"], df["end_customer_demand"], label="End-Customer Demand")     plt.plot(df["week"], df["order_retailer"], label="Retailer Orders")     plt.plot(df["week"], df["order_regional"], label="Regional Orders")     plt.plot(df["week"], df["order_dc"], label="DC Orders")     plt.plot(df["week"], df["order_producer"], label="Producer Orders")      plt.xlabel("Week")     plt.ylabel("Units")     plt.title(f"Bullwhip – Orders Over Time{title_suffix}")      if y_max is not None:         plt.ylim(0, y_max)      plt.legend()     st.pyplot(fig1)      # Waste chart (same logic)     fig2 = plt.figure()     plt.plot(df["week"], df["retailer_waste_cum"], label="Retailer Waste (cum)")     plt.plot(df["week"], df["regional_waste_cum"], label="Regional Waste (cum)")     plt.plot(df["week"], df["dc_waste_cum"], label="DC Waste (cum)")     plt.plot(df["week"], df["producer_waste_cum"], label="Producer Waste (cum)")      plt.xlabel("Week")     plt.ylabel("Units wasted")     plt.title(f"Food Waste (Spoilage) – Cumulative{title_suffix}")      if y_max is not None:         plt.ylim(0, y_max)      plt.legend()     st.pyplot(fig2)
    # Bullwhip chart
    fig1 = plt.figure()
    plt.plot(df["week"], df["end_customer_demand"], label="End-Customer Demand")
    plt.plot(df["week"], df["order_retailer"], label="Retailer Orders")
    plt.plot(df["week"], df["order_regional"], label="Regional Orders")
    plt.plot(df["week"], df["order_dc"], label="DC Orders")
    plt.plot(df["week"], df["order_producer"], label="Producer Orders")
    plt.xlabel("Week")
    plt.ylabel("Units")
    plt.title(f"Bullwhip – Orders Over Time{title_suffix}")
    plt.legend()
    st.pyplot(fig1)

    # Waste chart (cumulative)
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
    total_waste = last["retailer_waste_cum"] + last["regional_waste_cum"] + last["dc_waste_cum"] + last["producer_waste_cum"]
    total_cost = last["retailer_cost_cum"] + last["regional_cost_cum"] + last["dc_cost_cum"] + last["producer_cost_cum"]
    return total_waste, total_cost

# -----------------------------
# UI
# -----------------------------
st.title("📦 " + APP_TITLE)
st.caption("Beer Game-style simulation for food distribution: lead time, perishability, bullwhip, and an Agentic AI replay for waste reduction.")

if "sim" not in st.session_state:
    st.session_state.sim = init_state(n_weeks=12)

sim = st.session_state.sim

with st.sidebar:
    st.header("Run Controls")
    sim["n_weeks"] = st.slider("Weeks", 8, 20, sim["n_weeks"])
    if st.button("Reset Simulation"):
        st.session_state.sim = init_state(n_weeks=sim["n_weeks"])
        st.rerun()

    st.divider()
    sim["human_role"] = st.selectbox("You control (Human Round)", options=ROLE_KEYS, format_func=lambda k: ROLES[ROLE_KEYS.index(k)])
    sim["mode"] = st.selectbox("Mode", ["Human", "Agentic AI"])
    st.write("Lead time:", LEAD_TIME_WEEKS, "weeks")
    st.write("Spoilage:", f"{int(SPOILAGE_RATE*100)}%/week")
    st.write("Costs: holding=1, backorder=2, waste=1 per unit")

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Operations Feed")

    # Render chat-style log
    for who, msg in sim["log"][-40:]:
        with st.chat_message("assistant" if who == "system" else "user"):
            st.markdown(msg)

    st.divider()

    if not sim["finished"]:
        w = sim["week"]
        end_customer_demand = sim["demand_schedule"][w - 1]
        role = sim["human_role"]
        node = sim["nodes"][role]

        # What the human sees (limited visibility in Human mode)
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
            # Agentic AI mode: one click per week (no typing needed)
            if st.button("Advance Week (AI)"):
                advance_week(sim)
                st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown("**Simulation complete.** Use the charts + replay to compare Human vs Agentic AI outcomes.")

with right:
    st.subheader("System View (for professionals)")
    if len(sim["history"]) > 0:
        df = pd.DataFrame(sim["history"])

        # KPI snapshot
        total_waste, total_cost = summarize(df)
        st.metric("Total Food Waste (units)", f"{total_waste:.2f}")
        st.metric("Total System Cost", f"{total_cost:.2f}")

        st.caption("Most recent state")
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)

        st.divider()
        max_units = max(     df["order_retailer"].max(),     df["order_regional"].max(),     df["order_dc"].max(),     df["order_producer"].max() )  render_charts(df, y_max=max_units)

        st.divider()
        st.subheader("Agentic AI Replay (same demand)")
        st.caption("Runs a fresh simulation using coordinated agentic ordering policy, then compares waste & cost.")

        if st.button("Run Agentic AI Replay"):
            # Run a full replay from scratch in AI mode with same weeks + schedule
            replay = init_state(n_weeks=sim["n_weeks"])
            replay["mode"] = "Agentic AI"
            # Use the same demand schedule for a true comparison
            replay["demand_schedule"] = sim["demand_schedule"][:]

            while not replay["finished"]:
                advance_week(replay)

            df_ai = pd.DataFrame(replay["history"])

            # Compare outcomes
            waste_h, cost_h = summarize(df)
            waste_ai, cost_ai = summarize(df_ai)

            st.success("Replay complete.")
            c1, c2 = st.columns(2)
            c1.metric("Human Waste", f"{waste_h:.2f}")
            c1.metric("AI Waste", f"{waste_ai:.2f}", delta=f"{(waste_h - waste_ai):.2f} less")

            c2.metric("Human Cost", f"{cost_h:.2f}")
            c2.metric("AI Cost", f"{cost_ai:.2f}", delta=f"{(cost_h - cost_ai):.2f} lower")

            st.caption("Agentic AI charts")
            render_charts(df_ai, title_suffix=" (Agentic AI)", y_max=max_units)

    else:
        st.info("Start advancing weeks to see KPIs and charts.")

