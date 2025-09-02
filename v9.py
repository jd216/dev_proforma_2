import streamlit as st
import numpy as np
import pandas as pd
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
mode = st.sidebar.selectbox("Select Mode", ["Default Mode", "Manual Mode"])
left_col, middle_col, right_col = st.columns([1, 2, 1])

# ---------------------------
def dual_input_sidebar(label, min_val, max_val, default_val, step):
    number_val = st.sidebar.number_input(label + " (manual entry)", min_val, max_val, default_val, step=step)
    slider_val = st.sidebar.slider(label + " (slider)", min_val, max_val, number_val, step=step)
    return slider_val

# ---------------------------
# DEFAULT MODE
# ---------------------------
if mode == "Default Mode":
    st.sidebar.header("Assumptions")

    # Cost inputs
    preconstruction_costs = dual_input_sidebar("Pre-Construction Costs ($)", 0, 10_000_000, 1_200_000, 50_000)
    preconstruction_months = dual_input_sidebar("Pre-Construction Period (months)", 1, 24, 3, 1)
    purchase_price = dual_input_sidebar("Purchase Price ($)", 0, 5_000_000, 500_000, 10_000)
    construction_costs = dual_input_sidebar("Construction Costs ($)", 0, 10_000_000, 2_000_000, 50_000)
    construction_months = dual_input_sidebar("Construction Period (months)", 1, 60, 12, 1)

    # Revenue inputs
    revenue_start = dual_input_sidebar("Revenue Start Month", 1, 120, preconstruction_months + 2 + construction_months, 1)
    sales_price_per_lot = dual_input_sidebar("Sales Price per Lot ($)", 0, 1_000_000, 100_000, 5_000)
    num_lots = dual_input_sidebar("Number of Lots", 1, 1000, 10, 1)
    revenue_months = dual_input_sidebar("Revenue Spread Months", 1, 60, 3, 1)

    annual_rate = st.sidebar.number_input("Annual Interest Rate (%)", 0.0, 20.0, 6.0, 0.1)
    monthly_rate = (1 + annual_rate/100)**(1/12) - 1
    brokerage_fee = st.sidebar.number_input("Brokerage Fee (%)", 0.0, 20.0, 0.0, 0.1)
    # Total months to cover all events
    total_months = preconstruction_months + 1 + construction_months + revenue_months + 12
    months = np.arange(total_months)

    # ---------------------------
    # Build Costs
    costs = np.zeros(total_months)

    # 1. Pre-construction spread starting at t=1
    costs[1:1+preconstruction_months] = -preconstruction_costs / preconstruction_months

    # 2. Purchase price after pre-construction
    purchase_month = 1 + preconstruction_months
    costs[purchase_month] = -purchase_price

    # 3. Construction costs spread starting after purchase
    construction_start = purchase_month + 1
    costs[construction_start:construction_start + construction_months] = -construction_costs / construction_months

    # ---------------------------
    # Build Revenues
    revenues = np.zeros(total_months)
    total_revenue = sales_price_per_lot * num_lots *(1-brokerage_fee/100)
    revenues[revenue_start:revenue_start + revenue_months] = total_revenue / revenue_months

    # Net cash flow
    net_cf = revenues + costs
    cum_cf = np.cumsum(net_cf)
    npv_val = npf.npv(monthly_rate, net_cf)
    irr_val = npf.irr(net_cf)
    annual_irr = (1 + irr_val)**12 - 1 if irr_val is not None else None
    #dcf
    dcf = net_cf / (1 + monthly_rate) ** np.arange(total_months)
    # ---------------------------
    # Outputs
    with middle_col:
        st.subheader("Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IRR (Annualized)", f"{annual_irr*100:.2f}%" if annual_irr else "N/A")
        m2.metric("NPV", f"${npv_val:,.0f}")
        m3.metric("Revenue", f"${revenues.sum():,.0f}")
        m4.metric("Costs", f"${-costs.sum():,.0f}")
    
        st.subheader("Cash Flow Table")
        df = pd.DataFrame({
            "Month": months,
            "Costs": costs,
            "Revenues": revenues,
            "Net CF": net_cf,
            "DCF": dcf,
            "Cumulative": cum_cf
        })
        st.dataframe(df.style.format("{:,.0f}"), height=300)

        st.subheader("Cash Flow Chart")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(net_cf, label="Net CF")
        ax.plot(cum_cf, label="Cumulative CF")
        ax.axhline(0, color="black", linestyle="--")
        ax.legend()
        st.pyplot(fig)

        # ---------------------------
        # IRR Heatmap (Price vs Lots)
        st.subheader("IRR Sensitivity Heatmap")
        price_min, price_max = st.slider("Sales Price per Lot Range", 0, 1_000_000, (50_000, 200_000), step=5_000, key="sens_price")
        lots_min, lots_max = st.slider("Number of Lots Range", 1, 1000, (5, 50), step=1, key="sens_lots")

        price_array = np.arange(price_min, price_max+1, 10_000)
        lots_array = np.arange(lots_min, lots_max+1, 5)

        price_grid, lots_grid = np.meshgrid(price_array, lots_array)
        irr_matrix = np.full_like(price_grid, np.nan, dtype=float)

        for i in range(lots_grid.shape[0]):
            for j in range(lots_grid.shape[1]):
                test_price = price_grid[i, j]
                test_lots = lots_grid[i, j]
                cf = np.copy(costs)
                rev_start_idx = revenue_start
                cf[rev_start_idx:rev_start_idx+revenue_months] += (test_price * test_lots) / revenue_months
                irr_val = npf.irr(cf)
                if irr_val is not None and not np.isnan(irr_val):
                    irr_matrix[i, j] = ((1 + irr_val) ** 12 - 1) * 100

        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(irr_matrix, xticklabels=price_array, yticklabels=lots_array, cmap="RdYlGn",
                    cbar_kws={'label':'Annualized IRR (%)'}, ax=ax2)
        ax2.set_xlabel("Sales Price per Lot ($)")
        ax2.set_ylabel("Number of Lots")
        st.pyplot(fig2)

        # ---------------------------
        # Profit Sensitivity Table
        st.subheader("Profit Sensitivity Table (Price vs Lots)")
        metric_choice = st.selectbox("Select metric to display",
                                    options=["Revenue", "Net Cash Flow", "NPV", "IRR"])
   
        # Build metric matrix
        metric_matrix = np.zeros((len(lots_array), len(price_array)))
        for i, lots in enumerate(lots_array):
            for j, price in enumerate(price_array):
                cf = np.copy(costs)
                cf[rev_start_idx:rev_start_idx+revenue_months] += (price * lots) / revenue_months
                if metric_choice == "Revenue":
                    metric_matrix[i, j] = price * lots
                elif metric_choice == "Net Cash Flow":
                    metric_matrix[i, j] = cf.sum()
                elif metric_choice == "NPV":
                    metric_matrix[i, j] = npf.npv(monthly_rate, cf)
                elif metric_choice == "IRR":
                    try:
                        metric_matrix[i, j] = npf.irr(cf) * 100
                    except:
                        metric_matrix[i, j] = np.nan

        metric_df = pd.DataFrame(metric_matrix,
                                index=[f"{l} lots" for l in lots_array],
                                columns=[f"${p}" for p in price_array])

        st.dataframe(metric_df.style.background_gradient(cmap="RdYlGn", axis=None)
                    .format("${:,.0f}" if metric_choice in ["Revenue", "Net Cash Flow", "NPV"] else "{:.2f}%"), height=400)
        # -------
        # Monte Carlo Simulation (Risk)
        st.subheader("Monte Carlo Simulation (Risk Analysis)")

        num_simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)

        #ranges
        rev_var = st.slider("Revenue Variability (±%)", 0, 50, 20)
        cost_var = st.slider("Cost Variability (±%)", 0, 50, 10)

        npv_results, irr_results = [], []

        for _ in range(num_simulations):
            #rng
            rev_multiplier = np.random.uniform(1 - rev_var/100, 1 + rev_var/100)
            cost_multiplier = np.random.uniform(1 - cost_var/100, 1 + cost_var/100)

            # Adjust revenues & costs
            rev_sim = revenues * rev_multiplier * (1 - brokerage_fee/100)
            cost_sim = costs * cost_multiplier
            cash_flows = rev_sim + cost_sim

            # Calculate
            npv_sim = npf.npv(monthly_rate, cash_flows)
            irr_sim = npf.irr(cash_flows)

            npv_results.append(npv_sim)
            irr_results.append(irr_sim)

        # remove nan
        irr_results_annual = [(1 + irr)**12 - 1 for irr in irr_results if not np.isnan(irr)]

        # Base case 
        base_npv = npf.npv(monthly_rate, revenues + costs)
        base_irr_val = npf.irr(revenues + costs)
        base_irr_annual = (1 + base_irr_val)**12 - 1 if base_irr_val is not None else None

        # --- Plot Results
        col1, col2 = st.columns(2)
        with col1:
            st.write("**NPV Distribution**")
            fig, ax = plt.subplots()
            if len(npv_results) > 0:
                sns.histplot(npv_results, kde=True, ax=ax, color="skyblue")
                ax.axvline(base_npv, color="red", linestyle="--", label="Base Case")
                ax.set_xlabel("NPV ($)")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No valid NPV results", ha="center")
            st.pyplot(fig)
            plt.clf()

        with col2:
            st.write("**IRR Distribution (Annual %)**")
            fig, ax = plt.subplots()
            if len(irr_results_annual) > 0:
                sns.histplot(np.array(irr_results_annual)*100, kde=True, ax=ax, color="lightgreen")
                if base_irr_annual is not None:
                    ax.axvline(base_irr_annual*100, color="red", linestyle="--", label="Base Case")
                ax.set_xlabel("IRR (%)")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No valid IRR results", ha="center")
            st.pyplot(fig)
            plt.clf()

        # --- Summary Statistics
        st.write("### Simulation Results")
        if len(npv_results) > 0:
            st.write(f"NPV P10: ${np.percentile(npv_results, 10):,.0f} | "
                    f"P50: ${np.percentile(npv_results, 50):,.0f} | "
                    f"P90: ${np.percentile(npv_results, 90):,.0f}")
        else:
            st.write("⚠️ No valid NPV results.")

        if len(irr_results_annual) > 0:
            st.write(f"IRR P10: {np.percentile(irr_results_annual, 10)*100:.2f}% | "
                    f"P50: {np.percentile(irr_results_annual, 50)*100:.2f}% | "
                    f"P90: {np.percentile(irr_results_annual, 90)*100:.2f}%")
        else:
            st.write("⚠️ No valid IRR results.")


        # --- Target NPV / IRR Solver Section
        st.subheader("Target-Based Purchase Price Calculator")

        solver_type = st.radio("Choose target type:", ["Target NPV", "Target IRR"])

        if solver_type == "Target NPV":
            target_npv = st.number_input("Enter Target NPV ($)", value=0, step=1000, format="%d")

            def f(price):
                c = costs.copy()
                c[purchase_month] = -price
                return npf.npv(monthly_rate, revenues + c) - target_npv

            try:
                from scipy.optimize import brentq
                new_price = brentq(f, 1, 1e9) 
                st.write(f"To achieve NPV = ${target_npv:,.0f}")
                st.write(f"Maximum Purchase Price: **${new_price:,.0f}**")

            except Exception as e:
                st.write("⚠️ Could not solve for purchase price. Try a different target.")

        elif solver_type == "Target IRR":
            target_irr_pct = st.number_input("Enter Target IRR (Annual %)", value=15.00, step=0.50, format="%.2f")
            target_irr = target_irr_pct / 100 

            target_monthly_irr = (1 + target_irr) ** (1/12) - 1

            def f(price):
                c = costs.copy()
                c[purchase_month] = -price
                return npf.irr(revenues + c) - target_monthly_irr

            try:
                from scipy.optimize import brentq
                new_price = brentq(f, 1, 1e9)
                st.write(f"To achieve IRR = {target_irr_pct:.2f}% annually")
                st.write(f"Maximum Purchase Price: **${new_price:,.0f}**")

            except Exception as e:
                st.write("⚠️ Could not solve for purchase price. Try a different target.")
        # ---------------------------
        # Cash Flow Input Summary
        st.markdown("---")  
        st.subheader("Cash Flow Input Summary")
        summary_text = f"""
        **Purchase Price:** ${purchase_price:,.0f}<br>
        **Pre-Construction Costs:** ${preconstruction_costs:,.0f}<br>
        **Pre-Construction Months:** {preconstruction_months}<br>
        **Construction Costs:** ${construction_costs:,.0f}<br>
        **Construction Months:** {construction_months}<br>
        **Revenue Start Month:** {revenue_start}<br>
        **Sales Price per Lot:** ${sales_price_per_lot:,.0f}<br>
        **Number of Lots:** {num_lots}<br>
        **Revenue Spread Months:** {revenue_months}<br>
        **Annual Interest Rate:** {annual_rate:.2f}%<br>
        **Monthly Interest Rate:** {monthly_rate*100:.2f}%
        """
        st.markdown(f"<div style='background-color:#f0f0f0; padding:15px; border-radius:10px;'>{summary_text}</div>", unsafe_allow_html=True)

        # ---------------------------
# MANUAL MODE
# ---------------------------
elif mode == "Manual Mode":
    col_left, col_middle, col_right = st.columns([1, 2, 1])
    with left_col:
        num_months = st.sidebar.number_input("Number of Months", min_value=1, max_value=120, value=12, step=1)
        months = np.arange(num_months)
        st.sidebar.header("Interest Rate")

        annual_rate_input = st.sidebar.text_input("Annual Interest Rate (%)", value="8.00")
        try:
            annual_rate = float(annual_rate_input)
        except ValueError:
            annual_rate = 8.0  # fallback

        monthly_rate = annual_rate/100/12

        costs = np.zeros(num_months+1)
        revenues = np.zeros(num_months+1)
        for i in range(num_months+1):
            c1, c2 = st.columns(2)
            with c1:
                costs[i] = st.number_input(f"Cost Month:{i}", value=0, step=1000, key=f"c{i}")
            with c2:
                revenues[i] = st.number_input(f"Revenue Month:{i}", value=0, step=1000, key=f"r{i}")

        net_cf = revenues - costs
        cum_cf = np.cumsum(net_cf)
        npv_val = npf.npv(monthly_rate, net_cf)
        irr_val = npf.irr(net_cf)
        annual_irr = (1+irr_val)**12-1 if irr_val is not None else None

        with middle_col:
            st.subheader("Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("IRR (Annualized)", f"{annual_irr*100:.2f}%" if annual_irr else "N/A")
            m2.metric("NPV", f"${npv_val:,.0f}")
            m3.metric("Revenue", f"${revenues.sum():,.0f}")
            m4.metric("Costs", f"${costs.sum():,.0f}")

            st.subheader("Cash Flow Table")
            df = pd.DataFrame({
                "Month": np.arange(num_months+1),
                "Costs": costs,
                "Revenues": revenues,
                "Net CF": net_cf,
                "Cumulative": cum_cf
            })
            st.dataframe(df.style.format("{:,.0f}"), height=300)

            st.subheader("Cash Flow Chart")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(net_cf, label="Net CF")
            ax.plot(cum_cf, label="Cumulative CF")
            ax.axhline(0, color="black", linestyle="--")
            ax.legend()
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Input Summary")
            summary_text = f"""
            **Months:** {num_months+1}<br>
            **Annual Interest Rate:** {annual_rate:.2f}%<br>
            **Monthly Interest Rate:** {monthly_rate*100:.2f}%
            """
            st.markdown(f"<div style='background-color:#f0f0f0; padding:15px; border-radius:10px;'>{summary_text}</div>", unsafe_allow_html=True)
