import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Set page config at the very top
st.set_page_config(page_title="Pump & Vacuum Pump Sizing Sheet", layout="wide")

st.title("Pump & Vacuum Pump Sizing Sheet — Vendor Ready")

# Simple navigation between pages
page = st.sidebar.selectbox("Choose tool", ["Rotating Pumps (Centrifugal etc.)", "Vacuum Pump Calculator"])

# ------------------ Helper functions ------------------
def colebrook_f(Re, D, eps_rel, tol=1e-6, max_iter=50):
    """Calculate friction factor using Colebrook equation"""
    if Re <= 0:
        return np.nan
    if Re < 2300:
        return 64.0 / Re

    # Handle edge cases for log calculations
    try:
        A = eps_rel / D / 3.7
        B = 5.74 / (Re**0.9)
        f = 0.25 / (math.log10(A + B))**2

        for _ in range(max_iter):
            denominator = eps_rel / (3.7*D) + 2.51/(Re*math.sqrt(f))
            if denominator <= 0:
                break
            new = (-2.0 * math.log10(denominator))**-2
            if abs(new - f) < tol:
                f = new
                break
            f = new
        return f
    except (ValueError, ZeroDivisionError):
        return 0.02  # Default friction factor for turbulent flow


def darcy_head_loss(f, L, D, V, g=9.81):
    """Calculate head loss using Darcy-Weisbach equation"""
    if D <= 0:
        return 0
    return f * (L/D) * (V**2) / (2*g)


def reynolds(rho, V, D, mu):
    """Calculate Reynolds number"""
    if mu <= 0:
        return 0
    return (rho * V * D) / mu


def velocity_from_flow(Q, D):
    """Calculate velocity from flow rate and diameter"""
    if D <= 0:
        return 0
    A = math.pi * (D**2) / 4.0
    return Q / A if A > 0 else 0


def minor_loss_head(K_total, V, g=9.81):
    """Calculate minor losses head"""
    return K_total * (V**2) / (2*g)


def pump_power_required(rho, g, Q, H, pump_efficiency, motor_efficiency=0.95):
    """Calculate pump power requirements"""
    if pump_efficiency <= 0 or motor_efficiency <= 0:
        return 0, 0
    shaft_watts = rho * g * Q * H / pump_efficiency
    electrical_watts = shaft_watts / motor_efficiency
    return shaft_watts/1000.0, electrical_watts/1000.0


def suggest_impeller(material):
    """Suggest impeller type based on material"""
    mapping = {
        'Water, non-corrosive': 'Cast iron closed impeller',
        'Seawater': 'Bronze open impeller',
        'Acids': 'PVDF or Stainless steel semi-open impeller',
        'Slurry': 'High-chrome open impeller',
        'Food-grade': 'Stainless steel closed impeller'
    }
    return mapping.get(material, 'Consult vendor')


def compute_bep(Q_points, eff_curve):
    """Find Best Efficiency Point"""
    if len(eff_curve) == 0:
        return 0, 0, 0
    idx = np.nanargmax(eff_curve)
    return Q_points[idx], eff_curve[idx], idx


def generate_pump_curves(Q_design, total_head_design, static_head):
    """Generate representative pump and system curves"""
    if Q_design <= 0:
        Q_design = 1e-6  # Avoid division by zero

    Q_points = np.linspace(max(1e-9, Q_design*0.1), Q_design*1.6, 200)

    # System curve
    a = (total_head_design - static_head) / (Q_design**2) if Q_design > 0 else 0
    H_system = static_head + a * (Q_points**2)

    # Pump curve
    H0 = total_head_design*1.15
    k = H0 / ((Q_design*1.4)**2) if Q_design > 0 else 0
    H_pump = H0 - k * (Q_points**2)

    # Efficiency curve: peak near Q_design (BEP) with a bell shape
    eff_curve = np.clip(0.45 + 0.4 * np.exp(-((Q_points-Q_design)/(Q_design*0.25))**2), 0.1, 0.95)

    return Q_points, H_system, H_pump, eff_curve


def create_excel_report_rotating(df_summary, inputs_echo, fig_png_bytes):
    """Create Excel report for rotating pumps"""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, index=False, sheet_name='Summary')
            inputs_echo.to_excel(writer, index=False, sheet_name='Inputs')

            if fig_png_bytes is not None:
                workbook = writer.book
                worksheet = writer.sheets['Summary']
                # Insert image at cell G2
                worksheet.insert_image('G2', 'pump_curves.png', {'image_data': io.BytesIO(fig_png_bytes)})

        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel report: {e}")
        return None


def create_excel_report_vacuum(df_vac, fig_png_bytes):
    """Create Excel report for vacuum pumps"""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_vac.to_excel(writer, index=False, sheet_name='Vacuum')

            if fig_png_bytes is not None:
                workbook = writer.book
                worksheet = writer.sheets['Vacuum']
                worksheet.insert_image('G2', 'vacuum_curve.png', {'image_data': io.BytesIO(fig_png_bytes)})

        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel report: {e}")
        return None

# ------------------ Rotating Pumps Page ------------------
if page == "Rotating Pumps (Centrifugal etc.)":
    st.header("Rotating Pump Sizing & Selection")

    with st.form(key='rotating'):
        st.subheader("Process & Fluid Data")
        Q_input = st.number_input("Flow rate", value=100.0, min_value=0.0, format="%.6f")
        Q_unit = st.selectbox("Flow unit", ['m³/h', 'L/s', 'm³/s', 'm³/d', 'GPM (US)'], index=0)
        T = st.number_input("Fluid temperature (°C)", value=25.0)
        SG = st.number_input("Specific gravity (relative to water)", value=1.0, min_value=0.01)
        mu_cP = st.number_input("Viscosity (cP)", value=1.0, min_value=0.01)
        density = 1000.0 * SG

        if st.checkbox("Override density (kg/m³)?", value=False):
            density = st.number_input("Density (kg/m³)", value=1000.0, min_value=0.1)

        # Display calculated density
        st.write(f"**Calculated Density:** {density:.2f} kg/m³")

        st.markdown("---")
        st.subheader("Piping & Elevation")
        D_inner = st.number_input("Pipe inner diameter (mm)", value=100.0, min_value=1.0)
        L_pipe = st.number_input("Pipe length (m)", value=100.0, min_value=0.0)
        elevation_in = st.number_input("Suction elevation (m)", value=0.0)
        elevation_out = st.number_input("Discharge elevation (m)", value=10.0)
        K_fittings = st.number_input("Total equivalent K (sum of fittings)", value=2.0, min_value=0.0)
        eps_mm = st.number_input("Absolute roughness (mm)", value=0.045, min_value=0.0001)

        st.markdown("---")
        st.subheader("Pump & Motor Settings")
        pump_eff_user = st.number_input("Pump efficiency (%) [if known]", value=70.0, min_value=1.0, max_value=100.0)/100.0
        motor_eff = st.number_input("Motor efficiency (%)", value=95.0, min_value=10.0, max_value=100.0)/100.0
        safety_margin_head = st.number_input("Design margin on head (%)", value=10.0, min_value=0.0)/100.0
        safety_margin_flow = st.number_input("Design margin on flow (%)", value=10.0, min_value=0.0)/100.0
        service_factor = st.number_input("Service factor (e.g. 1.15)", value=1.15, min_value=1.0)

        st.markdown("---")
        st.subheader("Application & Materials")
        material_type = st.selectbox("Fluid type for impeller suggestion", ['Water, non-corrosive', 'Seawater', 'Acids', 'Slurry', 'Food-grade', 'Oil Transfer'])
        application = st.selectbox("Application Type", ['General Transfer', 'Chemical Handling', 'Slurry Transport', 'Oil Transfer', 'High Pressure', 'Metering'])


        st.markdown("---")
        st.subheader("NPSH & Vapor Data")
        atm_pressure_kPa = st.number_input("Atmospheric pressure (kPa)", value=101.325, min_value=50.0)
        vapor_pressure_kPa = st.number_input("Fluid vapor pressure (kPa)", value=2.3, min_value=0.0)
        friction_for_NPSH = st.number_input("Suction-side friction (head, m)", value=2.0, min_value=0.0)

        st.markdown("---")
        st.subheader("Vendor Catalog Matching (optional CSV upload)")
        uploaded = st.file_uploader("Upload pump catalog CSV (columns: name, flow_m3h, head_m, power_kW, speed_rpm, npshr_m)", type=['csv'])

        submitted = st.form_submit_button("Calculate")

    if submitted:
        try:
            # Flow unit conversion
            if Q_unit == 'm³/h':
                Q_m3s = Q_input / 3600.0
            elif Q_unit == 'L/s':
                Q_m3s = Q_input / 1000.0
            elif Q_unit == 'm³/d':
                Q_m3s = Q_input / (24*3600)
            elif Q_unit == 'GPM (US)':
                Q_m3s = Q_input * 0.00378541178 / 60.0
            else:
                Q_m3s = Q_input

            # Convert units and calculate
            mu = mu_cP / 1000.0
            D = D_inner / 1000.0
            V = velocity_from_flow(Q_m3s, D)
            Re = reynolds(density, V, D, mu)
            f = colebrook_f(Re, D, eps_mm/1000.0)
            hf = darcy_head_loss(f, L_pipe, D, V)
            hm = minor_loss_head(K_fittings, V)
            static_head = elevation_out - elevation_in
            total_head = static_head + hf + hm
            total_head_design = total_head * (1.0 + safety_margin_head)
            Q_design = Q_m3s * (1.0 + safety_margin_flow)

            # Pump efficiency
            pump_eff = pump_eff_user

            shaft_kW, electrical_kW = pump_power_required(density, 9.81, Q_design, total_head_design, pump_eff, motor_eff)

            # NPSH calculation
            P_atm_Pa = atm_pressure_kPa * 1000.0
            P_vap_Pa = vapor_pressure_kPa * 1000.0
            z_suction = elevation_in
            NPSHa = (P_atm_Pa - P_vap_Pa)/(density*9.81) + z_suction - friction_for_NPSH

            # Pump type suggestion
            pump_suggestions = {
                ('Water, non-corrosive', 'General Transfer'): 'Centrifugal Pump',
                ('Acids', 'Chemical Handling'): 'Magnetic Drive Pump',
                ('Slurry', 'Slurry Transport'): 'Slurry Pump',
                ('Food-grade', 'Oil Transfer'): 'Gear Pump',
                ('Slurry', 'High Pressure'): 'Positive Displacement Pump',
                ('Oil Transfer', 'General Transfer'): 'Centrifugal Pump',
                ('Oil Transfer', 'High Pressure'): 'Positive Displacement Pump'
            }
            pump_type = pump_suggestions.get((material_type, application), 'Consult vendor for best selection')

            # Generate curves and BEP
            Q_points, H_system, H_pump, eff_curve = generate_pump_curves(Q_design, total_head_design, static_head)
            Q_bep, eff_bep, bep_idx = compute_bep(Q_points, eff_curve)

            # Operating point: intersection approx (min squared diff)
            if len(H_pump) > 0 and len(H_system) > 0:
                idx_op = np.argmin((H_pump - H_system)**2)
                Q_op = Q_points[idx_op]
                H_op = H_pump[idx_op]
                eff_op = eff_curve[idx_op]
            else:
                Q_op = Q_design
                H_op = total_head_design
                eff_op = pump_eff

            # Specific Speed calculation
            rep_speed_rpm = 1450.0
            Ns = rep_speed_rpm * math.sqrt(Q_design*3600.0) / (H_op**0.75) if H_op > 0 else np.nan

            # BEP derating check
            pct_from_bep = abs((Q_op - Q_bep)/Q_bep) * 100.0 if Q_bep > 0 else np.nan
            if pct_from_bep <= 10:
                derating_recommendation = 'Good — operating point within 10% of BEP.'
            elif pct_from_bep <= 20:
                derating_recommendation = 'Acceptable — consider 5-10% derating or impeller trimming advice.'
            else:
                derating_recommendation = 'Poor — operating point far from BEP. Recommend different pump size or positive displacement pump.'

            # Service factor and motor sizing
            motor_rated_kW = electrical_kW * service_factor

            # Create summary DataFrame
            summary = {
                'Item': ['Flow (m3/s)', 'Flow (m3/h)', 'Design Flow (m3/s)', 'Velocity (m/s)', 'Reynolds number',
                         'Friction factor (Darcy)', 'Pipe friction head (m)', 'Minor losses head (m)', 'Static head (m)',
                         'Total dynamic head (m)', 'Design total head (m)', 'Shaft power (kW)', 'Electrical power (kW)',
                         'Motor rated (kW)', 'Pump efficiency (%)', 'Motor efficiency (%)', 'NPSH Available (m)',
                         'Specific speed (Ns)', 'BEP Flow (m3/s)', 'Operating Flow (m3/s)', 'Distance from BEP (%)',
                         'BEP Recommendation'],
                'Value': [Q_m3s, Q_m3s*3600.0, Q_design, V, Re, f, hf, hm, static_head, total_head, total_head_design,
                         shaft_kW, electrical_kW, motor_rated_kW, pump_eff*100.0, motor_eff*100.0, NPSHa, Ns,
                         Q_bep, Q_op, pct_from_bep, derating_recommendation]
            }
            df_summary = pd.DataFrame(summary)

            # Inputs echo
            inputs_echo = pd.DataFrame({
                'Input': ['Flow (user units)', 'Flow unit', 'Temperature (°C)', 'Specific gravity', 'Viscosity (cP)',
                         'Pipe D (mm)', 'Pipe L (m)', 'K fittings', 'Roughness (mm)', 'Pump eff (%)',
                         'Motor eff (%)', 'Service factor', 'Fluid type', 'Application'],
                'Value': [Q_input, Q_unit, T, SG, mu_cP, D_inner, L_pipe, K_fittings, eps_mm,
                         pump_eff_user*100.0, motor_eff*100.0, service_factor, material_type, application]
            })


            # Create plots
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(Q_points*3600.0, H_system, label='System curve', linewidth=2)
            ax.plot(Q_points*3600.0, H_pump, label='Pump curve', linewidth=2)
            ax.scatter([Q_bep*3600.0], [H_pump[bep_idx]], color='green', s=100, label='BEP', zorder=5)
            ax.scatter([Q_op*3600.0], [H_op], color='red', s=100, label='Operating point', zorder=5)
            ax.set_xlabel('Flow (m³/h)')
            ax.set_ylabel('Head (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Pump and System Curves')

            # Add annotations
            if not np.isnan(Q_bep) and not np.isnan(H_pump[bep_idx]):
                ax.annotate(f'BEP: {Q_bep*3600.0:.1f} m³/h',
                           xy=(Q_bep*3600.0, H_pump[bep_idx]),
                           xytext=(Q_bep*3600.0*1.1, H_pump[bep_idx]+max(H_pump)*0.05),
                           arrowprops=dict(arrowstyle='->', color='green'))

            if not np.isnan(Q_op) and not np.isnan(H_op):
                ax.annotate(f'OP: {Q_op*3600.0:.1f} m³/h',
                           xy=(Q_op*3600.0, H_op),
                           xytext=(Q_op*3600.0*0.7, H_op+max(H_pump)*0.1),
                           arrowprops=dict(arrowstyle='->', color='red'))

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            fig_png = buf.getvalue()
            buf.close()
            plt.close(fig)

            # Power and efficiency plots
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Power curve
            power_curve = density * 9.81 * Q_points * H_pump / pump_eff / 1000.0
            ax1.plot(Q_points*3600.0, power_curve, linewidth=2)
            ax1.set_title('Power vs Flow')
            ax1.set_xlabel('Flow (m³/h)')
            ax1.set_ylabel('Power (kW)')
            ax1.axvline(Q_op*3600.0, color='red', linestyle='--', alpha=0.7)
            ax1.grid(True, alpha=0.3)

            # Efficiency curve
            ax2.plot(Q_points*3600.0, eff_curve*100.0, linewidth=2)
            ax2.set_title('Estimated Efficiency vs Flow')
            ax2.set_xlabel('Flow (m³/h)')
            ax2.set_ylabel('Efficiency (%)')
            ax2.axvline(Q_bep*3600.0, color='green', linestyle='--', alpha=0.7)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
            fig2_png = buf2.getvalue()
            buf2.close()
            plt.close(fig2)

            # Inference texts
            inference_system = f"Operating point at {Q_op*3600.0:.1f} m³/h intersects pump and system curves at ~{H_op:.2f} m. BEP is {Q_bep*3600.0:.1f} m³/h. {derating_recommendation}"
            inference_power = f"At operating flow the estimated shaft power is {density*9.81*Q_op*H_op/pump_eff/1000.0:.2f} kW (shaft). Check motor rating and service factor ({service_factor})."
            inference_eff = f"Estimated efficiency at operating point: {eff_op*100.0:.1f}%. BEP efficiency: {eff_bep*100.0:.1f}% at {Q_bep*3600.0:.1f} m³/h."

            # Vendor catalog matching
            matched_pumps = None
            if uploaded is not None:
                try:
                    catalog = pd.read_csv(uploaded)
                    # Simple filter: find rows where flow and head within +/-20%
                    flow_m3h = Q_design*3600.0
                    head_m = total_head_design

                    # Check if required columns exist
                    required_cols = ['name', 'flow_m3h', 'head_m', 'power_kW', 'speed_rpm', 'npshr_m']
                    if all(col in catalog.columns for col in required_cols):
                        candidates = catalog[
                            (catalog['flow_m3h'] > 0.8*flow_m3h) &
                            (catalog['flow_m3h'] < 1.2*flow_m3h) &
                            (catalog['head_m'] > 0.8*head_m) &
                            (catalog['head_m'] < 1.2*head_m)
                        ]

                        if not candidates.empty:
                            matched_pumps = candidates.copy()
                            matched_pumps['score'] = 1 - (
                                abs(matched_pumps['flow_m3h'] - flow_m3h)/flow_m3h +
                                abs(matched_pumps['head_m'] - head_m)/head_m
                            )/2
                            matched_pumps = matched_pumps.sort_values('score', ascending=False)
                    else:
                        st.warning(f'CSV missing required columns. Expected: {required_cols}')

                except Exception as e:
                    st.warning(f'Failed to parse catalog CSV: {e}')

            # Display results
            st.success("Calculation complete — see summary below")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Calculated Results")
                st.dataframe(df_summary, use_container_width=True)

                # Add alerts for Process Data based on calculated values
                st.markdown("---")
                st.subheader("Process Data Insights")
                if Re < 2300:
                    st.info(f"Reynolds number ({Re:.1f}) indicates Laminar flow. Viscosity has a significant impact on head loss in this regime.")
                elif Re > 4000:
                    st.info(f"Reynolds number ({Re:.1f}) indicates Turbulent flow. Pipe roughness significantly impacts head loss.")
                else:
                    st.info(f"Reynolds number ({Re:.1f}) indicates Transitional flow.")

                if mu_cP > 100:
                    st.warning(f"High viscosity ({mu_cP:.1f} cP) can significantly reduce pump performance and efficiency, especially for centrifugal pumps. Consider positive displacement pumps for very high viscosities.")
                if density > 1200:
                    st.warning(f"High density ({density:.1f} kg/m³) will result in higher power consumption for the same head. Ensure motor is adequately sized.")
                if V > 3:
                    st.warning(f"High velocity ({V:.1f} m/s) can lead to increased pipe friction losses and potential erosion.")
                elif V < 0.5:
                    st.warning(f"Low velocity ({V:.1f} m/s) may lead to settling of solids if handling slurries.")


                st.markdown("---")
                st.subheader("Performance Curves")
                st.image(fig_png, use_column_width=True)
                st.markdown(f"**Inference:** {inference_system}")
                if pct_from_bep > 20:
                     st.warning(f"Operating point is {pct_from_bep:.1f}% away from BEP. This can lead to reduced efficiency, increased wear, and potential reliability issues.")

                st.markdown("---")
                st.subheader("Power & Efficiency")
                st.image(fig2_png, use_column_width=True)
                st.markdown(f"**Inference (power):** {inference_power}")
                st.markdown(f"**Inference (efficiency):** {inference_eff}")
                if pump_eff_user < 0.6:
                    st.warning(f"Low estimated pump efficiency ({pump_eff_user*100:.1f}%) suggests potential for significant energy savings with a more efficient pump.")

            with col2:
                st.subheader("Vendor-ready Summary")
                st.write(f"**Design flow:** {Q_design:.6f} m³/s ({Q_design*3600.0:.2f} m³/h)")
                st.write(f"**Design total head:** {total_head_design:.2f} m")
                st.write(f"**Required shaft power:** {shaft_kW:.2f} kW")
                st.write(f"**Electrical power (est.):** {electrical_kW:.2f} kW")
                st.write(f"**Motor rated (with service factor):** {motor_rated_kW:.2f} kW")
                st.write(f"**NPSH available:** {NPSHa:.2f} m")
                st.write(f"**Suggested impeller:** {suggest_impeller(material_type)}")
                st.write(f"**Suggested pump type:** {pump_type}")

                # Add fluid-specific notes and alerts
                if material_type == 'Acids':
                    st.warning("Handling acids requires careful material selection for pump and piping. Ensure compatibility with the specific acid and concentration.")
                    st.info("Consider magnetic drive pumps or pumps with corrosion-resistant materials like PVDF, PTFE, or specialized alloys.")
                elif material_type == 'Slurry':
                    st.warning("Pumping slurries can cause abrasive wear. Proper material selection and pump type are critical.")
                    st.info("Consider slurry pumps with robust construction, wear-resistant materials (e.g., high-chrome), and open or recessed impellers.")
                elif material_type == 'Seawater':
                     st.warning("Seawater is corrosive. Material selection to resist corrosion and fouling is important.")
                     st.info("Bronze or stainless steel pumps are often used for seawater, depending on the application and required lifespan.")
                elif material_type == 'Food-grade':
                    st.warning("Food-grade applications require sanitary design and specific materials.")
                    st.info("Use pumps designed for sanitary applications with polished stainless steel surfaces and FDA-approved elastomers.")
                elif material_type == 'Oil Transfer':
                    st.info("Viscosity and temperature significantly impact oil transfer. Ensure viscosity is within the pump's operating range.")


                st.markdown("---")
                st.markdown("**Checklist for vendor**")
                st.write("1. Provide pump curve (flow vs head) at quoted impeller size and speed.")
                st.write("2. Provide NPSHr curve and recommended margin.")
                st.write("3. Confirm motor frame, service factor, and starter type.")
                st.write("4. Provide mechanical seals, bearing arrangement, materials of construction and documentation.")

                if matched_pumps is not None:
                    st.markdown("---")
                    st.subheader("Matched catalog candidates")
                    st.dataframe(matched_pumps[['name','flow_m3h','head_m','power_kW','speed_rpm','npshr_m','score']].head(10))

                # Excel export
                if st.button("Generate Excel Report", type="primary"):
                    excel_bytes = create_excel_report_rotating(df_summary, inputs_echo, fig_png)
                    if excel_bytes:
                        st.download_button(
                            "📥 Download Rotating Pump Report (xlsx)",
                            data=excel_bytes,
                            file_name=f"rotating_pump_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )

        except Exception as e:
            st.error(f"Calculation error: {e}")
            st.error("Please check your input values and try again.")

# ------------------ Vacuum Pump Calculator Page ------------------
elif page == "Vacuum Pump Calculator":
    st.header("Vacuum Pump Sizing & Selection")

    with st.form(key='vacuum'):
        st.subheader("Chamber & Gas Load")
        chamber_volume_l = st.number_input("Chamber Volume (L)", value=100.0, min_value=0.1)
        leak_rate_mbarL_s = st.number_input("Estimated leak/outgassing (mbar·L/s)", value=0.1, min_value=0.0, format="%.6g")
        desired_pressure_mbar = st.number_input("Desired operating pressure (mbar)", value=1e-3, format="%.6g", min_value=1e-12)
        desired_pressure_unit = st.selectbox("Pressure unit", ['mbar', 'Pa'], index=0)

        if desired_pressure_unit == 'Pa':
            desired_pressure_mbar = desired_pressure_mbar / 100.0

        st.subheader("Foreline / Conductance (optional)")
        foreline_id_mm = st.number_input("Foreline inner diameter (mm)", value=25.0, min_value=1.0)
        foreline_length_m = st.number_input("Foreline length (m)", value=1.0, min_value=0.01)
        gas_molecular_mass = st.number_input("Representative gas molecular mass (g/mol)", value=28.97, min_value=1.0)

        st.subheader("Pump/Process Constraints")
        available_pumping_speed_Ls = st.number_input("Available pumping speed (L/s) if known (0 to auto-calc)", value=0.0, min_value=0.0)
        suggest_backing = st.checkbox("Suggest backing pump for high vacuum (turbo+backing)", value=True)

        submitted_vac = st.form_submit_button("Calculate Vacuum Requirements")

    if submitted_vac:
        try:
            # Convert units
            chamber_volume_m3 = chamber_volume_l / 1000.0
            Q_pa_m3_s = leak_rate_mbarL_s * 0.1  # 1 mbar·L/s = 0.1 Pa·m3/s
            P_target_Pa = desired_pressure_mbar * 100.0

            # Required pumping speed S = Q / P (in m3/s)
            if P_target_Pa > 0:
                S_required_m3_s = Q_pa_m3_s / P_target_Pa
            else:
                S_required_m3_s = np.inf
            S_required_Ls = S_required_m3_s * 1000.0

            # Conductance estimation for tube (molecular flow approx)
            d_cm = foreline_id_mm / 10.0
            if foreline_length_m > 0:
                C_molecular_Ls = 12.1 * d_cm**3 / foreline_length_m
            else:
                C_molecular_Ls = np.inf

            # Effective pumping speed calculation
            if not np.isinf(C_molecular_Ls) and not np.isnan(S_required_Ls) and S_required_Ls > 0:
                S_effective_Ls = (S_required_Ls * C_molecular_Ls) / (S_required_Ls + C_molecular_Ls)
            else:
                S_effective_Ls = S_required_Ls

            # If user provided available pumping speed, compute steady-state pressure
            if available_pumping_speed_Ls > 0:
                S_user_m3_s = available_pumping_speed_Ls / 1000.0
                P_steady_Pa = Q_pa_m3_s / S_user_m3_s if S_user_m3_s > 0 else np.inf
            else:
                P_steady_Pa = P_target_Pa

            # Pump-down time estimate (assuming S constant, ideal gas)
            p0_mbar = 1000.0
            p0_Pa = p0_mbar * 100.0
            if S_required_m3_s > 0 and not np.isinf(S_required_m3_s):
                tau_sec = chamber_volume_m3 / S_required_m3_s
                if P_target_Pa > 0:
                    t_to_target_sec = tau_sec * math.log(p0_Pa / P_target_Pa)
                else:
                    t_to_target_sec = np.inf
            else:
                tau_sec = np.inf
                t_to_target_sec = np.inf


            # Suggest pump types by pressure range and throughput
            pump_type_suggestion = []
            if desired_pressure_mbar >= 100:
                pump_type_suggestion.append('Rotary Lobe / Dry Screw (vacuum blower)')
            if 1 <= desired_pressure_mbar < 100:
                pump_type_suggestion.append('Rotary Vane or Dry Scroll (roughing pump)')
            if desired_pressure_mbar < 1e-3:
                pump_type_suggestion.append('Turbomolecular pump (requires backing pump)')
            elif desired_pressure_mbar < 1e-1:
                pump_type_suggestion.append('Roots (booster) + backing pump or high-capacity rotary vane')


            # Backing pump suggestion
            backing_suggestion = ''
            if suggest_backing:
                if any('Turbomolecular' in s for s in pump_type_suggestion):
                    backing_suggestion = 'Use dry rotary vane or dry screw pump as backing; choose backing speed >= 0.5x turbo foreline conductance-adjusted speed.'
                elif any('Roots' in s for s in pump_type_suggestion):
                    backing_suggestion = 'Use large capacity rotary vane or screw backing pump sized for roots pumping speed and pressure range.'
                else:
                    backing_suggestion = 'No special backing pump needed for this pressure range.'


            # Create results display
            st.subheader("Vacuum Sizing Results")

            # Create columns for better layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # Results table
                results_data = {
                    'Parameter': [
                        'Chamber volume (m³)',
                        'Chamber volume (L)',
                        'Leak/Outgassing (mbar·L/s)',
                        'Leak/Outgassing (Pa·m³/s)',
                        'Target pressure (mbar)',
                        'Target pressure (Pa)',
                        'Required pumping speed (L/s)',
                        'Required pumping speed (m³/s)',
                        'Molecular conductance (L/s)',
                        'Effective pumping speed (L/s)',
                        'Time constant τ (s)',
                        'Pump-down time (min)'
                    ],
                    'Value': [
                        f"{chamber_volume_m3:.4f}",
                        f"{chamber_volume_l:.1f}",
                        f"{leak_rate_mbarL_s:.4g}",
                        f"{Q_pa_m3_s:.4g}",
                        f"{desired_pressure_mbar:.4g}",
                        f"{P_target_Pa:.4g}",
                        f"{S_required_Ls:.2f}",
                        f"{S_required_m3_s:.6f}",
                        f"{C_molecular_Ls:.2f}" if not np.isinf(C_molecular_Ls) else "∞",
                        f"{S_effective_Ls:.2f}" if not np.isnan(S_effective_Ls) else "N/A",
                        f"{tau_sec:.1f}" if not np.isinf(tau_sec) else "∞",
                        f"{t_to_target_sec/60.0:.1f}" if not np.isinf(t_to_target_sec) else "∞"
                    ]
                }

                st.dataframe(pd.DataFrame(results_data), use_container_width=True)

                if available_pumping_speed_Ls > 0:
                    st.info(f"**With available speed {available_pumping_speed_Ls:.1f} L/s:** Steady pressure = {P_steady_Pa/100.0:.4g} mbar")


            with col2:
                st.subheader("Pump & Backing Suggestions")
                st.write("**Pump types suitable for this application:**")
                for s in pump_type_suggestion:
                    st.write(f"• {s}")

                if backing_suggestion:
                    st.write(f"**Backing pump:** {backing_suggestion}")


            # Conductance guidance & plot
            st.markdown("---")
            st.subheader("Conductance Analysis")

            # Plot conductance vs diameter for this length
            diameters = np.linspace(5, 150, 50)
            C_vs_d = 12.1 * (diameters/10.0)**3 / foreline_length_m if foreline_length_m > 0 else np.zeros_like(diameters)

            fig_c, ax_c = plt.subplots(figsize=(10, 6))
            ax_c.plot(diameters, C_vs_d, linewidth=2, color='blue')
            ax_c.set_xlabel('Foreline diameter (mm)')
            ax_c.set_ylabel('Molecular conductance (L/s)')
            ax_c.set_title(f'Conductance vs Diameter (Length = {foreline_length_m:.1f} m)')
            ax_c.grid(True, alpha=0.3)

            # Annotate current diameter
            current_C = 12.1 * (foreline_id_mm/10.0)**3 / foreline_length_m if foreline_length_m > 0 else 0
            ax_c.scatter([foreline_id_mm], [current_C], color='red', s=100, zorder=5, label=f'Current: {foreline_id_mm:.0f} mm')
            ax_c.legend()

            # Add diminishing returns annotation
            if len(C_vs_d) > 0 and max(C_vs_d) > 0:
                idx_80 = np.where(C_vs_d >= 0.8 * max(C_vs_d))[0]
                if len(idx_80) > 0:
                    d_hint = diameters[idx_80[0]]
                    ax_c.axvline(d_hint, color='orange', linestyle='--', alpha=0.7)
                    ax_c.text(d_hint + 5, max(C_vs_d) * 0.9, f'Diminishing returns\n~{d_hint:.0f} mm',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))

            plt.tight_layout()
            buf_c = io.BytesIO()
            fig_c.savefig(buf_c, format='png', dpi=150, bbox_inches='tight')
            fig_c_png = buf_c.getvalue()
            buf_c.close()
            plt.close(fig_c)

            st.image(fig_c_png, use_column_width=True)
            st.markdown("**Inference:** Increasing foreline diameter improves conductance rapidly for small diameters; beyond a point returns diminish and routing/space constraints dominate.")

            # Create vacuum report data
            df_vac = pd.DataFrame({
                'Parameter': [
                    'Chamber volume (L)',
                    'Leak (mbar·L/s)',
                    'Target pressure (mbar)',
                    'Required S (L/s)',
                    'Foreline conductance (L/s)',
                    'Effective S (L/s)',
                    'Time constant (s)',
                    'Pump-down time (min)'
                ],
                'Value': [
                    chamber_volume_l,
                    leak_rate_mbarL_s,
                    desired_pressure_mbar,
                    S_required_Ls,
                    C_molecular_Ls if not np.isinf(C_molecular_Ls) else 999999,
                    S_effective_Ls if not np.isnan(S_effective_Ls) else 0,
                    tau_sec if not np.isinf(tau_sec) else 999999,
                    t_to_target_sec/60.0 if not np.isinf(t_to_target_sec) else 999999
                ]
            })

            # Notes & Limitations
            st.markdown("---")
            st.subheader("Notes & Limitations")
            st.write("• This calculator uses simplified steady-state/first-order transient models.")
            st.write("• Conductance of piping, gas composition, water vapor loads and molecular vs viscous flow regime effects are NOT fully accounted for.")
            st.write("• For turbomolecular systems, consult vendor for foreline conductance, required backing speed, and ultimate pressure with gas load.")
            st.write("• Units: 1 mbar·L/s = 0.1 Pa·m³/s. Pumping speed is given in L/s (typical vacuum industry unit).")


            # Excel export for vacuum
            if st.button("Generate Vacuum Excel Report", type="primary"):
                excel_bytes = create_excel_report_vacuum(df_vac, fig_c_png)
                if excel_bytes:
                    st.download_button(
                        "📥 Download Vacuum Report (xlsx)",
                        data=excel_bytes,
                        file_name=f"vacuum_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

        except Exception as e:
            st.error(f"Calculation error: {e}")
            st.error("Please check your input values and try again.")

# Footer note
st.markdown("---")
st.caption("⚠️ This tool provides engineering estimates to help procurement and vendor discussions. Validate with vendor curves, conductance calculations and field data before procurement.")

# Add sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("This calculator provides preliminary sizing for:")
    st.markdown("• Centrifugal and positive displacement pumps")
    st.markdown("• Vacuum pumps and systems")
    st.markdown("• NPSH calculations")
    st.markdown("• Power requirements")
    st.markdown("• Vendor selection guidance")

    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("• Always validate results with vendor data")
    st.markdown("• Consider safety margins for critical applications")
    st.markdown("• Upload vendor catalogs for automatic matching")
    st.markdown("• Export results for vendor discussions")
