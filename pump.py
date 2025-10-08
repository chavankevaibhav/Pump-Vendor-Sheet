Okay, I've analyzed the provided Streamlit code for the "Advanced Pump & Vacuum Sizing Sheet". The code is largely complete, featuring multiple pages for Rotating Pumps, Vacuum Pumps, System Comparison, and Life Cycle Cost Analysis, along with numerous calculations, visualizations, and analysis tools.

However, there are a few minor issues and missing pieces that need to be addressed to make the code fully functional:

1.  **Missing Helper Functions:** Two helper functions (`calculate_vibration_severity`, `calculate_pressure_pulsation`, `estimate_seal_life`) are called in the "Rotating Pumps" section but are not defined anywhere in the provided code.
2.  **Incomplete Feature:** The "Export to Excel" button on the Rotating Pumps page mentions a `create_excel_report_rotating` function which is not defined.
3.  **Missing PDF Generation:** The "Generate PDF Report" button likely requires an external library like `reportlab`.
4.  **Minor Formatting:** There's a small formatting issue in the sidebar markdown (`st.markdown("**Vacuum Systems:**")` followed by `st.markdown("✓ Gas load scenarios")` without a newline `---` in between, and similarly in the "Analysis Tools" section).

Here is the complete, corrected code with the missing helper functions and the Excel export function added:

```python
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
from datetime import datetime
# For Excel export (requires openpyxl or xlsxwriter)
# For PDF export (requires reportlab)
# pip install openpyxl reportlab

# Set page config at the very top
st.set_page_config(page_title="Advanced Pump & Vacuum Sizing", layout="wide")
st.title("🔧 Advanced Pump & Vacuum Pump Sizing Sheet — Vendor Ready")

# Simple navigation between pages
page = st.sidebar.selectbox("Choose tool", [
    "Rotating Pumps (Centrifugal etc.)", 
    "Vacuum Pump Calculator",
    "Pump System Comparison",
    "Life Cycle Cost Analysis"
])

# ------------------ Helper functions ------------------
def colebrook_f(Re, D, eps_rel, tol=1e-6, max_iter=50):
    """Calculate friction factor using Colebrook equation"""
    if Re <= 0:
        return np.nan
    if Re < 2300:
        return 64.0 / Re
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
        return 0.02

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

def calculate_affinity_laws(Q1, H1, N1, N2):
    """Apply pump affinity laws for speed changes"""
    Q2 = Q1 * (N2/N1)
    H2 = H1 * (N2/N1)**2
    P2_ratio = (N2/N1)**3
    return Q2, H2, P2_ratio

def calculate_parallel_pumps(n_pumps, Q_single, H_single):
    """Calculate parallel pump operation"""
    Q_total = n_pumps * Q_single
    H_total = H_single  # Head remains same
    return Q_total, H_total

def calculate_series_pumps(n_pumps, Q_single, H_single):
    """Calculate series pump operation"""
    Q_total = Q_single  # Flow remains same
    H_total = n_pumps * H_single
    return Q_total, H_total

def calculate_cavitation_index(NPSHa, H):
    """Calculate Thoma cavitation number"""
    if H <= 0:
        return np.nan
    return NPSHa / H

def suggest_impeller(material):
    """Suggest impeller type based on material"""
    mapping = {
        'Water': 'Cast iron closed impeller',
        'Seawater': 'Bronze open impeller',
        'Acids': 'PVDF or Stainless steel semi-open impeller',
        'Slurry': 'High-chrome open impeller',
        'Food-grade': 'Stainless steel closed impeller',
        'Oil': 'Cast Iron or Stainless Steel',
        'Alkaline': 'Stainless Steel or hastelloy',
        'More': 'Consult vendor'
    }
    return mapping.get(material, 'Consult vendor')

def suggest_density_range(fluid_type):
    """Suggest density range based on fluid type"""
    ranges = {
        'Water': (990, 1000),
        'Seawater': (1020, 1030),
        'Acids': (1050, 1840),
        'Slurry': (1100, 2000),
        'Food-grade': (1000, 1500),
        'Oil': (700, 950),
        'Alkaline': (1050, 1500),
        'More': (None, None)
    }
    return ranges.get(fluid_type, (None, None))

def compute_bep(Q_points, eff_curve):
    """Find Best Efficiency Point"""
    if len(eff_curve) == 0:
        return 0, 0, 0
    idx = np.nanargmax(eff_curve)
    return Q_points[idx], eff_curve[idx], idx

def generate_pump_curves(Q_design, total_head_design, static_head):
    """Generate representative pump and system curves"""
    if Q_design <= 0:
        Q_design = 1e-6
    Q_points = np.linspace(max(1e-9, Q_design*0.1), Q_design*1.6, 200)
    # System curve
    a = (total_head_design - static_head) / (Q_design**2) if Q_design > 0 else 0
    H_system = static_head + a * (Q_points**2)
    # Pump curve
    H0 = total_head_design*1.15
    k = H0 / ((Q_design*1.4)**2) if Q_design > 0 else 0
    H_pump = H0 - k * (Q_points**2)
    # Efficiency curve
    eff_curve = np.clip(0.45 + 0.4 * np.exp(-((Q_points-Q_design)/(Q_design*0.25))**2), 0.1, 0.95)
    # Power curve
    power_curve = np.zeros_like(Q_points)
    for i, (q, h, eff) in enumerate(zip(Q_points, H_pump, eff_curve)):
        if eff > 0:
            power_curve[i] = 1000 * 9.81 * q * h / eff / 1000.0
    return Q_points, H_system, H_pump, eff_curve, power_curve

def calculate_suction_specific_speed(N, Q, NPSHr):
    """Calculate suction specific speed (Nss)"""
    if NPSHr <= 0:
        return np.nan
    # Nss = N * sqrt(Q) / (NPSHr)^0.75
    return N * np.sqrt(Q * 3600) / (NPSHr**0.75)

def estimate_wear_rate(material_type, V, particle_size=0):
    """Estimate relative wear rate"""
    base_wear = {'Water': 1, 'Seawater': 2, 'Acids': 3, 'Slurry': 10, 
                 'Food-grade': 1, 'Oil': 0.5, 'Alkaline': 2, 'More': 1}
    wear = base_wear.get(material_type, 1)
    # Velocity impact
    if V > 3:
        wear *= (1 + 0.5 * (V - 3))
    # Particle impact for slurry
    if material_type == 'Slurry' and particle_size > 0:
        wear *= (1 + particle_size/10)
    return wear

# --- NEW: Missing Helper Functions for Rotating Pumps ---
def calculate_vibration_severity(V, Re, material_type):
    """Estimate vibration severity based on velocity and flow regime"""
    severity = "Low"
    color = "green"
    if V > 4.0:
        severity = "High"
        color = "red"
    elif V > 3.0 or (Re > 10000 and material_type in ['Slurry', 'Acids']):
        severity = "Medium"
        color = "orange"
    return severity, color

def calculate_pressure_pulsation(pump_type, Q_op, Q_bep):
    """Estimate pulsation risk based on operating point vs BEP"""
    if pump_type.lower() in ['centrifugal', 'axial']:
        if abs(Q_op - Q_bep) / Q_bep > 0.3:
            return "High", "red"
        elif abs(Q_op - Q_bep) / Q_bep > 0.15:
            return "Medium", "orange"
        else:
            return "Low", "green"
    else: # Assume PD pumps have inherent pulsation
        return "Medium", "orange"

def estimate_seal_life(material_type, T, V):
    """Estimate mechanical seal life based on fluid, temp, velocity"""
    base_life = 20000 # hours
    temp_factor = max(0.5, min(1.5, 1.0 - (T - 25) * 0.005)) # Degrades above 25C
    velocity_factor = max(0.5, 1.0 - V * 0.05) # Degrades with velocity
    material_factor = {'Water': 1.0, 'Seawater': 0.8, 'Acids': 0.7, 'Slurry': 0.5,
                       'Food-grade': 0.9, 'Oil': 1.0, 'Alkaline': 0.7, 'More': 0.8}.get(material_type, 0.8)
    
    estimated_hours = base_life * temp_factor * velocity_factor * material_factor
    return estimated_hours
# --- END NEW FUNCTIONS ---

# --- NEW: Excel Export Function ---
def create_excel_report_rotating(results_data):
    """Create an Excel report for Rotating Pump calculations"""
    import pandas as pd
    from io import BytesIO

    # Create a BytesIO buffer
    buffer = BytesIO()

    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Write summary data to the first sheet
        summary_df = pd.DataFrame(list(results_data.items()), columns=['Parameter', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Add other sheets for curves, analysis if needed
        # Example: Add pump curves data
        if 'Q_points' in results_data and 'H_pump' in results_data:
            curve_df = pd.DataFrame({
                'Flow (m3/h)': [q * 3600 for q in results_data['Q_points']],
                'Head (m)': results_data['H_pump'],
                'Efficiency (%)': [e * 100 for e in results_data['eff_curve']],
                'Power (kW)': results_data['power_curve']
            })
            curve_df.to_excel(writer, sheet_name='Curves', index=False)

    # Get the value from the buffer
    buffer.seek(0)
    return buffer
# --- END EXCEL FUNCTION ---

# ------------------ Rotating Pumps Page ------------------
if page == "Rotating Pumps (Centrifugal etc.)":
    st.header("🔄 Rotating Pump Sizing & Selection")
    with st.form(key='rotating'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Process & Fluid Data")
            Q_input = st.number_input("Flow rate", value=100.0, min_value=0.0, format="%.6f")
            Q_unit = st.selectbox("Flow unit", ['m³/h', 'L/s', 'm³/s', 'm³/d', 'GPM (US)'], index=0)
            T = st.number_input("Fluid temperature (°C)", value=25.0)
            material_type = st.selectbox("Fluid type", ['Water', 'Seawater', 'Acids', 'Alkaline', 'Slurry', 'Food-grade', 'Oil', 'More'])
            SG = st.number_input("Specific gravity", value=1.0, min_value=0.01)
            mu_cP = st.number_input("Viscosity (cP)", value=1.0, min_value=0.01)
            if material_type == 'Slurry':
                particle_size = st.number_input("Average particle size (mm)", value=0.0, min_value=0.0)
            else:
                particle_size = 0
            density = 1000.0 * SG
            density_range = suggest_density_range(material_type)
            if density_range[0] is not None:
                st.info(f"Suggested density: {density_range[0]:.0f} - {density_range[1]:.0f} kg/m³")
            if st.checkbox("Override density (kg/m³)?", value=False):
                density = st.number_input("Density (kg/m³)", value=1000.0, min_value=0.1)
            st.write(f"**Calculated Density:** {density:.2f} kg/m³")
        with col2:
            st.subheader("Piping & Elevation")
            D_inner = st.number_input("Pipe inner diameter (mm)", value=100.0, min_value=1.0)
            L_pipe = st.number_input("Pipe length (m)", value=100.0, min_value=0.0)
            elevation_in = st.number_input("Suction elevation (m)", value=0.0)
            elevation_out = st.number_input("Discharge elevation (m)", value=10.0)
            K_fittings = st.number_input("Total K (fittings)", value=2.0, min_value=0.0)
            eps_mm = st.number_input("Roughness (mm)", value=0.045, min_value=0.0001)
            st.subheader("Multiple Pump Configuration")
            pump_config = st.selectbox("Pump arrangement", ['Single', 'Parallel (n pumps)', 'Series (n pumps)'])
            if pump_config != 'Single':
                n_pumps = st.number_input("Number of pumps", value=2, min_value=2, max_value=10)
            else:
                n_pumps = 1
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Pump & Motor Settings")
            pump_eff_user = st.number_input("Pump efficiency (%)", value=70.0, min_value=1.0, max_value=100.0)/100.0
            motor_eff = st.number_input("Motor efficiency (%)", value=95.0, min_value=10.0, max_value=100.0)/100.0
            safety_margin_head = st.number_input("Design margin on head (%)", value=10.0, min_value=0.0)/100.0
            safety_margin_flow = st.number_input("Design margin on flow (%)", value=10.0, min_value=0.0)/100.0
            service_factor = st.number_input("Service factor", value=1.15, min_value=1.0)
            pump_speed_rpm = st.number_input("Pump speed (RPM)", value=1450.0, min_value=100.0)
        with col4:
            st.subheader("Application & NPSH")
            application = st.selectbox("Application", ['General Transfer', 'Chemical Handling', 'Slurry Transport', 
                                                       'Oil Transfer', 'High Pressure', 'Metering'])
            atm_pressure_kPa = st.number_input("Atmospheric pressure (kPa)", value=101.325, min_value=50.0)
            vapor_pressure_kPa = st.number_input("Vapor pressure (kPa)", value=2.3, min_value=0.0)
            friction_for_NPSH = st.number_input("Suction friction head (m)", value=2.0, min_value=0.0)
            NPSHr_vendor = st.number_input("Vendor NPSHr (m) [optional]", value=0.0, min_value=0.0)
        st.markdown("---")
        st.subheader("Advanced Analysis Options")
        col5, col6 = st.columns(2)
        with col5:
            show_affinity = st.checkbox("Analyze speed variations", value=True)
            show_wear_analysis = st.checkbox("Show wear analysis", value=True)
        with col6:
            show_energy_cost = st.checkbox("Calculate energy costs", value=True)
            if show_energy_cost:
                electricity_cost = st.number_input("Electricity cost ($/kWh)", value=0.12, min_value=0.0)
                operating_hours = st.number_input("Operating hours/year", value=8000.0, min_value=0.0)
        submitted = st.form_submit_button("🚀 Calculate", type="primary")

    if submitted:
        try:
            # Flow conversion
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

            # Basic calculations
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

            # Multiple pump configuration
            if pump_config == 'Parallel (n pumps)':
                Q_total, H_total = calculate_parallel_pumps(n_pumps, Q_design, total_head_design)
                st.info(f"**Parallel Configuration:** {n_pumps} pumps × {Q_design*3600:.1f} m³/h each = {Q_total*3600:.1f} m³/h total")
                Q_design_per_pump = Q_design
                H_design_per_pump = total_head_design
            elif pump_config == 'Series (n pumps)':
                Q_total, H_total = calculate_series_pumps(n_pumps, Q_design, total_head_design)
                st.info(f"**Series Configuration:** {n_pumps} pumps × {total_head_design:.1f} m each = {H_total:.1f} m total")
                Q_design_per_pump = Q_design
                H_design_per_pump = total_head_design / n_pumps
            else:
                Q_design_per_pump = Q_design
                H_design_per_pump = total_head_design

            pump_eff = pump_eff_user
            shaft_kW, electrical_kW = pump_power_required(density, 9.81, Q_design, total_head_design, pump_eff, motor_eff)

            # NPSH calculations
            P_atm_Pa = atm_pressure_kPa * 1000.0
            P_vap_Pa = vapor_pressure_kPa * 1000.0
            z_suction = elevation_in
            NPSHa = (P_atm_Pa - P_vap_Pa)/(density*9.81) + z_suction - friction_for_NPSH
            sigma = calculate_cavitation_index(NPSHa, total_head_design)

            if NPSHr_vendor > 0:
                NPSH_margin = NPSHa - NPSHr_vendor
                if NPSH_margin < 0.5:
                    npsh_warning = "⚠️ CRITICAL: NPSH margin too low! Risk of cavitation."
                elif NPSH_margin < 1.0:
                    npsh_warning = "⚡ WARNING: Low NPSH margin. Consider design modifications."
                else:
                    npsh_warning = "✅ Adequate NPSH margin"
            else:
                NPSH_margin = np.nan
                npsh_warning = "No vendor NPSHr provided"

            # Specific speed
            Ns = pump_speed_rpm * math.sqrt(Q_design*3600.0) / (total_head_design**0.75) if total_head_design > 0 else np.nan

            # Suction specific speed
            if NPSHr_vendor > 0:
                Nss = calculate_suction_specific_speed(pump_speed_rpm, Q_design, NPSHr_vendor)
            else:
                Nss = np.nan

            # Generate curves
            Q_points, H_system, H_pump, eff_curve, power_curve = generate_pump_curves(Q_design, total_head_design, static_head)
            Q_bep, eff_bep, bep_idx = compute_bep(Q_points, eff_curve)

            # Operating point
            if len(H_pump) > 0 and len(H_system) > 0:
                idx_op = np.argmin((H_pump - H_system)**2)
                Q_op = Q_points[idx_op]
                H_op = H_pump[idx_op]
                eff_op = eff_curve[idx_op]
                power_op = power_curve[idx_op]
            else:
                Q_op = Q_design
                H_op = total_head_design
                eff_op = pump_eff
                power_op = shaft_kW

            pct_from_bep = abs((Q_op - Q_bep)/Q_bep) * 100.0 if Q_bep > 0 else np.nan

            # Wear analysis
            if show_wear_analysis:
                wear_rate = estimate_wear_rate(material_type, V, particle_size)
                estimated_service_life = 50000 / wear_rate  # hours

            # Vibration and reliability analysis
            vibration_severity, vib_color = calculate_vibration_severity(V, Re, material_type)
            pulsation_risk, pulse_color = calculate_pressure_pulsation('Centrifugal', Q_op, Q_bep)
            seal_life_hours = estimate_seal_life(material_type, T, V)
            motor_rated_kW = electrical_kW * service_factor

            # Display results
            st.success("✅ Calculation Complete")
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Results Summary", "📈 Performance Curves", "⚙️ Advanced Analysis", "💰 Cost Analysis"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Design Flow", f"{Q_design*3600:.2f} m³/h", f"{Q_m3s*3600:.2f} m³/h actual")
                    st.metric("Total Head", f"{total_head_design:.2f} m", f"+{safety_margin_head*100:.0f}% margin")
                    st.metric("Velocity", f"{V:.2f} m/s", 
                             "⚠️ High" if V > 3 else ("⚠️ Low" if V < 0.5 else "✅ OK"))
                with col2:
                    st.metric("Shaft Power", f"{shaft_kW:.2f} kW")
                    st.metric("Motor Rating", f"{motor_rated_kW:.2f} kW", f"SF: {service_factor}")
                    st.metric("Efficiency", f"{eff_op*100:.1f}%", f"BEP: {eff_bep*100:.1f}%")
                with col3:
                    st.metric("NPSHa", f"{NPSHa:.2f} m")
                    if NPSHr_vendor > 0:
                        st.metric("NPSH Margin", f"{NPSH_margin:.2f} m", npsh_warning)
                    st.metric("Reynolds #", f"{Re:.0f}", 
                             "Laminar" if Re < 2300 else "Turbulent")

                st.markdown("---")
                # Detailed results table
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Hydraulic Parameters")
                    hydraulic_data = pd.DataFrame({
                        'Parameter': ['Static Head', 'Friction Loss', 'Minor Losses', 'Total Head', 
                                     'Design Head', 'Friction Factor', 'Pipe Diameter'],
                        'Value': [f"{static_head:.2f} m", f"{hf:.2f} m", f"{hm:.2f} m", 
                                 f"{total_head:.2f} m", f"{total_head_design:.2f} m", 
                                 f"{f:.4f}", f"{D_inner:.1f} mm"]
                    })
                    st.dataframe(hydraulic_data, use_container_width=True, hide_index=True)
                with col_b:
                    st.subheader("Pump Characteristics")
                    pump_data = pd.DataFrame({
                        'Parameter': ['Specific Speed (Ns)', 'Suction Specific Speed', 'Cavitation Index (σ)', 
                                     'BEP Flow', 'Operating Flow', 'Deviation from BEP'],
                        'Value': [f"{Ns:.0f}" if not np.isnan(Ns) else "N/A",
                                 f"{Nss:.0f}" if not np.isnan(Nss) else "N/A",
                                 f"{sigma:.3f}" if not np.isnan(sigma) else "N/A",
                                 f"{Q_bep*3600:.2f} m³/h", f"{Q_op*3600:.2f} m³/h",
                                 f"{pct_from_bep:.1f}%"]
                    })
                    st.dataframe(pump_data, use_container_width=True, hide_index=True)

                # Recommendations
                st.markdown("---")
                st.subheader("🎯 Recommendations")
                rec_col1, rec_col2 = st.columns(2)
                with rec_col1:
                    st.write("**Pump Type:**", suggest_impeller(material_type))
                    st.write("**Application:**", application)
                    if pct_from_bep <= 10:
                        st.success("✅ Excellent: Operating within 10% of BEP")
                    elif pct_from_bep <= 20:
                        st.warning("⚡ Acceptable: Consider 5-10% derating")
                    else:
                        st.error("⚠️ Poor: Far from BEP. Consider different size")
                with rec_col2:
                    if show_wear_analysis:
                        st.write(f"**Relative Wear Rate:** {wear_rate:.1f}x baseline")
                        st.write(f"**Est. Service Life:** {estimated_service_life:.0f} hours")
                        if wear_rate > 5:
                            st.warning("⚠️ High wear expected. Consider hardened materials.")
                    st.write(f"**Vibration Risk:** :{vib_color}[{vibration_severity}]")
                    st.write(f"**Pulsation Risk:** :{pulse_color}[{pulsation_risk}]")
                    st.write(f"**Seal Life Est.:** {seal_life_hours:.0f} hours")


            with tab2:
                st.subheader("Performance Curves")
                # Main pump curve
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Head curves
                ax1.plot(Q_points*3600, H_system, 'b-', linewidth=2, label='System Curve')
                ax1.plot(Q_points*3600, H_pump, 'r-', linewidth=2, label='Pump Curve')
                ax1.scatter([Q_bep*3600], [H_pump[bep_idx]], color='green', s=150, 
                           marker='*', label='BEP', zorder=5, edgecolors='black')
                ax1.scatter([Q_op*3600], [H_op], color='orange', s=150, 
                           marker='D', label='Operating Point', zorder=5, edgecolors='black')
                ax1.fill_between(Q_points*3600, H_system, alpha=0.2, color='blue')
                ax1.axvline(Q_design*3600, color='gray', linestyle='--', alpha=0.5, label='Design Flow')
                ax1.set_xlabel('Flow Rate (m³/h)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Head (m)', fontsize=11, fontweight='bold')
                ax1.set_title('Pump & System Curves', fontsize=12, fontweight='bold')
                ax1.legend(loc='best', framealpha=0.9)
                ax1.grid(True, alpha=0.3, linestyle=':')

                # Efficiency curve
                ax2.plot(Q_points*3600, eff_curve*100, 'g-', linewidth=2, label='Efficiency')
                ax2.axvline(Q_bep*3600, color='green', linestyle='--', alpha=0.7, label='BEP')
                ax2.axvline(Q_op*3600, color='orange', linestyle='--', alpha=0.7, label='Operating Point')
                ax2.fill_between(Q_points*3600, eff_curve*100, alpha=0.3, color='green')
                ax2.set_xlabel('Flow Rate (m³/h)', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
                ax2.set_title('Pump Efficiency Curve', fontsize=12, fontweight='bold')
                ax2.legend(loc='best', framealpha=0.9)
                ax2.grid(True, alpha=0.3, linestyle=':')
                ax2.set_ylim([0, 100])

                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()

                # Power curve
                fig2, ax3 = plt.subplots(figsize=(10, 5))
                ax3.plot(Q_points*3600, power_curve, 'purple', linewidth=2, label='Power Required')
                ax3.axvline(Q_op*3600, color='orange', linestyle='--', alpha=0.7, label='Operating Point')
                ax3.axhline(motor_rated_kW, color='red', linestyle=':', alpha=0.7, label='Motor Rating')
                ax3.fill_between(Q_points*3600, power_curve, alpha=0.3, color='purple')
                ax3.set_xlabel('Flow Rate (m³/h)', fontsize=11, fontweight='bold')
                ax3.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
                ax3.set_title('Power Consumption vs Flow', fontsize=12, fontweight='bold')
                ax3.legend(loc='best', framealpha=0.9)
                ax3.grid(True, alpha=0.3, linestyle=':')

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()


            with tab3:
                st.subheader("⚙️ Advanced Analysis")
                if show_affinity:
                    st.markdown("#### Affinity Laws Analysis")
                    st.write("See how pump performance changes with speed variation:")
                    speed_variations = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
                    affinity_results = []
                    for speed_ratio in speed_variations:
                        Q_new, H_new, P_ratio = calculate_affinity_laws(
                            Q_design, total_head_design, pump_speed_rpm, pump_speed_rpm * speed_ratio
                        )
                        affinity_results.append({
                            'Speed (RPM)': pump_speed_rpm * speed_ratio,
                            'Speed %': speed_ratio * 100,
                            'Flow (m³/h)': Q_new * 3600,
                            'Head (m)': H_new,
                            'Power Ratio': P_ratio
                        })

                    df_affinity = pd.DataFrame(affinity_results)
                    col_af1, col_af2 = st.columns(2)
                    with col_af1:
                        st.dataframe(df_affinity, use_container_width=True, hide_index=True)
                    with col_af2:
                        fig_aff, (ax_aff1, ax_aff2) = plt.subplots(2, 1, figsize=(8, 8))
                        ax_aff1.plot(df_affinity['Speed %'], df_affinity['Flow (m³/h)'], 'bo-', linewidth=2)
                        ax_aff1.axvline(100, color='red', linestyle='--', alpha=0.5, label='Current Speed')
                        ax_aff1.set_xlabel('Speed (%)', fontweight='bold')
                        ax_aff1.set_ylabel('Flow (m³/h)', fontweight='bold')
                        ax_aff1.set_title('Flow vs Speed')
                        ax_aff1.grid(True, alpha=0.3)
                        ax_aff1.legend()

                        ax_aff2.plot(df_affinity['Speed %'], df_affinity['Head (m)'], 'ro-', linewidth=2)
                        ax_aff2.axvline(100, color='red', linestyle='--', alpha=0.5, label='Current Speed')
                        ax_aff2.set_xlabel('Speed (%)', fontweight='bold')
                        ax_aff2.set_ylabel('Head (m)', fontweight='bold')
                        ax_aff2.set_title('Head vs Speed')
                        ax_aff2.grid(True, alpha=0.3)
                        ax_aff2.legend()

                        plt.tight_layout()
                        st.pyplot(fig_aff)
                        plt.close()

                # NPSH Analysis
                st.markdown("---")
                st.markdown("#### NPSH Analysis")
                col_npsh1, col_npsh2 = st.columns(2)
                with col_npsh1:
                    # NPSH margin chart
                    flow_range = np.linspace(0.5, 1.5, 20)
                    NPSHa_curve = np.ones_like(flow_range) * NPSHa
                    if NPSHr_vendor > 0:
                        # Estimate NPSHr increases with flow
                        NPSHr_curve = NPSHr_vendor * (flow_range ** 1.5)
                    else:
                        NPSHr_curve = 3.0 * (flow_range ** 1.5)  # Typical curve

                    fig_npsh, ax_npsh = plt.subplots(figsize=(8, 5))
                    ax_npsh.plot(flow_range * Q_design * 3600, NPSHa_curve, 'b-', linewidth=2, label='NPSHa')
                    ax_npsh.plot(flow_range * Q_design * 3600, NPSHr_curve, 'r-', linewidth=2, label='NPSHr (estimated)')
                    ax_npsh.fill_between(flow_range * Q_design * 3600, NPSHr_curve, NPSHa_curve, 
                                        where=(NPSHa_curve > NPSHr_curve), alpha=0.3, color='green', label='Safe Zone')
                    ax_npsh.fill_between(flow_range * Q_design * 3600, NPSHr_curve, NPSHa_curve, 
                                        where=(NPSHa_curve <= NPSHr_curve), alpha=0.3, color='red', label='Cavitation Risk')
                    ax_npsh.axvline(Q_op * 3600, color='orange', linestyle='--', label='Operating Point')
                    ax_npsh.set_xlabel('Flow (m³/h)', fontweight='bold')
                    ax_npsh.set_ylabel('NPSH (m)', fontweight='bold')
                    ax_npsh.set_title('NPSH Available vs Required', fontweight='bold')
                    ax_npsh.legend(loc='best')
                    ax_npsh.grid(True, alpha=0.3)

                    st.pyplot(fig_npsh)
                    plt.close()
                with col_npsh2:
                    st.write("**NPSH Analysis Summary:**")
                    st.write(f"- NPSHa: {NPSHa:.2f} m")
                    if NPSHr_vendor > 0:
                        st.write(f"- NPSHr (vendor): {NPSHr_vendor:.2f} m")
                        st.write(f"- Margin: {NPSH_margin:.2f} m")
                        st.write(f"- Safety Factor: {NPSHa/NPSHr_vendor:.2f}x")
                    st.write(f"- Cavitation Index: {sigma:.3f}" if not np.isnan(sigma) else "N/A")

                    st.markdown("**Recommendations:**")
                    if NPSHa < 2:
                        st.error("⚠️ Very low NPSHa. Consider:")
                        st.write("  - Lowering pump installation")
                        st.write("  - Reducing suction pipe losses")
                        st.write("  - Increasing suction pipe diameter")
                    elif NPSHr_vendor > 0 and NPSH_margin < 1:
                        st.warning("⚡ Marginal NPSH. Monitor for cavitation.")
                    else:
                        st.success("✅ Adequate NPSH margin")


                # Pipe sizing optimization
                st.markdown("---")
                st.markdown("#### 🔧 Pipe Sizing Optimization")
                # Test different pipe sizes
                test_diameters = np.array([50, 75, 100, 125, 150, 200, 250, 300])  # mm
                head_losses = []
                velocities = []
                pipe_costs = []
                for d_test in test_diameters:
                    d_m = d_test / 1000.0
                    v_test = velocity_from_flow(Q_design, d_m)
                    re_test = reynolds(density, v_test, d_m, mu)
                    f_test = colebrook_f(re_test, d_m, eps_mm/1000.0)
                    hf_test = darcy_head_loss(f_test, L_pipe, d_m, v_test)
                    velocities.append(v_test)
                    head_losses.append(hf_test)
                    pipe_costs.append(d_test * 0.5 * L_pipe)  # Simplified cost model

                fig_pipe, (ax_pipe1, ax_pipe2) = plt.subplots(1, 2, figsize=(14, 5))

                # Head loss vs diameter
                ax_pipe1.plot(test_diameters, head_losses, 'b-', linewidth=2, marker='o')
                ax_pipe1.axvline(D_inner, color='r', linestyle='--', label=f'Current: {D_inner}mm')
                ax_pipe1.axhline(hf, color='r', linestyle=':', alpha=0.5)
                ax_pipe1.set_xlabel('Pipe Diameter (mm)', fontweight='bold')
                ax_pipe1.set_ylabel('Head Loss (m)', fontweight='bold')
                ax_pipe1.set_title('Head Loss vs Pipe Diameter', fontweight='bold')
                ax_pipe1.legend()
                ax_pipe1.grid(True, alpha=0.3)

                # Velocity vs diameter with recommended zones
                ax_pipe2.plot(test_diameters, velocities, 'g-', linewidth=2, marker='s')
                ax_pipe2.axvline(D_inner, color='r', linestyle='--', label=f'Current: {D_inner}mm')
                ax_pipe2.axhspan(0.5, 3.0, alpha=0.2, color='green', label='Recommended Range')
                ax_pipe2.axhspan(3.0, 5.0, alpha=0.2, color='yellow', label='Acceptable')
                ax_pipe2.axhspan(5.0, 10.0, alpha=0.2, color='red', label='Too High')
                ax_pipe2.set_xlabel('Pipe Diameter (mm)', fontweight='bold')
                ax_pipe2.set_ylabel('Velocity (m/s)', fontweight='bold')
                ax_pipe2.set_title('Velocity vs Pipe Diameter', fontweight='bold')
                ax_pipe2.legend(loc='best')
                ax_pipe2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig_pipe)
                plt.close()

                # Find optimal diameter
                optimal_idx = np.argmin(np.abs(np.array(velocities) - 2.0))  # Target 2 m/s
                optimal_diameter = test_diameters[optimal_idx]
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    st.info(f"**Recommended Diameter:** {optimal_diameter} mm")
                    st.write(f"- Current: {D_inner} mm ({V:.2f} m/s)")
                    st.write(f"- Optimal: {optimal_diameter} mm ({velocities[optimal_idx]:.2f} m/s)")
                    st.write(f"- Head loss reduction: {hf - head_losses[optimal_idx]:.2f} m")
                with col_opt2:
                    if optimal_diameter != D_inner:
                        power_saving = density * 9.81 * Q_design * (hf - head_losses[optimal_idx]) / pump_eff / 1000.0
                        if show_energy_cost:
                            annual_savings = power_saving * operating_hours * electricity_cost
                            st.success(f"**Potential Savings:**")
                            st.write(f"- Power: {power_saving:.2f} kW")
                            st.write(f"- Annual: ${annual_savings:,.2f}")
                        else:
                            st.write(f"**Power Savings:** {power_saving:.2f} kW")
                    else:
                        st.success("✅ Current diameter is optimal")

                # Reliability and Maintenance Prediction
                st.markdown("---")
                st.markdown("#### 🔧 Reliability & Maintenance Analysis")
                col_rel1, col_rel2 = st.columns(2)
                with col_rel1:
                    st.write("**Component Life Estimates:**")
                    # Bearing life estimation (simplified L10 life)
                    bearing_life_hours = 40000 / ((pump_speed_rpm/1450)**3) * (0.8 if material_type == 'Slurry' else 1.0)
                    st.write(f"- Bearings: {bearing_life_hours:.0f} hours")
                    st.write(f"- Mechanical Seal: {seal_life_hours:.0f} hours")
                    st.write(f"- Impeller (wear): {estimated_service_life:.0f} hours" if show_wear_analysis else "")
                    # Calculate MTBF
                    component_lives = [bearing_life_hours, seal_life_hours]
                    if show_wear_analysis:
                        component_lives.append(estimated_service_life)
                    mtbf = min(component_lives) * 0.8  # Conservative estimate
                    st.write(f"**Est. MTBF:** {mtbf:.0f} hours")
                with col_rel2:
                    st.write("**Maintenance Schedule:**")
                    inspection_interval = min(2000, mtbf * 0.1)
                    minor_service_interval = min(4000, mtbf * 0.2)
                    major_overhaul_interval = min(8000, mtbf * 0.4)
                    st.write(f"- Inspection: Every {inspection_interval:.0f} hours")
                    st.write(f"- Minor service: Every {minor_service_interval:.0f} hours")
                    st.write(f"- Major overhaul: Every {major_overhaul_interval:.0f} hours")
                    if operating_hours > 0:
                        annual_inspections = operating_hours / inspection_interval
                        st.write(f"**Annual requirements:**")
                        st.write(f"- {annual_inspections:.1f} inspections/year")

                # Failure mode analysis
                st.markdown("---")
                st.markdown("#### ⚠️ Critical Failure Modes")
                failure_modes = []
                if NPSHa < NPSHr_vendor + 1 and NPSHr_vendor > 0:
                    failure_modes.append(("Cavitation damage", "High", "Monitor NPSH, check for noise/vibration"))
                if pct_from_bep > 25:
                    failure_modes.append(("Recirculation damage", "Medium", "Resize pump or use VFD control"))
                if V > 4:
                    failure_modes.append(("Erosion", "Medium", "Consider larger pipe or flow restriction"))
                if material_type == 'Slurry' and V < 0.8:
                    failure_modes.append(("Settling/plugging", "High", "Increase velocity or use agitation"))
                if Re < 2300 and mu_cP > 100:
                    failure_modes.append(("Performance degradation", "Medium", "Consider PD pump or heating"))
                if seal_life_hours < 8000:
                    failure_modes.append(("Seal failure", "High", "Upgrade seal type or improve cooling"))

                if failure_modes:
                    failure_df = pd.DataFrame(failure_modes, columns=['Failure Mode', 'Risk', 'Mitigation'])
                    st.dataframe(failure_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No critical failure modes identified")


            with tab4:
                if show_energy_cost:
                    st.subheader("💰 Life Cycle Cost Analysis")
                    # Annual energy consumption
                    annual_energy_kWh = electrical_kW * operating_hours
                    annual_cost = annual_energy_kWh * electricity_cost
                    col_cost1, col_cost2, col_cost3 = st.columns(3)
                    with col_cost1:
                        st.metric("Annual Energy", f"{annual_energy_kWh:,.0f} kWh")
                        st.metric("Annual Cost", f"${annual_cost:,.2f}")
                    with col_cost2:
                        # Efficiency impact
                        if eff_op < 0.7:
                            potential_savings = annual_cost * (0.75 - eff_op) / eff_op
                            st.metric("Potential Savings", f"${potential_savings:,.2f}/yr", 
                                     "with 75% efficient pump")
                        else:
                            st.metric("Efficiency Status", "Good", "✅")
                    with col_cost3:
                        # 10-year projection
                        ten_year_cost = annual_cost * 10
                        st.metric("10-Year Energy Cost", f"${ten_year_cost:,.2f}")

                    st.markdown("---")
                    # Cost breakdown over time
                    years = np.arange(1, 11)
                    cumulative_cost = years * annual_cost
                    # Compare with higher efficiency pump
                    high_eff = 0.80
                    if eff_op < high_eff:
                        high_eff_power = electrical_kW * eff_op / high_eff
                        high_eff_cost = high_eff_power * operating_hours * electricity_cost * years
                        savings = cumulative_cost - high_eff_cost

                    fig_cost, (ax_cost1, ax_cost2) = plt.subplots(1, 2, figsize=(14, 5))

                    # Cumulative cost
                    ax_cost1.plot(years, cumulative_cost/1000, 'b-', linewidth=2, marker='o', label='Current Pump')
                    if eff_op < high_eff:
                        ax_cost1.plot(years, high_eff_cost/1000, 'g--', linewidth=2, marker='s', label='80% Efficient Pump')
                        ax_cost1.fill_between(years, high_eff_cost/1000, cumulative_cost/1000, alpha=0.3, color='green')
                    ax_cost1.set_xlabel('Years', fontweight='bold')
                    ax_cost1.set_ylabel('Cumulative Cost ($1000s)', fontweight='bold')
                    ax_cost1.set_title('10-Year Energy Cost Projection', fontweight='bold')
                    ax_cost1.legend()
                    ax_cost1.grid(True, alpha=0.3)

                    # Monthly breakdown
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_cost = np.ones(12) * (annual_cost / 12)
                    ax_cost2.bar(months, monthly_cost, color='steelblue', alpha=0.7)
                    ax_cost2.set_xlabel('Month', fontweight='bold')
                    ax_cost2.set_ylabel('Cost ($)', fontweight='bold')
                    ax_cost2.set_title('Estimated Monthly Energy Cost', fontweight='bold')
                    ax_cost2.grid(True, alpha=0.3, axis='y')
                    plt.setp(ax_cost2.xaxis.get_majorticklabels(), rotation=45)

                    plt.tight_layout()
                    st.pyplot(fig_cost)
                    plt.close()

                    # Payback analysis
                    if eff_op < high_eff:
                        st.markdown("---")
                        st.markdown("#### 💡 Efficiency Upgrade Analysis")
                        upgrade_cost_estimate = motor_rated_kW * 200  # $200/kW rough estimate
                        annual_savings = savings[0]
                        payback_years = upgrade_cost_estimate / annual_savings if annual_savings > 0 else np.inf
                        col_pay1, col_pay2, col_pay3 = st.columns(3)
                        with col_pay1:
                            st.metric("Estimated Upgrade Cost", f"${upgrade_cost_estimate:,.0f}")
                        with col_pay2:
                            st.metric("Annual Savings", f"${annual_savings:,.2f}")
                        with col_pay3:
                            if payback_years < 5:
                                st.metric("Payback Period", f"{payback_years:.1f} years", "✅ Good ROI")
                            elif payback_years < 10:
                                st.metric("Payback Period", f"{payback_years:.1f} years", "⚡ Marginal")
                            else:
                                st.metric("Payback Period", f"{payback_years:.1f} years", "❌ Poor ROI")
                else:
                    st.info("Enable 'Calculate energy costs' in the form to see this analysis")

            # Export functionality
            st.markdown("---")
            st.subheader("📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📄 Generate PDF Report", type="secondary"):
                    st.info("PDF generation feature requires integration with reportlab or similar library.")
                    # Example placeholder for PDF generation
                    # from reportlab.lib.pagesizes import letter
                    # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    # doc = SimpleDocTemplate("pump_report.pdf", pagesize=letter)
                    # story = [Paragraph("Pump Sizing Report", ...), ...]
                    # doc.build(story)
                    # with open("pump_report.pdf", "rb") as pdf_file:
                    #     st.download_button(label="Download PDF", data=pdf_file, file_name="pump_report.pdf", mime="application/pdf")

            with col_exp2:
                if st.button("📊 Export to Excel", type="primary"):
                    # Prepare results data for export
                    results_data = {
                        'Design Flow (m3/h)': Q_design * 3600,
                        'Design Head (m)': total_head_design,
                        'Velocity (m/s)': V,
                        'Shaft Power (kW)': shaft_kW,
                        'Motor Rating (kW)': motor_rated_kW,
                        'Efficiency (%)': eff_op * 100,
                        'NPSHa (m)': NPSHa,
                        'NPSH Margin (m)': NPSH_margin,
                        'Reynolds Number': Re,
                        'Specific Speed (Ns)': Ns,
                        'Suction Specific Speed (Nss)': Nss,
                        'Cavitation Index (σ)': sigma,
                        'BEP Flow (m3/h)': Q_bep * 3600,
                        'Operating Flow (m3/h)': Q_op * 3600,
                        'Deviation from BEP (%)': pct_from_bep,
                        'Relative Wear Rate': wear_rate if show_wear_analysis else 'N/A',
                        'Est. Service Life (hrs)': estimated_service_life if show_wear_analysis else 'N/A',
                        'Vibration Risk': vibration_severity,
                        'Pulsation Risk': pulsation_risk,
                        'Seal Life Est. (hrs)': seal_life_hours,
                        'Annual Energy (kWh)': annual_energy_kWh if show_energy_cost else 'N/A',
                        'Annual Cost ($)': annual_cost if show_energy_cost else 'N/A',
                        '10-Year Energy Cost ($)': ten_year_cost if show_energy_cost else 'N/A',
                        'Q_points': Q_points,
                        'H_pump': H_pump,
                        'eff_curve': eff_curve,
                        'power_curve': power_curve
                    }
                    excel_buffer = create_excel_report_rotating(results_data)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_buffer,
                        file_name=f"pump_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


        except Exception as e:
            st.error(f"❌ Calculation error: {e}")
            st.error("Please check your input values and try again.")

# ------------------ Vacuum Pump Calculator Page ------------------
elif page == "Vacuum Pump Calculator":
    st.header("🌀 Vacuum Pump Sizing & Selection")
    with st.form(key='vacuum'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Chamber & Gas Load")
            chamber_volume_l = st.number_input("Chamber Volume (L)", value=100.0, min_value=0.1)
            leak_rate_mbarL_s = st.number_input("Leak/outgassing (mbar·L/s)", value=0.1, min_value=0.0, format="%.6g")
            desired_pressure_mbar = st.number_input("Desired pressure (mbar)", value=1e-3, format="%.6g", min_value=1e-12)
            desired_pressure_unit = st.selectbox("Pressure unit", ['mbar', 'Pa', 'Torr'], index=0)
            if desired_pressure_unit == 'Pa':
                desired_pressure_mbar = desired_pressure_mbar / 100.0
            elif desired_pressure_unit == 'Torr':
                desired_pressure_mbar = desired_pressure_mbar * 1.33322
        with col2:
            st.subheader("Foreline / Conductance")
            foreline_id_mm = st.number_input("Foreline ID (mm)", value=25.0, min_value=1.0)
            foreline_length_m = st.number_input("Foreline length (m)", value=1.0, min_value=0.01)
            gas_molecular_mass = st.number_input("Gas molecular mass (g/mol)", value=28.97, min_value=1.0)
            temperature_K = st.number_input("Gas temperature (K)", value=293.0, min_value=1.0)

        st.subheader("Pump/Process Constraints")
        available_pumping_speed_Ls = st.number_input("Available pumping speed (L/s) [0=auto]", value=0.0, min_value=0.0)
        suggest_backing = st.checkbox("Suggest backing pump", value=True)
        process_type = st.selectbox("Process Type", ['General Vacuum', 'High Vacuum Research', 
                                                      'Coating/Deposition', 'Freeze Drying', 
                                                      'Semiconductor', 'Analytical Instrument'])
        submitted_vac = st.form_submit_button("🚀 Calculate", type="primary")

    if submitted_vac:
        try:
            # Convert units
            chamber_volume_m3 = chamber_volume_l / 1000.0
            Q_pa_m3_s = leak_rate_mbarL_s * 0.1
            P_target_Pa = desired_pressure_mbar * 100.0

            # Required pumping speed
            if P_target_Pa > 0:
                S_required_m3_s = Q_pa_m3_s / P_target_Pa
            else:
                S_required_m3_s = np.inf
            S_required_Ls = S_required_m3_s * 1000.0

            # Conductance (molecular flow)
            d_cm = foreline_id_mm / 10.0
            if foreline_length_m > 0:
                C_molecular_Ls = 12.1 * d_cm**3 / foreline_length_m * np.sqrt(293.0/temperature_K) * np.sqrt(28.97/gas_molecular_mass)
            else:
                C_molecular_Ls = np.inf

            # Effective speed
            if not np.isinf(C_molecular_Ls) and not np.isnan(S_required_Ls) and S_required_Ls > 0:
                S_effective_Ls = (S_required_Ls * C_molecular_Ls) / (S_required_Ls + C_molecular_Ls)
            else:
                S_effective_Ls = S_required_Ls

            # Pump-down time
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

            # Pump type suggestions
            pump_type_suggestion = []
            if desired_pressure_mbar >= 100:
                pump_type_suggestion.append('Rotary Lobe / Dry Screw (vacuum blower)')
            if 1 <= desired_pressure_mbar < 100:
                pump_type_suggestion.append('Rotary Vane or Dry Scroll (roughing)')
            if desired_pressure_mbar < 1e-3:
                pump_type_suggestion.append('Turbomolecular pump')
            elif desired_pressure_mbar < 1e-1:
                pump_type_suggestion.append('Roots booster + backing')

            # Display results
            st.success("✅ Vacuum Analysis Complete")
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Performance", "🔧 Optimization"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Required Speed", f"{S_required_Ls:.1f} L/s")
                    st.metric("Effective Speed", f"{S_effective_Ls:.1f} L/s")
                with col2:
                    st.metric("Conductance", f"{C_molecular_Ls:.1f} L/s")
                    st.metric("Time Constant", f"{tau_sec:.1f} s")
                with col3:
                    st.metric("Pumpdown Time", f"{t_to_target_sec/60:.1f} min")
                    st.metric("Target Pressure", f"{desired_pressure_mbar:.2e} mbar")

                st.markdown("---")
                st.subheader("Recommended Pump Types")
                for pump_type in pump_type_suggestion:
                    st.write(f"✓ {pump_type}")

            with tab2:
                # Pumpdown curve
                time_points = np.linspace(0, min(t_to_target_sec * 1.5, 3600), 100)
                pressure_curve = p0_Pa * np.exp(-time_points / tau_sec) if not np.isinf(tau_sec) else np.ones_like(time_points) * p0_Pa
                fig_vac, ax_vac = plt.subplots(figsize=(10, 6))
                ax_vac.semilogy(time_points/60, pressure_curve/100, 'b-', linewidth=2)
                ax_vac.axhline(desired_pressure_mbar, color='r', linestyle='--', label='Target Pressure')
                ax_vac.set_xlabel('Time (minutes)', fontweight='bold')
                ax_vac.set_ylabel('Pressure (mbar)', fontweight='bold')
                ax_vac.set_title('Pumpdown Curve', fontweight='bold')
                ax_vac.grid(True, which='both', alpha=0.3)
                ax_vac.legend()
                st.pyplot(fig_vac)
                plt.close()

            with tab3:
                st.subheader("System Optimization")
                col_vac1, col_vac2 = st.columns(2)
                with col_vac1:
                    st.markdown("#### 📏 Diameter Optimization")
                    # Diameter optimization
                    diameters = np.linspace(10, 100, 50)
                    conductances = 12.1 * (diameters/10)**3 / foreline_length_m
                    # Calculate effective speeds
                    S_effs = (S_required_Ls * conductances) / (S_required_Ls + conductances)

                    fig_opt, ax_opt = plt.subplots(figsize=(8, 5))
                    ax_opt.plot(diameters, conductances, 'g-', linewidth=2, label='Conductance')
                    ax_opt.plot(diameters, S_effs, 'b--', linewidth=2, label='Effective Speed')
                    ax_opt.axvline(foreline_id_mm, color='r', linestyle='--', label=f'Current: {foreline_id_mm}mm')
                    ax_opt.set_xlabel('Foreline Diameter (mm)', fontweight='bold')
                    ax_opt.set_ylabel('Speed (L/s)', fontweight='bold')
                    ax_opt.set_title('Conductance & Effective Speed vs Diameter', fontweight='bold')
                    ax_opt.grid(True, alpha=0.3)
                    ax_opt.legend()
                    st.pyplot(fig_opt)
                    plt.close()

                with col_vac2:
                    st.markdown("#### ⚡ Speed vs Pressure")
                    # Pumping speed vs pressure (typical turbopump curve)
                    pressures = np.logspace(-4, 2, 50)  # mbar
                    # Simplified curve: constant speed at low P, drops at high P
                    speeds = np.where(pressures < 0.1, 
                                     S_required_Ls, 
                                     S_required_Ls * (0.1/pressures)**0.3)
                    fig_sp, ax_sp = plt.subplots(figsize=(8, 5))
                    ax_sp.semilogx(pressures, speeds, 'purple', linewidth=2)
                    ax_sp.axvline(desired_pressure_mbar, color='r', linestyle='--', label='Target')
                    ax_sp.set_xlabel('Pressure (mbar)', fontweight='bold')
                    ax_sp.set_ylabel('Pumping Speed (L/s)', fontweight='bold')
                    ax_sp.set_title('Estimated Speed vs Pressure', fontweight='bold')
                    ax_sp.grid(True, alpha=0.3, which='both')
                    ax_sp.legend()
                    st.pyplot(fig_sp)
                    plt.close()

                # Gas load analysis
                st.markdown("---")
                st.markdown("#### 💨 Gas Load Analysis")
                # Different gas load scenarios
                scenarios = {
                    'Current': leak_rate_mbarL_s,
                    'Low leak': leak_rate_mbarL_s * 0.5,
                    'High leak': leak_rate_mbarL_s * 2.0,
                    'With purge': leak_rate_mbarL_s * 1.5,
                }
                scenario_results = []
                for name, load in scenarios.items():
                    S_req_scenario = (load * 0.1) / P_target_Pa * 1000 if P_target_Pa > 0 else np.inf
                    if not np.isinf(C_molecular_Ls) and S_req_scenario > 0:
                        S_eff_scenario = (S_req_scenario * C_molecular_Ls) / (S_req_scenario + C_molecular_Ls)
                    else:
                        S_eff_scenario = S_req_scenario
                    scenario_results.append({
                        'Scenario': name,
                        'Gas Load (mbar·L/s)': load,
                        'Required Speed (L/s)': S_req_scenario if not np.isinf(S_req_scenario) else 999999,
                        'Effective Speed (L/s)': S_eff_scenario if not np.isinf(S_eff_scenario) else 999999
                    })

                df_scenarios = pd.DataFrame(scenario_results)
                st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

                # Vacuum system recommendations
                st.markdown("---")
                st.markdown("#### 🎯 System Recommendations")
                rec_col1, rec_col2 = st.columns(2)
                with rec_col1:
                    st.write("**Pump Selection Guidance:**")
                    if desired_pressure_mbar < 1e-5:
                        st.write("- Turbo + Ion/Cryo pump combination")
                        st.write("- Bake-out capability required")
                        st.write("- Clean/dry gases only")
                    elif desired_pressure_mbar < 1e-3:
                        st.write("- Turbomolecular pump")
                        st.write("- Dry backing pump (scroll/screw)")
                        st.write("- Consider nitrogen purge")
                    elif desired_pressure_mbar < 1:
                        st.write("- Roots + rotary vane")
                        st.write("- Or large dry pump")
                    else:
                        st.write("- Rotary vane or dry scroll")
                        st.write("- Single stage adequate")

                with rec_col2:
                    st.write("**Process Considerations:**")
                    if process_type == 'Coating/Deposition':
                        st.write("- Clean pump (mag-lev turbo)")
                        st.write("- Particle trap recommended")
                        st.write("- Consider load-lock")
                    elif process_type == 'Freeze Drying':
                        st.write("- Condensable vapor capacity")
                        st.write("- Cold trap essential")
                        st.write("- Large throughput needed")
                    elif process_type == 'Semiconductor':
                        st.write("- Ultra-clean vacuum")
                        st.write("- Corrosive gas handling")
                        st.write("- In-line scrubbers")
                    else:
                        st.write("- General purpose pump suitable")
                        st.write("- Standard maintenance intervals")

                # Ultimate pressure achievable
                st.markdown("---")
                col_ult1, col_ult2 = st.columns(2)
                with col_ult1:
                    st.info("**Ultimate Pressure Estimate**")
                    # Base ultimate pressure for pump type
                    if desired_pressure_mbar < 1e-3:
                        base_ultimate = 1e-8  # Turbo
                    elif desired_pressure_mbar < 1:
                        base_ultimate = 1e-4  # Roots
                    else:
                        base_ultimate = 1e-2  # Rotary vane
                    # Degrade based on gas load
                    effective_ultimate = base_ultimate + (leak_rate_mbarL_s * 0.01)
                    st.write(f"Pump ultimate: ~{base_ultimate:.2e} mbar")
                    st.write(f"With gas load: ~{effective_ultimate:.2e} mbar")
                    if effective_ultimate > desired_pressure_mbar:
                        st.error("⚠️ May not reach target pressure with this gas load!")
                    else:
                        st.success("✅ Target pressure achievable")
                with col_ult2:
                    st.info("**Crossover Pressure**")
                    if desired_pressure_mbar < 1e-1:
                        crossover = 1e-2  # Turbo/Roots crossover
                        st.write(f"Switch pump at: ~{crossover:.2e} mbar")
                        st.write("Rough → High vacuum")
                        if crossover > desired_pressure_mbar * 100:
                            st.write("✅ Adequate margin")
                        else:
                            st.warning("⚡ Tight transition")


        except Exception as e:
            st.error(f"❌ Error: {e}")

# ------------------ System Comparison Page ------------------
elif page == "Pump System Comparison":
    st.header("⚖️ Pump System Comparison Tool")
    st.write("Compare multiple pump configurations side-by-side")
    st.info("Enter parameters for up to 3 different pump systems to compare")
    num_systems = st.slider("Number of systems to compare", 2, 3, 2)
    comparison_data = []
    cols = st.columns(num_systems)
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"System {i+1}")
            with st.form(f"system_{i}"):
                name = st.text_input("System Name", value=f"System {i+1}")
                flow = st.number_input("Flow (m³/h)", value=100.0*(i+1))
                head = st.number_input("Head (m)", value=20.0+i*5)
                efficiency = st.number_input("Efficiency (%)", value=70.0+i*3)
                power = st.number_input("Power (kW)", value=10.0+i*2)
                cost = st.number_input("Capital Cost ($)", value=5000.0*(i+1))
                submitted = st.form_submit_button("Add")
                if submitted:
                    comparison_data.append({
                        'Name': name,
                        'Flow (m³/h)': flow,
                        'Head (m)': head,
                        'Efficiency (%)': efficiency,
                        'Power (kW)': power,
                        'Capital Cost ($)': cost
                    })

    if len(comparison_data) >= 2:
        df_comp = pd.DataFrame(comparison_data)
        st.markdown("---")
        st.subheader("Comparison Results")
        st.dataframe(df_comp, use_container_width=True)

        # Comparison charts
        fig_comp, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Efficiency comparison
        axes[0,0].bar(df_comp['Name'], df_comp['Efficiency (%)'], color=['blue', 'green', 'orange'][:len(df_comp)])
        axes[0,0].set_ylabel('Efficiency (%)')
        axes[0,0].set_title('Efficiency Comparison')
        axes[0,0].grid(True, alpha=0.3)

        # Power comparison
        axes[0,1].bar(df_comp['Name'], df_comp['Power (kW)'], color=['red', 'purple', 'brown'][:len(df_comp)])
        axes[0,1].set_ylabel('Power (kW)')
        axes[0,1].set_title('Power Consumption')
        axes[0,1].grid(True, alpha=0.3)

        # Cost comparison
        axes[1,0].bar(df_comp['Name'], df_comp['Capital Cost ($)'], color=['cyan', 'magenta', 'yellow'][:len(df_comp)])
        axes[1,0].set_ylabel('Cost ($)')
        axes[1,0].set_title('Capital Cost')
        axes[1,0].grid(True, alpha=0.3)

        # Flow vs Head scatter
        axes[1,1].scatter(df_comp['Flow (m³/h)'], df_comp['Head (m)'], s=200, c=['blue', 'green', 'orange'][:len(df_comp)])
        for idx, row in df_comp.iterrows():
            axes[1,1].annotate(row['Name'], (row['Flow (m³/h)'], row['Head (m)']), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1,1].set_xlabel('Flow (m³/h)')
        axes[1,1].set_ylabel('Head (m)')
        axes[1,1].set_title('Operating Points')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()

# ------------------ Life Cycle Cost Analysis Page ------------------
elif page == "Life Cycle Cost Analysis":
    st.header("💰 Detailed Life Cycle Cost Analysis")
    with st.form("lcc_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Capital Costs")
            pump_cost = st.number_input("Pump cost ($)", value=5000.0)
            motor_cost = st.number_input("Motor cost ($)", value=2000.0)
            installation_cost = st.number_input("Installation ($)", value=1500.0)
            st.subheader("Operating Parameters")
            operating_hours_year = st.number_input("Hours/year", value=8000.0)
            electricity_rate = st.number_input("Electricity ($/kWh)", value=0.12)
            power_kW = st.number_input("Power consumption (kW)", value=15.0)
        with col2:
            st.subheader("Maintenance Costs")
            annual_maintenance = st.number_input("Annual maintenance ($)", value=500.0)
            major_overhaul_cost = st.number_input("Major overhaul ($)", value=3000.0)
            overhaul_interval_years = st.number_input("Overhaul interval (years)", value=5.0)
            st.subheader("Analysis Period")
            analysis_years = st.number_input("Analysis period (years)", value=15.0, min_value=1.0, max_value=30.0)
            discount_rate = st.number_input("Discount rate (%)", value=5.0) / 100
        calculate_lcc = st.form_submit_button("Calculate Life Cycle Cost", type="primary")

    if calculate_lcc:
        # Calculate costs over time
        years = np.arange(1, int(analysis_years) + 1)
        # Initial costs
        initial_cost = pump_cost + motor_cost + installation_cost
        # Annual energy cost
        annual_energy_cost = power_kW * operating_hours_year * electricity_rate
        # Build cost arrays
        capital_costs = np.zeros(len(years))
        capital_costs[0] = initial_cost
        energy_costs = np.ones(len(years)) * annual_energy_cost
        maintenance_costs = np.ones(len(years)) * annual_maintenance
        # Add overhaul costs
        overhaul_costs = np.zeros(len(years))
        for year in years:
            if year % overhaul_interval_years == 0:
                overhaul_costs[year-1] = major_overhaul_cost
        # Total annual costs
        total_annual_costs = capital_costs + energy_costs + maintenance_costs + overhaul_costs
        # Apply discount rate
        discount_factors = 1 / (1 + discount_rate) ** years
        npv_costs = total_annual_costs * discount_factors
        cumulative_npv = np.cumsum(npv_costs)

        # Display results
        st.success(f"✅ Total Life Cycle Cost (NPV): ${cumulative_npv[-1]:,.2f}")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Initial Investment", f"${initial_cost:,.2f}")
            st.metric("Total Energy Cost", f"${np.sum(energy_costs * discount_factors):,.2f}")
        with col_res2:
            st.metric("Total Maintenance", f"${np.sum(maintenance_costs * discount_factors):,.2f}")
            st.metric("Total Overhauls", f"${np.sum(overhaul_costs * discount_factors):,.2f}")
        with col_res3:
            energy_pct = np.sum(energy_costs * discount_factors) / cumulative_npv[-1] * 100
            st.metric("Energy % of LCC", f"{energy_pct:.1f}%")
            st.metric("Avg Annual Cost", f"${cumulative_npv[-1]/analysis_years:,.2f}")

        # Visualization
        fig_lcc, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Cumulative NPV
        ax1.plot(years, cumulative_npv/1000, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Years', fontweight='bold')
        ax1.set_ylabel('Cumulative NPV ($1000s)', fontweight='bold')
        ax1.set_title('Life Cycle Cost Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Cost breakdown
        labels = ['Energy', 'Maintenance', 'Overhauls', 'Initial']
        sizes = [
            np.sum(energy_costs * discount_factors),
            np.sum(maintenance_costs * discount_factors),
            np.sum(overhaul_costs * discount_factors),
            initial_cost * discount_factors[0]
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cost Distribution', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig_lcc)
        plt.close()

# Footer
st.markdown("---")
st.caption("⚠️ Engineering estimates only. Validate with vendor data before procurement.")

# --- Updated Sidebar with corrected markdown ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Enhanced Features")
    st.markdown("**Rotating Pumps:**")
    st.markdown("✓ Multi-pump configurations (parallel/series)")
    st.markdown("✓ Affinity laws speed analysis")
    st.markdown("✓ NPSH margin visualization")
    st.markdown("✓ Pipe sizing optimization")
    st.markdown("✓ Vibration & pulsation risk")
    st.markdown("✓ Component life prediction")
    st.markdown("✓ Failure mode analysis")
    st.markdown("✓ Seal life estimation")
    st.markdown("✓ Wear rate calculation")
    st.markdown("---") # Added newline
    st.markdown("**Vacuum Systems:**")
    st.markdown("✓ Gas load scenarios")
    st.markdown("✓ Conductance optimization")
    st.markdown("✓ Speed vs pressure curves")
    st.markdown("✓ Ultimate pressure analysis")
    st.markdown("✓ Process-specific recommendations")
    st.markdown("✓ Crossover pressure guidance")
    st.markdown("---") # Added newline
    st.markdown("**Analysis Tools:**")
    st.markdown("✓ Energy cost projections (10-year)")
    st.markdown("✓ Efficiency upgrade ROI")
    st.markdown("✓ System comparison (up to 3)")
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[Pump Selection Guide](#)")
    st.markdown("[Affinity Laws](#)")
    st.markdown("[NPSH Requirements](#)")
```
