import streamlit as st
import pandas as pd
import numpy as np
import math 
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

st.set_page_config(page_title="Advanced Pump & Vacuum Sizing Tool", layout="wide")
st.title("🔧 Advanced Pump & Vacuum Sizing Tool")

# --- Session State Initialization ---
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'mode' not in st.session_state:
    st.session_state.mode = "Pressure Pump"

# --- Mode Selection ---
mode = st.radio("Select Mode", ["Pressure Pump", "Vacuum Pump"], index=0)
st.session_state.mode = mode

# === VACUUM PUMP MODE ===
if mode == "Vacuum Pump":
    st.header("🧯 Vacuum Pump Sizing Calculator")

    # Sidebar inputs with enhanced parameters
    st.sidebar.header("Vacuum Pump Inputs")
    volume = st.sidebar.number_input("Evacuated Volume (m³)", value=2.0, min_value=0.01)
    initial_pressure = st.sidebar.number_input("Initial Pressure (kPa)", value=101.3, min_value=0.1)
    target_pressure = st.sidebar.number_input("Target Pressure (kPa)", value=10.0, min_value=0.1)

    # Gas Type Selection
    gas_type = st.sidebar.selectbox(
        "Gas Type",
        ["Air", "Water Vapor", "Nitrogen", "Oxygen", "Hydrogen", "CO2", "Steam"]
    )

    # Multi-Stage Option
    multi_stage = st.sidebar.checkbox("Use Multi-Stage System")
    if multi_stage:
        stage_1_target = st.sidebar.number_input("Stage 1 Target Pressure (kPa)", 
                                               value=50.0, 
                                               min_value=target_pressure + 1.0, 
                                               max_value=initial_pressure - 1.0)
        pump_1_type = st.sidebar.selectbox("Roughing Pump Type", 
                                         ["Rotary Vane", "Liquid Ring", "Claw"],
                                         key="pump1")
        pump_2_type = st.sidebar.selectbox("High-Vacuum Pump Type",
                                         ["Scroll", "Roots (Booster)", "Diffusion"],
                                         key="pump2")
    else:
        pump_type_vac = st.sidebar.selectbox(
            "Vacuum Pump Type",
            ["Rotary Vane", "Liquid Ring", "Scroll", "Claw", "Roots (Booster)"]
        )

    # Additional Parameters
    safety_factor_vac = st.sidebar.number_input("Safety Factor on Pumping Speed", 
                                              value=1.5, min_value=1.0, max_value=3.0)
    pipe_length_vac = st.sidebar.number_input("Suction Line Length (m)", value=5.0)
    pipe_diameter_vac = st.sidebar.number_input("Suction Line Diameter (m)", 
                                              value=0.05, min_value=0.01)

    # Condensable Vapor Handling
    has_condensable = st.sidebar.checkbox("Contains Condensable Vapor (e.g., Steam)")
    if has_condensable:
        vapor_fraction = st.sidebar.slider("Vapor Fraction in Gas Mix (%)", 
                                         0.0, 50.0, 10.0) / 100
        condensation_factor = 0.7
    else:
        vapor_fraction = 0
        condensation_factor = 1.0

    leakage_rate = st.sidebar.number_input("Estimated Leakage Rate (% vol/hour)", 
                                         value=2.0, min_value=0.0) / 100

    # Validation
    if target_pressure >= initial_pressure:
        st.error("Target pressure must be lower than initial pressure.")
        st.stop()

    # Gas correction factors
    gas_correction = {
        "Air": 1.0, "Nitrogen": 1.05, "Oxygen": 0.95,
        "Hydrogen": 2.8, "CO2": 0.7, "Water Vapor": 1.0, "Steam": 0.6
    }
    correction_factor = gas_correction[gas_type]

    # Enhanced calculations
    required_speed_nominal = (volume / 1) * np.log(initial_pressure / target_pressure)
    corrected_speed_for_gas = required_speed_nominal / correction_factor
    corrected_speed_with_vapor = (corrected_speed_for_gas / condensation_factor 
                                 if has_condensable else corrected_speed_for_gas)
    required_speed_with_sf = corrected_speed_with_vapor * safety_factor_vac

    # Leakage compensation
    leakage_volume_per_hour = volume * leakage_rate
    final_required_speed = required_speed_with_sf + leakage_volume_per_hour

    # Standard pump sizes
    standard_speeds = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
    selected_pump_speed = next((s for s in standard_speeds if s >= final_required_speed), 1000)

    # Performance simulation function
    def get_pumping_speed_efficiency(pressure, pump_type):
        """Get pumping speed efficiency based on pressure and pump type"""
        p_ratio = pressure / 101.3
        if pump_type in ["Rotary Vane", "Liquid Ring", "Claw"]:
            return max(0.1, 1.0 - 0.3 * p_ratio)
        elif pump_type == "Scroll":
            return max(0.2, 1.0 - 0.9 * p_ratio**0.5)
        elif pump_type in ["Roots (Booster)", "Diffusion"]:
            return 0.1 if p_ratio > 0.3 else 1.0
        return 1.0

    # Evacuation Time Calculation
    times = []
    pressures_sim = []
    time_elapsed = 0
    dt = 0.1  # time step in hours
    p = initial_pressure

    while p > target_pressure * 1.01 and time_elapsed < 10:  # Safety limit of 10 hours
        # Get current pump type for simulation
        current_pump_type = pump_type_vac if not multi_stage else (
            pump_1_type if p > stage_1_target else pump_2_type
        )
        
        # Effective pumping speed
        efficiency = get_pumping_speed_efficiency(p, current_pump_type)
        S = selected_pump_speed * efficiency
        
        # Calculate volume removed
        dV = S * dt
        
        # Update pressure using exponential decay
        if p > 0.1:  # Avoid division issues
            p = p * np.exp(-dV / volume)
        else:
            break
            
        time_elapsed += dt
        times.append(time_elapsed * 60)  # Convert to minutes
        pressures_sim.append(p)

    evacuation_time_at_target = times[-1] if times else 0

    # Calculate average velocity in suction line
    area = np.pi * (pipe_diameter_vac ** 2) / 4
    avg_pressure = (initial_pressure + target_pressure) / 2
    gas_flow_rate_scm = selected_pump_speed * (avg_pressure / 101.3)  # std m³/h
    velocity_vac = (gas_flow_rate_scm / 3600) / area  # m/s

    # Results Display
    st.subheader("📊 Vacuum Pump Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Required Pumping Speed", f"{required_speed_nominal:.1f} m³/h")
    col2.metric("Selected Pump Speed", f"{selected_pump_speed} m³/h")
    col3.metric("Estimated Evacuation Time", f"{evacuation_time_at_target:.1f} min")

    col1.metric("Gas Correction Factor", f"{correction_factor:.2f}")
    col2.metric("Leakage Compensation", f"{leakage_volume_per_hour:.2f} m³/h")
    col3.metric("Line Velocity (avg)", f"{velocity_vac:.2f} m/s")

    # Multi-stage display
    if multi_stage:
        st.info(f"Multi-Stage System: {pump_1_type} (roughing) + {pump_2_type} (high-vacuum)")
        st.write(f"Stage 1 target: {stage_1_target} kPa")
    else:
        st.info(f"Single Stage: {pump_type_vac} Vacuum Pump")

    # Plot: Pressure vs Time
    st.subheader("📈 Evacuation Profile")
    if times and pressures_sim:
        fig_vac = go.Figure()
        fig_vac.add_trace(go.Scatter(
            x=times,
            y=pressures_sim,
            mode='lines',
            name='Pressure Decay',
            line=dict(color='blue')
        ))
        
        # Add target pressure line
        fig_vac.add_hline(y=target_pressure, line_dash="dash", 
                         line_color="red", annotation_text="Target Pressure")
        
        if multi_stage:
            fig_vac.add_hline(y=stage_1_target, line_dash="dot", 
                             line_color="orange", annotation_text="Stage 1 Target")
        
        fig_vac.update_layout(
            title="Vacuum Build-Up Over Time",
            xaxis_title="Time (minutes)",
            yaxis_title="Absolute Pressure (kPa)",
            yaxis_type="log",
            hovermode="x unified"
        )
        st.plotly_chart(fig_vac, use_container_width=True)

    # Warnings and Recommendations
    if velocity_vac > 20:
        st.warning("⚠️ High suction line velocity. Consider larger diameter to reduce pressure drop.")
    if pipe_length_vac > 10:
        st.warning("⚠️ Long suction line may increase evacuation time. Minimize length.")
    if has_condensable and not multi_stage:
        st.warning("⚠️ Condensable vapors detected. Consider cold trap or condenser.")

    # Vendor Notes
    st.subheader("📋 Vacuum Pump Vendor Notes")
    pump_config = f"{pump_1_type} + {pump_2_type}" if multi_stage else pump_type_vac
    vac_notes = f"""
**VACUUM PUMP SIZING CALCULATION**
- Application: Vacuum System Evacuation
- Volume: {volume} m³
- Pressure Range: {initial_pressure} kPa → {target_pressure} kPa
- Gas Type: {gas_type} (Correction Factor: {correction_factor:.2f})
- Required Pumping Speed: {required_speed_nominal:.1f} m³/h
- Corrected Speed (gas + vapor + SF): {final_required_speed:.1f} m³/h
- Selected Pump Configuration: {pump_config}
- Selected Pump Speed: {selected_pump_speed} m³/h
- Estimated Evacuation Time: {evacuation_time_at_target:.1f} minutes
- Suction Line: {pipe_length_vac} m length, {pipe_diameter_vac*100:.1f} cm diameter
- Average Gas Velocity: {velocity_vac:.2f} m/s
- Leakage Rate: {leakage_rate*100:.1f}% of volume per hour
- Condensable Vapor: {'Yes' if has_condensable else 'No'}
- Multi-Stage System: {'Yes' if multi_stage else 'No'}
- Safety Factor: {safety_factor_vac:.1f}
- Design Date: {datetime.now().strftime('%Y-%m-%d')}

**NOTES:**
- Performance varies with temperature, gas type, and pump wear
- Confirm pump curves with vendor for actual gas composition
- Consider inlet filtration for dirty applications
- Verify electrical requirements and utilities
    """
    st.text_area("Copy for RFQ:", vac_notes, height=300)

    # Export Vacuum Data
    st.subheader("📤 Export Vacuum Data")
    if times and pressures_sim:
        df_vac = pd.DataFrame({
            "Time (min)": times,
            "Pressure (kPa)": pressures_sim
        })
        csv_vac = df_vac.to_csv(index=False)
        b64 = base64.b64encode(csv_vac.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="vacuum_evacuation_curve.csv">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# === PRESSURE PUMP MODE ===
else:
    st.header("💧 Pressure Pump Sizing Calculator")

    # --- Unit Selection ---
    unit_system = st.radio("Unit System", ["Metric (SI)", "Imperial (US)"], index=0)

    # Helper conversion functions
    def to_imperial_si(val, conv_type):
        factors = {
            'flow': 15.8503,      # m3/hr to GPM
            'head': 3.28084,      # m to ft
            'diameter': 39.3701,  # m to in
            'density': 0.062428,  # kg/m3 to lb/ft3
            'viscosity': 1,       # cP (same)
            'power': 1.34102,     # kW to HP
            'length': 3.28084,    # m to ft
            'pressure': 0.145038, # kPa to psi
        }
        return val * factors[conv_type]

    def to_si_imperial(val, conv_type):
        factors = {
            'flow': 15.8503,
            'head': 3.28084,
            'diameter': 39.3701,
            'density': 0.062428,
            'viscosity': 1,
            'power': 1.34102,
            'length': 3.28084,
            'pressure': 0.145038,
        }
        return val / factors[conv_type]

    # --- User Inputs ---
    st.sidebar.header("💧 Input Parameters")

    if unit_system == "Metric (SI)":
        flow_rate = st.sidebar.number_input("Flow Rate (m³/hr)", value=50.0, min_value=0.1)
        head = st.sidebar.number_input("Head (m)", value=30.0, min_value=0.1)
        pipe_diameter = st.sidebar.number_input("Pipe Diameter (m)", value=0.1, min_value=0.01)
        pipe_length = st.sidebar.number_input("Pipe Length (m)", value=10.0)
        vapor_pressure = st.sidebar.number_input("Vapor Pressure (kPa)", value=2.3)
        suction_head = st.sidebar.number_input("Suction Head Available (m)", value=5.0)
    else:
        flow_rate_imp = st.sidebar.number_input("Flow Rate (GPM)", value=211.3, min_value=0.1)
        head_imp = st.sidebar.number_input("Head (ft)", value=98.4, min_value=0.1)
        pipe_diameter_imp = st.sidebar.number_input("Pipe Diameter (in)", value=3.94, min_value=0.1)
        pipe_length_imp = st.sidebar.number_input("Pipe Length (ft)", value=32.8)
        vapor_pressure_imp = st.sidebar.number_input("Vapor Pressure (psi)", value=0.33)
        suction_head_imp = st.sidebar.number_input("Suction Head Available (ft)", value=16.4)
        
        # Convert to SI
        flow_rate = to_si_imperial(flow_rate_imp, 'flow')
        head = to_si_imperial(head_imp, 'head')
        pipe_diameter = to_si_imperial(pipe_diameter_imp, 'diameter')
        pipe_length = to_si_imperial(pipe_length_imp, 'length')
        vapor_pressure = to_si_imperial(vapor_pressure_imp, 'pressure')
        suction_head = to_si_imperial(suction_head_imp, 'head')

    density = st.sidebar.number_input("Fluid Density (kg/m³)", value=1000.0, min_value=100.0)
    efficiency = st.sidebar.number_input("Pump Efficiency (%)", value=75.0, min_value=1.0, max_value=100.0) / 100
    material = st.sidebar.selectbox("Fluid Type", ["Water", "Acid", "Slurry", "Oil", "Viscous Fluid", "Chemical Mixture"])
    fluid_viscosity = st.sidebar.number_input("Fluid Viscosity (cP)", value=1.0, min_value=0.1)
    application = st.sidebar.selectbox("Application Type", ["General Transfer", "Chemical Handling", "Slurry Transport", "Oil Transfer", "High Pressure", "Metering"])

    with st.sidebar.expander("Advanced Parameters"):
        safety_factor = st.number_input("Safety Factor on Power", value=1.15, min_value=1.0, max_value=2.0)
        service_factor = st.number_input("Motor Service Factor", value=1.15, min_value=1.0)
        configuration = st.selectbox("Pump Configuration", ["Single", "Parallel (2x)", "Series (2x)"])

    # --- Calculations ---
    flow_m3s = flow_rate / 3600
    velocity = (4 * flow_m3s) / (np.pi * pipe_diameter ** 2)
    reynolds_number = (density * velocity * pipe_diameter) / (fluid_viscosity / 1000)
    
    # Friction factor calculation
    if reynolds_number > 4000:
        friction_factor = 0.02  # Simplified for turbulent flow
    else:
        friction_factor = 64 / reynolds_number if reynolds_number > 0 else 0.02
    
    friction_loss = friction_factor * (pipe_length / pipe_diameter) * (velocity ** 2 / (2 * 9.81))
    TDH = head + friction_loss

    # Power calculation
    power_kw = (flow_rate * TDH * density * 9.81) / (3.6e6 * efficiency)
    power_kw_with_sf = power_kw * safety_factor

    # Motor selection
    standard_motors = [0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110]
    motor_size = next((m for m in standard_motors if m >= power_kw_with_sf), 110.0)

    # NPSH calculations
    NPSHa = suction_head - (vapor_pressure * 1000 / (density * 9.81))
    NPSHr_estimated = 0.3 * (TDH / 10) + 1
    NPSH_margin = NPSHa - NPSHr_estimated
    npsh_ok = "✅ OK" if NPSH_margin > 0.5 else "⚠️ Risk of Cavitation"

    # Equipment selection logic
    impeller_map = {
        "Water": "Closed Impeller",
        "Acid": "Non-Metallic / Stainless Steel Impeller",
        "Slurry": "Semi-Open or Open Impeller",
        "Oil": "Closed Impeller with special seals",
        "Viscous Fluid": "Screw or Helical Impeller",
        "Chemical Mixture": "Non-Metallic Lined Impeller"
    }
    impeller_type = impeller_map.get(material, "Consult vendor")

    pump_suggestions = {
        ("Water", "General Transfer"): "Centrifugal Pump",
        ("Acid", "Chemical Handling"): "Magnetic Drive Pump",
        ("Slurry", "Slurry Transport"): "Slurry Pump",
        ("Oil", "Oil Transfer"): "Gear Pump",
        ("Viscous Fluid", "High Pressure"): "Screw Pump",
        ("Chemical Mixture", "Metering"): "Diaphragm Metering Pump"
    }
    pump_type = pump_suggestions.get((material, application), "Consult vendor for best selection")

    material_compatibility = {
        "Water": "Stainless Steel, Bronze",
        "Acid": "Hastelloy, PVDF, PTFE",
        "Slurry": "Hardened Steel, Ceramic Linings",
        "Oil": "Carbon Steel, Buna-N seals",
        "Viscous Fluid": "Stainless Steel with heat tracing",
        "Chemical Mixture": "Alloy 20, Teflon Lined"
    }
    wetted_materials = material_compatibility.get(material, "Consult vendor")

    # Configuration adjustments
    if configuration == "Parallel (2x)":
        flow_rate_single = flow_rate / 2
        TDH_single = TDH
    elif configuration == "Series (2x)":
        flow_rate_single = flow_rate
        TDH_single = TDH / 2
    else:
        flow_rate_single = flow_rate
        TDH_single = TDH

    # Store results in session state
    results = {
        "Flow Rate (m³/hr)": flow_rate,
        "Head (m)": head,
        "TDH (m)": TDH,
        "Friction Loss (m)": friction_loss,
        "Required Power (kW)": power_kw,
        "Power with SF (kW)": power_kw_with_sf,
        "Selected Motor (kW)": motor_size,
        "Motor HP": motor_size * 1.341,
        "Efficiency (%)": efficiency * 100,
        "Velocity (m/s)": velocity,
        "Reynolds Number": reynolds_number,
        "NPSHa (m)": NPSHa,
        "NPSHr (est. m)": NPSHr_estimated,
        "NPSH Margin": NPSH_margin,
        "Cavitation Risk": npsh_ok,
        "Impeller Type": impeller_type,
        "Pump Type": pump_type,
        "Wetted Materials": wetted_materials,
        "Configuration": configuration
    }
    st.session_state.results = results

    # --- Display Results ---
    st.subheader("📊 Results Summary")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Required Power", f"{power_kw:.2f} kW")
    col1.metric("Selected Motor", f"{motor_size:.1f} kW ({motor_size * 1.341:.1f} HP)")
    col1.metric("TDH", f"{TDH:.2f} m")

    col2.metric("Flow Velocity", f"{velocity:.2f} m/s", 
               delta="Ideal: 1–3 m/s" if 1 <= velocity <= 3 else "High/Low")
    col2.metric("Reynolds Number", f"{reynolds_number:.0f}", 
               delta="Turbulent" if reynolds_number > 4000 else "Laminar")
    col2.metric("NPSHa", f"{NPSHa:.2f} m", delta=npsh_ok)

    col3.metric("Pump Type", pump_type)
    col3.metric("Impeller Type", impeller_type)
    col3.metric("Wetted Materials", wetted_materials)

    # Performance curves
    flow_range = np.linspace(0.5 * flow_rate, 1.5 * flow_rate, 50)
    head_curve = TDH * (flow_rate / np.clip(flow_range, 0.1, None)) ** 2
    power_curve = (flow_range * head_curve * density * 9.81) / (3.6e6 * efficiency)
    eff_curve = efficiency * (1 - ((flow_range - flow_rate) ** 2) / (flow_rate ** 2)) * 100
    eff_curve = np.clip(eff_curve, 0, 100)  # Limit efficiency to realistic range

    # Pump curve estimation
    pump_coeff = TDH_single / (flow_rate_single ** 2) if flow_rate_single > 0 else 0
    pump_head_curve = TDH_single * 1.2 - pump_coeff * (flow_range - flow_rate_single / 2) ** 2
    pump_head_curve = np.maximum(pump_head_curve, 0)

    # Plot: System vs Pump Curve
    st.subheader("📈 Performance Curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flow_range, y=head_curve, mode='lines', 
                            name='System Curve', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=flow_range, y=pump_head_curve, mode='lines', 
                            name='Pump Curve (Est.)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[flow_rate], y=[TDH], mode='markers', 
                            name='Operating Point', marker=dict(size=10, color='green')))
    fig.update_layout(title="System vs Pump Curve", 
                     xaxis_title="Flow Rate (m³/hr)", 
                     yaxis_title="Head (m)")
    st.plotly_chart(fig, use_container_width=True)

    # Power & Efficiency Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=flow_range, y=power_curve, name="Power (kW)", 
                             yaxis="y", line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=flow_range, y=eff_curve, name="Efficiency (%)", 
                             yaxis="y2", line=dict(color='green')))
    fig2.update_layout(
        title="Power and Efficiency vs Flow",
        xaxis_title="Flow Rate (m³/hr)",
        yaxis=dict(title="Power (kW)", side="left"),
        yaxis2=dict(title="Efficiency (%)", side="right", overlaying="y")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Vendor Notes
    st.subheader("📋 Pump Vendor Notes")
    vendor_notes = f"""
**PUMP SIZING CALCULATION**
- Application: {application} | Fluid: {material}
- Flow Rate: {flow_rate:.1f} m³/hr
- Total Dynamic Head (TDH): {TDH:.2f} m
- Static Head: {head:.1f} m
- Friction Loss: {friction_loss:.2f} m
- Pump Type Recommendation: {pump_type}
- Impeller Type: {impeller_type}
- Wetted Parts Material: {wetted_materials}
- Motor Size Required: {motor_size:.1f} kW ({motor_size * 1.341:.1f} HP)
- Motor Service Factor: {service_factor}
- Pump Efficiency: {efficiency*100:.1f}%
- NPSHa Available: {NPSHa:.2f} m
- NPSHr Estimated: {NPSHr_estimated:.2f} m
- NPSH Margin: {NPSH_margin:.2f} m → {npsh_ok}
- Flow Velocity: {velocity:.2f} m/s
- Reynolds Number: {reynolds_number:.0f}
- System Configuration: {configuration}
- Safety Factor Applied: {safety_factor:.2f}
- Design Date: {datetime.now().strftime('%Y-%m-%d')}

**TECHNICAL NOTES:**
- All calculations are preliminary estimates
- Confirm performance with vendor pump curves
- Verify material compatibility with actual fluid
- Consider temperature effects on viscosity and vapor pressure
- Installation requirements: proper suction piping, alignment, foundation
- Recommended spare parts: mechanical seals, impeller, bearings
    """
    st.text_area("Copy these notes for vendor RFQ:", vendor_notes, height=300)

    # Export functionality
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    # CSV export
    df_export = pd.DataFrame({
        "Flow Rate (m³/hr)": flow_range,
        "System Head (m)": head_curve,
        "Pump Head (m)": pump_head_curve,
        "Power (kW)": power_curve,
        "Efficiency (%)": eff_curve
    })
    csv = df_export.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="pump_curve_data.csv">📥 Download CSV</a>'
    col1.markdown(href, unsafe_allow_html=True)

    # Text report export
    b64_txt = base64.b64encode(vendor_notes.encode()).decode()
    href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="pump_sizing_report.txt">📄 Download Report</a>'
    col2.markdown(href_txt, unsafe_allow_html=True)

    # Imperial unit display
    if unit_system == "Imperial (US)":
        st.sidebar.subheader("Imperial Conversions")
        st.sidebar.write(f"Flow: {to_imperial_si(flow_rate, 'flow'):.1f} GPM")
        st.sidebar.write(f"Head: {to_imperial_si(TDH, 'head'):.1f} ft")
        st.sidebar.write(f"Pipe Dia: {to_imperial_si(pipe_diameter, 'diameter'):.2f} in")
        st.sidebar.write(f"Power: {to_imperial_si(power_kw, 'power'):.1f} HP")
        st.sidebar.write(f"Pressure: {to_imperial_si(vapor_pressure, 'pressure'):.2f} psi")
