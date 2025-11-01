import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import json
from datetime import datetime

# Set page config at the very top
st.set_page_config(page_title="Advanced Pump & Vacuum Sizing", layout="wide")
st.title("🔧 Advanced Pump & Vacuum Pump Sizing Sheet — Vendor Ready")

# Simple navigation between pages
page = st.sidebar.selectbox("Choose tool", [
    "Rotating Pumps (Centrifugal etc.)", 
    "Pump System Comparison",
    "Life Cycle Cost Analysis"
])

# ------------------ Helper functions ------------------
def colebrook_newton(Re, eps_abs, D, tol=1e-12, maxiter=50):
    """Solve Colebrook for friction factor f using Newton-Raphson."""
    if Re <= 0 or D <= 0:
        raise ValueError("Re and D must be positive")
    if Re < 2300:
        return 64.0 / Re

    A = eps_abs / (3.7 * D)
    B = 2.51 / Re
    try:
        f = 0.25 / (math.log10(A + 2.51 / Re) ** 2)
    except Exception:
        f = 0.02

    for i in range(maxiter):
        sqrtf = math.sqrt(f)
        arg = A + B / sqrtf
        if arg <= 0:
            arg = 1e-16
        F = 1.0 / sqrtf + 2.0 * math.log10(arg)

        ln10 = math.log(10.0)
        dF_df = -0.5 * f ** (-1.5) - (B / ln10) * f ** (-1.5) / arg

        if dF_df == 0:
            break

        delta = -F / dF_df
        f_new = f + delta

        if f_new <= 0 or not math.isfinite(f_new):
            f_new = max(1e-12, f * 0.5)

        if abs(f_new - f) < tol * max(1.0, f):
            return f_new

        f = f_new

    raise RuntimeError(f"Newton-Raphson did not converge after {maxiter} iterations; last f={f}")


def colebrook_brent_like(Re, eps_abs, D, tol=1e-12, maxiter=100):
    """Bracketed secant/bisection hybrid solver for Colebrook (robust)."""
    if Re <= 0 or D <= 0:
        raise ValueError("Re and D must be positive")
    if Re < 2300:
        return 64.0 / Re

    A = eps_abs / (3.7 * D)
    B = 2.51 / Re

    def F_of_f(f):
        sqrtf = math.sqrt(f)
        arg = A + B / sqrtf
        if arg <= 0:
            arg = 1e-16
        return 1.0 / sqrtf + 2.0 * math.log10(arg)

    lo = 1e-12
    hi = 1.0
    Flo = F_of_f(lo)
    Fhi = F_of_f(hi)

    attempts = 0
    while Flo * Fhi > 0 and attempts < 50:
        hi *= 2.0
        Fhi = F_of_f(hi)
        attempts += 1

    if Flo * Fhi > 0:
        try:
            f_guess = 0.25 / (math.log10(A + 2.51 / Re) ** 2)
            val = F_of_f(f_guess)
            if abs(val) < 1e-12:
                return f_guess
        except Exception:
            f_guess = 0.02
        lo = max(1e-12, f_guess * 0.1)
        hi = max(f_guess * 10.0, 1e-3)
        Flo = F_of_f(lo)
        Fhi = F_of_f(hi)

    candidate = 0.5 * (lo + hi)
    for i in range(maxiter):
        if (Fhi - Flo) != 0:
            f_secant = (lo * Fhi - hi * Flo) / (Fhi - Flo)
        else:
            f_secant = 0.5 * (lo + hi)

        if lo < f_secant < hi:
            candidate = f_secant
        else:
            candidate = 0.5 * (lo + hi)

        Fc = F_of_f(candidate)
        if abs(Fc) < 1e-14 or (hi - lo) < tol * max(1.0, candidate):
            return max(candidate, 1e-16)

        if Flo * Fc <= 0:
            hi = candidate
            Fhi = Fc
        else:
            lo = candidate
            Flo = Fc

    raise RuntimeError(f"Bracketed solver did not converge after {maxiter} iterations; last candidate={candidate}")


def colebrook_f(Re, D, eps_abs, method='brent', tol=1e-6, max_iter=100):
    """Unified Colebrook friction-factor solver."""
    if Re <= 0 or D <= 0:
        return np.nan
    if Re < 2300:
        return 64.0 / Re

    method = (method or 'brent').lower()
    if method.startswith('swamee'):
        try:
            return 0.25 / (math.log10(eps_abs/(3.7*D) + 5.74/(Re**0.9)) ** 2)
        except Exception:
            return 0.02
    elif method.startswith('newton'):
        return colebrook_newton(Re, eps_abs, D, tol=min(tol,1e-8), maxiter=max_iter)
    else:
        return colebrook_brent_like(Re, eps_abs, D, tol=min(tol,1e-8), maxiter=max_iter)

def darcy_head_loss(f, L, D, V, g=9.81):
    """Calculate head loss using Darcy-Weisbach equation"""
    if D <= 0:
        return 0
    return f * (L/D) * (V**2) / (2*g)

def reynolds(rho, V, D, mu):
    """Calculate Reynolds number"""
    if mu <= 0:
        return 0
    return rho * V * D / mu

def velocity_from_flow(Q, D):
    """Calculate velocity from flow rate and diameter"""
    if D <= 0:
        return 0
    A = math.pi * (D**2) / 4
    return Q / A if A > 0 else 0

def minor_loss_head(K, V, g=9.81):
    """Calculate minor losses"""
    return K * (V**2) / (2*g)

def suggest_density_range(material_type):
    """Suggest density range based on material type"""
    ranges = {
        'Water': (995, 1005),
        'Seawater': (1020, 1030),
        'Acids': (1100, 1850),
        'Alkaline': (1050, 1300),
        'Slurry': (1200, 2000),
        'Food-grade': (1000, 1200),
        'Oil': (850, 950)
    }
    return ranges.get(material_type, (None, None))

def pump_power_required(rho, g, Q, H, eta_pump, eta_motor):
    """Calculate pump power"""
    shaft_kW = (rho * g * Q * H) / (eta_pump * 1000)
    electrical_kW = shaft_kW / eta_motor
    return shaft_kW, electrical_kW

def calculate_cavitation_index(NPSHa, H):
    """Calculate cavitation index sigma"""
    if H <= 0:
        return np.nan
    return NPSHa / H

def calculate_suction_specific_speed(N, Q, NPSHr):
    """Calculate suction specific speed"""
    if NPSHr <= 0:
        return np.nan
    return N * math.sqrt(Q*3600) / (NPSHr**0.75)

def generate_pump_curves(Q_design, H_design, H_static):
    """Generate pump and system curves"""
    Q_points = np.linspace(0, Q_design * 1.5, 50)
    # System curve (parabolic)
    H_system = H_static + (H_design - H_static) * (Q_points / Q_design) ** 2
    # Pump curve (polynomial fit)
    H_pump = H_design * (1.2 - 0.4 * (Q_points / Q_design) ** 2)
    # Efficiency curve (parabolic with peak)
    eff_curve = 0.7 * (1 - 0.3 * ((Q_points / Q_design) - 1) ** 2)
    eff_curve = np.clip(eff_curve, 0.3, 0.85)
    # Power curve
    power_curve = (1000 * 9.81 * Q_points * H_pump) / (eff_curve * 1000)
    return Q_points, H_system, H_pump, eff_curve, power_curve

def compute_bep(Q_points, eff_curve):
    """Compute best efficiency point"""
    bep_idx = np.argmax(eff_curve)
    Q_bep = Q_points[bep_idx]
    eff_bep = eff_curve[bep_idx]
    return Q_bep, eff_bep, bep_idx

def calculate_parallel_pumps(n, Q_single, H_single):
    """Calculate parallel pump configuration"""
    return n * Q_single, H_single

def calculate_series_pumps(n, Q_single, H_single):
    """Calculate series pump configuration"""
    return Q_single, n * H_single

def calculate_affinity_laws(Q1, H1, N1, N2):
    """Apply affinity laws for speed change"""
    ratio = N2 / N1
    Q2 = Q1 * ratio
    H2 = H1 * (ratio ** 2)
    P_ratio = ratio ** 3
    return Q2, H2, P_ratio

def estimate_wear_rate(material_type, velocity, particle_size):
    """Estimate wear rate"""
    base_wear = 1.0
    if material_type == 'Slurry':
        base_wear = 5.0 + particle_size * 2.0
    base_wear *= (velocity / 2.0) ** 2
    return base_wear

def slurry_erosion_rate(conc, particle_size, velocity, hardness_ratio=1.0):
    """Estimate slurry erosion in mm/year"""
    return 0.001 * conc * particle_size * (velocity ** 2.5) * hardness_ratio

def cavitation_erosion_index(NPSHa, NPSHr, sigma, flow_dev_pct):
    """Calculate cavitation erosion risk"""
    if NPSHr <= 0:
        return 0.0, 'Low', 'No vendor NPSHr data'
    margin = NPSHa - NPSHr
    if margin < 0.5:
        score = 0.9
    elif margin < 1.0:
        score = 0.6
    else:
        score = 0.2
    score += flow_dev_pct / 100
    score = min(score, 1.0)
    if score < 0.33:
        return score, 'Low', 'Continue monitoring'
    elif score < 0.66:
        return score, 'Medium', 'Consider design review'
    else:
        return score, 'High', 'Immediate action recommended'

def clearance_to_performance_loss(initial_mm, current_mm, geometry_factor=1.0):
    """Estimate performance loss from clearance increase"""
    delta = current_mm - initial_mm
    if delta <= 0:
        return 0.0, 0.0
    head_loss_pct = delta * 2.0 * geometry_factor
    eff_loss_pct = delta * 1.5 * geometry_factor
    return head_loss_pct, eff_loss_pct

def remaining_life(thickness_mm, wear_rate_mm_per_year, operating_hours_per_year):
    """Estimate remaining component life"""
    if wear_rate_mm_per_year <= 0 or thickness_mm <= 0:
        return np.inf, np.inf
    years_left = thickness_mm / wear_rate_mm_per_year
    hours_left = years_left * operating_hours_per_year
    return years_left, hours_left

def calculate_vibration_severity(velocity, Re, material_type):
    """Calculate vibration severity"""
    score = 0
    if velocity > 4:
        score += 2
    elif velocity > 3:
        score += 1
    if Re < 2300:
        score += 1
    if material_type == 'Slurry':
        score += 1
    
    if score >= 3:
        return 'High', 'red'
    elif score >= 2:
        return 'Medium', 'orange'
    else:
        return 'Low', 'green'

def calculate_pressure_pulsation(pump_type, Q_op, Q_bep):
    """Calculate pressure pulsation risk"""
    if Q_bep <= 0:
        return 'Unknown', 'gray'
    deviation = abs(Q_op - Q_bep) / Q_bep * 100
    if deviation > 25:
        return 'High', 'red'
    elif deviation > 15:
        return 'Medium', 'orange'
    else:
        return 'Low', 'green'

def estimate_seal_life(material_type, temperature, velocity):
    """Estimate mechanical seal life"""
    base_life = 20000  # hours
    if material_type == 'Slurry':
        base_life *= 0.5
    if temperature > 80:
        base_life *= 0.7
    if velocity > 3:
        base_life *= 0.8
    return base_life

def pump_health_score(vibration, erosion, pulsation):
    """Calculate composite health score (0-100, lower is better)"""
    vib_score = {'Low': 10, 'Medium': 30, 'High': 50}.get(vibration, 20)
    erosion_score_num = erosion * 30
    pulsation_score_num = pulsation * 20
    return vib_score + erosion_score_num + pulsation_score_num

def _safe_get(locals_dict, key, default=None):
    return locals_dict.get(key, default)

def _to_json_safe(v):
    """Convert common numpy types to native Python types for JSON serialization."""
    try:
        if v is None:
            return None
        if hasattr(v, 'item') and not isinstance(v, (str, bytes, dict, list, tuple)):
            try:
                return v.item()
            except Exception:
                pass
        if hasattr(v, 'tolist') and not isinstance(v, (str, bytes, dict)):
            try:
                return v.tolist()
            except Exception:
                pass
        return v
    except Exception:
        return str(v)

def generate_api610_checklist(locals_dict):
    """Create a small API-610 style checklist based on key diagnostics."""
    checklist = []
    NPSHa = _safe_get(locals_dict, 'NPSHa', None)
    NPSHr = _safe_get(locals_dict, 'NPSHr_vendor', None)
    if NPSHa is None or NPSHr is None or NPSHr == 0:
        checklist.append({'Item': 'NPSH Margin', 'Status': 'Review', 'Remarks': 'Vendor NPSHr missing or insufficient data'})
    else:
        margin = NPSHa - NPSHr
        if margin >= 1.0:
            checklist.append({'Item': 'NPSH Margin', 'Status': 'Pass', 'Remarks': f'Margin {margin:.2f} m'})
        elif margin >= 0.5:
            checklist.append({'Item': 'NPSH Margin', 'Status': 'Review', 'Remarks': f'Margin {margin:.2f} m (low)'})
        else:
            checklist.append({'Item': 'NPSH Margin', 'Status': 'Fail', 'Remarks': f'Margin {margin:.2f} m (insufficient)'})

    vib = _safe_get(locals_dict, 'vibration_severity', None)
    if vib is None:
        checklist.append({'Item': 'Vibration (Overall)', 'Status': 'Review', 'Remarks': 'No measurement'})
    else:
        if vib == 'Low':
            checklist.append({'Item': 'Vibration (Overall)', 'Status': 'Pass', 'Remarks': 'Within acceptable limits'})
        elif vib == 'Medium':
            checklist.append({'Item': 'Vibration (Overall)', 'Status': 'Review', 'Remarks': 'Monitor and investigate trend'})
        else:
            checklist.append({'Item': 'Vibration (Overall)', 'Status': 'Fail', 'Remarks': 'Exceeds recommended levels'})

    erosion_score = _safe_get(locals_dict, 'erosion_score', None)
    if erosion_score is None:
        checklist.append({'Item': 'Cavitation / Erosion', 'Status': 'Review', 'Remarks': 'Insufficient data'})
    else:
        if erosion_score < 0.33:
            checklist.append({'Item': 'Cavitation / Erosion', 'Status': 'Pass', 'Remarks': 'Low risk'})
        elif erosion_score < 0.66:
            checklist.append({'Item': 'Cavitation / Erosion', 'Status': 'Review', 'Remarks': 'Medium risk; consider mitigation'})
        else:
            checklist.append({'Item': 'Cavitation / Erosion', 'Status': 'Fail', 'Remarks': 'High risk; action recommended'})

    pct_from_bep = _safe_get(locals_dict, 'pct_from_bep', None)
    if pct_from_bep is None:
        checklist.append({'Item': 'Operating point vs BEP', 'Status': 'Review', 'Remarks': 'No BEP data'})
    else:
        if pct_from_bep <= 10:
            checklist.append({'Item': 'Operating point vs BEP', 'Status': 'Pass', 'Remarks': f'{pct_from_bep:.1f}% from BEP'})
        elif pct_from_bep <= 20:
            checklist.append({'Item': 'Operating point vs BEP', 'Status': 'Review', 'Remarks': f'{pct_from_bep:.1f}% from BEP (consider derating)'})
        else:
            checklist.append({'Item': 'Operating point vs BEP', 'Status': 'Fail', 'Remarks': f'{pct_from_bep:.1f}% from BEP (poor)'})

    seal_life = _safe_get(locals_dict, 'seal_life_hours', None)
    if seal_life is None:
        checklist.append({'Item': 'Seal Life', 'Status': 'Review', 'Remarks': 'No estimate'})
    else:
        if seal_life >= 20000:
            checklist.append({'Item': 'Seal Life', 'Status': 'Pass', 'Remarks': f'Estimated {seal_life:.0f} h'})
        elif seal_life >= 8000:
            checklist.append({'Item': 'Seal Life', 'Status': 'Review', 'Remarks': f'Estimated {seal_life:.0f} h'})
        else:
            checklist.append({'Item': 'Seal Life', 'Status': 'Fail', 'Remarks': f'Estimated {seal_life:.0f} h (low)'})

    health_score = _safe_get(locals_dict, 'health_score', None)
    if health_score is not None:
        if health_score < 30:
            checklist.append({'Item': 'Composite Health Score', 'Status': 'Pass', 'Remarks': f'Score {health_score:.0f}/100'})
        elif health_score < 60:
            checklist.append({'Item': 'Composite Health Score', 'Status': 'Review', 'Remarks': f'Score {health_score:.0f}/100'})
        else:
            checklist.append({'Item': 'Composite Health Score', 'Status': 'Fail', 'Remarks': f'Score {health_score:.0f}/100'})

    return checklist

def prepare_diagnostics_payload(locals_dict):
    """Prepare a JSON-serializable diagnostics payload from local variables."""
    payload = {
        'timestamp': datetime.now().isoformat(),
        'process': {
            'Q_design_m3_s': _to_json_safe(_safe_get(locals_dict, 'Q_design', None)),
            'Q_op_m3_s': _to_json_safe(_safe_get(locals_dict, 'Q_op', None)),
            'Q_bep_m3_s': _to_json_safe(_safe_get(locals_dict, 'Q_bep', None)),
            'total_head_design_m': _to_json_safe(_safe_get(locals_dict, 'total_head_design', None)),
            'static_head_m': _to_json_safe(_safe_get(locals_dict, 'static_head', None)),
            'velocity_m_s': _to_json_safe(_safe_get(locals_dict, 'V', None)),
            'reynolds': _to_json_safe(_safe_get(locals_dict, 'Re', None)),
        },
        'diagnostics': {
            'wear_rate_relative': _to_json_safe(_safe_get(locals_dict, 'wear_rate', None)),
            'slurry_wear_mm_per_year': _to_json_safe(_safe_get(locals_dict, 'slurry_wear_mm_per_year', None)),
            'erosion_score': _to_json_safe(_safe_get(locals_dict, 'erosion_score', None)),
            'erosion_label': _to_json_safe(_safe_get(locals_dict, 'erosion_label', None)),
            'head_loss_pct_from_clearance': _to_json_safe(_safe_get(locals_dict, 'head_loss_pct', None)),
            'eff_loss_pct_from_clearance': _to_json_safe(_safe_get(locals_dict, 'eff_loss_pct', None)),
            'remaining_life_years': _to_json_safe(_safe_get(locals_dict, 'years_left', None)),
            'remaining_life_hours': _to_json_safe(_safe_get(locals_dict, 'hours_left', None)),
            'vibration_severity': _to_json_safe(_safe_get(locals_dict, 'vibration_severity', None)),
            'pulsation_risk': _to_json_safe(_safe_get(locals_dict, 'pulsation_risk', None)),
            'seal_life_hours': _to_json_safe(_safe_get(locals_dict, 'seal_life_hours', None)),
            'health_score': _to_json_safe(_safe_get(locals_dict, 'health_score', None)),
        }
    }
    payload['api610_checklist'] = generate_api610_checklist(locals_dict)
    return payload

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
                particle_conc = st.number_input("Particle concentration (wt%)", value=0.0, min_value=0.0)
                impeller_thickness_mm = st.number_input("Impeller thickness (mm)", value=5.0, min_value=0.1)
                measured_clearance_mm = st.number_input("Measured clearance (mm)", value=0.5, min_value=0.0)
            else:
                particle_size = 0
                particle_conc = 0.0
                impeller_thickness_mm = 0.0
                measured_clearance_mm = 0.0
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
                electricity_cost = st.number_input("Electricity cost (₹/kWh)", value=10.0, min_value=0.0)
                operating_hours = st.number_input("Operating hours/year", value=8000.0, min_value=0.0)
            colebrook_method = st.selectbox(
                "Colebrook solver",
                ['Swamee-Jain (explicit)', 'Newton-Raphson', 'Bracketed (robust)'],
                index=2
            )
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
            f = colebrook_f(Re, D, eps_mm/1000.0, method=colebrook_method, tol=1e-8, max_iter=100)
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
            vibration_severity, vib_color = calculate_vibration_severity(V, Re, material_type)
            pulsation_risk, pulse_color = calculate_pressure_pulsation('Centrifugal', Q_op, Q_bep)
            seal_life_hours = estimate_seal_life(material_type, T, V)
            motor_rated_kW = electrical_kW * service_factor

            # Wear analysis
            if show_wear_analysis:
                wear_rate = estimate_wear_rate(material_type, V, particle_size)
                if material_type == 'Slurry':
                    slurry_wear_mm_per_year = slurry_erosion_rate(particle_conc, particle_size, V, hardness_ratio=1.0)
                else:
                    slurry_wear_mm_per_year = 0.01 * wear_rate

                flow_dev_pct = pct_from_bep if not np.isnan(pct_from_bep) else 0.0
                erosion_score, erosion_label, erosion_recommendation = cavitation_erosion_index(NPSHa, NPSHr_vendor, sigma, flow_dev_pct)

                initial_clearance_mm = 0.3 if impeller_thickness_mm <= 0 else max(0.1, impeller_thickness_mm * 0.05)
                head_loss_pct, eff_loss_pct = clearance_to_performance_loss(initial_clearance_mm, measured_clearance_mm, geometry_factor=1.0)

                op_hours = operating_hours if show_energy_cost else 8000.0
                years_left, hours_left = remaining_life(impeller_thickness_mm, slurry_wear_mm_per_year, op_hours)

                pulsation_numeric = 1.0 if pulsation_risk == 'High' else (0.5 if pulsation_risk == 'Medium' else 0.0)
                health_score = pump_health_score(vibration_severity, erosion_score, pulsation_numeric)

                estimated_service_life = hours_left if not np.isinf(hours_left) else 50000

            # Display results
            st.success("✅ Calculation Complete")
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

                st.markdown("---")
                st.subheader("🎯 Recommendations")
                rec_col1, rec_col2 = st.columns(2)
                with col2:
                    st.subheader("Process Parameters")
                    proc_params = {
                        'Design Flow (m³/h)': f"{Q_design * 3600:.2f}",
                        'Operating Flow (m³/h)': f"{Q_op * 3600:.2f}",
                        'BEP Flow (m³/h)': f"{Q_bep * 3600:.2f}",
                        'Design Head (m)': f"{total_head_design:.2f}",
                        'Static Head (m)': f"{static_head:.2f}",
                    }
                    df_proc = pd.DataFrame(list(proc_params.items()), columns=["Parameter", "Value"])
                    st.dataframe(df_proc, use_container_width=True, hide_index=True)

                with rec_col1:
                    if pct_from_bep <= 10:
                        st.success("✅ Excellent: Operating within 10% of BEP")
                    elif pct_from_bep <= 20:
                        st.warning("⚡ Acceptable: Consider 5-10% derating")
                    else:
                        st.error("⚠️ Poor: Far from BEP. Consider different size")
                        
                with rec_col2:
                    if show_wear_analysis:
                        st.write(f"**Relative Wear Rate:** {wear_rate:.1f}x baseline")
                        st.write(f"**Slurry Wear Estimate:** {slurry_wear_mm_per_year:.4f} mm/year")
                        st.write(f"**Est. Service Life:** {estimated_service_life:.0f} hours ({years_left:.1f} years)")
                        st.write(f"**Pump Health Score:** {health_score:.0f}/100")
                    st.write(f"**Vibration Risk:** :{vib_color}[{vibration_severity}]")
                    st.write(f"**Pulsation Risk:** :{pulse_color}[{pulsation_risk}]")
                    st.write(f"**Seal Life Est.:** {seal_life_hours:.0f} hours")

            with tab2:
                st.subheader("Performance Curves")
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                ax1.plot(Q_points*3600, H_system, 'b-', linewidth=2, label='System Curve')
                ax1.plot(Q_points*3600, H_pump, 'r-', linewidth=2, label='Pump Curve')
                ax1.scatter([Q_bep*3600], [H_pump[bep_idx]], color='green', s=150, 
                           marker='*', label='BEP', zorder=5, edgecolors='black')
                ax1.scatter([Q_op*3600], [H_op], color='orange', s=150, 
                           marker='D', label='Operating Point', zorder=5, edgecolors='black')
                ax1.fill_between(Q_points*3600, H_system, alpha=0.2, color='blue')
                ax1.set_xlabel('Flow Rate (m³/h)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Head (m)', fontsize=11, fontweight='bold')
                ax1.set_title('Pump & System Curves', fontsize=12, fontweight='bold')
                ax1.legend(loc='best', framealpha=0.9)
                ax1.grid(True, alpha=0.3, linestyle=':')

                ax2.plot(Q_points*3600, eff_curve*100, 'g-', linewidth=2, label='Efficiency')
                ax2.axvline(Q_bep*3600, color='green', linestyle='--', alpha=0.7, label='BEP')
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

            with tab3:
                st.subheader("⚙️ Advanced Analysis")
                if show_affinity:
                    st.markdown("#### Affinity Laws Analysis")
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
                    st.dataframe(df_affinity, use_container_width=True, hide_index=True)

            with tab4:
                if show_energy_cost:
                    st.subheader("💰 Life Cycle Cost Analysis")
                    annual_energy_kWh = electrical_kW * operating_hours
                    annual_cost = annual_energy_kWh * electricity_cost
                    col_cost1, col_cost2, col_cost3 = st.columns(3)
                    with col_cost1:
                        st.metric("Annual Energy", f"{annual_energy_kWh:,.0f} kWh")
                        st.metric("Annual Cost", f"₹{annual_cost:,.2f}")
                    with col_cost2:
                        if eff_op < 0.7:
                            potential_savings = annual_cost * (0.75 - eff_op) / eff_op
                            st.metric("Potential Savings", f"₹{potential_savings:,.2f}/yr", 
                                     "with 75% efficient pump")
                        else:
                            st.metric("Efficiency Status", "Good", "✅")
                    with col_cost3:
                        ten_year_cost = annual_cost * 10
                        st.metric("10-Year Energy Cost", f"₹{ten_year_cost:,.2f}")
                else:
                    st.info("Enable 'Calculate energy costs' in the form to see this analysis")

            # Export functionality
            st.markdown("---")
            st.subheader("📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                try:
                    diagnostics_payload = prepare_diagnostics_payload(locals())
                    json_bytes = json.dumps(diagnostics_payload, indent=2).encode('utf-8')
                    csv_row = {}
                    csv_row.update(diagnostics_payload.get('process', {}))
                    csv_row.update(diagnostics_payload.get('diagnostics', {}))
                    csv_df = pd.DataFrame([csv_row])
                    csv_bytes = csv_df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Diagnostics (JSON)",
                        data=json_bytes,
                        file_name=f"pump_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as _err:
                    st.warning(f"Diagnostics export unavailable: {_err}")

            with col_exp2:
                try:
                    diagnostics_payload = prepare_diagnostics_payload(locals())
                    checklist = diagnostics_payload.get('api610_checklist', [])
                    with st.expander("🛡️ API-610 Checklist", expanded=True):
                        if checklist:
                            df_chk = pd.DataFrame(checklist)
                            st.dataframe(df_chk, use_container_width=True, hide_index=True)
                        else:
                            st.write("No checklist data available.")
                except Exception as _err:
                    st.warning(f"Unable to build checklist: {_err}")

        except Exception as e:
            st.error(f"❌ Calculation error: {e}")
            st.error("Please check your input values and try again.")

# ------------------ System Comparison Page ------------------
elif page == "Pump System Comparison":
    st.header("⚖️ Pump System Comparison Tool")
    st.write("Compare multiple pump configurations side-by-side")
    
    if 'comparison_data' not in st.session_state:
        st.session_state['comparison_data'] = []
    comparison_data = st.session_state['comparison_data']

    num_systems = st.slider("Number of systems to compare", 2, 3, 2)
    
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
                cost = st.number_input("Capital Cost (₹)", value=5000.0*(i+1))
                submitted = st.form_submit_button("Add System")
                if submitted:
                    st.session_state['comparison_data'].append({
                        'Name': name,
                        'Flow (m³/h)': flow,
                        'Head (m)': head,
                        'Efficiency (%)': efficiency,
                        'Power (kW)': power,
                        'Capital Cost (₹)': cost
                    })
                    st.rerun()

    if len(comparison_data) >= 2:
        df_comp = pd.DataFrame(comparison_data)
        st.markdown("---")
        st.subheader("Comparison Results")
        st.dataframe(df_comp, use_container_width=True)

        if st.button("Calculate Best Pump"):
            df_comp['Efficiency Score'] = df_comp['Efficiency (%)'] / df_comp['Efficiency (%)'].max()
            df_comp['Power Score'] = 1 - (df_comp['Power (kW)'] / df_comp['Power (kW)'].max())
            df_comp['Cost Score'] = 1 - (df_comp['Capital Cost (₹)'] / df_comp['Capital Cost (₹)'].max())
            
            weights = {'Efficiency': 0.4, 'Power': 0.35, 'Cost': 0.25}
            
            df_comp['Total Score'] = (
                df_comp['Efficiency Score'] * weights['Efficiency'] +
                df_comp['Power Score'] * weights['Power'] +
                df_comp['Cost Score'] * weights['Cost']
            )
            
            best_pump = df_comp.loc[df_comp['Total Score'].idxmax()]
            st.success(f"🏆 Recommended Pump: {best_pump['Name']}")

# ------------------ Life Cycle Cost Analysis Page ------------------
elif page == "Life Cycle Cost Analysis":
    st.header("💰 Detailed Life Cycle Cost Analysis")
    with st.form("lcc_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Capital Costs")
            pump_cost = st.number_input("Pump cost (₹)", value=5000.0)
            motor_cost = st.number_input("Motor cost (₹)", value=2000.0)
            installation_cost = st.number_input("Installation (₹)", value=1500.0)
            st.subheader("Operating Parameters")
            operating_hours_year = st.number_input("Hours/year", value=8000.0)
            electricity_rate = st.number_input("Electricity (₹/kWh)", value=10.0)
            power_kW = st.number_input("Power consumption (kW)", value=15.0)
        with col2:
            st.subheader("Maintenance Costs")
            annual_maintenance = st.number_input("Annual maintenance (₹)", value=500.0)
            major_overhaul_cost = st.number_input("Major overhaul (₹)", value=3000.0)
            overhaul_interval_years = st.number_input("Overhaul interval (years)", value=5.0)
            st.subheader("Analysis Period")
            analysis_years = st.number_input("Analysis period (years)", value=15.0, min_value=1.0, max_value=30.0)
            discount_rate = st.number_input("Discount rate (%)", value=5.0) / 100
        
        calculate_lcc = st.form_submit_button("Calculate Life Cycle Cost", type="primary")

    if calculate_lcc:
        years = np.arange(1, int(analysis_years) + 1)
        initial_cost = pump_cost + motor_cost + installation_cost
        annual_energy_cost = power_kW * operating_hours_year * electricity_rate
        
        capital_costs = np.zeros(len(years))
        capital_costs[0] = initial_cost
        energy_costs = np.ones(len(years)) * annual_energy_cost
        maintenance_costs = np.ones(len(years)) * annual_maintenance
        
        overhaul_costs = np.zeros(len(years))
        for year in years:
            if year % overhaul_interval_years == 0:
                overhaul_costs[year-1] = major_overhaul_cost
        
        total_annual_costs = capital_costs + energy_costs + maintenance_costs + overhaul_costs
        discount_factors = 1 / (1 + discount_rate) ** years
        npv_costs = total_annual_costs * discount_factors
        cumulative_npv = np.cumsum(npv_costs)

        st.success(f"✅ Total Life Cycle Cost (NPV): ₹{cumulative_npv[-1]:,.2f}")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Initial Investment", f"₹{initial_cost:,.2f}")
            st.metric("Total Energy Cost", f"₹{np.sum(energy_costs * discount_factors):,.2f}")
        with col_res2:
            st.metric("Total Maintenance", f"₹{np.sum(maintenance_costs * discount_factors):,.2f}")
            st.metric("Total Overhauls", f"₹{np.sum(overhaul_costs * discount_factors):,.2f}")
        with col_res3:
            energy_pct = np.sum(energy_costs * discount_factors) / cumulative_npv[-1] * 100
            st.metric("Energy % of LCC", f"{energy_pct:.1f}%")
            st.metric("Avg Annual Cost", f"₹{cumulative_npv[-1]/analysis_years:,.2f}")

st.markdown("---")
st.caption("⚠️ Engineering estimates only. Validate with vendor data before procurement.")
