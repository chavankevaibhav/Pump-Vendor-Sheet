import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime

# Set page config at the very top
st.set_page_config(page_title="Advanced Pump & Vacuum Sizing", layout="wide")
st.title("🔧 Advanced Pump & Vacuum Pump Sizing Sheet — Vendor Ready")

# --- Enhanced UI controls (Auto-update, presets, form helper) ---
auto_update = st.sidebar.checkbox("Auto-update (real-time recalculation)", value=True)

def maybe_form(key: str):
    """Return a context manager: st (immediate) or st.form(key) when auto_update is off."""
    if auto_update:
        return st
    return st.form(key)

# Presets for quick configuration
PRESETS = {
    'Custom': {},
    'Small Water Transfer': {
        'Q_input': 10.0, 'Q_unit': 'm³/h', 'T': 25.0, 'material_type': 'Water', 'D_inner': 50.0, 'L_pipe': 10.0
    },
    'Seawater Cooling (Medium)': {
        'Q_input': 200.0, 'Q_unit': 'm³/h', 'T': 35.0, 'material_type': 'Seawater', 'D_inner': 150.0, 'L_pipe': 30.0
    },
    'Slurry - Pilot Plant': {
        'Q_input': 50.0, 'Q_unit': 'm³/h', 'T': 30.0, 'material_type': 'Slurry', 'D_inner': 100.0, 'L_pipe': 15.0
    }
}

# Simple navigation between pages
page = st.sidebar.selectbox("Choose tool", [
    "Rotating Pumps (Centrifugal etc.)", 
    "Pump System Comparison",
    "Life Cycle Cost Analysis"
])

# ------------------ Helper functions ------------------
def prepare_diagnostics_payload(context):
    """Prepare a structured diagnostics payload from calculation context."""
    try:
        payload = {
            'timestamp': datetime.now().isoformat(),
            'process': {
                'flow_design': context['Q_design'] * 3600 if context['Q_design'] else None,  # Convert to m³/h
                'flow_operating': context['Q_op'] * 3600 if context['Q_op'] else None,
                'flow_bep': context['Q_bep'] * 3600 if context['Q_bep'] else None,
                'head_design': context['total_head_design'],
                'static_head': context['static_head'],
                'fluid_velocity': context['V'],
                'reynolds_number': context['Re'],
            },
            'diagnostics': {
                'wear_rate': context['wear_rate'],
                'wear_per_year': context['slurry_wear_mm_per_year'],
                'erosion_score': context['erosion_score'],
                'erosion_severity': context['erosion_label'],
                'head_loss_percent': context['head_loss_pct'],
                'efficiency_loss_percent': context['eff_loss_pct'],
                'remaining_years': context['years_left'],
                'remaining_hours': context['hours_left'],
                'vibration_severity': context['vibration_severity'],
                'pulsation_risk': context['pulsation_risk'],
                'seal_life_hours': context['seal_life_hours'],
                'health_score': context['health_score'],
                'percent_from_bep': context['pct_from_bep'],
                'npsha': context['NPSHa'],
                'npshr': context['NPSHr_vendor']
            }
        }
        return {k: v for k, v in payload.items() if v is not None}
    except Exception as e:
        raise ValueError(f"Failed to prepare diagnostics payload: {str(e)}")

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


@st.cache_data
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

@st.cache_data
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

# --- API 610 Functions ---

def select_mechanical_seal_api610(temperature_C, pressure_bar, fluid_type, shaft_speed_rpm, fluid_properties=None):
    """Select mechanical seal configuration based on API 610 requirements."""
    if fluid_properties is None:
        fluid_properties = {'abrasive': False, 'volatile': False, 'toxic': False, 'crystallizing': False}
    
    # Base configuration
    seal_config = {
        'arrangement': 'Single',
        'seal_type': 'Pusher',
        'face_material': 'Carbon/SiC',
        'elastomer': 'FKM',
        'api_plan': '11',
        'contains_solids': False
    }
    
    # Temperature considerations
    if temperature_C > 200:
        seal_config['arrangement'] = 'Dual pressurized'
        seal_config['api_plan'] = '53A'
        seal_config['face_material'] = 'SiC/SiC'
        seal_config['elastomer'] = 'FFKM'
    elif temperature_C > 150:
        seal_config['elastomer'] = 'FFKM'
    
    # Pressure considerations
    if pressure_bar > 40:
        seal_config['arrangement'] = 'Dual unpressurized'
        seal_config['api_plan'] = '52'
        seal_config['face_material'] = 'TC/TC'
    
    # Fluid type considerations
    if fluid_type == 'Slurry':
        seal_config['contains_solids'] = True
        seal_config['face_material'] = 'TC/TC'
        seal_config['api_plan'] = '32'
    elif fluid_type in ['Acids', 'Alkaline']:
        seal_config['face_material'] = 'SiC/SiC'
        seal_config['elastomer'] = 'FFKM'
    
    # Speed considerations
    if shaft_speed_rpm > 3600:
        seal_config['seal_type'] = 'Non-pusher'
    
    # Special conditions
    if fluid_properties.get('volatile', False):
        seal_config['arrangement'] = 'Dual pressurized'
        seal_config['api_plan'] = '53B'
    if fluid_properties.get('toxic', False):
        seal_config['arrangement'] = 'Dual pressurized'
        seal_config['api_plan'] = '53B'
    if fluid_properties.get('crystallizing', False):
        seal_config['api_plan'] += '/32'
    
    return seal_config

def calculate_bearing_cooling_api610(shaft_speed_rpm, bearing_temp_rise, ambient_temp=40):
    """Calculate bearing housing cooling requirements per API 610."""
    cooling_req = {
        'cooling_required': False,
        'method': 'None',
        'flow_rate': 0,
        'heat_load': 0
    }
    
    # API 610 temperature limits
    max_oil_temp = 82  # °C
    max_bearing_temp = 95  # °C
    
    # Calculate heat generation (simplified)
    bearing_power_loss = (shaft_speed_rpm/1000)**1.6 * 100  # Watts
    
    # Temperature rise calculation
    total_temp = ambient_temp + bearing_temp_rise
    
    if total_temp > max_bearing_temp or bearing_temp_rise > 40:
        cooling_req['cooling_required'] = True
        if bearing_power_loss > 2000:
            cooling_req['method'] = 'Water cooling'
            cooling_req['flow_rate'] = bearing_power_loss / (4186 * 10)  # L/min
        else:
            cooling_req['method'] = 'Air cooling'
            cooling_req['flow_rate'] = bearing_power_loss / 1000  # m³/min
    
    cooling_req['heat_load'] = bearing_power_loss
    return cooling_req

def recommend_material_upgrades_api610(base_material, temperature_C, pressure_bar, 
                                     corrosive=False, erosive=False):
    """Provide material upgrade recommendations based on API 610."""
    upgrades = {
        'current_class': '',
        'recommended_class': '',
        'reason': [],
        'estimated_life_improvement': 1.0,
        'specific_recommendations': {}
    }
    
    # Base material class identification
    if base_material == 'Cast Iron':
        upgrades['current_class'] = 'I-1'
    elif base_material == 'Carbon Steel':
        upgrades['current_class'] = 'I-2'
    elif base_material == '12% Chrome':
        upgrades['current_class'] = 'S-1'
    elif base_material == 'Duplex SS':
        upgrades['current_class'] = 'S-6'
    
    # Temperature-based upgrades
    if temperature_C > 150 and upgrades['current_class'] in ['I-1', 'I-2']:
        upgrades['recommended_class'] = 'S-1'
        upgrades['reason'].append('High temperature service')
        upgrades['estimated_life_improvement'] = 2.0
    
    # Pressure-based upgrades
    if pressure_bar > 40 and upgrades['current_class'] == 'I-1':
        upgrades['recommended_class'] = 'I-2'
        upgrades['reason'].append('High pressure service')
        upgrades['estimated_life_improvement'] = 1.5
    
    # Corrosion/erosion upgrades
    if corrosive and upgrades['current_class'] in ['I-1', 'I-2']:
        upgrades['recommended_class'] = 'S-1'
        upgrades['reason'].append('Corrosive service')
        upgrades['estimated_life_improvement'] *= 2.5
    
    if erosive and upgrades['current_class'] in ['I-1', 'I-2', 'S-1']:
        upgrades['recommended_class'] = 'S-6'
        upgrades['reason'].append('Erosive service')
        upgrades['estimated_life_improvement'] *= 2.0
    
    # Specific component recommendations
    upgrades['specific_recommendations'] = {
        'impeller': 'Upgrade to next material class' if erosive else 'Standard',
        'wear_rings': 'Hard-faced' if erosive else 'Standard',
        'shaft': '12% Chrome minimum' if corrosive else 'Standard',
        'case': upgrades['recommended_class'] if upgrades['recommended_class'] else 'Standard'
    }
    
    return upgrades

def generate_installation_specs_api610(pump_power_kW, base_plate_length_mm):
    """Generate detailed installation and alignment specifications per API 610."""
    specs = {
        'foundation': {
            'type': 'Concrete',
            'minimum_mass': pump_power_kW * 2.5,  # tonnes
            'minimum_thickness': max(200, pump_power_kW * 25),  # mm
            'reinforcement': 'Required' if pump_power_kW > 50 else 'Optional'
        },
        'grouting': {
            'type': 'Epoxy' if pump_power_kW > 100 else 'Cementitious',
            'thickness_mm': min(50, max(25, pump_power_kW/2)),
            'cure_time_hours': 48 if pump_power_kW > 100 else 24
        },
        'alignment': {
            'cold_alignment': {
                'parallel_mm': 0.05,
                'angular_mm_per_100mm': 0.04
            },
            'hot_alignment': {
                'parallel_mm': 0.075,
                'angular_mm_per_100mm': 0.06
            },
            'measurement_points': 4 if base_plate_length_mm > 2000 else 3,
            'check_intervals': [
                'After grouting cure',
                'After pipe connection',
                'At normal operating temperature'
            ]
        },
        'piping': {
            'suction_support': {
                'first_support_distance': min(500, base_plate_length_mm/4),  # mm
                'maximum_span': 3000  # mm
            },
            'discharge_support': {
                'first_support_distance': min(300, base_plate_length_mm/6),  # mm
                'maximum_span': 2500  # mm
            },
            'allowable_forces': {
                'suction_flange': pump_power_kW * 50,  # N
                'discharge_flange': pump_power_kW * 40  # N
            }
        },
        'monitoring': {
            'vibration_points': ['Inboard Bearing', 'Outboard Bearing', 'Casing'],
            'alignment_intervals': {
                'initial_hours': 50,
                'routine_months': 6
            }
        }
    }
    return specs

# --- End of new API 610 Functions ---
def calculate_first_critical_speed(shaft_length_mm, shaft_diameter_mm, mass_kg):
    """Calculate first critical speed using Dunkerley's method."""
    # Convert to meters
    L = shaft_length_mm / 1000
    d = shaft_diameter_mm / 1000
    
    # Moment of inertia
    I = (math.pi * d**4) / 64
    
    # Area
    A = math.pi * (d/2)**2
    
    # Material properties (steel)
    E = 200e9  # Young's modulus
    rho = 7850  # Density
    
    # Natural frequency
    fn = (1/(2*math.pi)) * math.sqrt((48*E*I)/(mass_kg*L**3))
    
    # Convert to RPM
    return fn * 60

def calculate_shaft_deflection_api610(shaft_length_mm, shaft_diameter_mm, radial_load_N, 
                                    elastic_modulus_GPa=200):
    """Calculate shaft deflection at seal locations per API 610."""
    L = shaft_length_mm / 1000
    d = shaft_diameter_mm / 1000
    E = elastic_modulus_GPa * 1e9
    
    # Moment of inertia
    I = (math.pi * d**4) / 64
    
    # Maximum deflection (simplified beam calculation)
    deflection = (radial_load_N * L**3) / (48 * E * I)
    
    # API 610 limits
    api_limit_seal = 0.05/1000  # 0.05mm at seal faces
    api_limit_wear_rings = 0.08/1000  # 0.08mm at wear rings
    
    return {
        'deflection_m': deflection,
        'seal_limit_m': api_limit_seal,
        'wear_ring_limit_m': api_limit_wear_rings,
        'seal_compliant': deflection <= api_limit_seal,
        'wear_ring_compliant': deflection <= api_limit_wear_rings
    }

def calculate_bearing_life_api610_detailed(radial_load_N, axial_load_N, rpm, bearing_type='ball'):
    """Calculate detailed bearing life according to API 610."""
    # Basic dynamic load ratings (approximate)
    if bearing_type == 'ball':
        C = radial_load_N * 4  # Typical ratio for ball bearings
        X = 1  # Radial factor
        Y = 0.5  # Axial factor
        life_factor = 3  # Exponent for ball bearings
    else:  # roller
        C = radial_load_N * 6  # Typical ratio for roller bearings
        X = 0.8
        Y = 0.6
        life_factor = 10/3  # Exponent for roller bearings
    
    # Equivalent dynamic load
    P = X * radial_load_N + Y * axial_load_N
    
    # Basic rating life in millions of revolutions
    L10 = (C/P)**life_factor
    
    # Convert to hours
    hours = (L10 * 1e6) / (60 * rpm)
    
    # API 610 minimum requirement
    api_minimum = 25000  # hours
    
    return {
        'L10_hours': hours,
        'api_minimum': api_minimum,
        'compliant': hours >= api_minimum,
        'bearing_type': bearing_type,
        'load_ratio': P/C
    }

def calculate_seal_chamber_api610(shaft_diameter_mm):
    """Calculate API 610 seal chamber dimensions."""
    d = shaft_diameter_mm
    
    # API 610 seal chamber dimensions
    chamber = {
        'bore_diameter': max(d + 40, d * 1.5),  # mm
        'depth': max(d + 25, d * 1.25),  # mm
        'rabbet_diameter': max(d + 60, d * 1.8),  # mm
        'min_radius': 3.0,  # mm
        'surface_finish': 0.8  # μm Ra
    }
    
    # Calculate cooling requirements
    chamber['cooling_required'] = d > 75
    
    # Recommended flush plans based on size
    if d <= 60:
        chamber['recommended_plans'] = ['Plan 11', 'Plan 13']
    else:
        chamber['recommended_plans'] = ['Plan 23', 'Plan 32']
    
    return chamber

def calculate_baseplate_requirements_api610(pump_power_kW, pump_length_mm):
    """Calculate API 610 baseplate requirements."""
    # Minimum thickness based on pump power
    if pump_power_kW <= 100:
        min_thickness_mm = 12
    elif pump_power_kW <= 300:
        min_thickness_mm = 20
    else:
        min_thickness_mm = 25
    
    # Stiffness requirements
    length = pump_length_mm
    width = length * 0.4  # Approximate
    
    # Deflection limit under full load
    max_deflection_mm = length / 1000  # API typical limit
    
    # Grouting requirements
    grout_thickness_mm = min(min_thickness_mm * 2, 50)
    
    return {
        'min_thickness_mm': min_thickness_mm,
        'recommended_thickness_mm': min_thickness_mm * 1.2,
        'grout_thickness_mm': grout_thickness_mm,
        'max_deflection_mm': max_deflection_mm,
        'leveling_requirements': {
            'max_deviation_mm_per_m': 0.2,
            'flatness_requirement_mm': 0.1
        },
        'anchor_bolt_spec': {
            'min_diameter_mm': min(min_thickness_mm * 1.5, 30),
            'material': 'ASTM A307 Grade B',
            'embedment_mm': min_thickness_mm * 15
        }
    }

def calculate_rotor_dynamics_api610(pump_speed_rpm, impeller_mass_kg, shaft_length_mm, 
                                  shaft_diameter_mm):
    """Calculate rotor dynamics parameters per API 610."""
    # First critical speed
    first_critical = calculate_first_critical_speed(shaft_length_mm, shaft_diameter_mm, 
                                                  impeller_mass_kg)
    
    # Separation margins
    max_cont_speed = pump_speed_rpm * 1.05
    min_cont_speed = pump_speed_rpm * 0.95
    
    # API 610 requirements
    margins = {
        'first_critical_speed': first_critical,
        'margin_above': (first_critical - max_cont_speed) / max_cont_speed * 100,
        'margin_below': (min_cont_speed - first_critical) / min_cont_speed * 100,
        'api_required_margin': 20  # %
    }
    
    # Damped natural frequency analysis (simplified)
    damping_ratio = 0.05  # Typical value
    damped_frequency = first_critical * math.sqrt(1 - damping_ratio**2)
    
    margins['damped_first_critical'] = damped_frequency
    
    return margins

def calculate_por(Q_bep):
    """Calculate Preferred Operating Region (POR) according to API 610.
    Returns flow ranges as fraction of BEP flow."""
    return 0.7 * Q_bep, 1.2 * Q_bep

def calculate_aor(Q_bep):
    """Calculate Allowable Operating Region (AOR) according to API 610.
    Returns minimum continuous stable flow and maximum flow."""
    return 0.5 * Q_bep, 1.3 * Q_bep

def calculate_mcsf(Ns, Q_bep):
    """Calculate Minimum Continuous Stable Flow based on specific speed.
    Returns flow rate as fraction of BEP."""
    if Ns < 1500:
        return 0.4  # 40% of BEP
    elif Ns < 3000:
        return 0.45  # 45% of BEP
    else:
        return 0.5  # 50% of BEP

def api610_material_class(temperature_C, pressure_bar):
    """Determine API 610 material class based on operating conditions."""
    if temperature_C <= 150 and pressure_bar <= 20:
        return "I-1", "Cast Iron"
    elif temperature_C <= 200 and pressure_bar <= 40:
        return "I-2", "Carbon Steel"
    elif temperature_C <= 350 and pressure_bar <= 100:
        return "S-1", "12% Chrome"
    else:
        return "S-6", "Duplex Stainless Steel"

def calculate_bearing_life_api610(radial_load_N, axial_load_N, rpm, bearing_size_mm):
    """Calculate bearing life according to API 610 requirements."""
    # Basic dynamic load rating (approximate)
    C = bearing_size_mm * 500  # Simplified correlation
    
    # Equivalent dynamic load
    P = 0.56 * radial_load_N + 1.2 * axial_load_N  # Simplified
    
    # Basic rating life in millions of revolutions
    L10 = (C/P)**3
    
    # Convert to hours
    hours = (L10 * 1e6) / (60 * rpm)
    
    return hours

def api610_shaft_deflection(shaft_length_mm, shaft_diameter_mm, load_N, elastic_modulus_GPa=200):
    """Calculate shaft deflection and compare to API 610 limits."""
    # Convert to meters
    L = shaft_length_mm / 1000
    d = shaft_diameter_mm / 1000
    E = elastic_modulus_GPa * 1e9
    
    # Moment of inertia
    I = (math.pi * d**4) / 64
    
    # Maximum deflection for simply supported beam (simplified)
    deflection = (load_N * L**3) / (48 * E * I)
    
    # API 610 limit (typically 0.05mm at seal)
    api_limit = 0.05/1000
    
    return deflection, api_limit, deflection <= api_limit

def calculate_nozzle_loads_api610(discharge_diameter_mm):
    """Calculate allowable nozzle loads per API 610."""
    # Base moment (simplified calculation)
    base_moment = (discharge_diameter_mm/25.4)**3 * 1.5  # Convert to inches for calculation
    
    # API 610 load limits
    limits = {
        'Fx': base_moment * 2.0,  # N
        'Fy': base_moment * 1.5,  # N
        'Fz': base_moment * 1.7,  # N
        'Mx': base_moment * 1.0,  # N·m
        'My': base_moment * 0.5,  # N·m
        'Mz': base_moment * 0.7   # N·m
    }
    return limits

def api610_critical_speed_margins(first_critical_speed):
    """Calculate critical speed margins per API 610."""
    running_speed = 1450  # Example running speed
    
    # API 610 requirements:
    # - First critical speed should be at least 20% above max continuous speed
    # - Or at least 20% below min continuous speed
    margin_above = (first_critical_speed - running_speed) / running_speed * 100
    margin_below = (running_speed - first_critical_speed) / running_speed * 100
    
    if margin_above >= 20:
        return True, f"Above by {margin_above:.1f}%"
    elif margin_below >= 20:
        return True, f"Below by {margin_below:.1f}%"
    else:
        return False, f"Insufficient margin: {min(margin_above, margin_below):.1f}%"

def api610_testing_requirements():
    """Return API 610 testing requirements checklist."""
    return {
        'Hydrostatic': {'required': True, 'pressure': '1.5 × max working pressure'},
        'Performance': {'required': True, 'points': 'Minimum 5 points including shutoff'},
        'NPSH': {'required': True, 'method': 'Three-point step'},
        'Bearing Temperature': {'required': True, 'limit': '82°C maximum'},
        'Vibration': {'required': True, 'limits': {
            'Velocity': '3.0 mm/s RMS',
            'Displacement': '50 micrometers peak-to-peak'
        }},
        'Sound Level': {'required': True, 'limit': '85 dBA at 1m'},
        'Nozzle Load': {'required': False, 'comment': 'When specified'},
        'Mechanical Run': {'required': True, 'duration': '4 hours minimum'}
    }

# --- END helpers ---

# ------------------ Rotating Pumps Page ------------------
if page == "Rotating Pumps (Centrifugal etc.)":
    st.header("🔄 Rotating Pump Sizing & Selection")
    form_ctx = maybe_form('rotating')
    with form_ctx:
        # Presets selector
        preset_choice = st.selectbox("Presets", list(PRESETS.keys()), index=0)
        preset_defaults = PRESETS.get(preset_choice, {})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Process & Fluid Data")
            Q_input = st.number_input("Flow rate", value=preset_defaults.get('Q_input', 100.0), min_value=0.0, format="%.6f")
            units_list = ['m³/h', 'L/s', 'm³/s', 'm³/d', 'GPM (US)']
            default_unit = preset_defaults.get('Q_unit', 'm³/h')
            try:
                default_unit_idx = units_list.index(default_unit)
            except Exception:
                default_unit_idx = 0
            Q_unit = st.selectbox("Flow unit", units_list, index=default_unit_idx)
            T = st.number_input("Fluid temperature (°C)", value=preset_defaults.get('T', 25.0))
            material_options = ['Water', 'Seawater', 'Acids', 'Alkaline', 'Slurry', 'Food-grade', 'Oil', 'More']
            default_mat = preset_defaults.get('material_type', 'Water')
            try:
                mat_idx = material_options.index(default_mat)
            except Exception:
                mat_idx = 0
            material_type = st.selectbox("Fluid type", material_options, index=mat_idx)
            SG = st.number_input("Specific gravity", value=1.0 if 'SG' not in preset_defaults else preset_defaults.get('SG', 1.0), min_value=0.01)
            mu_cP = st.number_input("Viscosity (cP)", value=preset_defaults.get('mu_cP', 1.0), min_value=0.01)
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
            D_inner = st.number_input("Pipe inner diameter (mm)", value=preset_defaults.get('D_inner', 100.0), min_value=1.0)
            L_pipe = st.number_input("Pipe length (m)", value=preset_defaults.get('L_pipe', 100.0), min_value=0.0)
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

            # If using immediate auto-update, treat as submitted; otherwise wait for Apply
            if auto_update:
                submitted = True
            else:
                submitted = st.form_submit_button("🚀 Calculate", type="primary")

    if submitted:
        try:
            progress = None
            # Basic input validation
            if Q_input <= 0:
                st.error("Flow rate must be > 0")
                st.stop()
            if D_inner <= 0:
                st.error("Pipe inner diameter must be > 0")
                st.stop()

            # Provide progress feedback for heavier calculations
            with st.spinner("Running hydraulic calculations..."):
                progress = st.progress(0)

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
                progress.progress(10)

                # Basic calculations
                mu = mu_cP / 1000.0
                D = D_inner / 1000.0
                V = velocity_from_flow(Q_m3s, D)
                progress.progress(30)

                Re = reynolds(density, V, D, mu)
                # Colebrook may be expensive for many calls; cache applied
                f = colebrook_f(Re, D, eps_mm/1000.0, method=colebrook_method, tol=1e-8, max_iter=100)
                progress.progress(55)

                hf = darcy_head_loss(f, L_pipe, D, V)
                hm = minor_loss_head(K_fittings, V)
                static_head = elevation_out - elevation_in
                total_head = static_head + hf + hm
                total_head_design = total_head * (1.0 + safety_margin_head)
                Q_design = Q_m3s * (1.0 + safety_margin_flow)
                progress.progress(75)

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
            
            # Generate installation specs early since we have shaft_kW
            estimated_shaft_length = math.sqrt(shaft_kW) * 0.3  # Approximate correlation
            installation_specs = generate_installation_specs_api610(
                shaft_kW,
                estimated_shaft_length * 3  # Approximate pump length
            )

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
            try:
                progress.progress(95)
            except Exception:
                pass
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

            # Store calculation context
            calc_context = {
                # Process
                'Q_design': Q_design,
                'Q_op': Q_op,
                'Q_bep': Q_bep,
                'total_head_design': total_head_design,
                'static_head': static_head,
                'V': V,
                'Re': Re,

                # Diagnostics
                'wear_rate': wear_rate if show_wear_analysis else None,
                'slurry_wear_mm_per_year': slurry_wear_mm_per_year if show_wear_analysis else None,
                'erosion_score': erosion_score if show_wear_analysis else None,
                'erosion_label': erosion_label if show_wear_analysis else None,
                'head_loss_pct': head_loss_pct if show_wear_analysis else None,
                'eff_loss_pct': eff_loss_pct if show_wear_analysis else None,
                'years_left': years_left if show_wear_analysis else None,
                'hours_left': hours_left if show_wear_analysis else None,
                'vibration_severity': vibration_severity,
                'pulsation_risk': pulsation_risk,
                'seal_life_hours': seal_life_hours,
                'health_score': health_score if show_wear_analysis else None,
                'pct_from_bep': pct_from_bep,
                'NPSHa': NPSHa,
                'NPSHr_vendor': NPSHr_vendor,
            }

            with tab2:
                st.subheader("Performance Curves")

                # Interactive system vs pump curve (Plotly)
                try:
                    flow_h = (Q_points * 3600).tolist()
                except Exception:
                    flow_h = list(Q_points * 3600)

                fig_sys = go.Figure()
                fig_sys.add_trace(go.Scatter(x=flow_h, y=list(H_system), mode='lines', name='System Curve', line=dict(color='blue')))
                fig_sys.add_trace(go.Scatter(x=flow_h, y=list(H_pump), mode='lines', name='Pump Curve', line=dict(color='red')))
                # BEP and operating point markers
                fig_sys.add_trace(go.Scatter(x=[Q_bep*3600], y=[H_pump[bep_idx]], mode='markers', name='BEP', marker=dict(color='green', size=12, symbol='star')))
                fig_sys.add_trace(go.Scatter(x=[Q_op*3600], y=[H_op], mode='markers', name='Operating Point', marker=dict(color='orange', size=10, symbol='diamond')))
                fig_sys.update_layout(title='Pump & System Curves', xaxis_title='Flow Rate (m³/h)', yaxis_title='Head (m)', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='left', x=0))
                fig_sys.update_xaxes(showgrid=True)
                fig_sys.update_yaxes(showgrid=True)
                st.plotly_chart(fig_sys, use_container_width=True)

                # Efficiency and Power vs Flow (Plotly with secondary y-axis)
                fig_perf = make_subplots(specs=[[{"secondary_y": True}]])
                fig_perf.add_trace(go.Scatter(x=flow_h, y=(eff_curve*100).tolist(), name='Efficiency (%)', line=dict(color='green')), secondary_y=False)
                fig_perf.add_trace(go.Scatter(x=flow_h, y=list(power_curve), name='Power (kW)', line=dict(color='red')), secondary_y=True)
                # Vertical line for BEP
                fig_perf.add_vline(x=Q_bep*3600, line=dict(color='green', dash='dash'), annotation_text='BEP', annotation_position='top')
                fig_perf.update_xaxes(title_text='Flow Rate (m³/h)')
                fig_perf.update_yaxes(title_text='Efficiency (%)', secondary_y=False, range=[0, 100])
                fig_perf.update_yaxes(title_text='Power (kW)', secondary_y=True)
                fig_perf.update_layout(title='Power & Efficiency vs Flow', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='left', x=0))
                st.plotly_chart(fig_perf, use_container_width=True)

                # Export curve data as CSV
                try:
                    df_export = pd.DataFrame({
                        'Flow (m³/h)': flow_h,
                        'System Head (m)': list(H_system),
                        'Pump Head (m)': list(H_pump),
                        'Efficiency (%)': (eff_curve*100).tolist(),
                        'Power (kW)': list(power_curve)
                    })
                    csv = df_export.to_csv(index=False)
                    st.download_button(label='📥 Download Curve Data (CSV)', data=csv, file_name=f'pump_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mime='text/csv')
                except Exception:
                    pass

            with tab3:
                st.subheader("⚙️ Advanced Analysis")
                
                # API 610 Analysis Section
                st.markdown("#### 📋 API 610 Compliance Analysis")
                
                # Calculate API 610 operating regions
                Q_por_min, Q_por_max = calculate_por(Q_bep)
                Q_aor_min, Q_aor_max = calculate_aor(Q_bep)
                mcsf = calculate_mcsf(Ns, Q_bep) * Q_bep
                
                # Operating regions analysis
                api610_col1, api610_col2 = st.columns(2)
                with api610_col1:
                    st.write("**Operating Regions:**")
                    st.write(f"- POR: {Q_por_min*3600:.1f} - {Q_por_max*3600:.1f} m³/h")
                    st.write(f"- AOR: {Q_aor_min*3600:.1f} - {Q_aor_max*3600:.1f} m³/h")
                    st.write(f"- MCSF: {mcsf*3600:.1f} m³/h")
                    
                    # Operating point analysis
                    if Q_op >= Q_por_min and Q_op <= Q_por_max:
                        st.success("✅ Operating point within POR")
                    elif Q_op >= Q_aor_min and Q_op <= Q_aor_max:
                        st.warning("⚠️ Operating point in AOR (outside POR)")
                    else:
                        st.error("❌ Operating point outside AOR")
                
                with api610_col2:
                    # Material class recommendation
                    pressure_bar = total_head_design * density * 9.81 / 1e5
                    material_class, material_type = api610_material_class(T, pressure_bar)
                    st.write("**Material Requirements:**")
                    st.write(f"- API 610 Class: {material_class}")
                    st.write(f"- Material Type: {material_type}")
                    st.write(f"- Based on: {T:.1f}°C, {pressure_bar:.1f} bar")

                # Testing requirements
                with st.expander("🔍 API 610 Testing Requirements", expanded=False):
                    test_reqs = api610_testing_requirements()
                    test_df = pd.DataFrame([
                        {"Test": k, "Required": v['required'], "Specification": v.get('pressure') or v.get('points') or v.get('limit') or v.get('duration') or v.get('comment')}
                        for k, v in test_reqs.items()
                    ])
                    st.dataframe(test_df, use_container_width=True, hide_index=True)

                # Nozzle loads
                with st.expander("💪 Allowable Nozzle Loads", expanded=False):
                    # Assuming discharge diameter is 80% of suction diameter for estimation
                    discharge_diameter = 0.8 * D_inner
                    nozzle_limits = calculate_nozzle_loads_api610(discharge_diameter)
                    nozzle_df = pd.DataFrame([
                        {"Force/Moment": k, "Allowable Value": f"{v:.1f} {'N' if k[0]=='F' else 'N·m'}"}
                        for k, v in nozzle_limits.items()
                    ])
                    st.dataframe(nozzle_df, use_container_width=True, hide_index=True)

                # Shaft and Rotor Analysis
                st.markdown("---")
                st.markdown("#### 🔄 Shaft & Rotor Analysis")

                def create_rotor_dynamics_plot(critical_speed, operating_speed, margins, shaft_data=None):
                    """Create a comprehensive visualization of rotor dynamics analysis."""
                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(2, 2)
                    
                    # Main response curve plot
                    ax1 = fig.add_subplot(gs[0, :])
                    
                    # Speed range for plotting
                    speeds = np.linspace(0, critical_speed * 1.5, 1000)
                    
                    # Enhanced amplitude response with multiple damping ratios
                    def amplitude_response(speed, critical, damping):
                        return 1 / np.sqrt((1 - (speed/critical)**2)**2 + (2*damping*(speed/critical))**2)
                    
                    # Plot response curves for different damping ratios
                    damping_ratios = [0.05, 0.1, 0.2]
                    colors = ['r--', 'b-', 'g--']
                    
                    for damping, color in zip(damping_ratios, colors):
                        response = amplitude_response(speeds, critical_speed, damping)
                        ax1.plot(speeds, response, color, 
                               label=f'Damping Ratio = {damping:.2f}',
                               alpha=0.7)
                    
                    # Mark critical and operating speeds
                    ax1.axvline(critical_speed, color='r', linestyle='--', 
                              label=f'First Critical: {critical_speed:.0f} RPM')
                    ax1.axvline(operating_speed, color='g', linestyle='--',
                              label=f'Operating: {operating_speed:.0f} RPM')
                    
                    # Add separation margins
                    margin_low = operating_speed * (1 - margins['below']/100)
                    margin_high = operating_speed * (1 + margins['above']/100)
                    ax1.axvspan(margin_low, margin_high, color='y', alpha=0.2, 
                              label='Operating Range')
                    
                    ax1.set_xlabel('Rotor Speed (RPM)')
                    ax1.set_ylabel('Relative Amplitude')
                    ax1.set_title('Rotor Dynamic Response Analysis')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='upper right', fontsize=8)
                    
                    # Add Campbell diagram
                    ax2 = fig.add_subplot(gs[1, 0])
                    speeds_campbell = np.linspace(0, critical_speed * 1.2, 100)
                    
                    # Excitation frequencies (1X, 2X, 0.5X)
                    ax2.plot(speeds_campbell, speeds_campbell, 'b-', label='1X')
                    ax2.plot(speeds_campbell, speeds_campbell*2, 'g--', label='2X')
                    ax2.plot(speeds_campbell, speeds_campbell*0.5, 'r--', label='0.5X')
                    
                    # Add natural frequency line
                    ax2.axhline(y=critical_speed, color='r', linestyle='-.',
                              label='First Critical')
                    
                    ax2.set_xlabel('Rotor Speed (RPM)')
                    ax2.set_ylabel('Frequency (RPM)')
                    ax2.set_title('Campbell Diagram')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper left', fontsize=8)
                    
                    # Add API compliance analysis
                    ax3 = fig.add_subplot(gs[1, 1])
                    
                    # API 610 requirements
                    api_requirements = {
                        'Sep. Margin': margins['above'],
                        'Damping': 0.1,
                        'Mode Shape': 0.9,
                        'Stability': 0.85
                    }
                    
                    compliance_scores = list(api_requirements.values())
                    requirement_labels = list(api_requirements.keys())
                    
                    compliance_colors = ['g' if score >= 0.8 else 'r' for score in compliance_scores]
                    bars = ax3.bar(requirement_labels, compliance_scores, color=compliance_colors)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height*100:.0f}%',
                                ha='center', va='bottom')
                    
                    ax3.set_ylim(0, 1.2)
                    ax3.set_title('API 610 Compliance')
                    ax3.grid(True, alpha=0.3)
                    
                    # Add overall assessment
                    overall_score = np.mean(compliance_scores)
                    assessment = ('✅ Compliant' if overall_score >= 0.8 
                                else '⚠️ Marginal' if overall_score >= 0.6 
                                else '❌ Non-compliant')
                    ax3.text(0.5, -0.1, f'Overall: {assessment}',
                            ha='center', va='center', transform=ax3.transAxes)
                    
                    plt.tight_layout()
                    return fig
                    
                    # Mark operating speed
                    ax.axvline(operating_speed, color='g', linestyle='-', label='Operating Speed')
                    
                    # Mark separation margins
                    ax.axvspan(operating_speed * 0.8, operating_speed * 1.2, 
                             alpha=0.2, color='g', label='Operating Range')
                    
                    ax.set_xlabel('Speed (RPM)')
                    ax.set_ylabel('Relative Amplitude')
                    ax.set_title('Rotor Dynamic Response')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper left')
                    
                    return fig

                def create_shaft_deflection_diagram(length_mm, diameter_mm, deflection_m):
                    """Create a visual representation of shaft deflection."""
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Create shaft outline
                    x = np.linspace(0, length_mm, 100)
                    
                    # Simplified deflection curve (parabolic)
                    def deflection_curve(x, max_deflection):
                        return -4 * max_deflection * (x/length_mm) * (1 - x/length_mm)
                    
                    y_deflection = deflection_curve(x, deflection_m * 1000)  # Convert to mm
                    
                    # Plot shaft outline
                    ax.fill_between(x, -diameter_mm/2, diameter_mm/2, color='gray', alpha=0.3, label='Shaft')
                    
                    # Plot deflection curve
                    ax.plot(x, y_deflection, 'r--', linewidth=2, label='Deflection (exaggerated)')
                    
                    # Add annotations
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax.set_xlabel('Shaft Length (mm)')
                    ax.set_ylabel('Deflection (mm)')
                    ax.set_title('Shaft Deflection Diagram')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Set reasonable y-limits to show deflection clearly
                    max_defl = max(abs(y_deflection))
                    ax.set_ylim(-max(diameter_mm/2, max_defl*2), max(diameter_mm/2, max_defl*2))
                    
                    return fig
                
                # Estimate shaft parameters based on power
                estimated_shaft_diameter = max(25, math.sqrt(shaft_kW) * 10)  # mm
                estimated_shaft_length = D_inner * 4  # mm
                estimated_impeller_mass = shaft_kW * 0.5  # kg
                
                shaft_col1, shaft_col2 = st.columns(2)
                with shaft_col1:
                    st.write("**Shaft Parameters:**")
                    st.write(f"- Estimated diameter: {estimated_shaft_diameter:.1f} mm")
                    st.write(f"- Estimated length: {estimated_shaft_length:.1f} mm")
                    st.write(f"- Est. impeller mass: {estimated_impeller_mass:.1f} kg")
                    
                    # Calculate radial load (simplified)
                    radial_load = (shaft_kW * 1000 * 0.5) / (pump_speed_rpm * math.pi / 30)  # N
                    axial_load = radial_load * 0.5  # Estimated
                    
                    # Shaft deflection analysis
                    deflection_results = calculate_shaft_deflection_api610(
                        estimated_shaft_length, 
                        estimated_shaft_diameter,
                        radial_load
                    )
                    
                    st.write("**Shaft Deflection:**")
                    deflection_mm = deflection_results['deflection_m'] * 1000
                    st.write(f"- Calculated: {deflection_mm:.3f} mm")
                    if deflection_results['seal_compliant']:
                        st.success("✅ Meets API 610 seal limit")
                    else:
                        st.error("❌ Exceeds API 610 seal limit")

                with shaft_col2:
                    # Rotor dynamics
                    rotor_dynamics = calculate_rotor_dynamics_api610(
                        pump_speed_rpm, 
                        estimated_impeller_mass,
                        estimated_shaft_length, 
                        estimated_shaft_diameter
                    )
                    
                    st.write("**Critical Speed Analysis:**")
                    st.write(f"- First critical: {rotor_dynamics['first_critical_speed']:.0f} RPM")
                    st.write(f"- Margin above: {rotor_dynamics['margin_above']:.1f}%")
                    if rotor_dynamics['margin_above'] >= rotor_dynamics['api_required_margin']:
                        st.success("✅ Meets API 610 separation margin")
                    else:
                        st.error("❌ Insufficient separation margin")

                # Bearing Analysis
                st.markdown("---")
                st.markdown("#### 🛠️ Bearing Analysis")

                def create_bearing_life_plot(L10_hours, api_minimum, operating_hours, speed_rpm=None, load_radial=None, load_axial=None):
                    """Create a comprehensive visual representation of bearing life analysis."""
                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(2, 2)
                    
                    # Reliability curve plot (main plot)
                    ax1 = fig.add_subplot(gs[0, :])
                    time_points = np.linspace(0, max(L10_hours * 1.2, api_minimum * 1.2), 100)
                    
                    def reliability_curve(t, L10):
                        return np.exp(-(t/L10)**1.5)  # 1.5 is typical Weibull shape parameter for bearings
                    
                    reliability = reliability_curve(time_points, L10_hours)
                    ax1.plot(time_points/1000, reliability*100, 'b-', linewidth=2, label='Reliability Curve')
                    
                    # Add reference lines
                    ax1.axvline(api_minimum/1000, color='r', linestyle='--', label='API Minimum')
                    ax1.axvline(L10_hours/1000, color='g', linestyle='--', label='L10 Life')
                    if operating_hours > 0:
                        ax1.axvline(operating_hours/1000, color='orange', linestyle='--', label='Operating Hours')
                    
                    ax1.set_xlabel('Thousands of Hours')
                    ax1.set_ylabel('Reliability (%)')
                    ax1.set_title('Bearing Reliability Analysis')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    # Load distribution plot (if available)
                    if load_radial is not None and load_axial is not None:
                        ax2 = fig.add_subplot(gs[1, 0])
                        loads = [load_radial, load_axial]
                        load_labels = ['Radial', 'Axial']
                        
                        wedges, texts, autotexts = ax2.pie(loads, labels=load_labels, 
                                                         autopct='%1.1f%%',
                                                         colors=['lightblue', 'lightgreen'])
                        plt.setp(autotexts, size=8, weight="bold")
                        ax2.set_title('Load Distribution')
                        
                        # Calculate and display load ratio
                        if load_radial > 0:
                            load_ratio = load_axial / load_radial
                            ax2.text(0, -1.2, f'Load Ratio (A/R): {load_ratio:.2f}',
                                   ha='center', va='center')
                    
                    # Speed impact analysis (if speed available)
                    if speed_rpm is not None:
                        ax3 = fig.add_subplot(gs[1, 1])
                        speeds = np.linspace(0.7*speed_rpm, 1.3*speed_rpm, 100)
                        # Simplified life calculation model
                        lives = L10_hours * (speed_rpm/speeds)**3
                        
                        ax3.plot(speeds, lives/1000, 'b-')
                        ax3.plot(speed_rpm, L10_hours/1000, 'ro', label='Operating Point')
                        ax3.set_xlabel('Speed (RPM)')
                        ax3.set_ylabel('Life (thousands of hours)')
                        ax3.set_title('Speed Impact on Bearing Life')
                        ax3.grid(True, alpha=0.3)
                        
                        # Add critical reference lines
                        ax3.axhline(y=api_minimum/1000, color='r', linestyle='--', 
                                  alpha=0.5, label='API Minimum')
                        if operating_hours > 0:
                            ax3.axhline(y=operating_hours/1000, color='orange', 
                                      linestyle='--', alpha=0.5, label='Operating Hours')
                        ax3.legend(loc='best', fontsize=8)
                    
                    plt.tight_layout()
                    return fig
                    if operating_hours > 0:
                        ax.axvline(operating_hours/1000, color='y', linestyle='--', label='Annual Operation')
                    
                    ax.set_xlabel('Operating Time (thousands of hours)')
                    ax.set_ylabel('Reliability (%)')
                    ax.set_title('Bearing Life Prediction')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    return fig

                def create_temperature_distribution(bearing_temp_rise, ambient_temp):
                    """Create a temperature distribution visualization."""
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Create position points
                    positions = np.linspace(0, 10, 100)
                    
                    # Create temperature distribution (simplified)
                    def temp_distribution(x):
                        return ambient_temp + bearing_temp_rise * np.exp(-((x-5)**2)/8)
                    
                    temps = temp_distribution(positions)
                    
                    # Create colormap
                    norm = plt.Normalize(ambient_temp, max(temps))
                    cmap = plt.cm.RdYlBu_r
                    
                    # Plot temperature distribution
                    points = ax.scatter(positions, temps, c=temps, cmap=cmap, norm=norm)
                    ax.plot(positions, temps, 'k-', alpha=0.3)
                    
                    # Add colorbar
                    plt.colorbar(points, label='Temperature (°C)')
                    
                    ax.set_xlabel('Bearing Housing Position')
                    ax.set_ylabel('Temperature (°C)')
                    ax.set_title('Temperature Distribution')
                    ax.grid(True, alpha=0.3)
                    
                    return fig
                bearing_col1, bearing_col2 = st.columns(2)
                with bearing_col1:
                    # Calculate bearing life
                    bearing_results = calculate_bearing_life_api610_detailed(
                        radial_load, axial_load, pump_speed_rpm, 'ball'
                    )
                    
                    st.write("**Bearing Life Calculation:**")
                    st.write(f"- L10 life: {bearing_results['L10_hours']:.0f} hours")
                    st.write(f"- API minimum: {bearing_results['api_minimum']} hours")
                    st.write(f"- Load ratio: {bearing_results['load_ratio']:.2f}")
                    
                    if bearing_results['compliant']:
                        st.success("✅ Meets API 610 minimum life requirement")
                    else:
                        st.error("❌ Below API 610 minimum life requirement")

                with bearing_col2:
                    # Alternative bearing type analysis
                    roller_results = calculate_bearing_life_api610_detailed(
                        radial_load, axial_load, pump_speed_rpm, 'roller'
                    )
                    
                    st.write("**Alternative Configuration:**")
                    st.write("Roller bearing analysis:")
                    st.write(f"- L10 life: {roller_results['L10_hours']:.0f} hours")
                    st.write(f"- Load ratio: {roller_results['load_ratio']:.2f}")

                # Seal Chamber Analysis
                st.markdown("---")
                st.markdown("#### 🔐 Seal Chamber Design")

                def create_seal_chamber_diagram(chamber_dims, operating_conditions=None):
                    """Create a comprehensive visualization of the seal chamber with analysis."""
                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(2, 2)
                    
                    # Main chamber diagram
                    ax1 = fig.add_subplot(gs[0, :])
                    
                    # Chamber dimensions
                    bore_d = chamber_dims['bore_diameter']
                    depth = chamber_dims['depth']
                    rabbet_d = chamber_dims['rabbet_diameter']
                    shaft_d = chamber_dims.get('shaft_diameter', bore_d * 0.6)  # Estimated if not provided
                    
                    # Calculate important ratios and clearances
                    radial_clearance = (bore_d - shaft_d) / 2
                    depth_ratio = depth / bore_d  # L/D ratio
                    chamber_volume = np.pi * (bore_d/2)**2 * depth
                    
                    # Draw the main chamber diagram
                    def draw_chamber():
                        # Background grid for reference
                        ax1.grid(True, linestyle='--', alpha=0.3)
                        
                        # Outer chamber (rabbet)
                        ax1.add_patch(plt.Rectangle((-rabbet_d/2, 0), rabbet_d, depth, 
                                                  fill=False, color='blue', linewidth=2))
                        # Inner bore
                        ax1.add_patch(plt.Rectangle((-bore_d/2, 0), bore_d, depth,
                                                  fill=True, color='lightgray', alpha=0.3))
                        # Shaft
                        ax1.add_patch(plt.Rectangle((-shaft_d/2, -depth*0.2), shaft_d, depth*1.4,
                                                  fill=True, color='darkgray'))
                        
                        # Flow paths (indicative)
                        arrow_props = dict(arrowstyle='->', color='red', linestyle='--', alpha=0.5)
                        ax1.annotate('', xy=(-bore_d/3, depth*0.8), xytext=(-bore_d/3, depth*0.2),
                                   arrowprops=arrow_props)
                        ax1.annotate('', xy=(bore_d/3, depth*0.2), xytext=(bore_d/3, depth*0.8),
                                   arrowprops=arrow_props)
                        
                        # Add dimensions and annotations
                        def add_dimension(start, end, text, yoffset, color='black'):
                            ax1.annotate('', xy=start, xytext=end,
                                       arrowprops=dict(arrowstyle='<->', color=color))
                            ax1.text((start[0] + end[0])/2, yoffset, text,
                                   ha='center', va='bottom', color=color)
                        
                        # Bore diameter
                        add_dimension((-bore_d/2, depth*1.1), (bore_d/2, depth*1.1),
                                    f'Bore: {bore_d:.1f}mm', depth*1.15, 'blue')
                        # Shaft diameter
                        add_dimension((-shaft_d/2, -depth*0.1), (shaft_d/2, -depth*0.1),
                                    f'Shaft: {shaft_d:.1f}mm', -depth*0.15, 'darkgray')
                        # Depth
                        add_dimension((rabbet_d*0.6, 0), (rabbet_d*0.6, depth),
                                    f'Depth: {depth:.1f}mm', depth/2, 'green')
                        
                        # Design analysis annotations
                        analysis_text = (
                            f'Design Analysis:\n'
                            f'• Radial Clearance: {radial_clearance:.2f}mm\n'
                            f'• L/D Ratio: {depth_ratio:.2f}\n'
                            f'• Chamber Volume: {chamber_volume/1000:.1f}L'
                        )
                        ax1.text(rabbet_d*0.7, depth*0.5, analysis_text,
                               bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                               ha='left', va='center', fontsize=8)
                    
                    # Draw the chamber
                    draw_chamber()
                    
                    # Set up the main diagram
                    ax1.set_aspect('equal')
                    ax1.set_xlim(-rabbet_d*0.8, rabbet_d*1.2)
                    ax1.set_ylim(-depth*0.2, depth*1.2)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_title('API 610 Seal Chamber Design')
                    
                    # Add design compliance indicators
                    design_scores = {
                        'Clearance Ratio': min((radial_clearance/(bore_d/2))/0.008, 1),  # API recommends 0.008
                        'L/D Ratio': min(depth_ratio/1.8, 1),  # API typical range 1.5-2.0
                        'Volume': min(chamber_volume/(np.pi * (bore_d/2)**2 * bore_d), 1)  # Relative to basic cylinder
                    }
                    
                    # Create compliance gauge chart
                    ax2 = fig.add_subplot(gs[1, 0], projection='polar')
                    angles = np.linspace(0, 2*np.pi, len(design_scores), endpoint=False)
                    values = list(design_scores.values())
                    values.append(values[0])
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    ax2.plot(angles, values, 'o-', linewidth=2)
                    ax2.fill(angles, values, alpha=0.25)
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(list(design_scores.keys()), size=8)
                    ax2.set_ylim(0, 1)
                    ax2.set_title('Design Compliance', pad=15)
                    
                    # Add operating conditions analysis if provided
                    if operating_conditions:
                        ax3 = fig.add_subplot(gs[1, 1])
                        
                        # Process operating conditions
                        temp = operating_conditions.get('temperature', 25)
                        pressure = operating_conditions.get('pressure', 1)
                        speed = operating_conditions.get('speed', 0)
                        
                        # Create operating envelope plot
                        x = np.linspace(0, max(pressure*1.5, 20), 100)
                        y = np.linspace(0, max(temp*1.5, 150), 100)
                        X, Y = np.meshgrid(x, y)
                        
                        # Simple severity score (example)
                        Z = (X/20)**2 + (Y/150)**2  # Normalized severity
                        
                        contour = ax3.contourf(X, Y, Z, levels=np.linspace(0, 2, 20),
                                             cmap='RdYlGn_r')
                        ax3.plot(pressure, temp, 'ro', label='Operating Point')
                        
                        ax3.set_xlabel('Pressure (bar)')
                        ax3.set_ylabel('Temperature (°C)')
                        ax3.set_title('Operating Envelope')
                        plt.colorbar(contour, ax=ax3, label='Severity')
                        ax3.legend()
                    
                    plt.tight_layout()
                    return fig

                def create_seal_flush_diagram(api_plan):
                    """Create a schematic of the seal flush plan."""
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Basic pump outline
                    def draw_pump_outline():
                        # Pump casing
                        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='black'))
                        # Shaft
                        ax.add_patch(plt.Rectangle((-1.5, -0.1), 3, 0.2, color='gray'))
                        # Seal chamber
                        ax.add_patch(plt.Rectangle((-0.3, -0.3), 0.6, 0.6, fill=False, color='blue'))
                    
                    draw_pump_outline()
                    
                    # Add flush plan specific features
                    if '11' in api_plan:
                        # Plan 11 recirculation
                        ax.arrow(0.3, 0, 0.5, 0.5, head_width=0.1, color='red')
                        ax.arrow(0.8, 0.5, -0.5, -0.5, head_width=0.1, color='blue')
                        ax.text(0.9, 0.6, 'Plan 11\nRecirculation', fontsize=8)
                    elif '32' in api_plan:
                        # Plan 32 external flush
                        ax.arrow(-1, 0.5, 0.7, -0.5, head_width=0.1, color='green')
                        ax.text(-1.2, 0.6, 'Plan 32\nExternal Flush', fontsize=8)
                    
                    # Set equal aspect ratio and limits
                    ax.set_aspect('equal')
                    ax.set_xlim(-2, 2)
                    ax.set_ylim(-1.5, 1.5)
                    
                    # Remove axes for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    ax.set_title(f'API Flush Plan {api_plan}')
                    return fig
                seal_col1, seal_col2 = st.columns(2)
                with seal_col1:
                    # Calculate seal chamber dimensions
                    seal_chamber = calculate_seal_chamber_api610(estimated_shaft_diameter)
                    
                    st.write("**API 610 Seal Chamber Dimensions:**")
                    st.write(f"- Bore diameter: {seal_chamber['bore_diameter']:.1f} mm")
                    st.write(f"- Chamber depth: {seal_chamber['depth']:.1f} mm")
                    st.write(f"- Rabbet diameter: {seal_chamber['rabbet_diameter']:.1f} mm")
                    st.write(f"- Surface finish: {seal_chamber['surface_finish']} μm Ra")

                with seal_col2:
                    st.write("**Seal Configuration:**")
                    st.write(f"- Cooling required: {'Yes' if seal_chamber['cooling_required'] else 'No'}")
                    st.write("Recommended flush plans:")
                    for plan in seal_chamber['recommended_plans']:
                        st.write(f"- {plan}")

                # Baseplate Design
                st.markdown("---")
                st.markdown("#### 🏗️ Baseplate Design")
                base_col1, base_col2 = st.columns(2)
                with base_col1:
                    # Calculate baseplate requirements
                    baseplate = calculate_baseplate_requirements_api610(
                        shaft_kW,
                        estimated_shaft_length * 3  # Approximate pump length
                    )
                    
                    st.write("**API 610 Baseplate Requirements:**")
                    st.write(f"- Min. thickness: {baseplate['min_thickness_mm']} mm")
                    st.write(f"- Recommended: {baseplate['recommended_thickness_mm']} mm")
                    st.write(f"- Grout thickness: {baseplate['grout_thickness_mm']} mm")
                    st.write(f"- Max deflection: {baseplate['max_deflection_mm']} mm")

                with base_col2:
                    st.write("**Installation Requirements:**")
                    st.write(f"- Max deviation: {baseplate['leveling_requirements']['max_deviation_mm_per_m']} mm/m")
                    st.write(f"- Flatness: {baseplate['leveling_requirements']['flatness_requirement_mm']} mm")
                    st.write("Anchor bolt specification:")
                    st.write(f"- Diameter: {baseplate['anchor_bolt_spec']['min_diameter_mm']} mm")
                    st.write(f"- Material: {baseplate['anchor_bolt_spec']['material']}")

                # Mechanical Seal Selection
                st.markdown("---")
                st.markdown("#### 🔒 Mechanical Seal Selection")
                
                # Get operating conditions for seal selection
                pressure_bar = total_head_design * density * 9.81 / 1e5
                fluid_properties = {
                    'abrasive': material_type == 'Slurry',
                    'volatile': material_type in ['Oil', 'Acids'],
                    'toxic': material_type in ['Acids', 'Alkaline'],
                    'crystallizing': False
                }
                
                seal_recommendation = select_mechanical_seal_api610(
                    T, pressure_bar, material_type, 
                    pump_speed_rpm, fluid_properties
                )
                
                seal_col1, seal_col2 = st.columns(2)
                with seal_col1:
                    st.write("**Seal Configuration:**")
                    st.write(f"- Arrangement: {seal_recommendation['arrangement']}")
                    st.write(f"- Seal Type: {seal_recommendation['seal_type']}")
                    st.write(f"- Face Materials: {seal_recommendation['face_material']}")
                    st.write(f"- Elastomer: {seal_recommendation['elastomer']}")
                
                with seal_col2:
                    st.write("**API Piping Plan:**")
                    st.write(f"- Plan: {seal_recommendation['api_plan']}")
                    if seal_recommendation['contains_solids']:
                        st.warning("⚠️ Solids handling features required")
                    if seal_recommendation['arrangement'] == 'Dual pressurized':
                        st.info("ℹ️ Barrier fluid system required")
                    
                    # Add seal chamber visualization
                    chamber_fig = create_seal_chamber_diagram(seal_chamber)
                    st.pyplot(chamber_fig)
                    plt.close()
                    
                    # Add flush plan visualization
                    flush_fig = create_seal_flush_diagram(seal_recommendation['api_plan'])
                    st.pyplot(flush_fig)
                    plt.close()
                    
                    # Add seal arrangement details
                    st.write("\n**Seal Arrangement Details:**")
                    seal_details = pd.DataFrame({
                        'Parameter': [
                            'Arrangement Type',
                            'Face Material',
                            'Secondary Seals',
                            'Face Pressure',
                            'Face Loading',
                            'Heat Generation'
                        ],
                        'Specification': [
                            seal_recommendation['arrangement'],
                            seal_recommendation['face_material'],
                            seal_recommendation['elastomer'],
                            f"{pressure_bar:.1f} bar",
                            'Hydraulically Balanced' if pressure_bar > 20 else 'Standard',
                            f"{(shaft_kW * 0.02):.1f} kW (estimated)"
                        ]
                    })
                    st.dataframe(seal_details, hide_index=True)

                # Bearing Housing Cooling
                st.markdown("---")
                st.markdown("#### ❄️ Bearing Housing Cooling")
                
                # Estimate bearing temperature rise based on speed
                bearing_temp_rise = (pump_speed_rpm/1450)**1.2 * 20
                cooling_requirements = calculate_bearing_cooling_api610(
                    pump_speed_rpm, bearing_temp_rise
                )
                
                cooling_col1, cooling_col2 = st.columns(2)
                with cooling_col1:
                    st.write("**Temperature Analysis:**")
                    st.write(f"- Estimated temp rise: {bearing_temp_rise:.1f}°C")
                    st.write(f"- Heat load: {cooling_requirements['heat_load']:.0f} W")
                    if cooling_requirements['cooling_required']:
                        st.warning("⚠️ Cooling system required")
                    else:
                        st.success("✅ Natural cooling sufficient")
                
                with cooling_col2:
                    if cooling_requirements['cooling_required']:
                        st.write("**Cooling System:**")
                        st.write(f"- Method: {cooling_requirements['method']}")
                        st.write(f"- Required flow: {cooling_requirements['flow_rate']:.2f} "
                                f"{'L/min' if cooling_requirements['method']=='Water cooling' else 'm³/min'}")
                        
                    # Add temperature distribution visualization
                    temp_fig = create_temperature_distribution(bearing_temp_rise, 40)  # 40°C ambient
                    st.pyplot(temp_fig)
                    plt.close()
                
                # Add bearing life visualization
                op_hours_for_plot = operating_hours if show_energy_cost else 8000.0
                bearing_life_fig = create_bearing_life_plot(
                    bearing_results['L10_hours'],
                    bearing_results['api_minimum'],
                    op_hours_for_plot
                )
                st.pyplot(bearing_life_fig)
                plt.close()

                # Material Upgrades
                st.markdown("---")
                st.markdown("#### 🛡️ Material Upgrade Recommendations")
                
                # Add visualization for material improvements
                def create_interactive_material_analysis(current_class, recommended_class, improvements):
                    """Create interactive material analysis with multiple visualization options."""
                    
                    # Material properties for different API classes
                    material_props = {
                        'I-1': {
                            'max_temp': 150,
                            'max_pressure': 20,
                            'corrosion_resistance': 2,
                            'erosion_resistance': 2,
                            'relative_cost': 1
                        },
                        'I-2': {
                            'max_temp': 200,
                            'max_pressure': 40,
                            'corrosion_resistance': 3,
                            'erosion_resistance': 3,
                            'relative_cost': 1.5
                        },
                        'S-1': {
                            'max_temp': 350,
                            'max_pressure': 100,
                            'corrosion_resistance': 4,
                            'erosion_resistance': 4,
                            'relative_cost': 2.5
                        },
                        'S-6': {
                            'max_temp': 400,
                            'max_pressure': 150,
                            'corrosion_resistance': 5,
                            'erosion_resistance': 5,
                            'relative_cost': 4
                        }
                    }
                    
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Performance Comparison", "Cost-Benefit Analysis", "Operating Limits"])
                    
                    with viz_tabs[0]:
                        # Radar chart for performance comparison
                        fig_radar = plt.figure(figsize=(8, 6))
                        ax_radar = fig_radar.add_subplot(111, projection='polar')
                        
                        # Parameters to compare
                        params = ['Temperature', 'Pressure', 'Corrosion', 'Erosion', 'Cost']
                        angles = np.linspace(0, 2*np.pi, len(params), endpoint=False)
                        
                        # Plot for current material
                        current_values = [
                            material_props[current_class]['max_temp']/400,
                            material_props[current_class]['max_pressure']/150,
                            material_props[current_class]['corrosion_resistance']/5,
                            material_props[current_class]['erosion_resistance']/5,
                            1/material_props[current_class]['relative_cost']
                        ]
                        current_values.append(current_values[0])
                        angles_plot = np.concatenate((angles, [angles[0]]))
                        
                        ax_radar.plot(angles_plot, current_values, 'b-', label=f'Current ({current_class})')
                        ax_radar.fill(angles_plot, current_values, alpha=0.25)
                        
                        # Plot for recommended material if different
                        if recommended_class and recommended_class != current_class:
                            recommended_values = [
                                material_props[recommended_class]['max_temp']/400,
                                material_props[recommended_class]['max_pressure']/150,
                                material_props[recommended_class]['corrosion_resistance']/5,
                                material_props[recommended_class]['erosion_resistance']/5,
                                1/material_props[recommended_class]['relative_cost']
                            ]
                            recommended_values.append(recommended_values[0])
                            ax_radar.plot(angles_plot, recommended_values, 'r-', label=f'Recommended ({recommended_class})')
                            ax_radar.fill(angles_plot, recommended_values, alpha=0.25)
                        
                        ax_radar.set_xticks(angles)
                        ax_radar.set_xticklabels(params)
                        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        
                        st.pyplot(fig_radar)
                        plt.close()

                    with viz_tabs[1]:
                        # Cost-benefit analysis
                        if recommended_class and recommended_class != current_class:
                            cost_increase = material_props[recommended_class]['relative_cost'] / material_props[current_class]['relative_cost']
                            performance_increase = (
                                material_props[recommended_class]['corrosion_resistance'] +
                                material_props[recommended_class]['erosion_resistance']
                            ) / (
                                material_props[current_class]['corrosion_resistance'] +
                                material_props[current_class]['erosion_resistance']
                            )
                            
                            fig_cost = plt.figure(figsize=(8, 4))
                            ax_cost = fig_cost.add_subplot(111)
                            
                            x = np.array([1, 2])
                            metrics = {
                                'Cost': [1, cost_increase],
                                'Life': [1, improvements],
                                'Performance': [1, performance_increase]
                            }
                            
                            width = 0.25
                            for i, (metric, values) in enumerate(metrics.items()):
                                ax_cost.bar(x + i*width - width, values, width, label=metric)
                            
                            ax_cost.set_xticks([1, 2])
                            ax_cost.set_xticklabels(['Current', 'Recommended'])
                            ax_cost.legend()
                            ax_cost.set_ylabel('Relative Value')
                            ax_cost.grid(True, alpha=0.3)
                            
                            st.pyplot(fig_cost)
                            plt.close()
                            
                            # ROI Calculator
                            st.write("**ROI Calculator**")
                            initial_cost = st.number_input("Initial Equipment Cost (₹)", value=100000)
                            yearly_maintenance = st.number_input("Yearly Maintenance Cost (₹)", value=10000)
                            
                            current_life = 10  # years
                            improved_life = current_life * improvements
                            
                            roi_data = pd.DataFrame({
                                'Metric': ['Equipment Cost (₹)', 'Annual Maintenance (₹)', 'Expected Life (years)', 'Lifetime Cost (₹)'],
                                'Current': [
                                    initial_cost,
                                    yearly_maintenance,
                                    current_life,
                                    initial_cost + yearly_maintenance * current_life
                                ],
                                'Recommended': [
                                    initial_cost * cost_increase,
                                    yearly_maintenance / performance_increase,
                                    improved_life,
                                    initial_cost * cost_increase + (yearly_maintenance / performance_increase) * improved_life
                                ]
                            })
                            st.dataframe(roi_data, hide_index=True)

                    with viz_tabs[2]:
                        # Operating limits visualization
                        fig_limits = plt.figure(figsize=(8, 6))
                        ax_limits = fig_limits.add_subplot(111)
                        
                        for mat_class, props in material_props.items():
                            ax_limits.add_patch(plt.Rectangle(
                                (0, 0),
                                props['max_pressure'],
                                props['max_temp'],
                                alpha=0.3,
                                label=f'Class {mat_class}'
                            ))
                        
                        # Mark operating point
                        pressure_bar = st.session_state.get('pressure_bar', 50)
                        temp_c = st.session_state.get('T', 100)
                        ax_limits.plot(pressure_bar, temp_c, 'ro', label='Operating Point')
                        
                        ax_limits.set_xlabel('Pressure (bar)')
                        ax_limits.set_ylabel('Temperature (°C)')
                        ax_limits.set_title('Operating Limits by Material Class')
                        ax_limits.grid(True, alpha=0.3)
                        ax_limits.legend()
                        
                        st.pyplot(fig_limits)
                        plt.close()
                        
                        # Safety margins
                        current_margin_temp = (material_props[current_class]['max_temp'] - temp_c) / material_props[current_class]['max_temp'] * 100
                        current_margin_pressure = (material_props[current_class]['max_pressure'] - pressure_bar) / material_props[current_class]['max_pressure'] * 100
                        
                        st.write("**Safety Margins:**")
                        margins_data = pd.DataFrame({
                            'Parameter': ['Temperature', 'Pressure'],
                            'Current Margin (%)': [f"{current_margin_temp:.1f}%", f"{current_margin_pressure:.1f}%"],
                            'Status': [
                                '✅ Adequate' if current_margin_temp >= 20 else '⚠️ Marginal' if current_margin_temp > 0 else '❌ Exceeded',
                                '✅ Adequate' if current_margin_pressure >= 20 else '⚠️ Marginal' if current_margin_pressure > 0 else '❌ Exceeded'
                            ]
                        })
                        st.dataframe(margins_data, hide_index=True)
                    
                    return None  # Since we're using st.pyplot directly

                def create_material_comparison_chart(current_class, recommended_class, improvements):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    properties = ['Corrosion', 'Erosion', 'Temperature', 'Pressure']
                    current_values = [0.4, 0.4, 0.5, 0.6]  # Base material properties
                    
                    if recommended_class:
                        # Enhanced properties based on recommended class
                        recommended_values = [
                            0.8 if 'S' in recommended_class else 0.6,
                            0.9 if 'S-6' in recommended_class else 0.7,
                            0.9 if 'S' in recommended_class else 0.7,
                            0.9 if 'S' in recommended_class else 0.8
                        ]
                        
                        x = np.arange(len(properties))
                        width = 0.35
                        
                        ax.bar(x - width/2, current_values, width, label=f'Current ({current_class})',
                              color='lightgray', alpha=0.7)
                        ax.bar(x + width/2, recommended_values, width, label=f'Recommended ({recommended_class})',
                              color='green', alpha=0.7)
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(properties)
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Relative Performance')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        return fig
                    return None
                
                # Determine if service is corrosive/erosive
                is_corrosive = material_type in ['Acids', 'Alkaline', 'Seawater']
                is_erosive = material_type == 'Slurry' or V > 3.0
                
                material_upgrades = recommend_material_upgrades_api610(
                    'Carbon Steel',  # Base material
                    T, pressure_bar,
                    corrosive=is_corrosive,
                    erosive=is_erosive
                )
                
                material_col1, material_col2 = st.columns(2)
                with material_col1:
                    st.write("**Material Analysis:**")
                    st.write(f"- Current class: {material_upgrades['current_class']}")
                    if material_upgrades['recommended_class']:
                        st.write(f"- Recommended: {material_upgrades['recommended_class']}")
                        st.write("Upgrade reasons:")
                        for reason in material_upgrades['reason']:
                            st.write(f"  • {reason}")
                    else:
                        st.success("✅ Current material class is suitable")
                
                with material_col2:
                    st.write("**Component Recommendations:**")
                    for component, rec in material_upgrades['specific_recommendations'].items():
                        st.write(f"- {component.title()}: {rec}")
                    if material_upgrades['estimated_life_improvement'] > 1.0:
                        st.info(f"🔄 Estimated life improvement: "
                               f"{material_upgrades['estimated_life_improvement']:.1f}x")
                
                # Material comparison visualization
                if material_upgrades['recommended_class']:
                    comparison_fig = create_material_comparison_chart(
                        material_upgrades['current_class'],
                        material_upgrades['recommended_class'],
                        material_upgrades['estimated_life_improvement']
                    )
                    if comparison_fig:
                        st.pyplot(comparison_fig)
                        plt.close()
                
                # Export capabilities for API 610 specifications
                st.markdown("---")
                st.markdown("#### 📤 Export API 610 Specifications")
                
                # Prepare export data
                api610_export_data = {
                    "General Information": {
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Project": "Pump Specification",
                        "Service": material_type
                    },
                    "Operating Conditions": {
                        "Flow Rate (m³/h)": Q_design * 3600,
                        "Total Head (m)": total_head_design,
                        "Temperature (°C)": T,
                        "Pressure (bar)": pressure_bar,
                        "Speed (RPM)": pump_speed_rpm
                    },
                    "Material Specifications": {
                        "Current Class": material_upgrades['current_class'],
                        "Recommended Class": material_upgrades['recommended_class'],
                        "Upgrade Reasons": material_upgrades['reason'],
                        "Component Recommendations": material_upgrades['specific_recommendations']
                    },
                    "Mechanical Seal": {
                        "Arrangement": seal_recommendation['arrangement'],
                        "Type": seal_recommendation['seal_type'],
                        "Face Materials": seal_recommendation['face_material'],
                        "API Plan": seal_recommendation['api_plan']
                    },
                    "Installation Specifications": {
                        "Foundation": installation_specs['foundation'],
                        "Grouting": installation_specs['grouting'],
                        "Alignment": installation_specs['alignment'],
                        "Piping Support": installation_specs['piping']
                    }
                }
                
                # Export buttons
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    # JSON export
                    json_str = json.dumps(api610_export_data, indent=2)
                    st.download_button(
                        label="📥 Download API 610 Specifications (JSON)",
                        data=json_str,
                        file_name=f"api610_specifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with export_col2:
                    # Create PDF report
                    def create_api610_pdf():
                        pdf_buffer = io.BytesIO()
                        # Create PDF content (simplified)
                        content = [
                            "API 610 Pump Specifications Report",
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "\nOperating Conditions:",
                            f"Flow Rate: {Q_design * 3600:.2f} m³/h",
                            f"Total Head: {total_head_design:.2f} m",
                            f"Temperature: {T:.1f}°C",
                            f"Pressure: {pressure_bar:.1f} bar",
                            "\nMaterial Specifications:",
                            f"Current Class: {material_upgrades['current_class']}",
                            f"Recommended Class: {material_upgrades['recommended_class']}",
                            "\nMechanical Seal:",
                            f"Arrangement: {seal_recommendation['arrangement']}",
                            f"API Plan: {seal_recommendation['api_plan']}"
                        ]
                        return '\n'.join(content).encode('utf-8')
                    
                    pdf_content = create_api610_pdf()
                    st.download_button(
                        label="📥 Download API 610 Report (PDF)",
                        data=pdf_content,
                        file_name=f"api610_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                # Installation & Alignment
                st.markdown("---")
                st.markdown("#### 🔧 Installation & Alignment Specifications")
                
                # Installation specs have been calculated earlier
                
                install_col1, install_col2 = st.columns(2)
                with install_col1:
                    st.write("**Foundation Requirements:**")
                    st.write(f"- Type: {installation_specs['foundation']['type']}")
                    st.write(f"- Min. mass: {installation_specs['foundation']['minimum_mass']:.1f} tonnes")
                    st.write(f"- Thickness: {installation_specs['foundation']['minimum_thickness']} mm")
                    st.write(f"- Reinforcement: {installation_specs['foundation']['reinforcement']}")
                    
                    st.write("\n**Grouting:**")
                    st.write(f"- Type: {installation_specs['grouting']['type']}")
                    st.write(f"- Thickness: {installation_specs['grouting']['thickness_mm']} mm")
                    st.write(f"- Cure time: {installation_specs['grouting']['cure_time_hours']} hours")
                
                with install_col2:
                    st.write("**Alignment Tolerances:**")
                    st.write("Cold alignment:")
                    st.write(f"- Parallel: {installation_specs['alignment']['cold_alignment']['parallel_mm']} mm")
                    st.write(f"- Angular: {installation_specs['alignment']['cold_alignment']['angular_mm_per_100mm']} mm/100mm")
                    
                    st.write("\n**Piping Support:**")
                    st.write("Suction pipe:")
                    st.write(f"- First support: {installation_specs['piping']['suction_support']['first_support_distance']} mm")
                    st.write("Discharge pipe:")
                    st.write(f"- First support: {installation_specs['piping']['discharge_support']['first_support_distance']} mm")

                st.markdown("---")
                # Continue with existing Affinity Laws Analysis
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
                    diagnostics_payload = prepare_diagnostics_payload(calc_context)
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
                    diagnostics_payload = prepare_diagnostics_payload(calc_context)
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
