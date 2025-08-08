# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:29:49 2024

@author: USER
"""
#methanol_flow: 10,000kW based hourly production = 864.9881513923024kg/h
#title: Design, Modelling, and Thermodynamic Analysis of a Novel Marine Power System Based on Methanol Solid Oxide Fuel
#       Cells, Integrated Proton Exchange Membrane Fuel Cells, and Combined Heat and Power Production

import numpy as np
import os

def SOFC(flow_rate):
    # Fixed parameters
    F = 96485  # Faraday's constant in C/mol
    n = 2  # Number of electrons transferred
    molar_mass_methanol = 32.04  # g/mol
    cell_voltage = 0.73  # V 
    current_density = 1400  # A/m^2 (i)
    sofc_efficiency = 0.52  
    converter_efficiency = 0.98  # DC/AC converter efficiency
    fuel_utilization_factor = 0.85  # 85% utilization
    reforming_efficiency = 0.95 
    # X_flow = 864.9881513923024  # kg/hr

    # Step 1: Convert methanol flow rate to mol/hr
    methanol_flow_mol_hr = flow_rate * 1000 / molar_mass_methanol  # mol/hr

    # Step 2: Calculate hydrogen production rate (mol/hr) from methanol reforming
    # Methanol steam reforming reaction: CH3OH + H2O → CO2 + 3H2 
    hydrogen_production_rate_mol_hr = 3 * methanol_flow_mol_hr *reforming_efficiency  # mol/hr

    # Step 3: Calculate hydrogen reacted (85% utilization)
    hydrogen_reacted_mol_hr = hydrogen_production_rate_mol_hr * fuel_utilization_factor  # mol/hr

    # Step 4: Calculate the total current produced by the SOFC 
    # 1A = 1C/s; total_current(I)
    total_current = hydrogen_reacted_mol_hr * n * F / 3600  # C/hr -> C/s = 1A

    # Step 5: Calculate the effective cell area (A)
    surface_area = total_current / current_density  # m^2

    # Step 6: Calculate the power output from the SOFC stack
    W_stack = current_density * surface_area * cell_voltage * converter_efficiency  # A * V = Watt

    # Step 7: Convert to kW and apply SOFC efficiency
    power_output_kw = W_stack / 1000  # kW
    net_power_output_kw = power_output_kw * sofc_efficiency

    return net_power_output_kw

def SOFC_output(target_net_power_output_kw):
    # Fixed parameters
    F = 96485  # Faraday's constant in C/mol
    n = 2  # Number of electrons transferred
    molar_mass_methanol = 32.04  # g/mol
    cell_voltage = 0.73  # V
    current_density = 1400  # A/m^2 (i)
    sofc_efficiency = 0.52
    converter_efficiency = 0.98  # DC/AC converter efficiency
    fuel_utilization_factor = 0.85  # 85% utilization
    reforming_efficiency = 0.95

    # Step 2: Calculate total current required to achieve target power output
    total_current = (
        target_net_power_output_kw * 1000
        / (cell_voltage * converter_efficiency * sofc_efficiency)
    )

    # Step 3: Calculate hydrogen reacted in mol/hr
    hydrogen_reacted_mol_hr = (total_current * 3600) / (n * F)

    # Step 4: Calculate hydrogen production rate in mol/hr
    hydrogen_production_rate_mol_hr = hydrogen_reacted_mol_hr / fuel_utilization_factor

    # Step 5: Calculate methanol flow in mol/hr
    methanol_flow_mol_hr = hydrogen_production_rate_mol_hr / (3 * reforming_efficiency)

    # Step 6: Convert methanol flow to kg/hr
    flow_rate = methanol_flow_mol_hr * molar_mass_methanol / 1000  # kg/hr

    return flow_rate
