#!/usr/bin/env python3
"""
Absolute minimal Isaac Sim test - just imports
"""
print("Starting minimal Isaac Sim test...")

# Isaac Sim imports ONLY  
from isaacsim.simulation_app import SimulationApp

print("Creating SimulationApp...")
simulation_app = SimulationApp({"headless": False})

print("Isaac Sim loaded successfully!")

import time
time.sleep(5)  # Keep open for 5 seconds

print("Closing...")
simulation_app.close()